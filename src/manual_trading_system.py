#!/usr/bin/env python3
"""
Manual Trading System for ForexSwing AI
Provides detailed AI recommendations for manual execution in MT5

Features:
- Detailed trading signals with exact parameters
- Position sizing calculations for GBP account
- Risk management recommendations  
- Performance tracking of manual trades
- Step-by-step execution instructions
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
import os

class ManualTradingSystem:
    """Manual trading system with AI recommendations"""
    
    def __init__(self, account_balance: float = 1000.0, account_currency: str = "GBP"):
        self.account_balance = account_balance
        self.account_currency = account_currency
        self.max_risk_per_trade = 0.02  # 2%
        self.trades_log = []
        
        # Load existing trades log if available
        self.load_trades_log()
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                              stop_loss: float, risk_percentage: float = 0.02) -> Dict:
        """Calculate optimal position size based on risk management"""
        try:
            # Risk amount in account currency
            risk_amount = self.account_balance * risk_percentage
            
            # Calculate pip value for different symbols (more accurate)
            # Base pip value in USD for 1 standard lot
            if symbol in ['EUR/USD', 'GBP/USD', 'AUD/USD', 'NZD/USD']:
                # USD is quote currency - direct calculation
                pip_value_usd = 10.0  # $10 per pip for 1 lot
            elif symbol in ['USD/JPY', 'USD/CHF', 'USD/CAD']:
                # USD is base currency - inverse calculation
                pip_value_usd = 10.0 / entry_price if entry_price > 0 else 10.0
            else:
                # Cross currencies - simplified
                pip_value_usd = 10.0
            
            # Convert to GBP if account currency is GBP
            if self.account_currency == "GBP":
                # Rough GBP conversion rates (in real trading, use live rates)
                usd_gbp_rate = 0.79  # Approximate rate
                pip_value = pip_value_usd * usd_gbp_rate
            else:
                pip_value = pip_value_usd
            
            # Calculate stop loss distance in pips
            if 'JPY' in symbol:
                # JPY pairs: 1 pip = 0.01
                pip_distance = abs(entry_price - stop_loss) * 100
            else:
                # Other pairs: 1 pip = 0.0001
                pip_distance = abs(entry_price - stop_loss) * 10000
            
            # Position size calculation
            # Risk Amount = Position Size × Pip Value × Pip Distance
            # Therefore: Position Size = Risk Amount / (Pip Value × Pip Distance)
            
            if pip_distance > 0:
                position_size = risk_amount / (pip_value * pip_distance)
                
                # Convert to GBP if needed (simplified conversion)
                if self.account_currency == "GBP":
                    # Rough conversion - in real trading, use current exchange rates
                    gbp_conversion = {
                        'EUR/USD': 0.85, 'GBP/USD': 1.0, 'USD/JPY': 0.85,
                        'USD/CHF': 0.85, 'AUD/USD': 0.85, 'USD/CAD': 0.85, 'NZD/USD': 0.85
                    }
                    conversion_rate = gbp_conversion.get(symbol, 0.85)
                    position_size *= conversion_rate
                
                # Round to appropriate lot sizes
                position_size = round(position_size, 2)
                
                # Ensure minimum and maximum limits
                position_size = max(0.01, min(position_size, 10.0))
                
                return {
                    'position_size': position_size,
                    'risk_amount': risk_amount,
                    'pip_distance': round(pip_distance, 1),
                    'pip_value': pip_value,
                    'risk_percentage': risk_percentage * 100
                }
            else:
                return {
                    'position_size': 0.01,  # Minimum
                    'risk_amount': risk_amount,
                    'pip_distance': 0,
                    'pip_value': pip_value,
                    'risk_percentage': risk_percentage * 100,
                    'error': 'Invalid stop loss distance'
                }
                
        except Exception as e:
            return {
                'position_size': 0.01,
                'risk_amount': self.account_balance * risk_percentage,
                'error': str(e)
            }
    
    def generate_trading_signal(self, ai_recommendation: Dict) -> Dict:
        """Generate detailed trading signal from AI recommendation"""
        try:
            symbol = ai_recommendation.get('symbol', 'EUR/USD')
            action = ai_recommendation.get('action', 'BUY')
            confidence = ai_recommendation.get('confidence', 0.0)
            entry_price = ai_recommendation.get('entry_price')
            stop_loss = ai_recommendation.get('stop_loss')
            take_profit = ai_recommendation.get('take_profit')
            
            # Calculate position sizing
            if stop_loss and entry_price:
                position_calc = self.calculate_position_size(symbol, entry_price, stop_loss)
            else:
                position_calc = {'position_size': 0.01, 'risk_amount': 20.0}
            
            # Create detailed signal
            signal = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': action,
                'confidence': confidence,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_calc['position_size'],
                'risk_amount': position_calc.get('risk_amount', 20.0),
                'pip_distance': position_calc.get('pip_distance', 0),
                'risk_percentage': position_calc.get('risk_percentage', 2.0),
                'account_balance': self.account_balance,
                'currency': self.account_currency,
                'execution_instructions': self.generate_execution_instructions(
                    symbol, action, position_calc['position_size'], entry_price, stop_loss, take_profit
                )
            }
            
            return signal
            
        except Exception as e:
            return {'error': f"Signal generation error: {e}"}
    
    def generate_execution_instructions(self, symbol: str, action: str, 
                                      position_size: float, entry_price: float,
                                      stop_loss: float, take_profit: float) -> List[str]:
        """Generate step-by-step execution instructions for MT5"""
        
        instructions = [
            f"=== MANUAL EXECUTION INSTRUCTIONS ===",
            f"1. Open MetaTrader 5 terminal",
            f"2. Locate {symbol} in Market Watch",
            f"3. Right-click {symbol} → 'New Order' or press F9",
            f"4. Set Order Type: Market Execution",
            f"5. Set Volume: {position_size} lots",
            f"6. Click '{action}' button",
            f"7. Wait for order confirmation",
            f"8. Right-click the new position → 'Modify or Delete Order'",
        ]
        
        if stop_loss:
            instructions.append(f"9. Set Stop Loss: {stop_loss:.5f}")
        if take_profit:
            instructions.append(f"10. Set Take Profit: {take_profit:.5f}")
        
        instructions.extend([
            f"11. Click 'Modify' to apply stop loss and take profit",
            f"12. Monitor position in 'Trade' tab",
            f"13. Record trade result in ForexSwing AI app"
        ])
        
        return instructions
    
    def record_manual_trade(self, signal: Dict, executed: bool = True, 
                          actual_entry: float = None, notes: str = "") -> Dict:
        """Record a manually executed trade"""
        try:
            trade_record = {
                'id': len(self.trades_log) + 1,
                'timestamp': datetime.now().isoformat(),
                'signal': signal,
                'executed': executed,
                'actual_entry': actual_entry or signal.get('entry_price'),
                'notes': notes,
                'status': 'open' if executed else 'skipped'
            }
            
            self.trades_log.append(trade_record)
            self.save_trades_log()
            
            return {
                'success': True,
                'trade_id': trade_record['id'],
                'message': f"Trade recorded: {signal['action']} {signal['symbol']}"
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def update_trade_result(self, trade_id: int, exit_price: float, 
                           exit_time: datetime = None, profit_loss: float = None) -> Dict:
        """Update trade with exit results"""
        try:
            for trade in self.trades_log:
                if trade['id'] == trade_id:
                    trade['exit_price'] = exit_price
                    trade['exit_time'] = (exit_time or datetime.now()).isoformat()
                    trade['profit_loss'] = profit_loss
                    trade['status'] = 'closed'
                    
                    self.save_trades_log()
                    
                    return {
                        'success': True,
                        'message': f"Trade {trade_id} updated with exit results"
                    }
            
            return {'success': False, 'error': f"Trade {trade_id} not found"}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_performance_stats(self) -> Dict:
        """Calculate performance statistics"""
        try:
            if not self.trades_log:
                return {
                    'total_trades': 0,
                    'executed_trades': 0,
                    'win_rate': 0.0,
                    'total_profit': 0.0,
                    'average_trade': 0.0
                }
            
            executed_trades = [t for t in self.trades_log if t['executed']]
            closed_trades = [t for t in executed_trades if t.get('status') == 'closed']
            winning_trades = [t for t in closed_trades if t.get('profit_loss', 0) > 0]
            
            total_profit = sum(t.get('profit_loss', 0) for t in closed_trades)
            
            return {
                'total_signals': len(self.trades_log),
                'executed_trades': len(executed_trades),
                'closed_trades': len(closed_trades),
                'open_trades': len(executed_trades) - len(closed_trades),
                'winning_trades': len(winning_trades),
                'win_rate': (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0.0,
                'total_profit': total_profit,
                'average_trade': total_profit / len(closed_trades) if closed_trades else 0.0,
                'execution_rate': (len(executed_trades) / len(self.trades_log) * 100) if self.trades_log else 0.0
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get recent trades for display"""
        return sorted(self.trades_log, key=lambda x: x['timestamp'], reverse=True)[:limit]
    
    def save_trades_log(self):
        """Save trades log to file"""
        try:
            os.makedirs('data', exist_ok=True)
            with open('data/manual_trades.json', 'w') as f:
                json.dump(self.trades_log, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving trades log: {e}")
    
    def load_trades_log(self):
        """Load trades log from file"""
        try:
            if os.path.exists('data/manual_trades.json'):
                with open('data/manual_trades.json', 'r') as f:
                    self.trades_log = json.load(f)
        except Exception as e:
            print(f"Error loading trades log: {e}")
            self.trades_log = []

# Example usage and testing
def demo_manual_trading():
    """Demo the manual trading system"""
    print("=== FOREXSWING AI MANUAL TRADING DEMO ===")
    
    # Initialize system
    manual_system = ManualTradingSystem(account_balance=1000.0, account_currency="GBP")
    
    # Example AI recommendation
    ai_recommendation = {
        'symbol': 'EUR/USD',
        'action': 'BUY',
        'confidence': 0.678,
        'entry_price': 1.08450,
        'stop_loss': 1.08200,   # 25 pips
        'take_profit': 1.08950  # 50 pips
    }
    
    # Generate trading signal
    signal = manual_system.generate_trading_signal(ai_recommendation)
    
    print(f"\n🤖 AI RECOMMENDATION:")
    print(f"Symbol: {signal['symbol']}")
    print(f"Action: {signal['action']}")
    print(f"Confidence: {signal['confidence']:.1%}")
    print(f"Entry: {signal['entry_price']:.5f}")
    print(f"Stop Loss: {signal['stop_loss']:.5f}")
    print(f"Take Profit: {signal['take_profit']:.5f}")
    print(f"Position Size: {signal['position_size']} lots")
    print(f"Risk: £{signal['risk_amount']:.2f} ({signal['risk_percentage']:.1f}%)")
    
    print(f"\n📋 EXECUTION INSTRUCTIONS:")
    for instruction in signal['execution_instructions']:
        print(instruction)
    
    # Record the trade
    result = manual_system.record_manual_trade(signal, executed=True, notes="Demo trade")
    print(f"\n✅ Trade recorded: {result}")
    
    # Show performance stats
    stats = manual_system.get_performance_stats()
    print(f"\n📊 Performance Stats:")
    print(f"Total Signals: {stats['total_signals']}")
    print(f"Executed Trades: {stats['executed_trades']}")
    print(f"Execution Rate: {stats['execution_rate']:.1f}%")

if __name__ == "__main__":
    demo_manual_trading()