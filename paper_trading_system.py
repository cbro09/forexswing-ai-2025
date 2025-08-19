#!/usr/bin/env python3
"""
ForexSwing AI - Paper Trading System
Risk-free testing environment for strategy validation
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import time

@dataclass
class Position:
    """Represents a trading position"""
    id: str
    pair: str
    side: str  # 'BUY' or 'SELL'
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    status: str = 'OPEN'  # 'OPEN', 'CLOSED'
    
    def update_pnl(self, current_price: float):
        """Update unrealized P&L based on current price"""
        self.current_price = current_price
        
        if self.side == 'BUY':
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
        else:  # SELL
            self.unrealized_pnl = (self.entry_price - current_price) * self.size

@dataclass
class Trade:
    """Represents a completed trade"""
    id: str
    pair: str
    side: str
    size: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    commission: float
    reason: str  # 'MANUAL', 'STOP_LOSS', 'TAKE_PROFIT', 'SIGNAL'

class PaperTradingAccount:
    """Paper trading account with portfolio management"""
    
    def __init__(self, initial_balance: float = 10000.0, leverage: int = 1):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.positions = {}  # position_id -> Position
        self.trade_history = []  # List of completed trades
        self.equity = initial_balance
        self.margin_used = 0.0
        self.free_margin = initial_balance
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.max_equity = initial_balance
        
        # Risk settings
        self.max_position_size = 0.1  # 10% of balance per trade
        self.commission_rate = 0.0001  # 0.01% commission
        
        print(f"Paper Trading Account initialized:")
        print(f"  Initial Balance: ${initial_balance:,.2f}")
        print(f"  Leverage: {leverage}:1")
        print(f"  Max Position Size: {self.max_position_size:.1%} of balance")
    
    def get_position_size(self, pair: str, confidence: float) -> float:
        """Calculate position size based on confidence and risk management"""
        # Base position size on confidence
        confidence_multiplier = max(0.5, min(1.5, confidence))
        
        # Calculate position size
        position_value = self.balance * self.max_position_size * confidence_multiplier
        
        # Convert to units (simplified - assumes 1 unit = 1 USD equivalent)
        position_size = position_value
        
        return round(position_size, 2)
    
    def open_position(self, pair: str, side: str, current_price: float, confidence: float, 
                     stop_loss_pips: int = 50, take_profit_pips: int = 100) -> Optional[str]:
        """Open a new position"""
        
        # Calculate position size
        size = self.get_position_size(pair, confidence)
        
        # Check if we have enough free margin
        margin_required = size / self.leverage
        if margin_required > self.free_margin:
            print(f"[REJECTED] Insufficient margin. Required: ${margin_required:.2f}, Available: ${self.free_margin:.2f}")
            return None
        
        # Create position
        position_id = f"{pair}_{side}_{int(time.time())}"
        
        # Calculate stop loss and take profit
        pip_value = 0.0001  # Simplified pip value
        
        if side == 'BUY':
            stop_loss = current_price - (stop_loss_pips * pip_value) if stop_loss_pips else None
            take_profit = current_price + (take_profit_pips * pip_value) if take_profit_pips else None
        else:  # SELL
            stop_loss = current_price + (stop_loss_pips * pip_value) if stop_loss_pips else None
            take_profit = current_price - (take_profit_pips * pip_value) if take_profit_pips else None
        
        position = Position(
            id=position_id,
            pair=pair,
            side=side,
            size=size,
            entry_price=current_price,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Update account
        self.positions[position_id] = position
        self.margin_used += margin_required
        self.free_margin -= margin_required
        
        commission = size * self.commission_rate
        self.balance -= commission
        
        print(f"[OPENED] {side} {size:,.0f} units of {pair} at {current_price:.5f}")
        print(f"  Position ID: {position_id}")
        print(f"  Stop Loss: {stop_loss:.5f}" if stop_loss else "  No Stop Loss")
        print(f"  Take Profit: {take_profit:.5f}" if take_profit else "  No Take Profit")
        print(f"  Commission: ${commission:.2f}")
        
        return position_id
    
    def close_position(self, position_id: str, current_price: float, reason: str = "MANUAL") -> bool:
        """Close an existing position"""
        
        if position_id not in self.positions:
            print(f"[ERROR] Position {position_id} not found")
            return False
        
        position = self.positions[position_id]
        
        # Calculate P&L
        if position.side == 'BUY':
            pnl = (current_price - position.entry_price) * position.size
        else:  # SELL
            pnl = (position.entry_price - current_price) * position.size
        
        # Calculate commission
        commission = position.size * self.commission_rate
        net_pnl = pnl - commission
        
        # Update account
        self.balance += net_pnl
        margin_released = position.size / self.leverage
        self.margin_used -= margin_released
        self.free_margin += margin_released
        
        # Create trade record
        trade = Trade(
            id=position.id,
            pair=position.pair,
            side=position.side,
            size=position.size,
            entry_price=position.entry_price,
            exit_price=current_price,
            entry_time=position.entry_time,
            exit_time=datetime.now(),
            pnl=net_pnl,
            commission=commission,
            reason=reason
        )
        
        self.trade_history.append(trade)
        
        # Update statistics
        self.total_trades += 1
        self.total_pnl += net_pnl
        
        if net_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Remove position
        del self.positions[position_id]
        
        print(f"[CLOSED] {position.side} {position.size:,.0f} units of {position.pair}")
        print(f"  Entry: {position.entry_price:.5f} -> Exit: {current_price:.5f}")
        print(f"  P&L: ${net_pnl:+.2f} (Reason: {reason})")
        
        return True
    
    def update_positions(self, price_data: Dict[str, float]):
        """Update all positions with current prices"""
        
        positions_to_close = []
        
        for position_id, position in self.positions.items():
            if position.pair in price_data:
                current_price = price_data[position.pair]
                position.update_pnl(current_price)
                
                # Check stop loss and take profit
                if position.stop_loss and position.side == 'BUY' and current_price <= position.stop_loss:
                    positions_to_close.append((position_id, current_price, 'STOP_LOSS'))
                elif position.stop_loss and position.side == 'SELL' and current_price >= position.stop_loss:
                    positions_to_close.append((position_id, current_price, 'STOP_LOSS'))
                elif position.take_profit and position.side == 'BUY' and current_price >= position.take_profit:
                    positions_to_close.append((position_id, current_price, 'TAKE_PROFIT'))
                elif position.take_profit and position.side == 'SELL' and current_price <= position.take_profit:
                    positions_to_close.append((position_id, current_price, 'TAKE_PROFIT'))
        
        # Close positions that hit stop loss or take profit
        for position_id, price, reason in positions_to_close:
            self.close_position(position_id, price, reason)
        
        # Update equity
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        self.equity = self.balance + unrealized_pnl
        
        # Update drawdown
        if self.equity > self.max_equity:
            self.max_equity = self.equity
        
        current_drawdown = (self.max_equity - self.equity) / self.max_equity
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
    
    def get_account_summary(self) -> Dict:
        """Get comprehensive account summary"""
        
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            'balance': self.balance,
            'equity': self.equity,
            'unrealized_pnl': unrealized_pnl,
            'margin_used': self.margin_used,
            'free_margin': self.free_margin,
            'total_pnl': self.total_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'max_drawdown': self.max_drawdown * 100,
            'open_positions': len(self.positions),
            'return_pct': ((self.equity - self.initial_balance) / self.initial_balance) * 100
        }
    
    def print_account_status(self):
        """Print formatted account status"""
        summary = self.get_account_summary()
        
        print(f"\n" + "=" * 50)
        print("PAPER TRADING ACCOUNT STATUS")
        print("=" * 50)
        print(f"Balance: ${summary['balance']:,.2f}")
        print(f"Equity: ${summary['equity']:,.2f}")
        print(f"Unrealized P&L: ${summary['unrealized_pnl']:+.2f}")
        print(f"Free Margin: ${summary['free_margin']:,.2f}")
        print(f"Open Positions: {summary['open_positions']}")
        print()
        print(f"Total P&L: ${summary['total_pnl']:+.2f}")
        print(f"Total Return: {summary['return_pct']:+.2f}%")
        print(f"Total Trades: {summary['total_trades']}")
        print(f"Win Rate: {summary['win_rate']:.1f}%")
        print(f"Max Drawdown: {summary['max_drawdown']:.2f}%")
    
    def save_state(self, filename: str = "paper_trading_state.json"):
        """Save account state to file"""
        state = {
            'account': self.get_account_summary(),
            'positions': [asdict(pos) for pos in self.positions.values()],
            'trades': [asdict(trade) for trade in self.trade_history]
        }
        
        # Convert datetime objects to strings
        for trade in state['trades']:
            trade['entry_time'] = trade['entry_time'].isoformat()
            trade['exit_time'] = trade['exit_time'].isoformat()
        
        for pos in state['positions']:
            pos['entry_time'] = pos['entry_time'].isoformat()
        
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"Account state saved to {filename}")

class PaperTradingBot:
    """Paper trading bot that integrates ForexBot with paper trading account"""
    
    def __init__(self, initial_balance: float = 10000.0):
        from ForexBot import ForexBot
        
        self.forex_bot = ForexBot()
        self.account = PaperTradingAccount(initial_balance)
        self.active_pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY']
        
        print(f"\nPaper Trading Bot initialized")
        print(f"Monitoring pairs: {', '.join(self.active_pairs)}")
    
    def simulate_trading_day(self, market_data: Dict[str, pd.DataFrame]):
        """Simulate a full trading day with the bot"""
        
        print(f"\n" + "=" * 60)
        print(f"SIMULATING TRADING DAY - {datetime.now().strftime('%Y-%m-%d')}")
        print("=" * 60)
        
        for pair_name, pair_data in market_data.items():
            if len(pair_data) < 100:
                print(f"[SKIP] {pair_name}: Insufficient data")
                continue
            
            print(f"\nAnalyzing {pair_name}...")
            
            # Get bot recommendation
            try:
                recommendation = self.forex_bot.get_final_recommendation(pair_data, pair_name)
                action = recommendation['action']
                confidence = recommendation['confidence']
                
                print(f"Bot recommendation: {action} (confidence: {confidence:.1%})")
                
                # Get current price (last close price)
                current_price = float(pair_data['close'].iloc[-1])
                
                # Check if we have an existing position for this pair
                existing_position = None
                for pos_id, pos in self.account.positions.items():
                    if pos.pair == pair_name:
                        existing_position = pos
                        break
                
                # Trading logic
                if action == 'BUY' and confidence > 0.6:
                    if existing_position and existing_position.side == 'SELL':
                        # Close opposing position
                        self.account.close_position(existing_position.id, current_price, 'SIGNAL')
                    
                    if not existing_position or existing_position.side != 'BUY':
                        # Open new BUY position
                        self.account.open_position(pair_name, 'BUY', current_price, confidence)
                
                elif action == 'SELL' and confidence > 0.6:
                    if existing_position and existing_position.side == 'BUY':
                        # Close opposing position
                        self.account.close_position(existing_position.id, current_price, 'SIGNAL')
                    
                    if not existing_position or existing_position.side != 'SELL':
                        # Open new SELL position
                        self.account.open_position(pair_name, 'SELL', current_price, confidence)
                
                elif action == 'HOLD' and existing_position:
                    # Consider closing if confidence is low
                    if confidence < 0.4:
                        self.account.close_position(existing_position.id, current_price, 'LOW_CONFIDENCE')
                
                # Update positions with current price
                self.account.update_positions({pair_name: current_price})
                
            except Exception as e:
                print(f"[ERROR] Failed to process {pair_name}: {e}")
        
        # Print account status
        self.account.print_account_status()
        
        return self.account.get_account_summary()

def run_paper_trading_demo():
    """Run a paper trading demonstration"""
    
    print("FOREXSWING AI - PAPER TRADING DEMONSTRATION")
    print("=" * 60)
    
    # Initialize paper trading bot
    bot = PaperTradingBot(initial_balance=10000.0)
    
    # Load market data
    market_data = {}
    pairs = [
        ('EUR/USD', 'EUR_USD_real_daily.csv'),
        ('GBP/USD', 'GBP_USD_real_daily.csv'),
        ('USD/JPY', 'USD_JPY_real_daily.csv')
    ]
    
    for pair_name, filename in pairs:
        file_path = f"data/MarketData/{filename}"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Use last 100 days for simulation
            market_data[pair_name] = df.tail(100)
            print(f"Loaded {len(df)} candles for {pair_name}")
        else:
            print(f"[WARNING] Data file not found: {filename}")
    
    if not market_data:
        print("[ERROR] No market data available for simulation")
        return
    
    # Run simulation
    results = bot.simulate_trading_day(market_data)
    
    # Save results
    bot.account.save_state("demo_trading_results.json")
    
    print(f"\n" + "=" * 60)
    print("PAPER TRADING DEMO COMPLETE")
    print("=" * 60)
    print(f"Final Balance: ${results['balance']:,.2f}")
    print(f"Total Return: {results['return_pct']:+.2f}%")
    print(f"Win Rate: {results['win_rate']:.1f}%")
    print(f"Results saved to: demo_trading_results.json")

if __name__ == "__main__":
    run_paper_trading_demo()