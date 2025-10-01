#!/usr/bin/env python3
"""
Live Trading Manager for ForexSwing AI 2025
Integrates MT5 broker connectivity with AI trading system

Features:
- MT5 broker integration
- Real-time position monitoring
- AI recommendation execution
- Risk management enforcement
- Safety controls and circuit breakers
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from .mt5_connector import MT5Connector
from .mt5_config import MT5Config, MT5SafetyControls
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json

class LiveTradingManager:
    """Manages live trading operations with MT5 integration"""
    
    def __init__(self, config_file: str = "config/mt5_demo.json"):
        self.logger = logging.getLogger(__name__)
        
        # Initialize MT5 components
        self.config = MT5Config(config_file)
        self.safety_controls = MT5SafetyControls(self.config)
        self.mt5_connector = MT5Connector()
        
        # Trading state
        self.is_connected = False
        self.is_trading_enabled = False
        self.last_health_check = None
        
        # Performance tracking
        self.session_stats = {
            'trades_executed': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit': 0.0,
            'session_start': datetime.now()
        }
    
    def initialize(self) -> Tuple[bool, str]:
        """Initialize live trading connection"""
        try:
            # Get active account configuration
            active_account = self.config.get_active_account()
            if not active_account:
                return False, "No active MT5 account configured"
            
            # Initialize MT5 connection
            if not self.mt5_connector.initialize_connection():
                return False, "Failed to initialize MT5 connection"
            
            # Note: Authentication requires user credentials
            # For security, credentials should be entered at runtime
            self.logger.info("MT5 connection initialized - ready for authentication")
            
            self.is_connected = True
            return True, f"Live trading manager initialized (Account: {self.config.active_account})"
            
        except Exception as e:
            self.logger.error(f"Error initializing live trading: {e}")
            return False, f"Initialization error: {e}"
    
    def authenticate(self, password: str) -> Tuple[bool, str]:
        """Authenticate with MT5 using stored account configuration"""
        try:
            active_account = self.config.get_active_account()
            if not active_account:
                return False, "No active account configured"
            
            # Validate password against stored hash
            if not self.config.validate_password(self.config.active_account, password):
                return False, "Invalid password"
            
            # Authenticate with MT5
            success = self.mt5_connector.authenticate(
                login=active_account['login'],
                password=password,
                server=active_account['server']
            )
            
            if success:
                self.is_trading_enabled = True
                account_type = "DEMO" if active_account['is_demo'] else "LIVE"
                return True, f"Authenticated with {account_type} account {active_account['login']}"
            else:
                return False, "MT5 authentication failed"
                
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return False, f"Authentication error: {e}"
    
    def execute_ai_recommendation(self, recommendation: Dict) -> Tuple[bool, str, Dict]:
        """Execute an AI trading recommendation"""
        try:
            if not self.is_trading_enabled:
                return False, "Trading not enabled - authenticate first", {}
            
            # Extract recommendation details
            symbol = recommendation.get('symbol', 'EUR/USD')
            action = recommendation.get('action', 'BUY')
            confidence = recommendation.get('confidence', 0.0)
            entry_price = recommendation.get('entry_price')
            stop_loss = recommendation.get('stop_loss')
            take_profit = recommendation.get('take_profit')
            
            # Get account information for position sizing
            account_info = self.mt5_connector.get_account_info()
            if not account_info:
                return False, "Cannot retrieve account information", {}
            
            # Calculate position size based on risk management
            risk_amount = account_info['balance'] * 0.02  # 2% risk per trade
            
            # Safety check
            can_trade, safety_message = self.safety_controls.can_place_trade(
                risk_amount, account_info['balance']
            )
            
            if not can_trade:
                return False, f"Trade blocked by safety controls: {safety_message}", {}
            
            # Get current market price
            current_price_data = self.mt5_connector.get_live_price(symbol)
            if not current_price_data:
                return False, f"Cannot get current price for {symbol}", {}
            
            # Calculate position size based on stop loss distance
            if stop_loss:
                price = current_price_data['ask'] if action == 'BUY' else current_price_data['bid']
                stop_distance = abs(price - stop_loss)
                # Position size = Risk Amount / (Stop Distance * Point Value)
                # Simplified calculation - adjust based on your broker's specifications
                volume = round(risk_amount / (stop_distance * 100000), 2)  # Rough calculation
                volume = max(0.01, min(volume, 10.0))  # Limit volume between 0.01 and 10 lots
            else:
                volume = 0.01  # Minimum volume if no stop loss
            
            # Execute the trade
            order_result = self.mt5_connector.place_order(
                symbol=symbol,
                order_type=action,
                volume=volume,
                sl=stop_loss,
                tp=take_profit,
                comment=f"ForexSwing AI - {confidence:.1%} confidence"
            )
            
            if not order_result:
                self.session_stats['failed_trades'] += 1
                return False, "Order execution failed", {}
            
            if order_result.get('success'):
                # Record successful trade
                self.safety_controls.record_trade({
                    'success': True,
                    'risk_amount': risk_amount,
                    'symbol': symbol,
                    'volume': volume
                })
                
                self.session_stats['trades_executed'] += 1
                self.session_stats['successful_trades'] += 1
                
                result_data = {
                    'order_id': order_result['order_id'],
                    'symbol': symbol,
                    'action': action,
                    'volume': volume,
                    'entry_price': order_result['price'],
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': confidence,
                    'risk_amount': risk_amount
                }
                
                return True, f"Trade executed: {action} {volume} lots of {symbol}", result_data
            else:
                self.session_stats['failed_trades'] += 1
                return False, f"Order rejected: {order_result.get('error', 'Unknown error')}", {}
                
        except Exception as e:
            self.logger.error(f"Error executing AI recommendation: {e}")
            self.session_stats['failed_trades'] += 1
            return False, f"Execution error: {e}", {}
    
    def get_live_positions(self) -> List[Dict]:
        """Get all live trading positions"""
        try:
            if not self.is_trading_enabled:
                return []
            
            positions = self.mt5_connector.get_positions()
            
            # Enhance position data with additional info
            for position in positions:
                # Calculate current P&L percentage
                if position['price_open'] > 0:
                    if position['type'] == 'BUY':
                        pnl_pips = (position['price_current'] - position['price_open']) * 10000
                    else:
                        pnl_pips = (position['price_open'] - position['price_current']) * 10000
                    
                    position['pnl_pips'] = round(pnl_pips, 1)
                    position['pnl_percentage'] = (position['profit'] / (position['volume'] * position['price_open'])) * 100
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Error getting live positions: {e}")
            return []
    
    def close_position(self, ticket: int, reason: str = "Manual close") -> Tuple[bool, str]:
        """Close a specific position"""
        try:
            if not self.is_trading_enabled:
                return False, "Trading not enabled"
            
            result = self.mt5_connector.close_position(ticket)
            
            if result and result.get('success'):
                # Record position closure
                self.safety_controls.record_position_close(result.get('profit', 0.0))
                self.session_stats['total_profit'] += result.get('profit', 0.0)
                
                return True, f"Position {ticket} closed successfully"
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'Close operation failed'
                return False, f"Failed to close position {ticket}: {error_msg}"
                
        except Exception as e:
            self.logger.error(f"Error closing position {ticket}: {e}")
            return False, f"Close error: {e}"
    
    def get_trading_status(self) -> Dict:
        """Get comprehensive trading status"""
        try:
            account_info = self.mt5_connector.get_account_info() if self.is_trading_enabled else None
            safety_status = self.safety_controls.get_safety_status()
            positions = self.get_live_positions()
            
            return {
                'connection_status': {
                    'connected': self.is_connected,
                    'trading_enabled': self.is_trading_enabled,
                    'last_health_check': self.last_health_check
                },
                'account_info': account_info,
                'safety_status': safety_status,
                'positions': {
                    'count': len(positions),
                    'total_profit': sum(pos['profit'] for pos in positions),
                    'positions': positions
                },
                'session_stats': self.session_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error getting trading status: {e}")
            return {'error': str(e)}
    
    def emergency_stop(self, reason: str = "Manual emergency stop"):
        """Trigger emergency stop - halt all trading"""
        try:
            self.safety_controls.trigger_emergency_stop(reason)
            
            # Close all open positions
            positions = self.get_live_positions()
            for position in positions:
                try:
                    self.close_position(position['ticket'], f"Emergency stop: {reason}")
                except Exception as e:
                    self.logger.error(f"Error closing position during emergency stop: {e}")
            
            # Disable trading
            self.is_trading_enabled = False
            
            self.logger.critical(f"EMERGENCY STOP EXECUTED: {reason}")
            
        except Exception as e:
            self.logger.error(f"Error during emergency stop: {e}")
    
    def health_check(self) -> Dict:
        """Perform system health check"""
        try:
            self.last_health_check = datetime.now()
            
            health_status = {
                'timestamp': self.last_health_check,
                'mt5_connection': False,
                'account_accessible': False,
                'market_data': False,
                'safety_controls': True,
                'overall_status': 'UNKNOWN'
            }
            
            # Check MT5 connection
            if self.is_connected:
                health_status['mt5_connection'] = True
                
                # Check account access
                account_info = self.mt5_connector.get_account_info()
                if account_info:
                    health_status['account_accessible'] = True
                    
                    # Check market data
                    price_data = self.mt5_connector.get_live_price('EUR/USD')
                    if price_data:
                        health_status['market_data'] = True
            
            # Overall status
            if all([health_status['mt5_connection'], health_status['account_accessible'], 
                   health_status['market_data'], health_status['safety_controls']]):
                health_status['overall_status'] = 'HEALTHY'
            elif health_status['mt5_connection'] and health_status['safety_controls']:
                health_status['overall_status'] = 'DEGRADED'
            else:
                health_status['overall_status'] = 'CRITICAL'
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            return {
                'timestamp': datetime.now(),
                'overall_status': 'ERROR',
                'error': str(e)
            }
    
    def disconnect(self):
        """Safely disconnect from MT5"""
        try:
            if self.is_trading_enabled:
                self.logger.warning("Disconnecting while trading is enabled")
            
            self.mt5_connector.disconnect()
            self.is_connected = False
            self.is_trading_enabled = False
            
            self.logger.info("Live trading manager disconnected")
            
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")

# Demo and testing functions
def demo_setup():
    """Setup demo trading environment"""
    try:
        # Setup demo configuration
        from mt5_config import setup_demo_account
        config = setup_demo_account()
        
        # Initialize trading manager
        manager = LiveTradingManager("config/mt5_demo.json")
        success, message = manager.initialize()
        
        if success:
            print(f"✅ {message}")
            print("📋 Live trading manager ready for demo trading")
            print("⚠️  Replace demo credentials with your actual MT5 demo account")
            return manager
        else:
            print(f"❌ {message}")
            return None
            
    except Exception as e:
        print(f"❌ Demo setup error: {e}")
        return None

if __name__ == "__main__":
    demo_setup()