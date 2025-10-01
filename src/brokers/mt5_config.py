#!/usr/bin/env python3
"""
MetaTrader 5 Configuration and Safety Controls
Secure credential management and risk controls for live trading
"""

import json
import os
from typing import Dict, Optional, Tuple
import logging
from dataclasses import dataclass
import hashlib
from datetime import datetime

@dataclass
class MT5Account:
    """MT5 account configuration"""
    login: int
    server: str
    name: str
    is_demo: bool = True
    max_risk_per_trade: float = 0.02  # 2% max risk per trade
    max_daily_loss: float = 0.05      # 5% max daily loss
    max_open_positions: int = 5        # Max simultaneous positions
    
class MT5Config:
    """MT5 configuration and credential management"""
    
    def __init__(self, config_file: str = "config/mt5_accounts.json"):
        self.config_file = config_file
        self.logger = logging.getLogger(__name__)
        self.accounts = {}
        self.active_account = None
        
        # Ensure config directory exists
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        # Load existing configuration
        self.load_config()
    
    def add_account(self, account_name: str, login: int, password: str, 
                   server: str, is_demo: bool = True,
                   max_risk_per_trade: float = 0.02,
                   max_daily_loss: float = 0.05,
                   max_open_positions: int = 5) -> bool:
        """Add a new MT5 account configuration"""
        try:
            # Hash password for storage (basic security)
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            account_config = {
                'login': login,
                'password_hash': password_hash,
                'server': server,
                'is_demo': is_demo,
                'max_risk_per_trade': max_risk_per_trade,
                'max_daily_loss': max_daily_loss,
                'max_open_positions': max_open_positions,
                'created_at': str(datetime.now()),
                'active': False
            }
            
            self.accounts[account_name] = account_config
            self.save_config()
            
            self.logger.info(f"Added MT5 account: {account_name} ({'Demo' if is_demo else 'Live'})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding account {account_name}: {e}")
            return False
    
    def get_account(self, account_name: str) -> Optional[Dict]:
        """Get account configuration by name"""
        return self.accounts.get(account_name)
    
    def set_active_account(self, account_name: str) -> bool:
        """Set the active trading account"""
        if account_name not in self.accounts:
            self.logger.error(f"Account {account_name} not found")
            return False
        
        # Deactivate all accounts
        for name in self.accounts:
            self.accounts[name]['active'] = False
        
        # Activate selected account
        self.accounts[account_name]['active'] = True
        self.active_account = account_name
        
        self.save_config()
        self.logger.info(f"Set active account: {account_name}")
        return True
    
    def get_active_account(self) -> Optional[Dict]:
        """Get the currently active account configuration"""
        if self.active_account:
            return self.accounts.get(self.active_account)
        
        # Find active account
        for name, config in self.accounts.items():
            if config.get('active', False):
                self.active_account = name
                return config
        
        return None
    
    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self.accounts = data.get('accounts', {})
                    
                # Find active account
                for name, config in self.accounts.items():
                    if config.get('active', False):
                        self.active_account = name
                        break
                        
                self.logger.info(f"Loaded {len(self.accounts)} MT5 account configurations")
            else:
                self.logger.info("No existing MT5 configuration found")
                
        except Exception as e:
            self.logger.error(f"Error loading MT5 configuration: {e}")
    
    def save_config(self):
        """Save configuration to file"""
        try:
            config_data = {
                'accounts': self.accounts,
                'last_updated': str(datetime.now())
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            self.logger.debug("MT5 configuration saved")
            
        except Exception as e:
            self.logger.error(f"Error saving MT5 configuration: {e}")
    
    def validate_password(self, account_name: str, password: str) -> bool:
        """Validate password for an account"""
        try:
            account = self.accounts.get(account_name)
            if not account:
                return False
            
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            return account['password_hash'] == password_hash
            
        except Exception as e:
            self.logger.error(f"Error validating password: {e}")
            return False

class MT5SafetyControls:
    """Safety controls and circuit breakers for live trading"""
    
    def __init__(self, config: MT5Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.daily_stats = {}
        self.reset_daily_stats()
    
    def reset_daily_stats(self):
        """Reset daily trading statistics"""
        from datetime import datetime
        today = datetime.now().strftime('%Y-%m-%d')
        
        if self.daily_stats.get('date') != today:
            self.daily_stats = {
                'date': today,
                'trades_executed': 0,
                'total_risk': 0.0,
                'realized_pnl': 0.0,
                'max_drawdown': 0.0,
                'positions_opened': 0,
                'positions_closed': 0,
                'emergency_stop_triggered': False
            }
    
    def can_place_trade(self, risk_amount: float, account_balance: float) -> Tuple[bool, str]:
        """Check if a trade can be placed based on safety rules"""
        try:
            self.reset_daily_stats()
            active_account = self.config.get_active_account()
            
            if not active_account:
                return False, "No active trading account configured"
            
            # Check if demo account for live trading
            if not active_account['is_demo']:
                self.logger.warning("Live account trading - extra safety checks")
            
            # Risk per trade limit
            risk_percent = risk_amount / account_balance
            max_risk = active_account['max_risk_per_trade']
            
            if risk_percent > max_risk:
                return False, f"Trade risk {risk_percent:.2%} exceeds limit {max_risk:.2%}"
            
            # Daily loss limit
            daily_loss_percent = abs(self.daily_stats['realized_pnl']) / account_balance
            max_daily_loss = active_account['max_daily_loss']
            
            if daily_loss_percent > max_daily_loss:
                return False, f"Daily loss {daily_loss_percent:.2%} exceeds limit {max_daily_loss:.2%}"
            
            # Maximum open positions
            if self.daily_stats['positions_opened'] >= active_account['max_open_positions']:
                return False, f"Maximum open positions ({active_account['max_open_positions']}) reached"
            
            # Emergency stop check
            if self.daily_stats['emergency_stop_triggered']:
                return False, "Emergency stop is active - trading halted"
            
            return True, "Trade approved"
            
        except Exception as e:
            self.logger.error(f"Error in trade safety check: {e}")
            return False, f"Safety check error: {e}"
    
    def record_trade(self, trade_result: Dict):
        """Record trade execution for monitoring"""
        try:
            self.reset_daily_stats()
            
            self.daily_stats['trades_executed'] += 1
            
            if trade_result.get('success'):
                self.daily_stats['positions_opened'] += 1
                
                # Record risk amount if provided
                if 'risk_amount' in trade_result:
                    self.daily_stats['total_risk'] += trade_result['risk_amount']
            
            self.logger.info(f"Recorded trade: {self.daily_stats['trades_executed']} trades today")
            
        except Exception as e:
            self.logger.error(f"Error recording trade: {e}")
    
    def record_position_close(self, profit: float):
        """Record position closure and P&L"""
        try:
            self.reset_daily_stats()
            
            self.daily_stats['positions_closed'] += 1
            self.daily_stats['realized_pnl'] += profit
            
            # Update max drawdown if negative
            if profit < 0:
                self.daily_stats['max_drawdown'] = min(
                    self.daily_stats['max_drawdown'], 
                    self.daily_stats['realized_pnl']
                )
            
            self.logger.info(f"Position closed: P&L {profit:.2f}, Daily P&L: {self.daily_stats['realized_pnl']:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error recording position close: {e}")
    
    def trigger_emergency_stop(self, reason: str = "Manual trigger"):
        """Trigger emergency stop - halt all trading"""
        self.daily_stats['emergency_stop_triggered'] = True
        self.logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
        
        # TODO: Close all open positions immediately
        # TODO: Cancel all pending orders
        # TODO: Send notifications
    
    def reset_emergency_stop(self):
        """Reset emergency stop (manual intervention required)"""
        self.daily_stats['emergency_stop_triggered'] = False
        self.logger.warning("Emergency stop has been reset - trading can resume")
    
    def get_safety_status(self) -> Dict:
        """Get current safety and risk status"""
        self.reset_daily_stats()
        active_account = self.config.get_active_account()
        
        return {
            'active_account': self.config.active_account if active_account else None,
            'is_demo': active_account['is_demo'] if active_account else None,
            'daily_trades': self.daily_stats['trades_executed'],
            'open_positions': self.daily_stats['positions_opened'] - self.daily_stats['positions_closed'],
            'daily_pnl': self.daily_stats['realized_pnl'],
            'max_drawdown': self.daily_stats['max_drawdown'],
            'emergency_stop': self.daily_stats['emergency_stop_triggered'],
            'risk_limits': {
                'max_risk_per_trade': active_account['max_risk_per_trade'] if active_account else None,
                'max_daily_loss': active_account['max_daily_loss'] if active_account else None,
                'max_positions': active_account['max_open_positions'] if active_account else None
            } if active_account else None
        }

# Demo configuration setup
def setup_demo_account():
    """Setup a demo MT5 account for testing"""
    from datetime import datetime
    
    config = MT5Config("config/mt5_demo.json")
    
    # Add demo account (these are example credentials)
    success = config.add_account(
        account_name="Demo_Account_1",
        login=51234567,  # Demo login
        password="demo_password_123",
        server="MetaQuotes-Demo",
        is_demo=True,
        max_risk_per_trade=0.02,  # 2%
        max_daily_loss=0.05,      # 5%
        max_open_positions=3
    )
    
    if success:
        config.set_active_account("Demo_Account_1")
        print("✅ Demo MT5 account configured")
        print("⚠️  Replace with your actual MT5 demo account credentials")
    else:
        print("❌ Failed to setup demo account")
    
    return config

if __name__ == "__main__":
    setup_demo_account()