#!/usr/bin/env python3
"""
MetaTrader 5 Broker Connector for ForexSwing AI 2025
Handles live trading operations through MT5 Python API

Features:
- Account authentication and management
- Real-time market data
- Order placement and management
- Position monitoring
- Risk management integration
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import time
import json

class MT5Connector:
    """MetaTrader 5 broker connectivity and trading operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.connected = False
        self.account_info = None
        self.login_credentials = None
        
    def initialize_connection(self) -> bool:
        """Initialize connection to MetaTrader 5 terminal"""
        try:
            if not mt5.initialize():
                error_code = mt5.last_error()
                self.logger.error(f"MT5 initialization failed: {error_code}")
                return False
                
            self.connected = True
            self.logger.info("MT5 connection initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 initialization error: {e}")
            return False
    
    def authenticate(self, login: int, password: str, server: str) -> bool:
        """Authenticate with MT5 trading account"""
        try:
            if not self.connected:
                if not self.initialize_connection():
                    return False
            
            # Attempt login
            if not mt5.login(login, password=password, server=server):
                error_code = mt5.last_error()
                self.logger.error(f"MT5 login failed: {error_code}")
                return False
            
            # Store credentials for reconnection
            self.login_credentials = {
                'login': login,
                'password': password,
                'server': server
            }
            
            # Get account information
            self.account_info = mt5.account_info()
            if self.account_info is None:
                self.logger.error("Failed to retrieve account information")
                return False
                
            self.logger.info(f"Successfully logged in to account {login}")
            self.logger.info(f"Account balance: {self.account_info.balance}")
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 authentication error: {e}")
            return False
    
    def get_account_info(self) -> Dict:
        """Get current account information"""
        try:
            if not self.connected:
                return None
                
            account = mt5.account_info()
            if account is None:
                return None
                
            return {
                'login': account.login,
                'trade_mode': account.trade_mode,
                'balance': account.balance,
                'equity': account.equity,
                'margin': account.margin,
                'free_margin': account.margin_free,
                'margin_level': account.margin_level,
                'profit': account.profit,
                'currency': account.currency,
                'server': account.server,
                'company': account.company
            }
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return None
    
    def get_live_price(self, symbol: str) -> Dict:
        """Get current live price for a symbol"""
        try:
            # Convert forex pair format (EUR/USD -> EURUSD)
            mt5_symbol = symbol.replace('/', '')
            
            tick = mt5.symbol_info_tick(mt5_symbol)
            if tick is None:
                self.logger.warning(f"No tick data for {symbol}")
                return None
                
            return {
                'symbol': symbol,
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'time': datetime.fromtimestamp(tick.time),
                'spread': tick.ask - tick.bid
            }
            
        except Exception as e:
            self.logger.error(f"Error getting live price for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, timeframe: str = "1D", count: int = 100) -> pd.DataFrame:
        """Get historical price data"""
        try:
            mt5_symbol = symbol.replace('/', '')
            
            # Map timeframes
            timeframe_map = {
                '1M': mt5.TIMEFRAME_M1,
                '5M': mt5.TIMEFRAME_M5,
                '15M': mt5.TIMEFRAME_M15,
                '30M': mt5.TIMEFRAME_M30,
                '1H': mt5.TIMEFRAME_H1,
                '4H': mt5.TIMEFRAME_H4,
                '1D': mt5.TIMEFRAME_D1
            }
            
            mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_D1)
            
            rates = mt5.copy_rates_from_pos(mt5_symbol, mt5_timeframe, 0, count)
            if rates is None:
                self.logger.warning(f"No historical data for {symbol}")
                return None
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            return df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
            
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
    
    def place_order(self, symbol: str, order_type: str, volume: float, 
                   price: float = None, sl: float = None, tp: float = None,
                   comment: str = "ForexSwing AI") -> Dict:
        """Place a trading order"""
        try:
            mt5_symbol = symbol.replace('/', '')
            
            # Check if symbol is available
            symbol_info = mt5.symbol_info(mt5_symbol)
            if symbol_info is None:
                self.logger.error(f"Symbol {mt5_symbol} not found")
                return None
            
            # Enable symbol in Market Watch if needed
            if not symbol_info.visible:
                if not mt5.symbol_select(mt5_symbol, True):
                    self.logger.error(f"Failed to enable symbol {mt5_symbol}")
                    return None
            
            # Map order types
            order_type_map = {
                'BUY': mt5.ORDER_TYPE_BUY,
                'SELL': mt5.ORDER_TYPE_SELL,
                'BUY_LIMIT': mt5.ORDER_TYPE_BUY_LIMIT,
                'SELL_LIMIT': mt5.ORDER_TYPE_SELL_LIMIT,
                'BUY_STOP': mt5.ORDER_TYPE_BUY_STOP,
                'SELL_STOP': mt5.ORDER_TYPE_SELL_STOP
            }
            
            mt5_order_type = order_type_map.get(order_type.upper())
            if mt5_order_type is None:
                self.logger.error(f"Invalid order type: {order_type}")
                return None
            
            # Get current price if not specified
            if price is None:
                tick = mt5.symbol_info_tick(mt5_symbol)
                if tick is None:
                    self.logger.error(f"Cannot get current price for {mt5_symbol}")
                    return None
                price = tick.ask if order_type.upper() == 'BUY' else tick.bid
            
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": mt5_symbol,
                "volume": volume,
                "type": mt5_order_type,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Add stop loss and take profit if specified
            if sl is not None:
                request["sl"] = sl
            if tp is not None:
                request["tp"] = tp
            
            # Send order
            result = mt5.order_send(request)
            
            if result is None:
                self.logger.error("Order send failed - no result")
                return None
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Order failed: {result.comment} (code: {result.retcode})")
                return {
                    'success': False,
                    'error': result.comment,
                    'retcode': result.retcode
                }
            
            self.logger.info(f"Order placed successfully: {result.order}")
            
            return {
                'success': True,
                'order_id': result.order,
                'deal_id': result.deal,
                'volume': result.volume,
                'price': result.price,
                'retcode': result.retcode,
                'comment': result.comment
            }
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None
    
    def get_positions(self) -> List[Dict]:
        """Get all open positions"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            position_list = []
            for pos in positions:
                position_list.append({
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL',
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'price_current': pos.price_current,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'profit': pos.profit,
                    'swap': pos.swap,
                    'comment': pos.comment,
                    'time': datetime.fromtimestamp(pos.time),
                    'magic': pos.magic
                })
            
            return position_list
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
    
    def close_position(self, ticket: int) -> Dict:
        """Close a specific position"""
        try:
            positions = mt5.positions_get(ticket=ticket)
            if positions is None or len(positions) == 0:
                self.logger.error(f"Position {ticket} not found")
                return None
            
            position = positions[0]
            
            # Determine opposite order type
            order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            
            # Get current price
            symbol_info = mt5.symbol_info(position.symbol)
            if symbol_info is None:
                self.logger.error(f"Cannot get symbol info for {position.symbol}")
                return None
            
            tick = mt5.symbol_info_tick(position.symbol)
            if tick is None:
                self.logger.error(f"Cannot get current price for {position.symbol}")
                return None
            
            price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
            
            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": order_type,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": "ForexSwing AI - Position Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send close order
            result = mt5.order_send(request)
            
            if result is None:
                self.logger.error("Position close failed - no result")
                return None
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Position close failed: {result.comment} (code: {result.retcode})")
                return {
                    'success': False,
                    'error': result.comment,
                    'retcode': result.retcode
                }
            
            self.logger.info(f"Position {ticket} closed successfully")
            
            return {
                'success': True,
                'deal_id': result.deal,
                'volume': result.volume,
                'price': result.price,
                'profit': position.profit
            }
            
        except Exception as e:
            self.logger.error(f"Error closing position {ticket}: {e}")
            return None
    
    def get_orders(self) -> List[Dict]:
        """Get all pending orders"""
        try:
            orders = mt5.orders_get()
            if orders is None:
                return []
            
            order_list = []
            for order in orders:
                order_list.append({
                    'ticket': order.ticket,
                    'symbol': order.symbol,
                    'type': order.type,
                    'volume_initial': order.volume_initial,
                    'volume_current': order.volume_current,
                    'price_open': order.price_open,
                    'sl': order.sl,
                    'tp': order.tp,
                    'time_setup': datetime.fromtimestamp(order.time_setup),
                    'comment': order.comment,
                    'magic': order.magic
                })
            
            return order_list
            
        except Exception as e:
            self.logger.error(f"Error getting orders: {e}")
            return []
    
    def cancel_order(self, ticket: int) -> Dict:
        """Cancel a pending order"""
        try:
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": ticket,
            }
            
            result = mt5.order_send(request)
            
            if result is None:
                return None
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    'success': False,
                    'error': result.comment,
                    'retcode': result.retcode
                }
            
            return {
                'success': True,
                'retcode': result.retcode
            }
            
        except Exception as e:
            self.logger.error(f"Error canceling order {ticket}: {e}")
            return None
    
    def disconnect(self):
        """Shutdown MT5 connection"""
        try:
            mt5.shutdown()
            self.connected = False
            self.logger.info("MT5 connection closed")
        except Exception as e:
            self.logger.error(f"Error disconnecting from MT5: {e}")

# Example usage and testing functions
def test_connection():
    """Test MT5 connection functionality"""
    mt5_connector = MT5Connector()
    
    print("Testing MT5 Connection...")
    
    # Initialize
    if not mt5_connector.initialize_connection():
        print("❌ Failed to initialize MT5 connection")
        return False
    
    print("✅ MT5 connection initialized")
    
    # Note: Actual login requires real credentials
    # print("Testing authentication...")
    # success = mt5_connector.authenticate(login=12345678, password="password", server="server-name")
    
    print("✅ MT5 connector ready for live trading")
    
    # Cleanup
    mt5_connector.disconnect()
    return True

if __name__ == "__main__":
    test_connection()