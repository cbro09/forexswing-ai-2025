# Broker API Integration Guide - Phase 4 Implementation

> **Objective**: Connect ForexSwing AI to live forex brokers for real-time trading

---

## 🎯 **Broker Selection Matrix**

### **Primary Candidates (Recommended)**

#### **1. MetaTrader 5 (MT5) API** ⭐⭐⭐
- **Pros**: Industry standard, extensive documentation, retail-friendly
- **Cons**: Limited programming languages, requires MT5 terminal
- **API Type**: Terminal-based API with Python integration
- **Live Trading**: ✅ Full support
- **Paper Trading**: ✅ Demo accounts available
- **Minimum Deposit**: $100-500 (varies by broker)
- **Implementation**: `pip install MetaTrader5`

```python
# MT5 Integration Example
import MetaTrader5 as mt5

class MT5Integration:
    def __init__(self):
        if not mt5.initialize():
            print("MT5 initialization failed")
        
    def place_order(self, symbol, order_type, volume, price=None):
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price or mt5.symbol_info_tick(symbol).ask,
            "deviation": 20,
            "magic": 234000,
            "comment": "ForexSwing AI Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        return result
```

#### **2. OANDA REST API** ⭐⭐⭐
- **Pros**: RESTful API, excellent documentation, institutional-grade
- **Cons**: Higher minimum deposits for live accounts
- **API Type**: HTTP REST API
- **Live Trading**: ✅ Full support
- **Paper Trading**: ✅ Practice accounts with virtual money
- **Minimum Deposit**: $0 (practice), $1000+ (live)
- **Implementation**: `pip install oandapyV20`

```python
# OANDA Integration Example
from oandapyV20 import API
from oandapyV20.endpoints import orders, trades, pricing

class OANDAIntegration:
    def __init__(self, api_key, account_id, environment="practice"):
        self.api = API(access_token=api_key, environment=environment)
        self.account_id = account_id
        
    def place_market_order(self, instrument, units):
        order_data = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "positionFill": "DEFAULT"
            }
        }
        
        r = orders.OrderCreate(self.account_id, data=order_data)
        response = self.api.request(r)
        return response
```

#### **3. Interactive Brokers (IBKR) API** ⭐⭐
- **Pros**: Professional-grade, low costs, extensive markets
- **Cons**: Complex setup, steep learning curve
- **API Type**: TWS API (Trader Workstation)
- **Live Trading**: ✅ Full support
- **Paper Trading**: ✅ Paper trading account
- **Minimum Deposit**: $0 (paper), $10,000+ (live)
- **Implementation**: `pip install ibapi`

### **Secondary Options**

#### **4. Alpaca Markets** ⭐⭐
- **Focus**: Primarily stocks, limited forex
- **API Type**: REST API
- **Good for**: Testing algorithmic trading concepts
- **Forex Support**: Limited (mostly major pairs)

#### **5. TD Ameritrade API** ⭐
- **Focus**: Primarily US markets
- **Forex Support**: Limited
- **Status**: Being acquired by Charles Schwab

---

## 🔧 **Implementation Architecture**

### **Broker Integration Layer**
```python
from abc import ABC, abstractmethod

class BrokerInterface(ABC):
    """Abstract base class for all broker integrations"""
    
    @abstractmethod
    def connect(self):
        """Establish connection to broker"""
        pass
    
    @abstractmethod
    def get_account_info(self):
        """Get account balance, equity, margin info"""
        pass
    
    @abstractmethod
    def get_positions(self):
        """Get current open positions"""
        pass
    
    @abstractmethod
    def place_market_order(self, symbol, quantity, side):
        """Place market order (BUY/SELL)"""
        pass
    
    @abstractmethod
    def place_limit_order(self, symbol, quantity, side, price):
        """Place limit order"""
        pass
    
    @abstractmethod
    def close_position(self, position_id):
        """Close specific position"""
        pass
    
    @abstractmethod
    def get_real_time_price(self, symbol):
        """Get real-time price data"""
        pass

class MT5Broker(BrokerInterface):
    """MetaTrader 5 implementation"""
    def connect(self):
        return mt5.initialize()
    
    def get_account_info(self):
        account_info = mt5.account_info()
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free
        }
    
    # ... implement other methods

class OANDABroker(BrokerInterface):
    """OANDA implementation"""
    def connect(self):
        # Test connection with account info request
        try:
            r = accounts.AccountDetails(self.account_id)
            self.api.request(r)
            return True
        except:
            return False
    
    # ... implement other methods
```

### **Risk Management Integration**
```python
class RiskManager:
    """Risk management for live trading"""
    
    def __init__(self, max_daily_loss=0.02, max_position_size=0.01):
        self.max_daily_loss = max_daily_loss  # 2% max daily loss
        self.max_position_size = max_position_size  # 1% account risk per trade
        self.daily_pnl = 0
        self.open_positions = {}
        
    def can_place_trade(self, account_balance, trade_risk):
        """Check if trade meets risk criteria"""
        
        # Check daily loss limit
        if self.daily_pnl <= -account_balance * self.max_daily_loss:
            return False, "Daily loss limit reached"
        
        # Check position size limit
        if trade_risk > account_balance * self.max_position_size:
            return False, "Position size too large"
        
        # Check maximum number of positions
        if len(self.open_positions) >= 5:
            return False, "Too many open positions"
            
        return True, "Trade approved"
    
    def calculate_position_size(self, account_balance, stop_loss_pips, pip_value):
        """Calculate position size based on risk management"""
        risk_amount = account_balance * self.max_position_size
        position_size = risk_amount / (stop_loss_pips * pip_value)
        return position_size
```

---

## 📊 **Real-Time Data Integration**

### **Price Data Feeds**
```python
class RealTimeDataFeed:
    """Real-time price data management"""
    
    def __init__(self, broker_api):
        self.broker_api = broker_api
        self.price_buffer = {}
        self.callbacks = []
        
    def subscribe_to_symbol(self, symbol, callback):
        """Subscribe to real-time price updates"""
        self.callbacks.append(callback)
        
        # Start price monitoring thread
        import threading
        thread = threading.Thread(
            target=self._price_monitor, 
            args=(symbol,), 
            daemon=True
        )
        thread.start()
    
    def _price_monitor(self, symbol):
        """Monitor price changes and trigger callbacks"""
        last_price = None
        
        while True:
            try:
                current_price = self.broker_api.get_real_time_price(symbol)
                
                if current_price != last_price:
                    # Price changed, notify callbacks
                    for callback in self.callbacks:
                        callback(symbol, current_price)
                    
                    last_price = current_price
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                print(f"Price monitoring error: {e}")
                time.sleep(5)
```

### **Market Data Integration with ForexBot**
```python
class LiveForexBot(ForexBot):
    """ForexBot enhanced for live trading"""
    
    def __init__(self, broker, risk_manager):
        super().__init__()
        self.broker = broker
        self.risk_manager = risk_manager
        self.data_feed = RealTimeDataFeed(broker)
        self.trading_enabled = False
        
    def start_live_trading(self, symbols):
        """Start live trading on specified symbols"""
        self.trading_enabled = True
        
        for symbol in symbols:
            self.data_feed.subscribe_to_symbol(
                symbol, 
                self._on_price_update
            )
    
    def _on_price_update(self, symbol, price_data):
        """Handle real-time price updates"""
        if not self.trading_enabled:
            return
            
        # Get recent market data (last 100 candles)
        market_data = self._get_recent_data(symbol, periods=100)
        
        # Get AI recommendation
        recommendation = self.get_final_recommendation(market_data, symbol)
        
        # Check if we should trade
        if recommendation['confidence'] > 0.7:  # High confidence threshold
            self._execute_trade_if_safe(symbol, recommendation)
    
    def _execute_trade_if_safe(self, symbol, recommendation):
        """Execute trade with risk management"""
        account_info = self.broker.get_account_info()
        
        # Calculate position size
        stop_loss_pips = 20  # 20 pip stop loss
        pip_value = self._calculate_pip_value(symbol)
        position_size = self.risk_manager.calculate_position_size(
            account_info['balance'], stop_loss_pips, pip_value
        )
        
        # Check risk management
        can_trade, reason = self.risk_manager.can_place_trade(
            account_info['balance'], position_size * pip_value * stop_loss_pips
        )
        
        if can_trade:
            # Execute trade
            if recommendation['action'] == 'BUY':
                result = self.broker.place_market_order(symbol, position_size, 'BUY')
            elif recommendation['action'] == 'SELL':
                result = self.broker.place_market_order(symbol, position_size, 'SELL')
            
            print(f"Trade executed: {symbol} {recommendation['action']} {position_size}")
        else:
            print(f"Trade blocked: {reason}")
```

---

## 🧪 **Testing Strategy**

### **Phase 1: Paper Trading Setup**
```python
class PaperTradingEnvironment:
    """Simulated trading environment for testing"""
    
    def __init__(self, initial_balance=10000):
        self.balance = initial_balance
        self.equity = initial_balance
        self.positions = []
        self.trade_history = []
        self.pnl = 0
        
    def simulate_trade(self, symbol, action, size, entry_price):
        """Simulate trade execution"""
        trade = {
            'id': len(self.trade_history) + 1,
            'symbol': symbol,
            'action': action,
            'size': size,
            'entry_price': entry_price,
            'entry_time': datetime.now(),
            'status': 'open'
        }
        
        self.positions.append(trade)
        self.trade_history.append(trade)
        
        return trade
    
    def close_trade(self, trade_id, exit_price):
        """Close simulated trade"""
        trade = next(t for t in self.positions if t['id'] == trade_id)
        
        if trade['action'] == 'BUY':
            pnl = (exit_price - trade['entry_price']) * trade['size']
        else:  # SELL
            pnl = (trade['entry_price'] - exit_price) * trade['size']
        
        trade['exit_price'] = exit_price
        trade['exit_time'] = datetime.now()
        trade['pnl'] = pnl
        trade['status'] = 'closed'
        
        self.balance += pnl
        self.equity = self.balance
        self.positions.remove(trade)
        
        return pnl
```

### **Phase 2: Gradual Live Trading**
1. **Micro lots**: Start with 0.01 lot sizes
2. **Single pair**: Begin with EUR/USD only
3. **Limited hours**: Trade only during London/NY overlap
4. **Conservative thresholds**: Higher confidence requirements (0.8+)

---

## 📋 **Implementation Checklist**

### **Pre-Development**
- [ ] Choose primary broker (MT5 vs OANDA)
- [ ] Open demo/paper trading account
- [ ] Obtain API credentials/keys
- [ ] Install broker-specific Python libraries
- [ ] Test basic API connectivity

### **Phase 1: Basic Integration (2 weeks)**
- [ ] Implement broker interface abstract class
- [ ] Create MT5 or OANDA concrete implementation
- [ ] Test account info retrieval
- [ ] Test market order placement (demo only)
- [ ] Implement position monitoring

### **Phase 2: Risk Management (1 week)**
- [ ] Implement risk manager class
- [ ] Add position sizing calculations
- [ ] Add daily loss limits
- [ ] Add maximum position limits
- [ ] Test risk management logic

### **Phase 3: Real-Time Integration (2 weeks)**
- [ ] Implement real-time data feed
- [ ] Connect ForexBot to live data
- [ ] Add price update callbacks
- [ ] Test end-to-end paper trading
- [ ] Performance monitoring and logging

### **Phase 4: Live Trading Preparation (1 week)**
- [ ] Extensive paper trading validation (30+ days)
- [ ] Risk management stress testing
- [ ] Performance analysis and optimization
- [ ] Live account setup and funding
- [ ] Go-live checklist completion

---

## 🚨 **Critical Success Factors**

### **Risk Management**
- Never risk more than 2% of account per day
- Never risk more than 1% of account per trade
- Always use stop losses (20-30 pips recommended)
- Monitor drawdown continuously

### **Technical Implementation**
- Handle network disconnections gracefully
- Implement trade execution retry logic
- Log all trades and decisions for analysis
- Monitor API rate limits and errors

### **Performance Monitoring**
- Track win rate, profit factor, Sharpe ratio
- Monitor maximum drawdown
- Calculate risk-adjusted returns
- Alert on unusual performance patterns

---

## 🎯 **Success Metrics for Phase 4**

### **Paper Trading Targets (6 months)**
- Win rate: >52%
- Profit factor: >1.3
- Maximum drawdown: <15%
- Sharpe ratio: >1.0
- Average risk per trade: <1%

### **Live Trading Targets (Year 1)**
- Consistent profitability for 6+ months
- Annual return: >15%
- Maximum drawdown: <20%
- No catastrophic losses (>5% single day)
- System uptime: >99%

---

*Next Update: After Phase 4 completion*
*Target: Q4 2025*