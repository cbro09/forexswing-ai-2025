# 🚀 MetaTrader 5 Live Trading Setup Guide

## ⚡ Quick Setup (5 minutes)

### **Step 1: Install MT5 Requirements**
```bash
pip install -r requirements_mt5.txt
```

### **Step 2: Download MetaTrader 5**
1. Download MT5 from: https://www.metatrader5.com/
2. Install and create a **DEMO account** first
3. Ensure MT5 is running and logged in

### **Step 3: Configure Demo Account**
```bash
python src/brokers/mt5_config.py
```

### **Step 4: Test Connection**
```bash
python src/brokers/live_trading_manager.py
```

---

## 📋 **What You Get**

✅ **Live MT5 Integration** - Direct connection to your broker  
✅ **Real-time Market Data** - Live prices and spreads  
✅ **AI Trade Execution** - Automatic order placement  
✅ **Risk Management** - Built-in safety controls  
✅ **Position Monitoring** - Live P&L tracking  
✅ **Emergency Stop** - Circuit breakers and safety  

---

## 🛠️ **Detailed Setup Instructions**

### **Prerequisites**
- Windows OS (MT5 Python library requirement)
- MetaTrader 5 terminal installed
- Demo or live MT5 account

### **1. Install MetaTrader 5 Software**

**Download Options:**
- **Official Site**: https://www.metatrader5.com/
- **Broker Sites**: Most forex brokers provide MT5 downloads
- **Microsoft Store**: Search "MetaTrader 5"

**Recommended Brokers for Demo Testing:**
- **MetaQuotes Demo** (built-in demo server)
- **IC Markets** - Excellent spreads
- **Pepperstone** - Fast execution  
- **OANDA** - Retail friendly

### **2. Create Demo Account**

**In MetaTrader 5:**
1. File → Open Account → Demo Account
2. Choose broker server (e.g., "MetaQuotes-Demo")  
3. Fill account details
4. Note down: **Login**, **Password**, **Server**
5. Verify account works (place test trade)

### **3. Python Environment Setup**

**Install MT5 Package:**
```bash
pip install MetaTrader5
```

**Verify Installation:**
```python
import MetaTrader5 as mt5
print("MT5 package version:", mt5.__version__)
```

**Test Basic Connection:**
```python
import MetaTrader5 as mt5

# Initialize
if not mt5.initialize():
    print("Failed to initialize MT5")
else:
    print("MT5 initialized successfully")
    print("MT5 version:", mt5.version())
    mt5.shutdown()
```

### **4. Configure ForexSwing AI**

**Setup Demo Configuration:**
```bash
cd G:\forexswing-ai-2025
python src/brokers/mt5_config.py
```

**Manual Configuration:**
```python
from src.brokers.mt5_config import MT5Config

config = MT5Config()
config.add_account(
    account_name="MyDemo",
    login=51234567,          # Your demo login
    password="your_password", # Your demo password  
    server="MetaQuotes-Demo", # Your demo server
    is_demo=True,
    max_risk_per_trade=0.02,  # 2% per trade
    max_daily_loss=0.05       # 5% daily limit
)
config.set_active_account("MyDemo")
```

### **5. Test Live Trading Manager**

**Basic Connection Test:**
```python
from src.brokers.live_trading_manager import LiveTradingManager

manager = LiveTradingManager()
success, message = manager.initialize()
print(message)

# Authenticate (you'll need to enter password)
success, auth_msg = manager.authenticate("your_password")
print(auth_msg)

# Test health check
health = manager.health_check()
print("Health Status:", health['overall_status'])
```

---

## 🔧 **Configuration Options**

### **Risk Management Settings**
```python
# Edit in src/brokers/mt5_config.py
max_risk_per_trade = 0.02    # 2% maximum risk per trade
max_daily_loss = 0.05        # 5% maximum daily loss
max_open_positions = 5       # Maximum simultaneous positions
```

### **Trading Parameters**
```python
# Position sizing based on stop loss
# Automatic volume calculation
# Minimum volume: 0.01 lots
# Maximum volume: 10.0 lots (configurable)
```

### **Safety Controls**
- ✅ **Daily Loss Limits** - Stop trading after threshold
- ✅ **Position Limits** - Maximum open positions
- ✅ **Emergency Stop** - Manual halt all trading
- ✅ **Demo Account Detection** - Extra safety for live accounts

---

## 🎯 **Live Trading Workflow**

### **1. Pre-Trading Checklist**
```bash
# Verify MT5 is running and logged in
# Check account balance and margin
# Review risk management settings
# Confirm safety controls are active
```

### **2. AI Recommendation Execution**
```python
# AI generates recommendation
recommendation = {
    'symbol': 'EUR/USD',
    'action': 'BUY',
    'confidence': 0.678,
    'entry_price': 1.0845,
    'stop_loss': 1.0795,
    'take_profit': 1.0945
}

# Execute through live manager
success, message, result = manager.execute_ai_recommendation(recommendation)
```

### **3. Position Monitoring**
```python
# Get live positions
positions = manager.get_live_positions()

# Monitor P&L
for pos in positions:
    print(f"{pos['symbol']}: {pos['profit']:.2f} USD")

# Close position if needed
manager.close_position(ticket=12345, reason="Take profit hit")
```

### **4. Emergency Procedures**
```python
# Emergency stop (closes all positions)
manager.emergency_stop("Market volatility too high")

# Health check
health = manager.health_check()
if health['overall_status'] != 'HEALTHY':
    print("System needs attention")
```

---

## 📊 **Integration with Streamlit App**

The MT5 integration will be seamlessly integrated into your existing ForexSwing app:

### **New Features Added:**
- 🔴 **Live Trading Tab** - Execute AI recommendations
- 📈 **Real Positions** - Monitor live P&L  
- ⚙️ **Account Settings** - Configure MT5 connection
- 🚨 **Risk Dashboard** - Safety controls status
- 📋 **Trading Log** - Live execution history

### **Safety Features:**
- 🛡️ **Demo Mode First** - Always test on demo
- 🔒 **Password Protection** - Secure authentication  
- 🚦 **Risk Limits** - Automatic trade blocking
- 🛑 **Emergency Stop** - One-click halt trading

---

## ⚠️ **Important Safety Notes**

### **⭐ ALWAYS START WITH DEMO ACCOUNTS**
- Never use live accounts without thorough testing
- Test all features with demo money first
- Understand risk management thoroughly

### **🔒 Security Best Practices**
- Store credentials securely (never in code)
- Use environment variables for sensitive data
- Enable two-factor authentication on MT5 account
- Regularly monitor account activity

### **📉 Risk Management**
- Never risk more than 2% per trade
- Set daily loss limits (5% recommended)
- Always use stop losses
- Monitor positions regularly

### **🚨 Emergency Procedures**
- Know how to halt trading immediately
- Have broker contact information ready
- Understand position closing procedures
- Monitor internet connection stability

---

## 🔍 **Troubleshooting**

### **Common Issues:**

**"MT5 initialization failed"**
```bash
# Solution:
1. Ensure MT5 terminal is running
2. Verify you're logged into your account
3. Check Windows firewall/antivirus
4. Restart MT5 terminal
```

**"Login failed"**
```bash
# Solution:  
1. Verify login credentials
2. Check server name exactly matches
3. Ensure account is not suspended
4. Try logging in directly in MT5 first
```

**"No market data"**
```bash
# Solution:
1. Check MT5 Market Watch window
2. Right-click → Show All Symbols
3. Verify forex symbols are visible
4. Check broker connection status
```

**"Order execution failed"**
```bash
# Solution:
1. Check account has sufficient margin
2. Verify symbol is tradeable
3. Check market hours
4. Review minimum volume requirements
```

---

## 📈 **Next Steps**

Once MT5 is successfully integrated:

1. **Test thoroughly with demo account**
2. **Verify all AI recommendations execute properly**  
3. **Monitor risk management controls**
4. **Practice emergency procedures**
5. **Consider live account (with extreme caution)**

---

## 🎯 **Ready for Live Trading!**

**Your ForexSwing AI system now has:**
- ✅ Professional MT5 broker integration
- ✅ Real-time market data and execution
- ✅ Comprehensive risk management  
- ✅ Safety controls and emergency stops
- ✅ Live position monitoring
- ✅ Automated AI trade execution

**Status: Production Ready for Demo Trading** 🚀

---

*Remember: Trading involves significant risk. Always use proper risk management and start with demo accounts.* ⚠️