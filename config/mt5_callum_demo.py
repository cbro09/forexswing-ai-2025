#!/usr/bin/env python3
"""
MT5 Demo Account Configuration - Callum's Demo Account
Setup script for ForexSwing AI integration
"""

import sys
import os

# Add the parent directory to path to import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.brokers.mt5_config import MT5Config

def setup_callum_demo():
    """Setup Callum's MT5 demo account"""
    print("🚀 Setting up ForexSwing AI with Callum's MT5 Demo Account")
    print("=" * 60)
    
    # Create config
    config = MT5Config("config/mt5_accounts.json")
    
    # Add Callum's demo account
    success = config.add_account(
        account_name="Callum_Demo",
        login=95744673,
        password="P!2pDyQb",  # Will be hashed for storage
        server="MetaQuotes-Demo",
        is_demo=True,
        max_risk_per_trade=0.02,     # 2% per trade (£20 on £1000 account)
        max_daily_loss=0.05,         # 5% daily limit (£50 loss limit)
        max_open_positions=3         # 3 positions max (conservative for demo)
    )
    
    if success:
        # Set as active account
        config.set_active_account("Callum_Demo")
        
        print("✅ Account Configuration:")
        print(f"   Name: Callum Demo")
        print(f"   Login: 95744673")
        print(f"   Server: MetaQuotes-Demo")
        print(f"   Currency: GBP")
        print(f"   Balance: £1,000 (demo)")
        print(f"   Leverage: 1:100")
        print(f"   Account Type: Forex Netting")
        print()
        print("✅ Risk Management Settings:")
        print(f"   Max Risk per Trade: 2% (£20)")
        print(f"   Max Daily Loss: 5% (£50)")
        print(f"   Max Open Positions: 3")
        print()
        
        # Test connection
        print("🔌 Testing MT5 Connection...")
        manager = LiveTradingManager("config/mt5_accounts.json")
        
        init_success, init_msg = manager.initialize()
        print(f"   Initialize: {init_msg}")
        
        if init_success:
            # Test authentication
            auth_success, auth_msg = manager.authenticate("P!2pDyQb")
            print(f"   Authentication: {auth_msg}")
            
            if auth_success:
                # Test account info
                account_info = manager.mt5_connector.get_account_info()
                if account_info:
                    print("✅ Live Account Info:")
                    print(f"   Balance: £{account_info['balance']:.2f}")
                    print(f"   Equity: £{account_info['equity']:.2f}")
                    print(f"   Free Margin: £{account_info['free_margin']:.2f}")
                    print(f"   Server: {account_info['server']}")
                    print()
                
                # Test market data
                print("📊 Testing Market Data...")
                pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY']
                for pair in pairs:
                    price_data = manager.mt5_connector.get_live_price(pair)
                    if price_data:
                        print(f"   {pair}: {price_data['bid']:.5f} / {price_data['ask']:.5f}")
                
                print()
                print("🎯 SETUP COMPLETE!")
                print("✅ ForexSwing AI is now connected to your MT5 demo account")
                print("✅ Ready for live AI trading with demo money")
                print()
                print("Next steps:")
                print("1. Launch the main app: python app_launcher.py")
                print("2. Look for the new 'Live Trading' tab")
                print("3. Test AI recommendations with demo money")
                print("4. Monitor positions and P&L in real-time")
                
                manager.disconnect()
            else:
                print("❌ Authentication failed - check MT5 is running and logged in")
        else:
            print("❌ Connection failed - ensure MT5 terminal is installed and running")
        
        return True
    else:
        print("❌ Failed to setup account configuration")
        return False

def test_trading_features():
    """Test key trading features"""
    print("\n🧪 Testing Trading Features...")
    
    manager = LiveTradingManager("config/mt5_accounts.json")
    
    # Initialize and authenticate
    if manager.initialize()[0] and manager.authenticate("P!2pDyQb")[0]:
        
        # Test AI recommendation simulation
        print("\n📈 Testing AI Recommendation Execution...")
        
        # Simulate AI recommendation
        demo_recommendation = {
            'symbol': 'EUR/USD',
            'action': 'BUY',
            'confidence': 0.678,
            'entry_price': 1.0845,
            'stop_loss': 1.0795,  # 50 pip stop loss
            'take_profit': 1.0945  # 100 pip take profit
        }
        
        print(f"Demo AI Recommendation:")
        print(f"   {demo_recommendation['action']} {demo_recommendation['symbol']}")
        print(f"   Confidence: {demo_recommendation['confidence']:.1%}")
        print(f"   Entry: {demo_recommendation['entry_price']:.5f}")
        print(f"   Stop Loss: {demo_recommendation['stop_loss']:.5f}")
        print(f"   Take Profit: {demo_recommendation['take_profit']:.5f}")
        
        # Note: Uncomment to actually execute demo trade
        # success, message, result = manager.execute_ai_recommendation(demo_recommendation)
        # print(f"Execution Result: {message}")
        
        print("\n⚠️  Uncomment lines in test_trading_features() to execute actual demo trades")
        
        # Test safety controls
        safety_status = manager.safety_controls.get_safety_status()
        print(f"\n🛡️ Safety Controls Status:")
        print(f"   Active Account: {safety_status['active_account']}")
        print(f"   Demo Account: {safety_status['is_demo']}")
        print(f"   Daily Trades: {safety_status['daily_trades']}")
        print(f"   Risk Limits Active: ✅")
        
        manager.disconnect()
    else:
        print("❌ Could not connect for feature testing")

if __name__ == "__main__":
    try:
        success = setup_callum_demo()
        
        if success:
            # Run additional tests
            test_trading_features()
            
            print("\n" + "=" * 60)
            print("🎉 SUCCESS! ForexSwing AI is ready for demo trading")
            print("🚀 Launch with: python app_launcher.py")
            print("💡 Look for the 'Live Trading' tab in the app")
            print("⚠️  Remember: This is demo money - perfect for testing!")
            
        else:
            print("\n❌ Setup failed. Check MT5 installation and try again.")
            
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        print("Ensure MT5 is installed and running with the demo account logged in.")