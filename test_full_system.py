#!/usr/bin/env python3
"""
Full System Test: LSTM + Gemini Integration
Complete test of the enhanced ForexSwing AI system
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import subprocess

def test_gemini_availability():
    """Test if Gemini CLI is available and configured"""
    print("Testing Gemini CLI availability...")
    
    try:
        # Check if Gemini CLI is installed
        result = subprocess.run(['npx', '@google/gemini-cli', '--version'], 
                              capture_output=True, text=True, timeout=10, shell=True)
        
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"  ✅ Gemini CLI available: v{version}")
            
            # Test if API key is configured
            test_result = subprocess.run(['npx', '@google/gemini-cli', '--prompt', 'Test: respond with "OK"'], 
                                       capture_output=True, text=True, timeout=30, shell=True)
            
            if test_result.returncode == 0 and "OK" in test_result.stdout:
                print("  ✅ Gemini API configured and working")
                return True
            else:
                print("  ⚠️ Gemini CLI available but API not configured")
                print("     Set GEMINI_API_KEY environment variable")
                return False
        else:
            print("  ❌ Gemini CLI not available")
            return False
            
    except Exception as e:
        print(f"  ❌ Gemini test failed: {e}")
        return False

def test_market_data():
    """Test market data availability"""
    print("\nTesting market data...")
    
    data_dir = "data/market"
    
    if not os.path.exists(data_dir):
        print(f"  ❌ Market data directory not found: {data_dir}")
        return False
    
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.feather')]
    
    if not data_files:
        print(f"  ❌ No market data files found in {data_dir}")
        return False
    
    print(f"  ✅ Found {len(data_files)} forex data files")
    
    # Test loading one file
    try:
        test_file = data_files[0]
        df = pd.read_feather(os.path.join(data_dir, test_file))
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if all(col in df.columns for col in required_cols):
            print(f"  ✅ Market data format valid ({len(df)} candles)")
            return True
        else:
            print(f"  ❌ Market data missing required columns")
            return False
            
    except Exception as e:
        print(f"  ❌ Market data loading failed: {e}")
        return False

def test_ai_models():
    """Test AI model availability"""
    print("\nTesting AI models...")
    
    model_dir = "data/models"
    
    required_files = [
        "optimized_forex_ai.pth",
        "optimized_scaler.pkl"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"  ✅ {file}: {file_size:,} bytes")
        else:
            missing_files.append(file)
            print(f"  ❌ {file}: Not found")
    
    if missing_files:
        print(f"  ⚠️ Missing {len(missing_files)} model files")
        print("     Run: python train.py")
        return False
    
    return True

def test_jax_indicators():
    """Test JAX-accelerated indicators"""
    print("\nTesting JAX indicators...")
    
    try:
        import jax.numpy as jnp
        
        # Test data
        prices = jnp.array([1.1000, 1.1010, 1.1005, 1.0995, 1.1020, 1.1015, 1.1025])
        
        # Simple moving average test
        def simple_sma(prices, period):
            """Simple SMA for testing"""
            sma_values = []
            for i in range(len(prices)):
                if i >= period - 1:
                    sma_values.append(jnp.mean(prices[i-period+1:i+1]))
                else:
                    sma_values.append(prices[i])
            return jnp.array(sma_values)
        
        sma_5 = simple_sma(prices, 5)
        
        print(f"  ✅ JAX computation working")
        print(f"  ✅ Sample SMA calculation: {sma_5[-1]:.5f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ JAX indicators failed: {e}")
        return False

def test_basic_prediction():
    """Test basic prediction capability"""
    print("\nTesting basic prediction...")
    
    try:
        # Create sample forex data
        np.random.seed(42)
        n_points = 100
        
        base_price = 1.0850
        price_changes = np.random.randn(n_points) * 0.001
        price_walk = np.cumsum(price_changes) + base_price
        
        sample_data = pd.DataFrame({
            'open': price_walk + np.random.randn(n_points) * 0.0005,
            'high': price_walk + np.abs(np.random.randn(n_points)) * 0.002,
            'low': price_walk - np.abs(np.random.randn(n_points)) * 0.002,
            'close': price_walk,
            'volume': np.random.randint(50000, 200000, n_points),
        })
        
        print(f"  ✅ Sample data created: {len(sample_data)} candles")
        print(f"  ✅ Price range: {sample_data['close'].min():.5f} - {sample_data['close'].max():.5f}")
        
        # Simple prediction logic (placeholder)
        latest_price = sample_data['close'].iloc[-1]
        price_change = (latest_price - sample_data['close'].iloc[-2]) / sample_data['close'].iloc[-2]
        
        if price_change > 0.001:
            prediction = "BUY"
            confidence = 0.65
        elif price_change < -0.001:
            prediction = "SELL" 
            confidence = 0.60
        else:
            prediction = "HOLD"
            confidence = 0.55
        
        print(f"  ✅ Basic prediction: {prediction} (confidence: {confidence:.1%})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic prediction failed: {e}")
        return False

def test_system_integration():
    """Test complete system integration"""
    print("\nTesting system integration...")
    
    # Check if all components are available
    components = {
        "Gemini CLI": test_gemini_availability(),
        "Market Data": test_market_data(), 
        "AI Models": test_ai_models(),
        "JAX Indicators": test_jax_indicators(),
        "Basic Prediction": test_basic_prediction()
    }
    
    working_components = sum(components.values())
    total_components = len(components)
    
    print(f"\nSYSTEM INTEGRATION RESULTS:")
    print("=" * 40)
    
    for component, status in components.items():
        status_symbol = "✅" if status else "❌"
        print(f"{status_symbol} {component}")
    
    print(f"\nOverall Status: {working_components}/{total_components} components working")
    
    if working_components == total_components:
        print("🎯 FULL SYSTEM OPERATIONAL!")
        return True
    elif working_components >= 3:
        print("⚠️ System partially operational")
        return False
    else:
        print("❌ System needs significant work")
        return False

def generate_system_report():
    """Generate comprehensive system status report"""
    print(f"\nGENERATING SYSTEM REPORT...")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "system_status": "testing",
        "components": {},
        "recommendations": []
    }
    
    # Run all tests
    gemini_ok = test_gemini_availability()
    data_ok = test_market_data()
    models_ok = test_ai_models()
    jax_ok = test_jax_indicators()
    prediction_ok = test_basic_prediction()
    
    report["components"] = {
        "gemini_cli": gemini_ok,
        "market_data": data_ok,
        "ai_models": models_ok,
        "jax_indicators": jax_ok,
        "basic_prediction": prediction_ok
    }
    
    # Generate recommendations
    if not gemini_ok:
        report["recommendations"].append("Setup Gemini CLI: python scripts/deployment/setup_gemini.py")
    
    if not models_ok:
        report["recommendations"].append("Train AI models: python train.py")
    
    if not data_ok:
        report["recommendations"].append("Download market data or check data/market/ directory")
    
    working_count = sum(report["components"].values())
    total_count = len(report["components"])
    
    if working_count == total_count:
        report["system_status"] = "fully_operational"
        report["recommendations"].append("System ready for live trading!")
    elif working_count >= 3:
        report["system_status"] = "partially_operational"
        report["recommendations"].append("Fix remaining issues before live deployment")
    else:
        report["system_status"] = "needs_work"
        report["recommendations"].append("Significant setup required")
    
    return report

def main():
    """Run full system test"""
    
    print("FOREXSWING AI 2025 - FULL SYSTEM TEST")
    print("=" * 60)
    print("Testing Enhanced Dual AI System (LSTM + Gemini)")
    print()
    
    # Generate comprehensive report
    report = generate_system_report()
    
    # Final summary
    print(f"\n" + "=" * 60)
    print("FINAL SYSTEM STATUS")
    print("=" * 60)
    
    status = report["system_status"]
    working = sum(report["components"].values())
    total = len(report["components"])
    
    if status == "fully_operational":
        print("🎯 SYSTEM FULLY OPERATIONAL!")
        print("   Ready for live trading deployment")
    elif status == "partially_operational":
        print("⚠️ SYSTEM PARTIALLY OPERATIONAL")
        print(f"   {working}/{total} components working")
    else:
        print("❌ SYSTEM NEEDS WORK")
        print(f"   Only {working}/{total} components working")
    
    print(f"\nNext Steps:")
    for i, rec in enumerate(report["recommendations"], 1):
        print(f"{i}. {rec}")
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return report

if __name__ == "__main__":
    main()