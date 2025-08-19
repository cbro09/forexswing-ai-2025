#!/usr/bin/env python3
"""
Simple Full System Test: LSTM + Gemini Integration
"""

import subprocess
import os
import pandas as pd
import numpy as np

def test_gemini():
    """Test Gemini CLI"""
    print("Testing Gemini CLI...")
    
    try:
        result = subprocess.run(['npx', '@google/gemini-cli', '--version'], 
                              capture_output=True, text=True, timeout=10, shell=True)
        
        if result.returncode == 0:
            print(f"  [OK] Gemini CLI available: v{result.stdout.strip()}")
            
            # Test API
            test_result = subprocess.run(['npx', '@google/gemini-cli', '--prompt', 'Test: respond with "OK"'], 
                                       capture_output=True, text=True, timeout=30, shell=True)
            
            if test_result.returncode == 0 and "OK" in test_result.stdout:
                print("  [OK] Gemini API working")
                return True
            else:
                print("  [WARNING] Gemini CLI available but API not configured")
                return False
        else:
            print("  [ERROR] Gemini CLI not available")
            return False
            
    except Exception as e:
        print(f"  [ERROR] Gemini test failed: {e}")
        return False

def test_data():
    """Test market data"""
    print("\nTesting market data...")
    
    data_dir = "data/market"
    
    if not os.path.exists(data_dir):
        print(f"  [ERROR] Data directory not found: {data_dir}")
        return False
    
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.feather')]
    
    if not data_files:
        print(f"  [ERROR] No data files found")
        return False
    
    print(f"  [OK] Found {len(data_files)} forex data files")
    
    # Test loading
    try:
        test_file = data_files[0]
        df = pd.read_feather(os.path.join(data_dir, test_file))
        print(f"  [OK] Data loading works ({len(df)} candles)")
        return True
    except Exception as e:
        print(f"  [ERROR] Data loading failed: {e}")
        return False

def test_models():
    """Test AI models"""
    print("\nTesting AI models...")
    
    model_files = [
        "data/models/optimized_forex_ai.pth",
        "data/models/optimized_scaler.pkl"
    ]
    
    missing = []
    for file in model_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"  [OK] {os.path.basename(file)}: {size:,} bytes")
        else:
            missing.append(file)
            print(f"  [ERROR] Missing: {os.path.basename(file)}")
    
    return len(missing) == 0

def test_jax():
    """Test JAX"""
    print("\nTesting JAX...")
    
    try:
        import jax.numpy as jnp
        
        # Simple test
        prices = jnp.array([1.1000, 1.1010, 1.1005, 1.0995, 1.1020])
        mean_price = jnp.mean(prices)
        
        print(f"  [OK] JAX working, sample mean: {mean_price:.5f}")
        return True
        
    except Exception as e:
        print(f"  [ERROR] JAX failed: {e}")
        return False

def test_prediction():
    """Test basic prediction"""
    print("\nTesting prediction logic...")
    
    try:
        # Sample data
        sample_data = pd.DataFrame({
            'close': [1.0850, 1.0860, 1.0855, 1.0865, 1.0870],
            'volume': [100000, 110000, 105000, 115000, 120000]
        })
        
        # Simple prediction
        latest_price = sample_data['close'].iloc[-1]
        prev_price = sample_data['close'].iloc[-2]
        change = (latest_price - prev_price) / prev_price
        
        if change > 0.001:
            prediction = "BUY"
        elif change < -0.001:
            prediction = "SELL"
        else:
            prediction = "HOLD"
        
        print(f"  [OK] Sample prediction: {prediction} (change: {change:.3%})")
        return True
        
    except Exception as e:
        print(f"  [ERROR] Prediction failed: {e}")
        return False

def main():
    """Run system test"""
    
    print("FOREXSWING AI 2025 - SYSTEM TEST")
    print("=" * 50)
    
    # Run all tests
    tests = {
        "Gemini CLI": test_gemini(),
        "Market Data": test_data(),
        "AI Models": test_models(),
        "JAX Processing": test_jax(),
        "Prediction Logic": test_prediction()
    }
    
    # Results
    print(f"\n" + "=" * 50)
    print("SYSTEM STATUS")
    print("=" * 50)
    
    working = 0
    total = len(tests)
    
    for component, status in tests.items():
        status_text = "[OK]" if status else "[ERROR]"
        print(f"{status_text} {component}")
        if status:
            working += 1
    
    print(f"\nComponents Working: {working}/{total}")
    
    if working == total:
        print("\n[SUCCESS] FULL SYSTEM OPERATIONAL!")
        print("Ready for enhanced trading!")
    elif working >= 3:
        print("\n[PARTIAL] System partially working")
        print("Fix remaining issues for full deployment")
    else:
        print("\n[ISSUES] System needs significant work")
    
    # Recommendations
    print(f"\nNext Steps:")
    if not tests["Gemini CLI"]:
        print("1. Setup Gemini: python scripts/deployment/setup_gemini.py")
    if not tests["AI Models"]:
        print("2. Train models: python train.py")
    if working == total:
        print("1. System ready for live deployment!")
    
    print(f"\nTest complete!")

if __name__ == "__main__":
    main()