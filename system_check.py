#!/usr/bin/env python3
"""
System Check - Test all ForexSwing AI components
"""

import os
import sys
import time
import traceback

def check_core_imports():
    """Check all critical imports"""
    print("1. CORE IMPORTS")
    print("-" * 30)
    
    results = {}
    
    # PyTorch
    try:
        import torch
        print(f"[OK] PyTorch: {torch.__version__}")
        results['pytorch'] = True
    except Exception as e:
        print(f"[FAIL] PyTorch: {e}")
        results['pytorch'] = False
    
    # Pandas/NumPy
    try:
        import pandas as pd
        import numpy as np
        print(f"[OK] Pandas: {pd.__version__}")
        print(f"[OK] NumPy: {np.__version__}")
        results['data'] = True
    except Exception as e:
        print(f"[FAIL] Data libraries: {e}")
        results['data'] = False
    
    # JAX
    try:
        import jax.numpy as jnp
        test_array = jnp.array([1, 2, 3])
        print(f"[OK] JAX: Working")
        results['jax'] = True
    except Exception as e:
        print(f"[FAIL] JAX: {e}")
        results['jax'] = False
    
    # ForexLSTM
    try:
        from models.ForexLSTM import SimpleOptimizedLSTM, create_simple_features
        print(f"[OK] ForexLSTM: Available")
        results['model'] = True
    except Exception as e:
        print(f"[FAIL] ForexLSTM: {e}")
        results['model'] = False
    
    return results

def check_forexbot():
    """Check ForexBot functionality"""
    print(f"\n2. FOREXBOT CORE")
    print("-" * 30)
    
    results = {}
    
    try:
        from ForexBot import ForexBot
        
        # Initialize
        print("Initializing ForexBot...")
        bot = ForexBot()
        print("[OK] ForexBot initialized")
        results['init'] = True
        
        # Test system status
        status = bot.get_system_status()
        print(f"[OK] System status: {status['deployment_status']}")
        results['status'] = True
        
        # Test prediction
        import pandas as pd
        import numpy as np
        
        sample_data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 1.0850,
            'volume': np.random.randint(50000, 200000, 100),
            'high': np.random.randn(100) * 0.002 + 1.0850,
            'low': np.random.randn(100) * 0.002 + 1.0850,
        })
        
        rec = bot.get_final_recommendation(sample_data, "EUR/USD")
        print(f"[OK] Prediction: {rec['action']} ({rec['confidence']:.1%})")
        results['prediction'] = True
        
    except Exception as e:
        print(f"[FAIL] ForexBot: {e}")
        results['init'] = False
        results['status'] = False
        results['prediction'] = False
    
    return results

def check_gemini():
    """Check Gemini integration"""
    print(f"\n3. GEMINI AI")
    print("-" * 30)
    
    results = {}
    
    try:
        from src.integrations.optimized_gemini import OptimizedGeminiInterpreter
        
        # Initialize
        gemini = OptimizedGeminiInterpreter(cache_size=5, cache_duration_minutes=1)
        print(f"[OK] Gemini initialized")
        print(f"[INFO] CLI available: {gemini.gemini_available}")
        results['init'] = True
        results['available'] = gemini.gemini_available
        
        if gemini.gemini_available:
            # Test market interpretation
            test_data = {
                "current_price": 1.0850,
                "price_change_24h": 0.15,
                "trend": "bullish",
                "volatility": 0.012
            }
            
            interpretation = gemini.interpret_market_quickly(test_data, "EUR/USD")
            print(f"[OK] Market interpretation: {interpretation.get('sentiment', 'N/A')}")
            results['interpretation'] = True
        else:
            print(f"[INFO] Gemini CLI not available - offline mode")
            results['interpretation'] = False
        
    except Exception as e:
        print(f"[FAIL] Gemini: {e}")
        results['init'] = False
        results['available'] = False
        results['interpretation'] = False
    
    return results

def check_jax_performance():
    """Check JAX indicators"""
    print(f"\n4. JAX INDICATORS")
    print("-" * 30)
    
    results = {}
    
    try:
        sys.path.append('src/indicators')
        import jax_advanced_indicators
        import jax.numpy as jnp
        
        # Test data
        prices = jnp.array([1.0, 1.1, 1.2, 1.15, 1.25, 1.3, 1.28, 1.35, 1.4, 1.38])
        
        # Test momentum
        start = time.time()
        momentum = jax_advanced_indicators.jax_momentum(prices, period=3)
        momentum_time = time.time() - start
        print(f"[OK] Momentum: {momentum_time:.4f}s")
        results['momentum'] = True
        
        # Test ATR
        high = prices + 0.01
        low = prices - 0.01
        atr = jax_advanced_indicators.jax_atr(high, low, prices, period=5)
        print(f"[OK] ATR: Working")
        results['atr'] = True
        
        # Test performance
        if momentum_time < 0.01:
            print(f"[EXCELLENT] JAX performance: {momentum_time:.4f}s")
        else:
            print(f"[OK] JAX performance: {momentum_time:.4f}s")
        results['performance'] = momentum_time < 0.1
        
    except Exception as e:
        print(f"[FAIL] JAX indicators: {e}")
        results['momentum'] = False
        results['atr'] = False
        results['performance'] = False
    
    return results

def check_data_files():
    """Check data files and model"""
    print(f"\n5. DATA & MODELS")
    print("-" * 30)
    
    results = {}
    
    # Model file
    model_path = "data/models/optimized_forex_ai.pth"
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        print(f"[OK] Model: {size:,} bytes")
        results['model'] = True
    else:
        print(f"[FAIL] Model not found: {model_path}")
        results['model'] = False
    
    # Market data
    data_files = [
        "data/MarketData/EUR_USD_real_daily.csv",
        "data/MarketData/GBP_USD_real_daily.csv", 
        "data/MarketData/USD_JPY_real_daily.csv"
    ]
    
    found_files = 0
    for file_path in data_files:
        if os.path.exists(file_path):
            found_files += 1
    
    print(f"[OK] Market data: {found_files}/{len(data_files)} files")
    results['data'] = found_files >= 2
    
    return results

def run_system_check():
    """Run complete system check"""
    print("FOREXSWING AI 2025 - SYSTEM CHECK")
    print("=" * 50)
    
    # Run all checks
    all_results = {}
    all_results['imports'] = check_core_imports()
    all_results['forexbot'] = check_forexbot()
    all_results['gemini'] = check_gemini()
    all_results['jax'] = check_jax_performance()
    all_results['data'] = check_data_files()
    
    # Summary
    print(f"\n" + "=" * 50)
    print("SYSTEM STATUS SUMMARY")
    print("=" * 50)
    
    total_tests = 0
    passed_tests = 0
    
    for category, results in all_results.items():
        category_passed = 0
        category_total = 0
        
        for test_name, passed in results.items():
            total_tests += 1
            category_total += 1
            if passed:
                passed_tests += 1
                category_passed += 1
        
        status = "OK" if category_passed == category_total else "ISSUES"
        print(f"{category.upper()}: {category_passed}/{category_total} - {status}")
    
    # Overall
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    print(f"\nOVERALL: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 90:
        print("STATUS: EXCELLENT - Ready for deployment")
    elif success_rate >= 75:
        print("STATUS: GOOD - Minor issues to fix")
    elif success_rate >= 60:
        print("STATUS: FAIR - Some components need attention")
    else:
        print("STATUS: POOR - Significant issues")
    
    return success_rate, all_results

if __name__ == "__main__":
    try:
        success_rate, results = run_system_check()
    except Exception as e:
        print(f"System check failed: {e}")
        traceback.print_exc()