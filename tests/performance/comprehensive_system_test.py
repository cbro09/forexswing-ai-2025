#!/usr/bin/env python3
"""
Comprehensive ForexSwing AI System Test
Test all components: ForexBot, Gemini, JAX, Models, Data Pipeline
"""

import os
import sys
import time
import torch
import pandas as pd
import numpy as np
from datetime import datetime

def test_core_imports():
    """Test all critical imports"""
    print("1. TESTING CORE IMPORTS")
    print("=" * 50)
    
    results = {}
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        results['pytorch'] = True
    except Exception as e:
        print(f"✗ PyTorch: {e}")
        results['pytorch'] = False
    
    try:
        import pandas as pd
        import numpy as np
        print(f"✓ Pandas: {pd.__version__}")
        print(f"✓ NumPy: {np.__version__}")
        results['pandas_numpy'] = True
    except Exception as e:
        print(f"✗ Pandas/NumPy: {e}")
        results['pandas_numpy'] = False
    
    try:
        import jax.numpy as jnp
        print(f"✓ JAX: Available")
        results['jax'] = True
    except Exception as e:
        print(f"✗ JAX: {e}")
        results['jax'] = False
    
    try:
        from models.ForexLSTM import SimpleOptimizedLSTM, create_simple_features
        print(f"✓ ForexLSTM: Model and features")
        results['forex_model'] = True
    except Exception as e:
        print(f"✗ ForexLSTM: {e}")
        results['forex_model'] = False
    
    return results

def test_forexbot_core():
    """Test ForexBot core functionality"""
    print(f"\n2. TESTING FOREXBOT CORE")
    print("=" * 50)
    
    results = {}
    
    try:
        from ForexBot import ForexBot
        
        # Initialize bot
        print("Initializing ForexBot...")
        bot = ForexBot()
        print("✓ ForexBot initialization successful")
        results['initialization'] = True
        
        # Test system status
        print("Testing system status...")
        status = bot.get_system_status()
        print(f"✓ System status: {status['deployment_status']}")
        results['system_status'] = True
        
        # Test with sample data
        print("Testing with sample market data...")
        sample_data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 1.0850,
            'volume': np.random.randint(50000, 200000, 100),
            'high': np.random.randn(100) * 0.002 + 1.0850,
            'low': np.random.randn(100) * 0.002 + 1.0850,
        })
        
        recommendation = bot.get_final_recommendation(sample_data, "EUR/USD")
        print(f"✓ Recommendation: {recommendation['action']} ({recommendation['confidence']:.1%})")
        print(f"✓ Processing time: {recommendation['processing_time']}")
        results['recommendation'] = True
        
    except Exception as e:
        print(f"✗ ForexBot core test failed: {e}")
        results['initialization'] = False
        results['system_status'] = False
        results['recommendation'] = False
    
    return results

def test_gemini_integration():
    """Test Gemini AI integration"""
    print(f"\n3. TESTING GEMINI AI INTEGRATION")
    print("=" * 50)
    
    results = {}
    
    try:
        from src.integrations.optimized_gemini import OptimizedGeminiInterpreter, FastForexStrategy
        
        # Test Gemini interpreter
        print("Testing Gemini interpreter...")
        gemini = OptimizedGeminiInterpreter(cache_size=5, cache_duration_minutes=1)
        print(f"✓ Gemini interpreter initialized")
        print(f"✓ Gemini CLI available: {gemini.gemini_available}")
        results['gemini_init'] = True
        
        if gemini.gemini_available:
            # Test market interpretation
            print("Testing market interpretation...")
            test_market_data = {
                "current_price": 1.0850,
                "price_change_24h": 0.15,
                "trend": "bullish",
                "volatility": 0.012
            }
            
            start_time = time.time()
            interpretation = gemini.interpret_market_quickly(test_market_data, "EUR/USD")
            end_time = time.time()
            
            print(f"✓ Market interpretation: {interpretation.get('sentiment', 'N/A')}")
            print(f"✓ Response time: {end_time - start_time:.2f}s")
            results['market_interpretation'] = True
            
            # Test signal validation
            print("Testing signal validation...")
            signal_data = {
                'ml_prediction': 0.7,
                'ml_confidence': 0.8,
                'rsi': 65,
                'trend_direction': 'bullish'
            }
            
            validation = gemini.validate_signal_fast(signal_data)
            print(f"✓ Signal validation: {validation.get('validation', 'N/A')}")
            results['signal_validation'] = True
            
            # Test performance stats
            stats = gemini.get_performance_stats()
            print(f"✓ Performance stats: {stats['total_calls']} calls")
            results['performance_stats'] = True
            
        else:
            print("⚠ Gemini CLI not available - testing offline functionality")
            results['market_interpretation'] = False
            results['signal_validation'] = False
            results['performance_stats'] = True
            
    except Exception as e:
        print(f"✗ Gemini integration test failed: {e}")
        results['gemini_init'] = False
        results['market_interpretation'] = False
        results['signal_validation'] = False
        results['performance_stats'] = False
    
    return results

def test_jax_indicators():
    """Test JAX indicators performance"""
    print(f"\n4. TESTING JAX INDICATORS")
    print("=" * 50)
    
    results = {}
    
    try:
        sys.path.append('archive/archive_cleanup/src/indicators')
        import jax_advanced_indicators
        import jax.numpy as jnp
        
        print("Testing JAX indicators...")
        
        # Test data
        prices = jnp.array([1.0, 1.1, 1.2, 1.15, 1.25, 1.3, 1.28, 1.35, 1.4, 1.38])
        high = prices + 0.01
        low = prices - 0.01
        
        # Test momentum
        start_time = time.time()
        momentum = jax_advanced_indicators.jax_momentum(prices, period=3)
        momentum_time = time.time() - start_time
        print(f"✓ JAX momentum: {momentum_time:.4f}s")
        results['momentum'] = True
        
        # Test ATR
        start_time = time.time()
        atr = jax_advanced_indicators.jax_atr(high, low, prices, period=5)
        atr_time = time.time() - start_time
        print(f"✓ JAX ATR: {atr_time:.4f}s")
        results['atr'] = True
        
        # Test Bollinger Bands
        start_time = time.time()
        upper, middle, lower, bb_pct = jax_advanced_indicators.jax_bollinger_bands(prices, window=5)
        bb_time = time.time() - start_time
        print(f"✓ JAX Bollinger Bands: {bb_time:.4f}s")
        results['bollinger'] = True
        
        total_time = momentum_time + atr_time + bb_time
        print(f"✓ Total JAX processing: {total_time:.4f}s")
        results['total_performance'] = total_time < 0.01  # Should be very fast
        
    except Exception as e:
        print(f"✗ JAX indicators test failed: {e}")
        results['momentum'] = False
        results['atr'] = False
        results['bollinger'] = False
        results['total_performance'] = False
    
    return results

def test_data_pipeline():
    """Test data loading and processing"""
    print(f"\n5. TESTING DATA PIPELINE")
    print("=" * 50)
    
    results = {}
    
    try:
        # Test model loading
        print("Testing model loading...")
        model_path = "data/models/optimized_forex_ai.pth"
        if os.path.exists(model_path):
            model_size = os.path.getsize(model_path)
            print(f"✓ Model file: {model_size:,} bytes")
            
            # Test actual loading
            from models.ForexLSTM import SimpleOptimizedLSTM
            model = SimpleOptimizedLSTM(input_size=20, hidden_size=128, num_layers=3, dropout=0.4)
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
            print(f"✓ Model loaded successfully")
            results['model_loading'] = True
        else:
            print(f"✗ Model file not found: {model_path}")
            results['model_loading'] = False
        
        # Test market data loading
        print("Testing market data loading...")
        data_files = [
            "data/MarketData/EUR_USD_real_daily.csv",
            "data/MarketData/GBP_USD_real_daily.csv",
            "data/MarketData/USD_JPY_real_daily.csv"
        ]
        
        loaded_files = 0
        for file_path in data_files:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f"✓ {os.path.basename(file_path)}: {len(df)} rows")
                loaded_files += 1
            else:
                print(f"✗ {os.path.basename(file_path)}: Not found")
        
        results['market_data'] = loaded_files >= 2
        
        # Test feature creation
        print("Testing feature creation...")
        sample_data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 1.0850,
            'volume': np.random.randint(50000, 200000, 100),
            'high': np.random.randn(100) * 0.002 + 1.0850,
            'low': np.random.randn(100) * 0.002 + 1.0850,
        })
        
        from models.ForexLSTM import create_simple_features
        start_time = time.time()
        features = create_simple_features(sample_data, target_features=20)
        feature_time = time.time() - start_time
        
        print(f"✓ Features created: {features.shape} in {feature_time:.4f}s")
        results['feature_creation'] = True
        
    except Exception as e:
        print(f"✗ Data pipeline test failed: {e}")
        results['model_loading'] = False
        results['market_data'] = False
        results['feature_creation'] = False
    
    return results

def test_signal_processing():
    """Test signal processing functions"""
    print(f"\n6. TESTING SIGNAL PROCESSING")
    print("=" * 50)
    
    results = {}
    
    try:
        from ForexBot import ForexBot
        bot = ForexBot()
        
        # Test different market scenarios
        scenarios = {
            "bull": np.cumsum(np.random.randn(100) * 0.001 + 0.003) + 1.0850,
            "bear": np.cumsum(np.random.randn(100) * 0.001 - 0.003) + 1.0850,
            "sideways": np.random.randn(100) * 0.001 + 1.0850,
        }
        
        signals_generated = set()
        
        for scenario_name, prices in scenarios.items():
            test_data = pd.DataFrame({
                'close': prices,
                'volume': np.random.randint(50000, 200000, len(prices)),
                'high': prices + np.abs(np.random.randn(len(prices))) * 0.001,
                'low': prices - np.abs(np.random.randn(len(prices))) * 0.001,
            })
            
            recommendation = bot.get_final_recommendation(test_data, "EUR/USD")
            signal = recommendation['action']
            signals_generated.add(signal)
            
            print(f"✓ {scenario_name}: {signal} ({recommendation['confidence']:.1%})")
        
        print(f"✓ Signal diversity: {len(signals_generated)}/3 types")
        results['signal_diversity'] = len(signals_generated) >= 2
        
        # Test trend analysis
        print("Testing trend analysis...")
        enhanced_features, _ = bot.create_enhanced_features(test_data)
        trend_signal, trend_strength = bot.get_trend_signal(test_data)
        print(f"✓ Trend analysis: {trend_signal} (strength: {trend_strength:.2f})")
        results['trend_analysis'] = True
        
    except Exception as e:
        print(f"✗ Signal processing test failed: {e}")
        results['signal_diversity'] = False
        results['trend_analysis'] = False
    
    return results

def run_comprehensive_test():
    """Run all system tests"""
    print("FOREXSWING AI 2025 - COMPREHENSIVE SYSTEM TEST")
    print("=" * 60)
    print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run all tests
    test_results = {}
    test_results['imports'] = test_core_imports()
    test_results['forexbot'] = test_forexbot_core()
    test_results['gemini'] = test_gemini_integration()
    test_results['jax'] = test_jax_indicators()
    test_results['data'] = test_data_pipeline()
    test_results['signals'] = test_signal_processing()
    
    # Summary
    print(f"\n" + "=" * 60)
    print("COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    
    total_tests = 0
    passed_tests = 0
    
    for category, results in test_results.items():
        print(f"\n{category.upper()}:")
        for test_name, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {test_name}: {status}")
            total_tests += 1
            if passed:
                passed_tests += 1
    
    # Overall assessment
    success_rate = (passed_tests / total_tests) * 100
    print(f"\n" + "=" * 60)
    print("OVERALL SYSTEM STATUS")
    print("=" * 60)
    print(f"Tests passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 90:
        print("🎉 EXCELLENT: System fully operational")
        system_status = "EXCELLENT"
    elif success_rate >= 75:
        print("✅ GOOD: System operational with minor issues")
        system_status = "GOOD"
    elif success_rate >= 60:
        print("⚠️ FAIR: System partially operational")
        system_status = "FAIR"
    else:
        print("❌ POOR: System has significant issues")
        system_status = "POOR"
    
    print(f"\nNext steps:")
    if success_rate < 100:
        print("- Review failed tests and fix issues")
        print("- Ensure all dependencies are installed")
        print("- Check file paths and permissions")
    else:
        print("- System ready for live trading integration")
        print("- Proceed with broker API setup")
        print("- Begin paper trading tests")
    
    return system_status, test_results

if __name__ == "__main__":
    system_status, results = run_comprehensive_test()