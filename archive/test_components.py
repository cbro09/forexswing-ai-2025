#!/usr/bin/env python3
"""
Test script to validate all ForexSwing AI 2025 components
"""

import pandas as pd
import numpy as np
import sys
import os

def test_jax_indicators():
    """Test JAX indicators performance"""
    print("=" * 50)
    print("Testing JAX Indicators")
    print("=" * 50)
    
    sys.path.append('src')
    from indicators.jax_indicators import jax_rsi, jax_sma, jax_macd, speed_test
    
    # Run speed test
    speed_test()
    print("JAX Indicators: PASSED\n")

def test_ml_model():
    """Test ML model prediction"""
    print("=" * 50)
    print("Testing ML Model")
    print("=" * 50)
    
    sys.path.append('src')
    from ml_models.forex_lstm import HybridForexPredictor
    
    # Create sample data
    np.random.seed(42)
    n_points = 200
    base_price = 1.1000
    price_walk = np.cumsum(np.random.randn(n_points) * 0.001) + base_price
    
    sample_data = pd.DataFrame({
        'close': price_walk,
        'volume': np.random.randint(1000, 5000, n_points),
        'high': price_walk + np.abs(np.random.randn(n_points)) * 0.002,
        'low': price_walk - np.abs(np.random.randn(n_points)) * 0.002,
    })
    
    # Test prediction
    predictor = HybridForexPredictor()
    predictions = predictor.predict(sample_data)
    
    print(f"Sample predictions: {predictions[-5:]}")
    print(f"Prediction range: {predictions.min():.3f} to {predictions.max():.3f}")
    print("ML Model: PASSED\n")

def test_integration():
    """Test integration between components"""
    print("=" * 50)
    print("Testing Component Integration")
    print("=" * 50)
    
    sys.path.append('src')
    from indicators.jax_indicators import jax_rsi, jax_sma, jax_macd
    from ml_models.forex_lstm import HybridForexPredictor
    import jax.numpy as jnp
    
    # Create realistic forex data
    np.random.seed(123)
    n_points = 100
    
    # Simulate EUR/USD-like price movement
    base_price = 1.0500
    price_changes = np.random.randn(n_points) * 0.0005  # Small forex movements
    prices = np.cumsum(price_changes) + base_price
    
    # Create DataFrame
    df = pd.DataFrame({
        'close': prices,
        'high': prices + np.abs(np.random.randn(n_points)) * 0.0008,
        'low': prices - np.abs(np.random.randn(n_points)) * 0.0008,
        'volume': np.random.randint(10000, 50000, n_points),
    })
    
    # Test JAX indicators
    jax_prices = jnp.array(prices)
    rsi = jax_rsi(jax_prices, 14)
    sma_20 = jax_sma(jax_prices, 20)
    macd_line, macd_signal, macd_hist = jax_macd(jax_prices)
    
    print(f"RSI range: {np.min(rsi):.1f} to {np.max(rsi):.1f}")
    print(f"SMA-20 sample: {sma_20[-3:]}")
    print(f"MACD sample: {macd_line[-3:]}")
    
    # Test ML predictions
    predictor = HybridForexPredictor()
    ml_predictions = predictor.predict(df)
    
    print(f"ML predictions range: {ml_predictions.min():.3f} to {ml_predictions.max():.3f}")
    
    # Test trading signals (simplified)
    signals = []
    for i in range(len(df)):
        if i < 20:  # Not enough data for indicators
            signals.append(0)
            continue
            
        # Simple signal generation
        current_rsi = rsi[i] if i < len(rsi) else 50
        current_ml = ml_predictions[i]
        current_price = prices[i]
        current_sma = sma_20[i] if i < len(sma_20) else current_price
        
        # Buy signal: RSI not overbought, ML bullish, price above SMA
        if current_rsi < 70 and current_ml > 0.6 and current_price > current_sma:
            signals.append(1)  # Buy
        # Sell signal: RSI overbought or ML bearish
        elif current_rsi > 80 or current_ml < 0.4:
            signals.append(-1)  # Sell
        else:
            signals.append(0)  # Hold
    
    buy_signals = sum(1 for s in signals if s == 1)
    sell_signals = sum(1 for s in signals if s == -1)
    
    print(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
    print("Component Integration: PASSED\n")

def main():
    """Run all tests"""
    print("ForexSwing AI 2025 - Component Testing")
    print("=====================================")
    
    try:
        test_jax_indicators()
        test_ml_model()
        test_integration()
        
        print("=" * 50)
        print("ALL TESTS PASSED!")
        print("=" * 50)
        print("Your ForexSwing AI 2025 system is ready!")
        print("\nNext steps:")
        print("1. Set up FreqTrade (install TA-Lib separately if needed)")
        print("2. Configure broker credentials in config/config.json")
        print("3. Start with dry-run mode to test the strategy")
        print("4. Consider training the ML model with real data")
        
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()