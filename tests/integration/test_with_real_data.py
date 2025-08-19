#!/usr/bin/env python3
"""
Test ForexBot with Real Historical Data
Show how the bot would have performed on actual market data
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_real_data(pair="EUR_USD", days=30):
    """Load real market data"""
    file_path = f"data/MarketData/{pair}_real_daily.csv"
    
    if not os.path.exists(file_path):
        print(f"[ERROR] Data file not found: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path)
        print(f"[OK] Loaded {len(df)} candles for {pair}")
        
        # Get recent data
        recent_data = df.tail(days + 80)  # Extra for model warmup
        print(f"[OK] Using last {len(recent_data)} candles for testing")
        
        return recent_data
    
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return None

def test_bot_on_real_data():
    """Test bot performance on real historical data"""
    print("TESTING FOREXBOT ON REAL HISTORICAL DATA")
    print("=" * 60)
    
    from ForexBot import ForexBot
    
    # Initialize bot
    try:
        bot = ForexBot()
        print("[OK] ForexBot initialized")
    except Exception as e:
        print(f"[ERROR] Bot initialization failed: {e}")
        return
    
    # Test multiple currency pairs
    currency_pairs = ["EUR_USD", "GBP_USD", "USD_JPY"]
    
    results = []
    
    for pair in currency_pairs:
        print(f"\n--- TESTING {pair} ---")
        
        # Load real data
        data = load_real_data(pair, days=30)
        if data is None:
            continue
        
        # Test on different time periods
        test_periods = [
            ("Recent Week", data.tail(87)),      # Last 7 days + warmup
            ("Two Weeks Ago", data.iloc[-94:-87]), # Week before
            ("Month Ago", data.iloc[-100:-93])   # Earlier period
        ]
        
        pair_results = []
        
        for period_name, period_data in test_periods:
            if len(period_data) < 80:
                print(f"  [SKIP] {period_name}: Insufficient data")
                continue
            
            try:
                # Get recommendation
                recommendation = bot.get_final_recommendation(period_data, pair.replace("_", "/"))
                
                # Calculate actual price movement
                start_price = float(period_data['close'].iloc[-10])
                end_price = float(period_data['close'].iloc[-1])
                actual_change = (end_price - start_price) / start_price
                
                # Determine actual market direction
                if actual_change > 0.001:
                    actual_direction = "BUY"
                elif actual_change < -0.001:
                    actual_direction = "SELL"
                else:
                    actual_direction = "HOLD"
                
                # Check if bot prediction aligned
                predicted_action = recommendation['action']
                correct = predicted_action == actual_direction
                
                result = {
                    'period': period_name,
                    'predicted': predicted_action,
                    'actual': actual_direction,
                    'correct': correct,
                    'confidence': recommendation['confidence'],
                    'actual_change': actual_change,
                    'processing_time': recommendation['processing_time']
                }
                
                pair_results.append(result)
                
                # Display result
                status = "✓ CORRECT" if correct else "✗ WRONG"
                print(f"  {period_name}: {predicted_action} vs {actual_direction} - {status}")
                print(f"    Confidence: {recommendation['confidence']:.1%}, Actual change: {actual_change:+.2%}")
                
            except Exception as e:
                print(f"  [ERROR] {period_name}: {e}")
        
        results.extend(pair_results)
    
    # Summary analysis
    if results:
        print(f"\n" + "=" * 60)
        print("PERFORMANCE SUMMARY ON REAL DATA")
        print("=" * 60)
        
        total_predictions = len(results)
        correct_predictions = sum(1 for r in results if r['correct'])
        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        
        avg_confidence = np.mean([r['confidence'] for r in results])
        avg_processing_time = np.mean([float(r['processing_time'].replace('s', '')) for r in results])
        
        print(f"Total Predictions: {total_predictions}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.1f}%")
        print(f"Average Confidence: {avg_confidence:.1%}")
        print(f"Average Processing Time: {avg_processing_time:.3f}s")
        
        # Breakdown by action
        action_breakdown = {}
        for result in results:
            action = result['predicted']
            if action not in action_breakdown:
                action_breakdown[action] = {'total': 0, 'correct': 0}
            action_breakdown[action]['total'] += 1
            if result['correct']:
                action_breakdown[action]['correct'] += 1
        
        print(f"\nAccuracy by Signal Type:")
        for action, stats in action_breakdown.items():
            action_accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
            print(f"  {action}: {stats['correct']}/{stats['total']} ({action_accuracy:.1f}%)")
        
        # Performance assessment
        print(f"\nPerformance Assessment:")
        if accuracy >= 60:
            print("🎯 EXCELLENT: Bot shows strong predictive ability")
        elif accuracy >= 50:
            print("✅ GOOD: Bot performs better than random")
        elif accuracy >= 40:
            print("⚠️ FAIR: Bot shows some predictive ability")
        else:
            print("❌ POOR: Bot needs improvement")
        
        print(f"\nNote: This test shows how the bot would have performed")
        print(f"on actual historical market movements.")

def test_specific_market_events():
    """Test bot on specific market events"""
    print(f"\n" + "=" * 60)
    print("TESTING ON SPECIFIC HISTORICAL PERIODS")
    print("=" * 60)
    
    from ForexBot import ForexBot
    bot = ForexBot()
    
    # Load EUR/USD data for specific testing
    eur_usd = load_real_data("EUR_USD", days=200)
    
    if eur_usd is None:
        print("[ERROR] Cannot load EUR/USD data for event testing")
        return
    
    # Test different market conditions
    test_windows = [
        ("Early Data", eur_usd.iloc[80:160]),    # Early period
        ("Middle Data", eur_usd.iloc[160:240]),  # Middle period  
        ("Recent Data", eur_usd.iloc[-160:-80]), # Recent period
        ("Latest Data", eur_usd.tail(160))       # Most recent
    ]
    
    for window_name, window_data in test_windows:
        if len(window_data) < 80:
            continue
            
        try:
            rec = bot.get_final_recommendation(window_data, "EUR/USD")
            
            # Calculate volatility of this period
            returns = window_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Price range
            price_range = (window_data['close'].max() - window_data['close'].min()) / window_data['close'].mean()
            
            print(f"{window_name}:")
            print(f"  Signal: {rec['action']} ({rec['confidence']:.1%})")
            print(f"  Volatility: {volatility:.1%}")
            print(f"  Price Range: {price_range:.1%}")
            
        except Exception as e:
            print(f"{window_name}: Error - {e}")

if __name__ == "__main__":
    test_bot_on_real_data()
    test_specific_market_events()