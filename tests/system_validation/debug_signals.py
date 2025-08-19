#!/usr/bin/env python3
"""
Debug Signal Balance - Analyze model probabilities
"""

import torch
import pandas as pd
import numpy as np
from models.ForexLSTM import SimpleOptimizedLSTM, create_simple_features

def debug_model_probabilities():
    """Debug what the model is actually outputting"""
    print("DEBUGGING MODEL PROBABILITIES")
    print("=" * 50)
    
    # Load model
    model = SimpleOptimizedLSTM(input_size=20, hidden_size=128, num_layers=3, dropout=0.4)
    try:
        checkpoint = torch.load("data/models/optimized_forex_ai.pth", map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        print("[OK] Model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        return
    
    # Create test scenarios
    test_scenarios = {
        "strong_bull": np.cumsum(np.random.randn(100) * 0.001 + 0.003) + 1.0850,  # Strong uptrend
        "strong_bear": np.cumsum(np.random.randn(100) * 0.001 - 0.003) + 1.0850,  # Strong downtrend
        "sideways": np.random.randn(100) * 0.001 + 1.0850,                        # No trend
        "volatile_up": np.cumsum(np.random.randn(100) * 0.005 + 0.001) + 1.0850,  # Volatile uptrend
        "volatile_down": np.cumsum(np.random.randn(100) * 0.005 - 0.001) + 1.0850, # Volatile downtrend
    }
    
    print(f"\nAnalyzing {len(test_scenarios)} market scenarios...")
    print("-" * 60)
    
    all_probs = []
    
    for scenario_name, prices in test_scenarios.items():
        # Create test data
        test_data = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(50000, 200000, len(prices)),
            'high': prices + np.abs(np.random.randn(len(prices))) * 0.001,
            'low': prices - np.abs(np.random.randn(len(prices))) * 0.001,
        })
        
        # Generate features and prediction
        features = create_simple_features(test_data, target_features=20)
        
        if len(features) >= 80:
            sequence = features[-80:]
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
            
            with torch.no_grad():
                output = model(sequence_tensor)
                probs = output[0].numpy()
                all_probs.append(probs)
                
                hold_prob, buy_prob, sell_prob = probs[0], probs[1], probs[2]
                
                print(f"{scenario_name:15} | HOLD: {hold_prob:.3f} | BUY: {buy_prob:.3f} | SELL: {sell_prob:.3f}")
    
    # Statistical analysis
    if all_probs:
        all_probs = np.array(all_probs)
        print(f"\n" + "=" * 60)
        print("PROBABILITY STATISTICS")
        print("=" * 60)
        
        hold_stats = all_probs[:, 0]
        buy_stats = all_probs[:, 1] 
        sell_stats = all_probs[:, 2]
        
        print(f"HOLD - Mean: {np.mean(hold_stats):.3f}, Std: {np.std(hold_stats):.3f}, Range: {np.min(hold_stats):.3f}-{np.max(hold_stats):.3f}")
        print(f"BUY  - Mean: {np.mean(buy_stats):.3f}, Std: {np.std(buy_stats):.3f}, Range: {np.min(buy_stats):.3f}-{np.max(buy_stats):.3f}")
        print(f"SELL - Mean: {np.mean(sell_stats):.3f}, Std: {np.std(sell_stats):.3f}, Range: {np.min(sell_stats):.3f}-{np.max(sell_stats):.3f}")
        
        # Suggest new thresholds
        print(f"\n" + "=" * 60)
        print("SUGGESTED THRESHOLDS")
        print("=" * 60)
        
        # Use percentiles for balanced thresholds
        buy_75th = np.percentile(buy_stats, 75)
        sell_75th = np.percentile(sell_stats, 75)
        hold_50th = np.percentile(hold_stats, 50)
        
        print(f"Current thresholds:")
        print(f"  buy_threshold: 0.25 (too low)")
        print(f"  sell_threshold: 0.20 (too low)")
        print(f"  hold_max: 0.588")
        
        print(f"\nSuggested balanced thresholds:")
        print(f"  buy_threshold: {buy_75th:.3f} (75th percentile)")
        print(f"  sell_threshold: {sell_75th:.3f} (75th percentile)")
        print(f"  hold_max: {hold_50th:.3f} (50th percentile)")
        
        return {
            'buy_threshold': round(buy_75th, 3),
            'sell_threshold': round(sell_75th, 3),
            'hold_max': round(hold_50th, 3)
        }
    
    return None

if __name__ == "__main__":
    suggested_thresholds = debug_model_probabilities()
    if suggested_thresholds:
        print(f"\nUse these thresholds in ForexBot.py:")
        print(f"self.thresholds = {suggested_thresholds}")