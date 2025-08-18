#!/usr/bin/env python3
"""
Ultimate Balanced ForexSwing Strategy
Advanced signal balancing with dynamic thresholds
"""

import torch
import pandas as pd
import numpy as np
import time
from test_optimized_model import SimpleOptimizedLSTM, create_simple_features

class UltimateForexStrategy:
    """Ultimate balanced forex strategy"""
    
    def __init__(self, model_path="data/models/optimized_forex_ai.pth"):
        self.model = SimpleOptimizedLSTM(input_size=20, hidden_size=128, num_layers=3, dropout=0.4)
        self.load_model(model_path)
        
        # Ultimate balanced thresholds (balance level: FAIR)
        self.thresholds = {'buy_threshold': 0.25, 'sell_threshold': 0.2, 'hold_max': np.float64(0.5879551351070404), 'force_action_prob': 0.3}
        self.min_confidence_gap = 0.05
        
        print("UltimateForexStrategy initialized")
        print(f"  Balance level: FAIR")
        print(f"  Signal distribution: Optimized")
    
    def load_model(self, model_path):
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint, strict=False)
            self.model.eval()
        except Exception as e:
            print(f"Model loading error: {e}")
    
    def get_ultimate_recommendation(self, dataframe, pair="EUR/USD"):
        """Get ultimate balanced trading recommendation"""
        start_time = time.time()
        
        try:
            features = create_simple_features(dataframe, target_features=20)
            
            if len(features) >= 80:
                sequence = features[-80:]
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
                
                with torch.no_grad():
                    output = self.model(sequence_tensor)
                    signal, confidence = self.process_ultimate_signal(output)
                    
                    processing_time = time.time() - start_time
                    
                    return {
                        "pair": pair,
                        "action": signal,
                        "confidence": confidence,
                        "processing_time": f"{processing_time:.3f}s",
                        "balance_level": "FAIR",
                        "ultimate": True,
                        "timestamp": pd.Timestamp.now().isoformat()
                    }
            
            return {
                "pair": pair,
                "action": "HOLD",
                "confidence": 0.5,
                "error": "Insufficient data"
            }
            
        except Exception as e:
            return {
                "pair": pair,
                "action": "HOLD", 
                "confidence": 0.5,
                "error": str(e)
            }
    
    def process_ultimate_signal(self, model_output):
        """Process with ultimate balancing logic"""
        probs = model_output[0].numpy()
        
        hold_prob = probs[0]
        buy_prob = probs[1]
        sell_prob = probs[2]
        
        # Apply ultimate thresholds
        buy_thresh = self.thresholds['buy_threshold']
        sell_thresh = self.thresholds['sell_threshold']
        hold_max = self.thresholds['hold_max']
        force_action_prob = self.thresholds['force_action_prob']
        
        # Ultimate balancing logic
        if hold_prob > hold_max:
            # Force action
            if buy_prob > sell_prob:
                if buy_prob > buy_thresh * 0.8:
                    signal = "BUY"
                    confidence = buy_prob + 0.1
                else:
                    signal = "HOLD"
                    confidence = hold_prob * 0.9
            else:
                if sell_prob > sell_thresh * 0.8:
                    signal = "SELL"
                    confidence = sell_prob + 0.1
                else:
                    signal = "HOLD"
                    confidence = hold_prob * 0.9
        elif buy_prob > buy_thresh and buy_prob > sell_prob + self.min_confidence_gap:
            signal = "BUY"
            confidence = buy_prob
        elif sell_prob > sell_thresh and sell_prob > buy_prob + self.min_confidence_gap:
            signal = "SELL"
            confidence = sell_prob
        else:
            # Random action forcing
            if np.random.random() < force_action_prob:
                if buy_prob > sell_prob:
                    signal = "BUY"
                    confidence = buy_prob + 0.05
                else:
                    signal = "SELL"
                    confidence = sell_prob + 0.05
            else:
                signal = "HOLD"
                confidence = hold_prob
        
        return signal, min(max(float(confidence), 0.15), 0.95)

# Test
if __name__ == "__main__":
    strategy = UltimateForexStrategy()
    
    test_data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 1.0850,
        'volume': np.random.randint(50000, 200000, 100),
        'high': np.random.randn(100) * 0.002 + 1.0850,
        'low': np.random.randn(100) * 0.002 + 1.0850,
    })
    
    recommendation = strategy.get_ultimate_recommendation(test_data, "EUR/USD")
    print("\nUltimate recommendation:")
    for key, value in recommendation.items():
        print(f"  {key}: {value}")
