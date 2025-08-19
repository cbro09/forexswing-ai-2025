#!/usr/bin/env python3
"""
Test Balanced Signal Generation
"""

import torch
import pandas as pd
import numpy as np
from models.ForexLSTM import SimpleOptimizedLSTM, create_simple_features

class BalancedForexBot:
    """ForexBot with aggressive balance testing"""
    
    def __init__(self):
        self.model = SimpleOptimizedLSTM(input_size=20, hidden_size=128, num_layers=3, dropout=0.4)
        self.load_model("data/models/optimized_forex_ai.pth")
        
        # More aggressive thresholds for balance testing
        self.thresholds = {
            'buy_threshold': 0.50,    # Higher threshold  
            'sell_threshold': 0.50,   # Higher threshold
            'hold_max': 0.55,         # Lower max to force more actions
            'confidence_boost': 1.1
        }
        
    def load_model(self, model_path):
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint, strict=False)
            self.model.eval()
        except Exception as e:
            print(f"Model loading error: {e}")
    
    def process_optimized_signal(self, model_output, enhanced_features, trend_signal, trend_strength):
        """Process signal with balanced logic"""
        
        probs = model_output[0].numpy()
        hold_prob = probs[0]
        buy_prob = probs[1]
        sell_prob = probs[2]
        
        # Debug output
        print(f"    Raw probs: HOLD={hold_prob:.3f}, BUY={buy_prob:.3f}, SELL={sell_prob:.3f}")
        
        # Apply calibrated thresholds
        buy_thresh = self.thresholds['buy_threshold']
        sell_thresh = self.thresholds['sell_threshold']
        hold_max = self.thresholds['hold_max']
        
        # Enhanced decision logic with debug
        base_signal = None
        base_confidence = 0
        
        # Check for HOLD dominance (force action if needed)
        if hold_prob > hold_max:
            print(f"    HOLD dominance detected ({hold_prob:.3f} > {hold_max})")
            if buy_prob > sell_prob:
                base_signal = "BUY"
                base_confidence = buy_prob * 1.1
                print(f"    Forced BUY (buy_prob {buy_prob:.3f} > sell_prob {sell_prob:.3f})")
            else:
                base_signal = "SELL"
                base_confidence = sell_prob * 1.1
                print(f"    Forced SELL (sell_prob {sell_prob:.3f} > buy_prob {buy_prob:.3f})")
        elif buy_prob > buy_thresh and buy_prob > sell_prob:
            base_signal = "BUY"
            base_confidence = buy_prob
            print(f"    Direct BUY ({buy_prob:.3f} > {buy_thresh} threshold)")
        elif sell_prob > sell_thresh and sell_prob > buy_prob:
            base_signal = "SELL"
            base_confidence = sell_prob
            print(f"    Direct SELL ({sell_prob:.3f} > {sell_thresh} threshold)")
        else:
            base_signal = "HOLD"
            base_confidence = hold_prob
            print(f"    Default HOLD (no thresholds met)")
        
        # Enhance with trend analysis
        final_signal = base_signal
        final_confidence = base_confidence
        
        if trend_signal == 'bullish' and base_signal == 'BUY':
            final_confidence *= (1.0 + trend_strength * 0.2)
        elif trend_signal == 'bearish' and base_signal == 'SELL':
            final_confidence *= (1.0 + trend_strength * 0.2)
        elif trend_signal == 'bullish' and base_signal == 'SELL':
            final_confidence *= 0.9
        elif trend_signal == 'bearish' and base_signal == 'BUY':
            final_confidence *= 0.9
        
        # Clamp confidence
        final_confidence = min(max(float(final_confidence), 0.1), 0.95)
        
        return final_signal, final_confidence
    
    def create_enhanced_features(self, data):
        """Create enhanced features for better accuracy"""
        prices = data['close'].values
        standard_features = create_simple_features(data, target_features=20)
        enhanced_features = []
        
        # Multi-timeframe momentum
        for period in [3, 7, 14]:
            if len(prices) > period:
                momentum = (prices[-1] - prices[-period-1]) / prices[-period-1]
                enhanced_features.append(momentum)
        
        return standard_features, enhanced_features
    
    def get_trend_signal(self, data):
        """Get trend-based signal strength"""
        prices = data['close'].values
        
        if len(prices) >= 21:
            ma_10 = np.mean(prices[-10:])
            ma_21 = np.mean(prices[-21:])
            current_price = prices[-1]
            
            score = 0
            if current_price > ma_10:
                score += 1
            if current_price > ma_21:
                score += 1
            if ma_10 > ma_21:
                score += 1
            
            if score >= 2:
                return 'bullish', score / 3.0
            elif score <= 1:
                return 'bearish', (3 - score) / 3.0
            else:
                return 'neutral', 0.5
        
        return 'neutral', 0.5
    
    def get_final_recommendation(self, dataframe, pair="EUR/USD"):
        """Get final optimized recommendation with detailed logging"""
        
        try:
            # Create enhanced features
            standard_features, enhanced_features = self.create_enhanced_features(dataframe)
            
            # Get trend signal
            trend_signal, trend_strength = self.get_trend_signal(dataframe)
            
            print(f"  Trend: {trend_signal} (strength: {trend_strength:.2f})")
            
            if len(standard_features) >= 80:
                # LSTM prediction
                sequence = standard_features[-80:]
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
                
                with torch.no_grad():
                    output = self.model(sequence_tensor)
                    
                    # Process with all optimizations
                    final_signal, final_confidence = self.process_optimized_signal(
                        output, enhanced_features, trend_signal, trend_strength
                    )
                
                return {
                    "action": final_signal,
                    "confidence": final_confidence,
                    "trend_signal": trend_signal,
                    "trend_strength": f"{trend_strength:.1%}",
                }
            
            return {"action": "HOLD", "confidence": 0.5, "error": "Insufficient data"}
            
        except Exception as e:
            return {"action": "HOLD", "confidence": 0.5, "error": str(e)}

def test_balanced_signals():
    """Test for balanced signal generation"""
    print("TESTING BALANCED SIGNAL GENERATION")
    print("=" * 60)
    
    bot = BalancedForexBot()
    
    # Create diverse test scenarios designed to trigger different signals
    test_scenarios = {
        "strong_bull": np.cumsum(np.random.randn(120) * 0.001 + 0.004) + 1.0850,     # Strong uptrend
        "strong_bear": np.cumsum(np.random.randn(120) * 0.001 - 0.004) + 1.0850,     # Strong downtrend  
        "moderate_bull": np.cumsum(np.random.randn(120) * 0.002 + 0.002) + 1.0850,   # Moderate uptrend
        "moderate_bear": np.cumsum(np.random.randn(120) * 0.002 - 0.002) + 1.0850,   # Moderate downtrend
        "sideways": np.random.randn(120) * 0.001 + 1.0850,                           # No trend
        "volatile_neutral": np.cumsum(np.random.randn(120) * 0.005) + 1.0850,        # High volatility
        "declining": np.linspace(1.0900, 1.0800, 120) + np.random.randn(120) * 0.0005, # Clear decline
        "rising": np.linspace(1.0800, 1.0900, 120) + np.random.randn(120) * 0.0005,    # Clear rise
    }
    
    print(f"Testing {len(test_scenarios)} market scenarios for signal diversity...\n")
    
    signals = []
    
    for scenario_name, prices in test_scenarios.items():
        print(f"{scenario_name.upper()}:")
        
        # Create test data
        test_data = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(50000, 200000, len(prices)),
            'high': prices + np.abs(np.random.randn(len(prices))) * 0.001,
            'low': prices - np.abs(np.random.randn(len(prices))) * 0.001,
        })
        
        # Get recommendation
        recommendation = bot.get_final_recommendation(test_data, "EUR/USD")
        signals.append(recommendation['action'])
        
        print(f"  Result: {recommendation['action']} (confidence: {recommendation['confidence']:.1%})")
        print()
    
    # Analyze balance
    print("=" * 60)
    print("SIGNAL BALANCE ANALYSIS")
    print("=" * 60)
    
    signal_counts = {"HOLD": 0, "BUY": 0, "SELL": 0}
    for signal in signals:
        signal_counts[signal] += 1
    
    total = len(signals)
    
    print(f"Signal Distribution:")
    for signal, count in signal_counts.items():
        percentage = (count / total) * 100
        print(f"  {signal}: {count}/{total} ({percentage:.1f}%)")
    
    diversity = len([count for count in signal_counts.values() if count > 0])
    print(f"\nSignal Diversity: {diversity}/3 types")
    
    if diversity == 3:
        print("[SUCCESS] All three signal types generated!")
    elif diversity == 2:
        print("[GOOD] Two signal types generated, needs improvement")
    else:
        print("[ISSUE] Signal bias still present")
    
    return signal_counts

if __name__ == "__main__":
    test_balanced_signals()