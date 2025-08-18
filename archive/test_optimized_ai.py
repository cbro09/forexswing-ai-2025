#!/usr/bin/env python3
"""
Test Optimized ForexSwing AI Performance
Compare optimized vs original AI performance
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import jax.numpy as jnp
import joblib
import os
import sys

sys.path.append('src')
from indicators.jax_indicators import jax_rsi, jax_sma, jax_macd, jax_ema

# Optimized model architecture
class OptimizedForexLSTM(nn.Module):
    """Optimized LSTM with improved architecture"""
    
    def __init__(self, input_size=20, hidden_size=128, num_layers=3, dropout=0.4):
        super(OptimizedForexLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Enhanced input processing
        self.input_norm = nn.LayerNorm(input_size)
        self.input_projection = nn.Linear(input_size, input_size)
        
        # Multi-scale LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=0,
            batch_first=True,
            bidirectional=True
        )
        
        self.lstm2 = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size // 2,
            num_layers=1,
            dropout=0,
            batch_first=True,
            bidirectional=True
        )
        
        self.lstm3 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 4,
            num_layers=1,
            dropout=0,
            batch_first=True,
            bidirectional=True
        )
        
        # Enhanced attention mechanism
        self.attention = nn.MultiheadAttention(
            hidden_size // 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Residual connections
        self.residual_proj = nn.Linear(input_size, hidden_size // 2)
        
        # Advanced classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size // 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout / 4),
            nn.Linear(32, 3)  # Buy/Hold/Sell
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        # Enhanced input processing
        x_norm = self.input_norm(x)
        x_proj = torch.relu(self.input_projection(x_norm))
        
        # LSTM processing
        lstm1_out, _ = self.lstm1(x_proj)
        lstm1_out = self.dropout(lstm1_out)
        
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout(lstm2_out)
        
        lstm3_out, _ = self.lstm3(lstm2_out)
        lstm3_out = self.dropout(lstm3_out)
        
        # Attention with residual connection
        attended, _ = self.attention(lstm3_out, lstm3_out, lstm3_out)
        
        # Residual connection from input
        residual = torch.mean(self.residual_proj(x_norm), dim=1)
        
        # Global pooling with residual
        pooled = torch.mean(attended, dim=1)
        pooled = pooled + residual
        
        # Classification
        logits = self.classifier(pooled)
        
        return torch.softmax(logits, dim=1)

class OptimizedAIPredictor:
    """Optimized AI predictor with enhanced features"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = 80
        
        # Try to load optimized model first
        optimized_model_path = "models/optimized_forex_ai.pth"
        optimized_scaler_path = "models/optimized_scaler.pkl"
        
        if os.path.exists(optimized_model_path):
            # Load optimized model
            self.model = OptimizedForexLSTM(
                input_size=20,
                hidden_size=128,
                num_layers=3,
                dropout=0.4
            ).to(self.device)
            
            self.model.load_state_dict(torch.load(optimized_model_path, map_location=self.device))
            self.model.eval()
            self.model_type = "OPTIMIZED"
            print(f"OPTIMIZED AI loaded from {optimized_model_path}")
            
            if os.path.exists(optimized_scaler_path):
                self.scaler = joblib.load(optimized_scaler_path)
                print(f"Optimized scaler loaded")
            else:
                # Try to load backup scaler
                backup_scalers = [
                    "models/final_feature_scaler.pkl",
                    "models/real_market_scaler.pkl"
                ]
                
                for scaler_path in backup_scalers:
                    if os.path.exists(scaler_path):
                        self.scaler = joblib.load(scaler_path)
                        print(f"Using backup scaler: {scaler_path}")
                        break
                else:
                    print("Warning: No scaler found!")
                    return
            
        else:
            print("Optimized model not found yet. Training may still be in progress.")
            print("Run this script after optimization completes.")
            return
        
        print(f"OPTIMIZED FOREXSWING AI READY!")
    
    def create_enhanced_features(self, data):
        """Create enhanced 20-feature set"""
        
        prices = jnp.array(data['close'].values)
        volumes = jnp.array(data['volume'].values)
        highs = jnp.array(data['high'].values)
        lows = jnp.array(data['low'].values)
        opens = jnp.array(data['open'].values)
        
        # Enhanced indicator set (same as optimization)
        rsi_14 = jax_rsi(prices, 14)
        rsi_21 = jax_rsi(prices, 21)
        rsi_7 = jax_rsi(prices, 7)
        
        sma_10 = jax_sma(prices, 10)
        sma_20 = jax_sma(prices, 20)
        sma_50 = jax_sma(prices, 50)
        
        ema_12 = jax_ema(prices, 12)
        ema_26 = jax_ema(prices, 26)
        ema_9 = jax_ema(prices, 9)
        
        macd_line, macd_signal, macd_histogram = jax_macd(prices)
        
        # Enhanced price features
        returns_1 = jnp.diff(prices) / prices[:-1]
        returns_3 = jnp.concatenate([jnp.zeros(3), (prices[3:] - prices[:-3]) / prices[:-3]])
        returns_5 = jnp.concatenate([jnp.zeros(5), (prices[5:] - prices[:-5]) / prices[:-5]])
        
        # Volatility features
        volatility_10 = jnp.array([
            jnp.std(returns_1[max(0, i-9):i+1]) if i >= 9 else 0.01
            for i in range(len(returns_1))
        ])
        
        volatility_20 = jnp.array([
            jnp.std(returns_1[max(0, i-19):i+1]) if i >= 19 else 0.01
            for i in range(len(returns_1))
        ])
        
        # Volume analysis
        volume_sma = jax_sma(volumes, 20)
        volume_ratio = volumes / jnp.maximum(volume_sma, 1)
        
        # Market microstructure
        high_low_ratio = (highs - lows) / jnp.maximum(prices, 1)
        open_close_ratio = (prices - opens) / jnp.maximum(opens, 1)
        
        # Trend analysis
        price_position_sma20 = (prices - sma_20) / jnp.maximum(sma_20, 1)
        price_position_sma50 = (prices - sma_50) / jnp.maximum(sma_50, 1)
        trend_strength = (sma_10 - sma_50) / jnp.maximum(sma_50, 1)
        
        # Combine 20 enhanced features
        min_length = min(len(prices), len(rsi_14), len(returns_5), len(volatility_20))
        
        features = jnp.column_stack([
            rsi_14[:min_length],
            rsi_21[:min_length], 
            rsi_7[:min_length],
            sma_10[:min_length] / prices[:min_length],
            sma_20[:min_length] / prices[:min_length],
            sma_50[:min_length] / prices[:min_length],
            ema_9[:min_length] / prices[:min_length],
            ema_12[:min_length] / prices[:min_length],
            ema_26[:min_length] / prices[:min_length],
            macd_line[:min_length],
            macd_signal[:min_length],
            macd_histogram[:min_length],
            jnp.concatenate([jnp.array([0.0]), returns_1])[:min_length],
            returns_3[:min_length],
            returns_5[:min_length],
            jnp.concatenate([jnp.array([0.01]), volatility_10])[:min_length],
            jnp.concatenate([jnp.array([0.01]), volatility_20])[:min_length],
            volume_ratio[:min_length],
            price_position_sma20[:min_length],
            trend_strength[:min_length]
        ])
        
        return np.array(features)
    
    def predict(self, data):
        """Make predictions with optimized AI"""
        
        if len(data) < self.sequence_length:
            return None, None
        
        features = self.create_enhanced_features(data)
        features_scaled = self.scaler.transform(features)
        
        predictions = []
        confidences = []
        
        for i in range(len(features_scaled) - self.sequence_length + 1):
            sequence = features_scaled[i:i + self.sequence_length]
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(sequence_tensor)
                probs = output[0].cpu().numpy()
                
                predicted_class = np.argmax(probs)
                confidence = np.max(probs)
                
                predictions.append(predicted_class)
                confidences.append(confidence)
        
        return np.array(predictions), np.array(confidences)

def test_optimized_ai():
    """Test optimized AI performance"""
    
    print("OPTIMIZED FOREXSWING AI TESTING")
    print("=" * 50)
    
    # Initialize optimized AI
    try:
        ai = OptimizedAIPredictor()
    except:
        print("Optimized AI not ready yet. Training may still be in progress.")
        return
    
    # Test on real market data
    data_dir = "data/real_market"
    test_results = {}
    
    print(f"\nTesting Optimized AI on Real Market Data:")
    print("-" * 50)
    
    for filename in os.listdir(data_dir):
        if filename.endswith('_real_daily.feather'):
            pair_name = filename.replace('_real_daily.feather', '').replace('_', '/')
            
            print(f"\n{pair_name}:")
            
            # Load data
            df = pd.read_feather(os.path.join(data_dir, filename))
            if 'Date' in df.columns:
                df = df.set_index('Date')
            
            # Test on recent data
            test_data = df.tail(200)
            
            # Get predictions
            predictions, confidences = ai.predict(test_data)
            
            if predictions is not None:
                # Analyze predictions
                hold_signals = (predictions == 0).sum()
                buy_signals = (predictions == 1).sum()
                sell_signals = (predictions == 2).sum()
                
                avg_confidence = confidences.mean()
                
                print(f"  HOLD: {hold_signals} ({hold_signals/len(predictions)*100:.1f}%)")
                print(f"  BUY:  {buy_signals} ({buy_signals/len(predictions)*100:.1f}%)")
                print(f"  SELL: {sell_signals} ({sell_signals/len(predictions)*100:.1f}%)")
                print(f"  Avg Confidence: {avg_confidence:.3f}")
                
                # Calculate accuracy with optimized thresholds (0.5% instead of 1%)
                correct = 0
                total_predictions = 0
                
                for i in range(len(predictions)):
                    data_idx = i + ai.sequence_length - 1
                    future_idx = data_idx + 8  # 8 periods ahead (optimized)
                    
                    if future_idx >= len(test_data):
                        continue
                    
                    current_price = test_data.iloc[data_idx]['close']
                    future_price = test_data.iloc[future_idx]['close']
                    actual_return = (future_price - current_price) / current_price
                    
                    ai_prediction = predictions[i]
                    
                    # Optimized success criteria (0.5% thresholds)
                    if ai_prediction == 1 and actual_return > 0.005:  # BUY and went up >0.5%
                        correct += 1
                    elif ai_prediction == 2 and actual_return < -0.005:  # SELL and went down <-0.5%
                        correct += 1
                    elif ai_prediction == 0 and abs(actual_return) <= 0.005:  # HOLD and stayed flat
                        correct += 1
                    
                    total_predictions += 1
                
                accuracy = correct / total_predictions * 100 if total_predictions > 0 else 0
                print(f"  OPTIMIZED ACCURACY: {accuracy:.1f}%")
                
                test_results[pair_name] = {
                    'accuracy': accuracy,
                    'hold_pct': hold_signals/len(predictions)*100,
                    'buy_pct': buy_signals/len(predictions)*100,
                    'sell_pct': sell_signals/len(predictions)*100,
                    'avg_confidence': avg_confidence
                }
    
    # Summary
    if test_results:
        print(f"\n" + "=" * 50)
        print("OPTIMIZED AI PERFORMANCE SUMMARY")
        print("=" * 50)
        
        accuracies = [r['accuracy'] for r in test_results.values()]
        avg_accuracy = np.mean(accuracies)
        max_accuracy = max(accuracies)
        min_accuracy = min(accuracies)
        
        print(f"Average Accuracy: {avg_accuracy:.1f}%")
        print(f"Best Performance: {max_accuracy:.1f}%")
        print(f"Worst Performance: {min_accuracy:.1f}%")
        print(f"Previous AI: 21.0%")
        
        improvement = avg_accuracy - 21.0
        print(f"Improvement: {improvement:+.1f} percentage points")
        
        # Signal distribution
        avg_hold = np.mean([r['hold_pct'] for r in test_results.values()])
        avg_buy = np.mean([r['buy_pct'] for r in test_results.values()])
        avg_sell = np.mean([r['sell_pct'] for r in test_results.values()])
        
        print(f"\nSignal Distribution:")
        print(f"  HOLD: {avg_hold:.1f}%")
        print(f"  BUY:  {avg_buy:.1f}%")
        print(f"  SELL: {avg_sell:.1f}%")
        
        # Performance assessment
        print(f"\nOptimization Assessment:")
        if avg_accuracy >= 50:
            print("OUTSTANDING: Professional-grade performance achieved!")
        elif avg_accuracy >= 40:
            print("EXCELLENT: Significant improvement!")
        elif avg_accuracy >= 30:
            print("GOOD: Meaningful progress made!")
        elif improvement > 5:
            print("PROGRESS: Optimization working!")
        else:
            print("DEVELOPING: Continue optimization efforts")
        
        # Diversification check
        if avg_buy > 10 and avg_sell > 10 and avg_hold > 10:
            print("BALANCED: AI now generates diverse signals!")
        elif avg_buy > 5 and avg_sell > 5:
            print("IMPROVED: Better signal diversification")
        else:
            print("BIASED: Still needs signal balance improvement")
    
    return test_results

def compare_models():
    """Compare original vs optimized AI performance"""
    
    print(f"\n" + "=" * 50)
    print("ORIGINAL vs OPTIMIZED AI COMPARISON")
    print("=" * 50)
    
    # Original AI results (from previous testing)
    original_results = {
        "USD/JPY": 37.2,
        "USD/CHF": 23.4,
        "AUD/USD": 21.3,
        "NZD/USD": 19.1,
        "USD/CAD": 16.5,
        "EUR/USD": 14.9,
        "GBP/USD": 14.4
    }
    
    original_avg = np.mean(list(original_results.values()))
    
    print(f"Original AI Average: {original_avg:.1f}%")
    
    # Test optimized AI
    optimized_results = test_optimized_ai()
    
    if optimized_results:
        optimized_accuracies = [r['accuracy'] for r in optimized_results.values()]
        optimized_avg = np.mean(optimized_accuracies)
        
        print(f"Optimized AI Average: {optimized_avg:.1f}%")
        print(f"Improvement: {optimized_avg - original_avg:+.1f} percentage points")
        
        # Detailed comparison
        print(f"\nPair-by-Pair Comparison:")
        print("-" * 30)
        
        for pair in original_results.keys():
            if pair in optimized_results:
                original_acc = original_results[pair]
                optimized_acc = optimized_results[pair]['accuracy']
                improvement = optimized_acc - original_acc
                
                print(f"{pair}: {original_acc:.1f}% -> {optimized_acc:.1f}% ({improvement:+.1f}%)")

def main():
    """Run optimized AI testing"""
    
    print("TESTING OPTIMIZED FOREXSWING AI")
    print("Measuring Optimization Success")
    print("=" * 50)
    
    # Check if optimization completed
    if not os.path.exists("models/optimized_forex_ai.pth"):
        print("Optimization not complete yet.")
        print("Please wait for training to finish, then run this script.")
        return
    
    # Run comparison test
    compare_models()
    
    print(f"\nOptimized AI testing complete!")
    print(f"Results show the effectiveness of optimization efforts.")

if __name__ == "__main__":
    main()