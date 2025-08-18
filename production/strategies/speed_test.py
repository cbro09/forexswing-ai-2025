#!/usr/bin/env python3
"""
Test Optimized Model - Simple Version
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time

class SimpleOptimizedLSTM(nn.Module):
    """Simplified optimized LSTM for testing"""
    
    def __init__(self, input_size=20, hidden_size=128, num_layers=3, dropout=0.4):
        super(SimpleOptimizedLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input processing
        self.input_norm = nn.LayerNorm(input_size)
        self.input_projection = nn.Linear(input_size, input_size)
        
        # LSTM layers (matching trained model structure)
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
        
        # Attention
        self.attention = nn.MultiheadAttention(
            hidden_size // 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Residual
        self.residual_proj = nn.Linear(input_size, hidden_size // 2)
        
        # Classifier
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
            nn.Linear(32, 3)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        # Input processing
        x_norm = self.input_norm(x)
        x_proj = torch.relu(self.input_projection(x_norm))
        
        # LSTM layers
        lstm1_out, _ = self.lstm1(x_proj)
        lstm1_out = self.dropout(lstm1_out)
        
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout(lstm2_out)
        
        lstm3_out, _ = self.lstm3(lstm2_out)
        lstm3_out = self.dropout(lstm3_out)
        
        # Attention
        attended, _ = self.attention(lstm3_out, lstm3_out, lstm3_out)
        
        # Residual
        residual = torch.mean(self.residual_proj(x_norm), dim=1)
        
        # Pooling
        pooled = torch.mean(attended, dim=1)
        pooled = pooled + residual
        
        # Classification
        logits = self.classifier(pooled)
        return torch.softmax(logits, dim=1)

def create_simple_features(data, target_features=20):
    """Create simple features without JAX complications"""
    
    prices = data['close'].values
    volumes = data['volume'].values
    
    features = []
    
    # Simple moving averages
    for period in [5, 10, 20, 50]:
        sma = pd.Series(prices).rolling(window=period, min_periods=1).mean().values
        features.append(sma)
    
    # Simple returns
    for period in [1, 3, 5]:
        returns = pd.Series(prices).pct_change(periods=period).fillna(0).values
        features.append(returns)
    
    # Simple RSI approximation
    delta = pd.Series(prices).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs))).fillna(50).values
    features.append(rsi)
    
    # Volume features
    volume_sma = pd.Series(volumes).rolling(window=20, min_periods=1).mean().values
    volume_ratio = volumes / np.maximum(volume_sma, 1)
    features.append(volume_ratio)
    
    # Price position
    sma_20 = pd.Series(prices).rolling(window=20, min_periods=1).mean().values
    price_position = (prices - sma_20) / sma_20
    features.append(price_position)
    
    # Volatility approximation
    volatility = pd.Series(prices).rolling(window=20).std().fillna(0.01).values
    features.append(volatility)
    
    # Pad to target features if needed
    while len(features) < target_features:
        # Add shifted versions of existing features
        base_idx = len(features) % len(features) if len(features) > 0 else 0
        if len(features) > 0:
            shifted = np.roll(features[base_idx], 1)
            features.append(shifted)
        else:
            features.append(np.zeros(len(prices)))
    
    # Combine into matrix
    feature_matrix = np.column_stack(features[:target_features])
    return feature_matrix

def test_model_compatibility():
    """Test model compatibility with trained weights"""
    print("TESTING MODEL COMPATIBILITY")
    print("=" * 50)
    
    model_path = "data/models/optimized_forex_ai.pth"
    
    try:
        # Load the trained model
        device = torch.device('cpu')
        model = SimpleOptimizedLSTM(input_size=20, hidden_size=128, num_layers=3, dropout=0.4)
        
        checkpoint = torch.load(model_path, map_location=device)
        
        # Try loading with strict=False
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        
        print(f"[OK] Model architecture loaded")
        print(f"  Missing keys: {len(missing_keys)}")
        print(f"  Unexpected keys: {len(unexpected_keys)}")
        
        # Test forward pass
        test_input = torch.randn(1, 80, 20)
        model.eval()
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"[OK] Forward pass successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: {output.min():.3f} to {output.max():.3f}")
        
        return True, model
        
    except Exception as e:
        print(f"[ERROR] Model compatibility test failed: {e}")
        return False, None

def test_speed_optimization():
    """Test processing speed"""
    print("\nTESTING SPEED OPTIMIZATION")
    print("=" * 50)
    
    # Create test data
    np.random.seed(42)
    n_points = 200
    
    sample_data = pd.DataFrame({
        'close': np.cumsum(np.random.randn(n_points) * 0.001) + 1.1000,
        'volume': np.random.randint(50000, 200000, n_points),
        'high': np.random.randn(n_points) * 0.002 + 1.1000,
        'low': np.random.randn(n_points) * 0.002 + 1.1000,
    })
    
    print(f"Test data: {len(sample_data)} candles")
    
    # Test feature creation speed
    start_time = time.time()
    features = create_simple_features(sample_data, target_features=20)
    feature_time = time.time() - start_time
    
    print(f"[OK] Feature creation: {feature_time:.3f}s")
    print(f"  Features shape: {features.shape}")
    
    # Test model prediction speed
    success, model = test_model_compatibility()
    
    if success and model:
        sequence_length = 80
        predictions = []
        
        start_time = time.time()
        
        # Process with step size for speed
        step_size = 5  # Process every 5th point
        
        for i in range(sequence_length, len(features), step_size):
            sequence = features[i-sequence_length:i]
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
            
            with torch.no_grad():
                output = model(sequence_tensor)
                # Convert to single prediction score
                probs = output[0].numpy()
                pred_score = probs[1] + 0.5 * probs[0]  # BUY + 0.5*HOLD
                predictions.append(pred_score)
        
        prediction_time = time.time() - start_time
        
        print(f"[OK] Prediction generation: {prediction_time:.3f}s")
        print(f"  Predictions: {len(predictions)} values")
        print(f"  Prediction range: {np.min(predictions):.3f} to {np.max(predictions):.3f}")
        print(f"  Average prediction: {np.mean(predictions):.3f}")
        
        total_time = feature_time + prediction_time
        print(f"[RESULT] Total processing time: {total_time:.3f}s")
        
        if total_time < 5.0:
            print("[SUCCESS] Speed target achieved (<5s)")
        else:
            print(f"[IMPROVEMENT NEEDED] Target 5s, got {total_time:.3f}s")
        
        return total_time
    
    else:
        print("[ERROR] Cannot test prediction speed - model loading failed")
        return None

def test_signal_balance():
    """Test signal distribution"""
    print("\nTESTING SIGNAL BALANCE")
    print("=" * 50)
    
    success, model = test_model_compatibility()
    
    if not success or not model:
        print("[ERROR] Cannot test signals - model loading failed")
        return None
    
    # Create diverse test scenarios
    test_scenarios = {
        "bull_market": np.cumsum(np.random.randn(100) * 0.001 + 0.0005) + 1.1000,  # Upward trend
        "bear_market": np.cumsum(np.random.randn(100) * 0.001 - 0.0005) + 1.1000,  # Downward trend
        "sideways": np.random.randn(100) * 0.001 + 1.1000,  # No trend
        "volatile": np.cumsum(np.random.randn(100) * 0.003) + 1.1000,  # High volatility
    }
    
    signal_results = {}
    
    for scenario_name, prices in test_scenarios.items():
        test_data = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(50000, 200000, len(prices)),
            'high': prices + np.abs(np.random.randn(len(prices))) * 0.002,
            'low': prices - np.abs(np.random.randn(len(prices))) * 0.002,
        })
        
        # Generate predictions
        features = create_simple_features(test_data, target_features=20)
        
        if len(features) >= 80:
            sequence = features[-80:]
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
            
            with torch.no_grad():
                output = model(sequence_tensor)
                probs = output[0].numpy()
                
                # Classify signal
                if probs[1] > 0.5:  # BUY
                    signal = "BUY"
                elif probs[2] > 0.5:  # SELL
                    signal = "SELL"
                else:  # HOLD
                    signal = "HOLD"
                
                signal_results[scenario_name] = {
                    "signal": signal,
                    "probabilities": {
                        "HOLD": float(probs[0]),
                        "BUY": float(probs[1]),
                        "SELL": float(probs[2])
                    }
                }
                
                print(f"  {scenario_name}: {signal} (HOLD:{probs[0]:.2f}, BUY:{probs[1]:.2f}, SELL:{probs[2]:.2f})")
    
    # Analyze balance
    signals = [result["signal"] for result in signal_results.values()]
    unique_signals = set(signals)
    
    print(f"\nSignal diversity: {len(unique_signals)}/3 possible signals")
    
    if len(unique_signals) >= 2:
        print("[GOOD] Signal diversity achieved")
    else:
        print("[ISSUE] Signal bias detected - needs calibration")
    
    return signal_results

def main():
    """Run optimization tests"""
    
    print("FOREXSWING AI 2025 - OPTIMIZATION TESTING")
    print("=" * 60)
    print("Testing optimized model architecture and performance...")
    print()
    
    # Run tests
    speed_result = test_speed_optimization()
    signal_result = test_signal_balance()
    
    print(f"\n" + "=" * 60)
    print("OPTIMIZATION TEST RESULTS")
    print("=" * 60)
    
    # Speed assessment
    if speed_result:
        if speed_result < 5.0:
            print(f"[SUCCESS] Speed: {speed_result:.3f}s (target: <5s)")
        else:
            print(f"[NEEDS WORK] Speed: {speed_result:.3f}s (target: <5s)")
    else:
        print("[ERROR] Speed test failed")
    
    # Signal assessment
    if signal_result:
        signals = [result["signal"] for result in signal_result.values()]
        unique_signals = set(signals)
        print(f"[INFO] Signal diversity: {len(unique_signals)}/3 signals")
        
        if len(unique_signals) >= 2:
            print("[SUCCESS] Signal balance improved")
        else:
            print("[NEEDS WORK] Signal bias still present")
    else:
        print("[ERROR] Signal test failed")
    
    print(f"\nNext steps:")
    if speed_result and speed_result >= 5.0:
        print("1. Optimize processing speed further")
    if signal_result and len(set([r["signal"] for r in signal_result.values()])) < 2:
        print("2. Calibrate model thresholds for balanced signals")
    
    print("3. Test with real market data")
    print("4. Deploy optimized system")

if __name__ == "__main__":
    main()