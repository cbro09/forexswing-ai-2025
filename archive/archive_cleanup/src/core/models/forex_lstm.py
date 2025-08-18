# user_data/ml_models/forex_lstm.py
"""
Forex LSTM Model with JAX-Accelerated Feature Engineering
Combines 125K+ calc/sec JAX indicators with PyTorch neural networks
"""

import torch
import torch.nn as nn
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
import os
import sys

# Import our existing JAX indicators
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from indicators.jax_indicators import jax_rsi, jax_sma, jax_macd

class ForexLSTM(nn.Module):
    """Advanced LSTM with Attention for Forex Prediction"""
    
    def __init__(self, input_size=10, hidden_size=64, num_layers=2, dropout=0.2):
        super(ForexLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM for better context
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism for focus on important patterns
        self.attention = nn.MultiheadAttention(
            hidden_size * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Output layers with residual connection
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.fc2 = nn.Linear(32, 3)  # Buy/Hold/Sell probabilities
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Self-attention mechanism
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take the last timestep
        final_output = attended[:, -1, :]
        
        # Classification layers
        out = torch.relu(self.fc1(final_output))
        out = self.dropout(out)
        logits = self.fc2(out)
        
        return torch.softmax(logits, dim=1)

class JAXFeatureEngine:
    """Ultra-fast feature engineering using JAX (125K+ calc/sec)"""
    
    @staticmethod
    @jit
    def create_technical_features(prices, volumes):
        """JAX-accelerated technical indicators"""
        
        # Core technical indicators (blazing fast with JAX)
        rsi_14 = jax_rsi(prices, 14)
        rsi_21 = jax_rsi(prices, 21)
        
        sma_20 = jax_sma(prices, 20)
        sma_50 = jax_sma(prices, 50)
        
        macd_line, signal_line, histogram = jax_macd(prices)
        
        # Price momentum features
        returns_1 = jnp.diff(prices) / prices[:-1]
        returns_5 = (prices[5:] - prices[:-5]) / prices[:-5]
        
        # Volatility features
        volatility_20 = jnp.array([
            jnp.std(returns_1[max(0, i-19):i+1]) if i >= 19 else 0.0
            for i in range(len(returns_1))
        ])
        
        # Volume features
        volume_sma = jax_sma(volumes, 20)
        volume_ratio = volumes / jnp.where(volume_sma == 0, 1, volume_sma)
        
        return {
            'rsi_14': rsi_14,
            'rsi_21': rsi_21,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram,
            'returns_1': jnp.concatenate([jnp.array([0.0]), returns_1]),
            'returns_5': jnp.concatenate([jnp.zeros(5), returns_5]),
            'volatility': jnp.concatenate([jnp.array([0.0]), volatility_20]),
            'volume_ratio': volume_ratio
        }
    
    @staticmethod
    def combine_features(feature_dict, min_length):
        """Combine all features into matrix for ML"""
        
        # Ensure all features are the same length
        features = []
        for key, values in feature_dict.items():
            if len(values) >= min_length:
                features.append(values[:min_length])
            else:
                # Pad with zeros if too short
                padded = jnp.concatenate([
                    jnp.zeros(min_length - len(values)),
                    values
                ])
                features.append(padded)
        
        return jnp.column_stack(features)

class HybridForexPredictor:
    """Combines JAX speed with PyTorch intelligence"""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ForexLSTM(input_size=11).to(self.device)  # 11 features
        self.feature_engine = JAXFeatureEngine()
        self.sequence_length = 60
        self.is_trained = False
        
        print(f"Initialized ForexLSTM on device: {self.device}")
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                self.is_trained = True
                print(f"Loaded pre-trained model from {model_path}")
            except Exception as e:
                print(f"Could not load model: {e}")
                print("Will use untrained model (random predictions)")
    
    def predict(self, dataframe):
        """Generate ML predictions using JAX + PyTorch hybrid approach"""
        
        if len(dataframe) < self.sequence_length:
            print(f"Insufficient data: {len(dataframe)} < {self.sequence_length}")
            return np.zeros(len(dataframe))
        
        try:
            # Ultra-fast feature creation with JAX (125K+ calc/sec)
            prices = jnp.array(dataframe['close'].values)
            volumes = jnp.array(dataframe['volume'].values)
            
            print(f"Creating features for {len(prices)} data points with JAX...")
            feature_dict = self.feature_engine.create_technical_features(prices, volumes)
            features_matrix = self.feature_engine.combine_features(feature_dict, len(prices))
            
            # Convert to numpy for PyTorch
            features_np = np.array(features_matrix)
            print(f"Features created: {features_np.shape}")
            
            predictions = []
            
            # Sliding window predictions with PyTorch LSTM
            for i in range(self.sequence_length, len(features_np)):
                sequence = features_np[i-self.sequence_length:i]
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.model(sequence_tensor)
                    # Get probability of bullish signal (buy class)
                    bull_prob = output[0][0].cpu().numpy()
                    predictions.append(float(bull_prob))
            
            # Pad with neutral values for initial sequence
            full_predictions = np.full(len(dataframe), 0.5)  # Neutral prediction
            full_predictions[self.sequence_length:] = predictions
            
            print(f"Generated {len(predictions)} predictions")
            return full_predictions
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return np.full(len(dataframe), 0.5)  # Return neutral predictions on error
    
    def get_model_info(self):
        """Return model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'device': str(self.device),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'is_trained': self.is_trained,
            'sequence_length': self.sequence_length
        }

# Test function
def test_model():
    """Test the hybrid model with sample data"""
    print("Testing HybridForexPredictor...")
    
    # Create sample dataframe
    import pandas as pd
    np.random.seed(42)
    
    n_points = 500
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
    
    print(f"Model test complete!")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Prediction range: {predictions.min():.3f} to {predictions.max():.3f}")
    print(f"Average prediction: {predictions.mean():.3f}")
    
    # Model info
    info = predictor.get_model_info()
    print(f"Model info: {info}")
    
    return predictor, predictions

if __name__ == "__main__":
    test_model()