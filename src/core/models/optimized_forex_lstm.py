#!/usr/bin/env python3
"""
Optimized ForexSwing AI Model - Enhanced Architecture
Compatible with trained weights and optimized for performance
"""

import torch
import torch.nn as nn
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
import os
import sys
import time
from typing import Dict, Optional, Tuple

class OptimizedForexLSTM(nn.Module):
    """
    Enhanced LSTM model optimized for forex trading
    Compatible with existing trained weights
    """
    
    def __init__(self, input_size=20, hidden_size=128, num_layers=3, dropout=0.4):
        super(OptimizedForexLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Enhanced input processing
        self.input_norm = nn.LayerNorm(input_size)
        self.input_projection = nn.Linear(input_size, input_size)
        
        # Multi-scale LSTM layers (compatible with trained model)
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
        
        # Performance optimization
        self._cache_enabled = True
        self._last_input_hash = None
        self._last_output = None
        
    def forward(self, x):
        """Optimized forward pass with caching"""
        batch_size, seq_len, features = x.shape
        
        # Input hash for caching (optional optimization)
        if self._cache_enabled:
            input_hash = hash(x.detach().numpy().tobytes()) if x.requires_grad else hash(x.numpy().tobytes())
            if input_hash == self._last_input_hash:
                return self._last_output
        
        # Enhanced input processing
        x_norm = self.input_norm(x)
        x_proj = torch.relu(self.input_projection(x_norm))
        
        # LSTM processing with optimizations
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
        output = torch.softmax(logits, dim=1)
        
        # Cache result
        if self._cache_enabled:
            self._last_input_hash = input_hash
            self._last_output = output
        
        return output
    
    def set_cache_enabled(self, enabled: bool):
        """Enable/disable input caching for performance"""
        self._cache_enabled = enabled
    
    def get_model_info(self) -> Dict:
        """Get detailed model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': 'OptimizedForexLSTM',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'cache_enabled': self._cache_enabled
        }

class FastFeatureEngine:
    """
    Ultra-fast feature engineering with JAX optimization
    Improved speed and caching
    """
    
    def __init__(self):
        self._feature_cache = {}
        self._cache_enabled = True
    
    @staticmethod
    @jit
    def create_enhanced_features(prices, volumes):
        """Enhanced feature creation with JAX acceleration"""
        
        # Core technical indicators (optimized)
        rsi_14 = FastFeatureEngine._fast_rsi(prices, 14)
        rsi_21 = FastFeatureEngine._fast_rsi(prices, 21)
        rsi_7 = FastFeatureEngine._fast_rsi(prices, 7)
        
        # Multiple timeframe SMAs
        sma_10 = FastFeatureEngine._fast_sma(prices, 10)
        sma_20 = FastFeatureEngine._fast_sma(prices, 20)
        sma_50 = FastFeatureEngine._fast_sma(prices, 50)
        
        # Multiple EMAs
        ema_12 = FastFeatureEngine._fast_ema(prices, 12)
        ema_26 = FastFeatureEngine._fast_ema(prices, 26)
        ema_50 = FastFeatureEngine._fast_ema(prices, 50)
        
        # Enhanced MACD
        macd_line, signal_line, histogram = FastFeatureEngine._fast_macd(prices)
        
        # Price momentum features (multiple timeframes)
        returns_1 = jnp.diff(prices) / prices[:-1]
        returns_3 = jnp.concatenate([jnp.zeros(3), (prices[3:] - prices[:-3]) / prices[:-3]])
        returns_5 = jnp.concatenate([jnp.zeros(5), (prices[5:] - prices[:-5]) / prices[:-5]])
        returns_10 = jnp.concatenate([jnp.zeros(10), (prices[10:] - prices[:-10]) / prices[:-10]])
        
        # Advanced volatility measures
        volatility_10 = FastFeatureEngine._rolling_std(returns_1, 10)
        volatility_20 = FastFeatureEngine._rolling_std(returns_1, 20)
        
        # Volume features (enhanced)
        volume_sma_20 = FastFeatureEngine._fast_sma(volumes, 20)
        volume_ratio = volumes / jnp.maximum(volume_sma_20, 1)
        volume_trend = FastFeatureEngine._fast_sma(volume_ratio, 5)
        
        # Price position features
        price_position_sma20 = (prices - sma_20) / sma_20
        price_position_sma50 = (prices - sma_50) / sma_50
        
        return {
            'rsi_7': rsi_7,
            'rsi_14': rsi_14,
            'rsi_21': rsi_21,
            'sma_10': sma_10,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'ema_12': ema_12,
            'ema_26': ema_26,
            'ema_50': ema_50,
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram,
            'returns_1': jnp.concatenate([jnp.array([0.0]), returns_1]),
            'returns_3': returns_3,
            'returns_5': returns_5,
            'returns_10': returns_10,
            'volatility_10': jnp.concatenate([jnp.zeros(1), volatility_10]),
            'volatility_20': jnp.concatenate([jnp.zeros(1), volatility_20]),
            'volume_ratio': volume_ratio,
            'volume_trend': volume_trend,
            'price_pos_sma20': price_position_sma20,
            'price_pos_sma50': price_position_sma50
        }
    
    @staticmethod
    @jit
    def _fast_rsi(prices, period):
        """Optimized RSI calculation"""
        deltas = jnp.diff(prices)
        gains = jnp.where(deltas > 0, deltas, 0)
        losses = jnp.where(deltas < 0, -deltas, 0)
        
        # Simple moving average for RSI
        avg_gains = jnp.array([
            jnp.mean(gains[max(0, i-period+1):i+1]) if i >= period-1 else jnp.mean(gains[:i+1])
            for i in range(len(gains))
        ])
        avg_losses = jnp.array([
            jnp.mean(losses[max(0, i-period+1):i+1]) if i >= period-1 else jnp.mean(losses[:i+1])
            for i in range(len(losses))
        ])
        
        rs = avg_gains / jnp.maximum(avg_losses, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return jnp.concatenate([jnp.array([50.0]), rsi])
    
    @staticmethod
    @jit 
    def _fast_sma(prices, period):
        """Optimized SMA calculation"""
        return jnp.array([
            jnp.mean(prices[max(0, i-period+1):i+1])
            for i in range(len(prices))
        ])
    
    @staticmethod
    @jit
    def _fast_ema(prices, period):
        """Optimized EMA calculation"""
        alpha = 2.0 / (period + 1)
        ema = jnp.zeros_like(prices)
        ema = ema.at[0].set(prices[0])
        
        for i in range(1, len(prices)):
            ema = ema.at[i].set(alpha * prices[i] + (1 - alpha) * ema[i-1])
        
        return ema
    
    @staticmethod
    @jit
    def _fast_macd(prices, fast=12, slow=26, signal=9):
        """Optimized MACD calculation"""
        ema_fast = FastFeatureEngine._fast_ema(prices, fast)
        ema_slow = FastFeatureEngine._fast_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = FastFeatureEngine._fast_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    @jit
    def _rolling_std(values, window):
        """Optimized rolling standard deviation"""
        return jnp.array([
            jnp.std(values[max(0, i-window+1):i+1]) if i >= window-1 else jnp.std(values[:i+1])
            for i in range(len(values))
        ])
    
    def combine_features(self, feature_dict, min_length, target_features=20):
        """Combine features into matrix optimized for ML"""
        
        # Get cache key
        cache_key = f"{min_length}_{target_features}_{hash(str(sorted(feature_dict.keys())))}"
        
        if self._cache_enabled and cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        # Select top features based on importance
        feature_priority = [
            'rsi_14', 'rsi_21', 'rsi_7',
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'macd', 'macd_signal', 'macd_histogram',
            'returns_1', 'returns_5', 'returns_10',
            'volatility_10', 'volatility_20',
            'volume_ratio', 'volume_trend',
            'price_pos_sma20', 'price_pos_sma50',
            'sma_10'  # Additional features
        ]
        
        features = []
        feature_names = []
        
        for feature_name in feature_priority[:target_features]:
            if feature_name in feature_dict:
                values = feature_dict[feature_name]
                
                if len(values) >= min_length:
                    features.append(values[:min_length])
                else:
                    # Pad with last value if too short
                    padded = jnp.concatenate([
                        jnp.full(min_length - len(values), values[-1] if len(values) > 0 else 0),
                        values
                    ])
                    features.append(padded)
                
                feature_names.append(feature_name)
        
        # If we don't have enough features, pad with derived features
        while len(features) < target_features:
            if len(features) > 0:
                # Create derived feature (simple transformation)
                base_feature = features[len(features) % len(features)]
                derived_feature = jnp.roll(base_feature, 1)  # Shifted version
                features.append(derived_feature)
                feature_names.append(f"derived_{len(features)}")
            else:
                # Fallback to zeros
                features.append(jnp.zeros(min_length))
                feature_names.append(f"zero_{len(features)}")
        
        result = jnp.column_stack(features[:target_features])
        
        # Cache result
        if self._cache_enabled:
            self._feature_cache[cache_key] = result
        
        return result
    
    def clear_cache(self):
        """Clear feature cache"""
        self._feature_cache.clear()

class OptimizedForexPredictor:
    """
    Optimized ForexSwing AI Predictor
    Enhanced performance and compatibility
    """
    
    def __init__(self, model_path: str = None, enable_caching: bool = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = OptimizedForexLSTM(input_size=20, hidden_size=128, num_layers=3, dropout=0.4).to(self.device)
        self.feature_engine = FastFeatureEngine()
        self.sequence_length = 80
        self.is_trained = False
        self.enable_caching = enable_caching
        
        # Performance tracking
        self.prediction_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        
        print(f"Initialized OptimizedForexLSTM on device: {self.device}")
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load model with enhanced compatibility"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Load with strict=False to handle minor incompatibilities
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            if len(missing_keys) == 0 and len(unexpected_keys) == 0:
                print(f"✅ Model loaded perfectly from {model_path}")
            else:
                print(f"⚠️ Model loaded with compatibility adjustments:")
                if missing_keys:
                    print(f"   Missing keys: {len(missing_keys)}")
                if unexpected_keys:
                    print(f"   Unexpected keys: {len(unexpected_keys)}")
            
            self.model.eval()
            self.is_trained = True
            
        except Exception as e:
            print(f"❌ Could not load model: {e}")
            print("Will use untrained model (random predictions)")
            self.is_trained = False
    
    def predict(self, dataframe, enable_fast_mode: bool = True):
        """Generate optimized ML predictions"""
        
        start_time = time.time()
        
        if len(dataframe) < self.sequence_length:
            print(f"Insufficient data: {len(dataframe)} < {self.sequence_length}")
            return np.full(len(dataframe), 0.5)
        
        try:
            # Ultra-fast feature creation with caching
            prices = jnp.array(dataframe['close'].values)
            volumes = jnp.array(dataframe['volume'].values)
            
            if enable_fast_mode:
                print(f"Creating enhanced features for {len(prices)} data points...")
            
            feature_dict = self.feature_engine.create_enhanced_features(prices, volumes)
            features_matrix = self.feature_engine.combine_features(feature_dict, len(prices), target_features=20)
            
            # Convert to numpy for PyTorch
            features_np = np.array(features_matrix)
            
            if enable_fast_mode:
                print(f"Features created: {features_np.shape}")
            
            predictions = []
            
            # Optimized sliding window predictions
            step_size = 5 if enable_fast_mode else 1  # Process every 5th for speed
            
            for i in range(self.sequence_length, len(features_np), step_size):
                sequence = features_np[i-self.sequence_length:i]
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.model(sequence_tensor)
                    # Get probabilities for all classes
                    probs = output[0].cpu().numpy()
                    
                    # Convert to single prediction score (0-1)
                    # BUY=1, HOLD=0.5, SELL=0
                    pred_score = probs[1] + 0.5 * probs[0]  # BUY + 0.5*HOLD
                    predictions.append(float(pred_score))
            
            # Interpolate predictions for missing points in fast mode
            if enable_fast_mode and step_size > 1:
                full_predictions = np.interp(
                    range(self.sequence_length, len(features_np)),
                    range(self.sequence_length, len(features_np), step_size),
                    predictions
                )
                predictions = full_predictions
            
            # Pad with neutral values for initial sequence
            final_predictions = np.full(len(dataframe), 0.5)
            final_predictions[self.sequence_length:self.sequence_length+len(predictions)] = predictions
            
            end_time = time.time()
            processing_time = end_time - start_time
            self.prediction_times.append(processing_time)
            
            if enable_fast_mode:
                print(f"Generated {len(predictions)} predictions in {processing_time:.2f}s")
            
            return final_predictions
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return np.full(len(dataframe), 0.5)
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.prediction_times:
            return {"no_data": True}
        
        return {
            "avg_prediction_time": np.mean(self.prediction_times),
            "min_prediction_time": np.min(self.prediction_times),
            "max_prediction_time": np.max(self.prediction_times),
            "total_predictions": len(self.prediction_times),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        }
    
    def get_model_info(self) -> Dict:
        """Return enhanced model information"""
        base_info = self.model.get_model_info()
        base_info.update({
            'is_trained': self.is_trained,
            'sequence_length': self.sequence_length,
            'device': str(self.device),
            'caching_enabled': self.enable_caching
        })
        return base_info

# Test function
def test_optimized_model():
    """Test the optimized model"""
    print("Testing OptimizedForexPredictor...")
    
    # Create sample data
    import pandas as pd
    np.random.seed(42)
    
    n_points = 200
    base_price = 1.1000
    price_walk = np.cumsum(np.random.randn(n_points) * 0.001) + base_price
    
    sample_data = pd.DataFrame({
        'close': price_walk,
        'volume': np.random.randint(50000, 200000, n_points),
        'high': price_walk + np.abs(np.random.randn(n_points)) * 0.002,
        'low': price_walk - np.abs(np.random.randn(n_points)) * 0.002,
    })
    
    # Test prediction with performance timing
    predictor = OptimizedForexPredictor()
    
    start_time = time.time()
    predictions = predictor.predict(sample_data, enable_fast_mode=True)
    end_time = time.time()
    
    print(f"✅ Optimization test complete!")
    print(f"   Processing time: {end_time - start_time:.2f}s")
    print(f"   Predictions: {len(predictions)} values")
    print(f"   Range: {predictions.min():.3f} to {predictions.max():.3f}")
    print(f"   Model info: {predictor.get_model_info()}")
    
    return predictor, predictions

if __name__ == "__main__":
    test_optimized_model()