#!/usr/bin/env python3
"""
ForexSwing AI Optimization Pipeline
Transform 21% accuracy into professional-grade performance!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import pandas as pd
import numpy as np
import jax.numpy as jnp
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import os
import sys
import joblib
from collections import Counter

sys.path.append('../../core')
from indicators.jax_indicators import jax_rsi, jax_sma, jax_macd, jax_ema

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

class AIOptimizer:
    """AI Optimization Pipeline"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = 80  # Increased for better patterns
        self.future_periods = 8    # Reduced for more frequent signals
        
        print("AI OPTIMIZATION PIPELINE INITIALIZED")
        print("=" * 50)
        print(f"Device: {self.device}")
        print(f"Target: Transform 21% to 50%+ accuracy")
        print(f"Strategy: Multi-level optimization")
    
    def load_and_prepare_data(self):
        """Load and prepare optimized training data"""
        
        print("\n1. LOADING & PREPARING DATA")
        print("-" * 30)
        
        # Load real market data
        data_dir = "data/real_market"
        all_data = []
        
        for filename in os.listdir(data_dir):
            if filename.endswith('_real_daily.feather'):
                df = pd.read_feather(os.path.join(data_dir, filename))
                
                if 'Date' in df.columns:
                    df = df.set_index('Date')
                
                pair_name = filename.replace('_real_daily.feather', '').replace('_', '/')
                df['pair'] = pair_name
                all_data.append(df)
                print(f"  Loaded {len(df)} days of {pair_name}")
        
        combined_data = pd.concat(all_data).sort_index()
        print(f"  Total: {len(combined_data)} candles")
        
        # OPTIMIZATION 1: Use more data periods
        # Instead of just recent data, use multiple periods
        train_periods = [
            combined_data.iloc[:800],     # Early period
            combined_data.iloc[400:1200], # Mid period  
            combined_data.tail(800)       # Recent period
        ]
        
        print(f"  Using 3 different time periods for training")
        return train_periods
    
    def create_enhanced_features(self, data):
        """Create enhanced feature set with 20 features"""
        
        features_list = []
        labels_list = []
        
        for pair in data['pair'].unique():
            pair_data = data[data['pair'] == pair].copy()
            
            if len(pair_data) < 100:  # Skip if too little data
                continue
                
            print(f"    Processing {pair}...")
            
            # Convert to JAX arrays
            prices = jnp.array(pair_data['close'].values)
            volumes = jnp.array(pair_data['volume'].values)
            highs = jnp.array(pair_data['high'].values)
            lows = jnp.array(pair_data['low'].values)
            opens = jnp.array(pair_data['open'].values)
            
            # Enhanced indicator set
            rsi_14 = jax_rsi(prices, 14)
            rsi_21 = jax_rsi(prices, 21)
            rsi_7 = jax_rsi(prices, 7)  # Short-term RSI
            
            sma_10 = jax_sma(prices, 10)
            sma_20 = jax_sma(prices, 20)
            sma_50 = jax_sma(prices, 50)
            
            ema_12 = jax_ema(prices, 12)
            ema_26 = jax_ema(prices, 26)
            ema_9 = jax_ema(prices, 9)   # Fast EMA
            
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
            
            # OPTIMIZATION 2: Better label thresholds
            # Use 0.5% instead of 1% for more balanced labels
            future_returns = []
            for i in range(len(prices) - self.future_periods):
                current_price = prices[i]
                future_price = prices[i + self.future_periods]
                return_pct = (future_price - current_price) / current_price
                
                # OPTIMIZED THRESHOLDS
                if return_pct > 0.005:      # >0.5% = Buy (class 1)
                    label = 1
                elif return_pct < -0.005:   # <-0.5% = Sell (class 2)
                    label = 2
                else:                       # Otherwise = Hold (class 0)
                    label = 0
                
                future_returns.append(label)
            
            # Align features with labels
            min_length = min(len(features), len(future_returns))
            aligned_features = np.array(features[:min_length])
            aligned_labels = np.array(future_returns[:min_length])
            
            features_list.append(aligned_features)
            labels_list.append(aligned_labels)
            
            print(f"      {len(aligned_features)} feature vectors")
        
        all_features = np.vstack(features_list)
        all_labels = np.concatenate(labels_list)
        
        # OPTIMIZATION 3: Balance classes
        label_counts = Counter(all_labels)
        print(f"    Label distribution: {dict(label_counts)}")
        
        # Calculate balance ratios
        total_samples = len(all_labels)
        for label, count in label_counts.items():
            pct = count / total_samples * 100
            print(f"      Class {label}: {count} ({pct:.1f}%)")
        
        return all_features, all_labels
    
    def create_sequences(self, features, labels):
        """Create optimized training sequences"""
        X, y = [], []
        
        for i in range(len(features) - self.sequence_length):
            X.append(features[i:i + self.sequence_length])
            y.append(labels[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def train_optimized_model(self, epochs=120, batch_size=64, learning_rate=0.0005):
        """Train the optimized AI model"""
        
        print("\n2. TRAINING OPTIMIZED MODEL")
        print("-" * 30)
        
        all_features = []
        all_labels = []
        
        # OPTIMIZATION 4: Multi-period training
        train_periods = self.load_and_prepare_data()
        
        for i, period_data in enumerate(train_periods):
            print(f"  Processing period {i+1}/3...")
            features, labels = self.create_enhanced_features(period_data)
            
            if len(features) > 0:
                all_features.append(features)
                all_labels.append(labels)
        
        # Combine all periods
        combined_features = np.vstack(all_features)
        combined_labels = np.concatenate(all_labels)
        
        print(f"  Total training samples: {len(combined_features)}")
        
        # Create sequences
        X, y = self.create_sequences(combined_features, combined_labels)
        
        print(f"  Training sequences: {len(X)}")
        print(f"  Feature dimensions: {X.shape}")
        
        # OPTIMIZATION 5: Robust scaling instead of standard scaling
        scaler = RobustScaler()  # Less sensitive to outliers
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        
        # Split data with stratification
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"  Training: {len(X_train)}, Validation: {len(X_val)}")
        
        # OPTIMIZATION 6: Enhanced class balancing
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        # Create weighted sampler for balanced batches
        sample_weights = [class_weights[label] for label in y_train]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        
        # Data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # OPTIMIZATION 7: Enhanced model architecture
        model = OptimizedForexLSTM(
            input_size=20,  # Enhanced features
            hidden_size=128,
            num_layers=3,
            dropout=0.4
        ).to(self.device)
        
        # OPTIMIZATION 8: Advanced training setup
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Cosine annealing with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=learning_rate/10
        )
        
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters())}")
        print("  Starting optimized training...")
        
        # Training loop with early stopping
        best_val_acc = 0.0
        patience = 25
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            class_correct = [0, 0, 0]
            class_total = [0, 0, 0]
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                    
                    # Per-class accuracy
                    for i in range(len(batch_y)):
                        label = batch_y[i]
                        class_correct[label] += (predicted[i] == label).item()
                        class_total[label] += 1
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_accuracy = correct / total
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            # Learning rate scheduling
            scheduler.step()
            
            # Early stopping and model saving
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                patience_counter = 0
                
                # Save best model
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), "models/optimized_forex_ai.pth")
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch+1:3d}/{epochs} | "
                      f"Train: {train_loss:.4f} | "
                      f"Val: {val_loss:.4f} | "
                      f"Acc: {val_accuracy:.4f} | "
                      f"Best: {best_val_acc:.4f}")
                
                # Per-class accuracy
                class_names = ['Hold', 'Buy', 'Sell']
                for i in range(3):
                    if class_total[i] > 0:
                        acc = class_correct[i] / class_total[i]
                        print(f"      {class_names[i]}: {acc:.3f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break
        
        # Save final artifacts
        joblib.dump(scaler, "models/optimized_scaler.pkl")
        
        print(f"\n3. OPTIMIZATION COMPLETE!")
        print("-" * 30)
        print(f"  Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")
        print(f"  Previous accuracy: 21.0%")
        
        improvement = (best_val_acc * 100) - 21.0
        print(f"  Improvement: {improvement:+.1f} percentage points")
        
        if best_val_acc > 0.35:
            print("  STATUS: SIGNIFICANT IMPROVEMENT!")
        elif best_val_acc > 0.25:
            print("  STATUS: Good progress made")
        else:
            print("  STATUS: Further optimization needed")
        
        # Plot results
        self.plot_optimization_results(train_losses, val_losses, val_accuracies, best_val_acc)
        
        return model, best_val_acc
    
    def plot_optimization_results(self, train_losses, val_losses, val_accuracies, best_acc):
        """Plot optimization results"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Loss curves
        axes[0].plot(train_losses, label='Training Loss', alpha=0.8)
        axes[0].plot(val_losses, label='Validation Loss', alpha=0.8)
        axes[0].set_title('Optimized Training - Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy curve
        axes[1].plot(val_accuracies, label=f'Optimized Accuracy\nBest: {best_acc:.1%}', 
                    color='green', linewidth=2)
        axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Target: 50%')
        axes[1].axhline(y=0.33, color='orange', linestyle='--', alpha=0.7, label='Random: 33%')
        axes[1].axhline(y=0.21, color='gray', linestyle='--', alpha=0.7, label='Previous: 21%')
        axes[1].set_title('Optimization Progress')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Performance comparison
        axes[2].bar(['Previous\nAI', 'Random\nBaseline', 'Optimized\nAI'], 
                   [0.21, 0.33, best_acc], 
                   color=['red', 'gray', 'green'], alpha=0.7)
        axes[2].set_title('Performance Improvement')
        axes[2].set_ylabel('Accuracy')
        axes[2].set_ylim(0, max(0.6, best_acc + 0.1))
        
        # Add percentage labels
        for i, v in enumerate([0.21, 0.33, best_acc]):
            axes[2].text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('models/optimization_results.png', dpi=300, bbox_inches='tight')
        
        print(f"  Optimization results saved: models/optimization_results.png")

def main():
    """Run AI optimization"""
    
    print("FOREXSWING AI OPTIMIZATION")
    print("Transform 21% to Professional Performance")
    print("=" * 50)
    
    optimizer = AIOptimizer()
    model, final_accuracy = optimizer.train_optimized_model()
    
    print(f"\nOPTIMIZATION SUMMARY:")
    print("=" * 30)
    print(f"Previous accuracy: 21.0%")
    print(f"Optimized accuracy: {final_accuracy*100:.1f}%")
    improvement = (final_accuracy*100 - 21.0)
    print(f"Improvement: {improvement:+.1f} percentage points")
    
    if final_accuracy > 0.50:
        print("ACHIEVEMENT UNLOCKED: Professional-grade performance!")
    elif final_accuracy > 0.40:
        print("EXCELLENT: Significant improvement achieved!")
    elif final_accuracy > 0.30:
        print("PROGRESS: Good optimization results!")
    else:
        print("DEVELOPMENT: Continue optimizing for better results")
    
    print(f"\nOptimized model saved: models/optimized_forex_ai.pth")
    print(f"Ready for testing and deployment!")

if __name__ == "__main__":
    main()