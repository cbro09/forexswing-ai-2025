#!/usr/bin/env python3
"""
REAL MARKET DATA AI Training - Professional Grade!
Train on actual forex market data for maximum accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import jax.numpy as jnp
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import os
import sys
import joblib

# Import our modules
sys.path.append('src')
from indicators.jax_indicators import jax_rsi, jax_sma, jax_macd, jax_ema

class RealDataForexLSTM(nn.Module):
    """Real market data optimized LSTM"""
    
    def __init__(self, input_size=15, hidden_size=128, num_layers=3, dropout=0.3):
        super(RealDataForexLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # Multi-layer bidirectional LSTM
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
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            hidden_size // 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size // 2, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(48, 3)  # Buy/Hold/Sell
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        # Input normalization
        x_reshaped = x.reshape(-1, features)
        x_norm = self.input_norm(x_reshaped)
        x = x_norm.reshape(batch_size, seq_len, features)
        
        # LSTM layers
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout(lstm1_out)
        
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout(lstm2_out)
        
        lstm3_out, _ = self.lstm3(lstm2_out)
        lstm3_out = self.dropout(lstm3_out)
        
        # Self-attention
        attended, _ = self.attention(lstm3_out, lstm3_out, lstm3_out)
        
        # Global pooling
        pooled = torch.mean(attended, dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return torch.softmax(logits, dim=1)

class RealDataTrainingPipeline:
    def __init__(self, sequence_length=60, future_periods=12):
        self.sequence_length = sequence_length
        self.future_periods = future_periods
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"REAL DATA Training Pipeline initialized")
        print(f"Device: {self.device}")
        print(f"Target: Professional-grade real market performance!")
    
    def load_real_market_data(self):
        """Load real forex market data"""
        data_dir = "data/real_market"
        
        if not os.path.exists(data_dir):
            raise ValueError("Real market data not found! Run download_real_data.py first.")
        
        all_data = []
        
        # Load all real market feather files
        for filename in os.listdir(data_dir):
            if filename.endswith('_real_daily.feather'):
                print(f"Loading {filename}...")
                
                df = pd.read_feather(os.path.join(data_dir, filename))
                
                # Set date index
                if 'date' in df.columns:
                    df = df.set_index('date')
                elif 'Date' in df.columns:
                    df = df.set_index('Date')
                
                # Extract pair name
                pair_name = filename.replace('_real_daily.feather', '').replace('_', '/')
                df['pair'] = pair_name
                all_data.append(df)
        
        combined_data = pd.concat(all_data).sort_index()
        print(f"Total real market dataset: {len(combined_data)} candles")
        
        return combined_data
    
    def create_real_market_features(self, data):
        """Create features optimized for real market conditions"""
        features_list = []
        labels_list = []
        
        for pair in data['pair'].unique():
            pair_data = data[data['pair'] == pair].copy()
            print(f"Processing {pair} real market data...")
            
            # JAX arrays for ultra-fast computation
            prices = jnp.array(pair_data['close'].values)
            volumes = jnp.array(pair_data['volume'].values)
            highs = jnp.array(pair_data['high'].values)
            lows = jnp.array(pair_data['low'].values)
            
            # Core indicators
            rsi_14 = jax_rsi(prices, 14)
            rsi_21 = jax_rsi(prices, 21)
            sma_20 = jax_sma(prices, 20)
            sma_50 = jax_sma(prices, 50)
            ema_12 = jax_ema(prices, 12)
            ema_26 = jax_ema(prices, 26)
            macd_line, macd_signal, macd_histogram = jax_macd(prices)
            
            # Real market specific features
            returns_1 = jnp.diff(prices) / prices[:-1]
            returns_5 = jnp.concatenate([jnp.zeros(5), (prices[5:] - prices[:-5]) / prices[:-5]])
            
            # Volatility (crucial for real markets)
            volatility = jnp.array([
                jnp.std(returns_1[max(0, i-19):i+1]) if i >= 19 else 0.01
                for i in range(len(returns_1))
            ])
            
            # Volume features
            volume_sma = jax_sma(volumes, 20)
            volume_ratio = volumes / jnp.maximum(volume_sma, 1)
            
            # Market structure
            price_position = (prices - sma_50) / jnp.maximum(sma_50, 1)
            trend_strength = (sma_20 - sma_50) / jnp.maximum(sma_50, 1)
            
            # Combine 15 features
            min_length = min(len(prices), len(rsi_14), len(returns_5), len(volatility))
            
            features = jnp.column_stack([
                rsi_14[:min_length],
                rsi_21[:min_length],
                sma_20[:min_length] / prices[:min_length],  # Normalized
                sma_50[:min_length] / prices[:min_length],  # Normalized
                ema_12[:min_length] / prices[:min_length],  # Normalized
                ema_26[:min_length] / prices[:min_length],  # Normalized
                macd_line[:min_length],
                macd_signal[:min_length],
                macd_histogram[:min_length],
                jnp.concatenate([jnp.array([0.0]), returns_1])[:min_length],
                returns_5[:min_length],
                jnp.concatenate([jnp.array([0.01]), volatility])[:min_length],
                volume_ratio[:min_length],
                price_position[:min_length],
                trend_strength[:min_length]
            ])
            
            # Create labels optimized for real market volatility
            future_returns = []
            for i in range(len(prices) - self.future_periods):
                current_price = prices[i]
                future_price = prices[i + self.future_periods]
                return_pct = (future_price - current_price) / current_price
                
                # Real market thresholds (more conservative)
                if return_pct > 0.008:      # >0.8% = Buy (class 1)
                    label = 1
                elif return_pct < -0.008:   # <-0.8% = Sell (class 2)
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
            
            print(f"  {len(aligned_features)} feature vectors")
        
        all_features = np.vstack(features_list)
        all_labels = np.concatenate(labels_list)
        
        # Show class distribution
        unique, counts = np.unique(all_labels, return_counts=True)
        print(f"\nReal market label distribution:")
        for u, c in zip(unique, counts):
            print(f"  Class {u}: {c} samples ({c/len(all_labels)*100:.1f}%)")
        
        return all_features, all_labels
    
    def create_sequences(self, features, labels):
        """Create training sequences"""
        X, y = [], []
        
        for i in range(len(features) - self.sequence_length):
            X.append(features[i:i + self.sequence_length])
            y.append(labels[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def train_real_market_model(self, epochs=150, batch_size=32, learning_rate=0.0008):
        """Train model on real market data"""
        
        print("\n" + "=" * 60)
        print("TRAINING AI ON REAL MARKET DATA")
        print("=" * 60)
        
        # Load real market data
        market_data = self.load_real_market_data()
        features, labels = self.create_real_market_features(market_data)
        X, y = self.create_sequences(features, labels)
        
        print(f"\nReal market training data:")
        print(f"  Sequences: {len(X)}")
        print(f"  Features per timestep: {X.shape[-1]}")
        print(f"  Sequence length: {X.shape[1]}")
        
        # Normalize features
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nData split:")
        print(f"  Training: {len(X_train)}")
        print(f"  Validation: {len(X_val)}")
        
        # Calculate class weights
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        print(f"  Class weights: {class_weights}")
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = RealDataForexLSTM(
            input_size=15,
            hidden_size=128,
            num_layers=3,
            dropout=0.3
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
        
        print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")
        print("Starting real market training...\n")
        
        # Training loop
        best_val_acc = 0.0
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training
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
            
            # Validation
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
            scheduler.step(val_loss)
            
            # Save best model
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                
                # Save model
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), "models/real_market_forex_lstm.pth")
            
            # Print progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1:3d}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss:   {val_loss:.4f}")
                print(f"  Val Acc:    {val_accuracy:.4f} (Best: {best_val_acc:.4f})")
                
                # Per-class accuracy
                for i in range(3):
                    if class_total[i] > 0:
                        class_names = ['Hold', 'Buy', 'Sell']
                        acc = class_correct[i] / class_total[i]
                        print(f"  {class_names[i]} Acc:   {acc:.4f}")
                
                print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")
                print()
        
        # Save final artifacts
        torch.save(model.state_dict(), "models/real_market_forex_lstm.pth")
        joblib.dump(self.scaler, "models/real_market_feature_scaler.pkl")
        
        print("=" * 60)
        print("REAL MARKET TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")
        print(f"Model saved: models/real_market_forex_lstm.pth")
        print(f"Scaler saved: models/real_market_feature_scaler.pkl")
        
        if best_val_acc >= 0.65:
            print("EXCELLENT: Real market performance achieved!")
        elif best_val_acc >= 0.55:
            print("GOOD: Significantly better than random on real data!")
        elif best_val_acc >= 0.45:
            print("PROGRESS: Learning real market patterns!")
        
        # Plot results
        self.plot_real_market_results(train_losses, val_losses, val_accuracies, best_val_acc)
        
        return model, best_val_acc
    
    def plot_real_market_results(self, train_losses, val_losses, val_accuracies, best_acc):
        """Plot real market training results"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Loss curves
        axes[0].plot(train_losses, label='Training Loss', alpha=0.8)
        axes[0].plot(val_losses, label='Validation Loss', alpha=0.8)
        axes[0].set_title('Real Market Training - Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy curve
        axes[1].plot(val_accuracies, label=f'Real Market Accuracy\nBest: {best_acc:.1%}', 
                    color='green', linewidth=2)
        axes[1].axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='Target: 60%')
        axes[1].axhline(y=0.33, color='orange', linestyle='--', alpha=0.7, label='Random: 33%')
        axes[1].set_title('Real Market Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Performance comparison
        axes[2].bar(['Random\nChance', 'Synthetic\nData AI', 'Real Market\nAI'], 
                   [0.33, 0.76, best_acc], 
                   color=['gray', 'orange', 'green'], alpha=0.7)
        axes[2].set_title('AI Performance Comparison')
        axes[2].set_ylabel('Accuracy')
        axes[2].set_ylim(0, 1)
        
        # Add percentage labels
        for i, v in enumerate([0.33, 0.76, best_acc]):
            axes[2].text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('models/real_market_training_results.png', dpi=300, bbox_inches='tight')
        
        print("Real market training results saved: models/real_market_training_results.png")

def main():
    """Run real market AI training"""
    print("REAL MARKET FOREX AI TRAINING")
    print("Professional-Grade Real Market Performance")
    print("=" * 60)
    
    trainer = RealDataTrainingPipeline()
    model, final_accuracy = trainer.train_real_market_model(
        epochs=150,
        batch_size=32,
        learning_rate=0.0008
    )
    
    print(f"\nFINAL REAL MARKET RESULT: {final_accuracy:.1%} accuracy!")
    
    if final_accuracy >= 0.60:
        print("PROFESSIONAL SUCCESS! Real market mastery achieved!")
    elif final_accuracy >= 0.50:
        print("EXCELLENT! Significantly better than random on real data!")
    else:
        print("Good progress! Real market learning in progress!")

if __name__ == "__main__":
    main()