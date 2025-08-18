#!/usr/bin/env python3
"""
Simple Real Market Data AI Training
Fast and focused approach
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import jax.numpy as jnp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os
import sys
import joblib

# Import our modules
sys.path.append('src')
from indicators.jax_indicators import jax_rsi, jax_sma, jax_macd, jax_ema

# Use the same model from our successful final training
class StreamlinedForexLSTM(nn.Module):
    """Final AI model architecture - 76% accuracy!\""""
    
    def __init__(self, input_size=15, hidden_size=96, num_layers=2, dropout=0.25):
        super(StreamlinedForexLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # Bidirectional LSTM with residual connection
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
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification layers with batch norm
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout // 2),
            nn.Linear(32, 3)  # Buy/Hold/Sell
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
        
        # Self-attention
        attended, _ = self.attention(lstm2_out, lstm2_out, lstm2_out)
        
        # Global pooling
        pooled = torch.mean(attended, dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return torch.softmax(logits, dim=1)

def load_real_data():
    """Load real market data efficiently"""
    print("Loading real forex market data...")
    
    data_dir = "data/real_market"
    all_data = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('_real_daily.feather'):
            df = pd.read_feather(os.path.join(data_dir, filename))
            
            # Set date index
            if 'Date' in df.columns:
                df = df.set_index('Date')
            
            # Extract pair name
            pair_name = filename.replace('_real_daily.feather', '').replace('_', '/')
            df['pair'] = pair_name
            all_data.append(df)
            print(f"  Loaded {len(df)} days of {pair_name}")
    
    combined_data = pd.concat(all_data).sort_index()
    print(f"Total: {len(combined_data)} candles from {len(all_data)} pairs")
    
    return combined_data

def create_features(data):
    """Create features for each currency pair"""
    print("Creating features...")
    
    features_list = []
    labels_list = []
    
    for pair in data['pair'].unique():
        pair_data = data[data['pair'] == pair].copy()
        print(f"Processing {pair}...")
        
        # JAX arrays
        prices = jnp.array(pair_data['close'].values)
        volumes = jnp.array(pair_data['volume'].values)
        
        # Core indicators
        rsi_14 = jax_rsi(prices, 14)
        rsi_21 = jax_rsi(prices, 21)
        sma_20 = jax_sma(prices, 20)
        sma_50 = jax_sma(prices, 50)
        ema_12 = jax_ema(prices, 12)
        ema_26 = jax_ema(prices, 26)
        macd_line, macd_signal, macd_histogram = jax_macd(prices)
        
        # Price features
        returns_1 = jnp.diff(prices) / prices[:-1]
        returns_5 = jnp.concatenate([jnp.zeros(5), (prices[5:] - prices[:-5]) / prices[:-5]])
        
        # Volatility
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
            sma_20[:min_length] / prices[:min_length],
            sma_50[:min_length] / prices[:min_length],
            ema_12[:min_length] / prices[:min_length],
            ema_26[:min_length] / prices[:min_length],
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
        
        # Create labels (same logic as successful training)
        future_returns = []
        for i in range(len(prices) - 12):  # 12 period ahead
            current_price = prices[i]
            future_price = prices[i + 12]
            return_pct = (future_price - current_price) / current_price
            
            if return_pct > 0.01:      # >1% = Buy (class 1)
                label = 1
            elif return_pct < -0.01:   # <-1% = Sell (class 2)
                label = 2
            else:                      # Otherwise = Hold (class 0)
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
    
    print(f"\nTotal samples: {len(all_features)}")
    unique, counts = np.unique(all_labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} samples ({c/len(all_labels)*100:.1f}%)")
    
    return all_features, all_labels

def create_sequences(features, labels, sequence_length=60):
    """Create training sequences"""
    X, y = [], []
    
    for i in range(len(features) - sequence_length):
        X.append(features[i:i + sequence_length])
        y.append(labels[i + sequence_length])
    
    return np.array(X), np.array(y)

def train_real_market_ai():
    """Train AI on real market data"""
    
    print("REAL MARKET AI TRAINING")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load and process data
    market_data = load_real_data()
    features, labels = create_features(market_data)
    X, y = create_sequences(features, labels)
    
    # Normalize features
    scaler = StandardScaler()
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(X.shape)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining: {len(X_train)}, Validation: {len(X_val)}")
    
    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Model
    model = StreamlinedForexLSTM(input_size=15, hidden_size=96, num_layers=2, dropout=0.25).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print("Starting training...\n")
    
    # Training loop
    best_val_acc = 0.0
    epochs = 80  # Reduced for faster training
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy = correct / total
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/real_market_ai.pth")
        
        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:2d}/{epochs} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"Acc: {val_accuracy:.4f} | "
                  f"Best: {best_val_acc:.4f}")
    
    # Save final artifacts
    joblib.dump(scaler, "models/real_market_scaler.pkl")
    
    print(f"\nREAL MARKET TRAINING COMPLETE!")
    print(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")
    print(f"Model saved: models/real_market_ai.pth")
    
    return best_val_acc

if __name__ == "__main__":
    accuracy = train_real_market_ai()