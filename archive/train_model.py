# user_data/ml_models/train_model.py
"""
Neural Network Training Pipeline for Forex LSTM
Uses JAX-accelerated features + PyTorch training for maximum performance
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
import matplotlib.pyplot as plt
import os
import sys
import json
from datetime import datetime

# Import our modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from indicators.jax_indicators import jax_rsi, jax_sma, jax_macd, jax_ema
from ml_models.forex_lstm import ForexLSTM, JAXFeatureEngine

class ForexTrainingPipeline:
    def __init__(self, sequence_length=60, future_periods=12):
        self.sequence_length = sequence_length  # Look back 60 periods (10 days at 4h)
        self.future_periods = future_periods    # Predict 12 periods ahead (2 days at 4h)
        self.feature_engine = JAXFeatureEngine()
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Training Pipeline initialized")
        print(f"Sequence length: {sequence_length} periods")
        print(f"Prediction horizon: {future_periods} periods ahead")
        print(f"Device: {self.device}")
    
    def load_market_data(self):
        """Load downloaded market data from Feather format"""
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'training')
        
        # Load forex data (Feather format)
        all_data = []
        
        # Look for all feather files in the training directory
        import glob
        feather_files = glob.glob(os.path.join(data_dir, '*_daily.feather'))
        
        if not feather_files:
            raise ValueError("No market data found! Run download_data.py first.")
        
        for file_path in feather_files:
            filename = os.path.basename(file_path)
            print(f"Loading {filename}...")
            
            # Read Feather file (much faster than JSON!)
            df = pd.read_feather(file_path)
            
            # Ensure datetime index
            if 'date' in df.columns:
                df = df.set_index('date')
            elif 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('datetime')
            
            # Convert to float if needed
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            
            # Extract pair name from filename
            pair_name = filename.split('_')[0] + '/' + filename.split('_')[1]
            df['pair'] = pair_name
            all_data.append(df)
            
            print(f"Loaded {len(df)} candles for {pair_name}")
        
        # Combine all data
        combined_data = pd.concat(all_data)
        combined_data = combined_data.sort_index()
        
        print(f"Total dataset: {len(combined_data)} candles")
        return combined_data
    
    def create_features_and_labels(self, data):
        """Create training features and labels using JAX acceleration"""
        
        features_list = []
        labels_list = []
        
        # Process each trading pair separately
        for pair in data['pair'].unique():
            pair_data = data[data['pair'] == pair].copy()
            
            print(f"Processing {pair} with JAX acceleration...")
            
            # JAX-accelerated feature creation (60K+ calc/sec!)
            prices = jnp.array(pair_data['close'].values)
            volumes = jnp.array(pair_data['volume'].values)
            highs = jnp.array(pair_data['high'].values)
            lows = jnp.array(pair_data['low'].values)
            
            # Create technical features with JAX
            feature_dict = self.feature_engine.create_technical_features(prices, volumes)
            
            # Additional price features
            returns = jnp.diff(prices) / prices[:-1]
            
            # Combine all features
            features = jnp.column_stack([
                feature_dict['rsi_14'][1:],      # Skip first element to align with returns
                feature_dict['rsi_21'][1:],
                feature_dict['sma_20'][1:],
                feature_dict['sma_50'][1:],
                feature_dict['macd'][1:],
                feature_dict['macd_signal'][1:],
                feature_dict['macd_histogram'][1:],
                feature_dict['returns_1'][1:],
                feature_dict['returns_5'][1:],
                feature_dict['volatility'][1:],
                feature_dict['volume_ratio'][1:]
            ])
            
            # Create labels (future price movements)
            # Label = 1 if price goes up significantly, 0 if neutral, -1 if down
            future_returns = []
            
            for i in range(len(prices) - self.future_periods):
                current_price = prices[i]
                future_price = prices[i + self.future_periods]
                
                return_pct = (future_price - current_price) / current_price
                
                # Classification thresholds
                if return_pct > 0.02:      # +2% = Strong Buy (class 2)
                    label = 2
                elif return_pct > 0.005:   # +0.5% = Buy (class 1)  
                    label = 1
                else:                      # Otherwise = Hold/Sell (class 0)
                    label = 0
                
                future_returns.append(label)
            
            # Align features with labels
            min_length = min(len(features) - 1, len(future_returns))
            aligned_features = np.array(features[:min_length])
            aligned_labels = np.array(future_returns[:min_length])
            
            features_list.append(aligned_features)
            labels_list.append(aligned_labels)
            
            print(f"{pair}: {len(aligned_features)} feature vectors created")
        
        # Combine all pairs
        all_features = np.vstack(features_list)
        all_labels = np.concatenate(labels_list)
        
        print(f"Total training samples: {len(all_features)}")
        print(f"Feature shape: {all_features.shape}")
        print(f"Label distribution: {np.bincount(all_labels)}")
        
        return all_features, all_labels
    
    def create_sequences(self, features, labels):
        """Create sequences for LSTM training"""
        
        X, y = [], []
        
        for i in range(len(features) - self.sequence_length):
            # Input: sequence of features
            sequence = features[i:i + self.sequence_length]
            # Output: label at end of sequence
            label = labels[i + self.sequence_length]
            
            X.append(sequence)
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Created {len(X)} sequences")
        print(f"Sequence shape: {X.shape}")
        
        return X, y
    
    def train_model(self, epochs=50, batch_size=32, learning_rate=0.001):
        """Train the LSTM model"""
        
        print("Starting Neural Network Training...")
        
        # Load and prepare data
        print("Loading market data...")
        market_data = self.load_market_data()
        
        print("Creating features with JAX acceleration...")
        features, labels = self.create_features_and_labels(market_data)
        
        print("Creating sequences...")
        X, y = self.create_sequences(features, labels)
        
        # Normalize features
        print("Normalizing features...")
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
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
        input_size = X.shape[-1]  # Number of features
        model = ForexLSTM(input_size=input_size, hidden_size=64, num_layers=2).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        print(f"Model initialized: {sum(p.numel() for p in model.parameters())} parameters")
        
        # Training loop
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        
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
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_accuracy = correct / total
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
                os.makedirs(model_dir, exist_ok=True)
                best_model_path = os.path.join(model_dir, 'forex_lstm_best.pth')
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val Accuracy: {val_accuracy:.4f}")
                print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
                print()
            
            # Early stopping
            if patience_counter >= 15:
                print("Early stopping triggered!")
                break
        
        # Save final model
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        final_model_path = os.path.join(model_dir, 'forex_lstm.pth')
        torch.save(model.state_dict(), final_model_path)
        
        # Save scaler
        import joblib
        scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Training complete!")
        print(f"Model saved to: {final_model_path}")
        print(f"Best validation accuracy: {max(val_accuracies):.4f}")
        print(f"Final validation accuracy: {val_accuracies[-1]:.4f}")
        
        # Plot training curves
        self.plot_training_curves(train_losses, val_losses, val_accuracies)
        
        return model, train_losses, val_losses, val_accuracies
    
    def plot_training_curves(self, train_losses, val_losses, val_accuracies):
        """Plot training progress"""
        
        plt.figure(figsize=(15, 5))
        
        # Loss curves
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training & Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy curve
        plt.subplot(1, 3, 2)
        plt.plot(val_accuracies, label='Validation Accuracy', color='green')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Learning curve
        plt.subplot(1, 3, 3)
        epochs = len(train_losses)
        plt.plot(range(epochs), train_losses, 'b-', alpha=0.7, label='Train')
        plt.plot(range(epochs), val_losses, 'r-', alpha=0.7, label='Validation')
        plt.fill_between(range(epochs), train_losses, alpha=0.3)
        plt.fill_between(range(epochs), val_losses, alpha=0.3)
        plt.title('Learning Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        
        print("Training curves saved to models/training_curves.png")

def main():
    """Main training function"""
    
    print("Forex Neural Network Training Pipeline")
    print("=" * 50)
    
    # Initialize training pipeline
    trainer = ForexTrainingPipeline(sequence_length=60, future_periods=12)
    
    # Train the model
    model, train_losses, val_losses, val_accuracies = trainer.train_model(
        epochs=100,      # More epochs for better learning
        batch_size=32,   # Good batch size for stability
        learning_rate=0.001  # Conservative learning rate
    )
    
    print("Neural network training complete!")
    print("Your model is now ready for professional trading!")

if __name__ == "__main__":
    main()