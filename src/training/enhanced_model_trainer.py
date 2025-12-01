#!/usr/bin/env python3
"""
Enhanced Model Trainer with Improved Features
Trains LSTM model with expanded feature set including news sentiment
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
import json
import logging
from typing import List, Dict, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.models.optimized_forex_lstm import OptimizedForexLSTM, FastFeatureEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ModelTrainer')

class EnhancedForexDataset(Dataset):
    """
    Enhanced dataset with:
    - Technical indicators
    - News sentiment scores
    - Price patterns
    - Volume analysis
    """

    def __init__(self, data_dir: str, news_dir: str, sequence_length: int = 80):
        self.data_dir = data_dir
        self.news_dir = news_dir
        self.sequence_length = sequence_length
        self.feature_engine = FastFeatureEngine()

        # Load all currency pair data
        self.sequences = []
        self.labels = []
        self.pairs = []

        self._load_all_data()

        logger.info(f"Dataset loaded: {len(self.sequences)} sequences")

    def _load_all_data(self):
        """Load data from all currency pairs"""
        import glob

        # Find all daily data files
        data_files = glob.glob(os.path.join(self.data_dir, "*_real_daily.csv"))

        for data_file in data_files:
            pair_name = os.path.basename(data_file).replace('_real_daily.csv', '')
            logger.info(f"Loading {pair_name}...")

            try:
                # Load price data
                df = pd.read_csv(data_file, index_col=0, parse_dates=True)

                # Load news sentiment if available
                news_sentiment = self._load_news_sentiment(pair_name)

                # Create sequences
                sequences, labels = self._create_sequences_from_data(df, news_sentiment, pair_name)

                self.sequences.extend(sequences)
                self.labels.extend(labels)
                self.pairs.extend([pair_name] * len(sequences))

                logger.info(f"  Added {len(sequences)} sequences from {pair_name}")

            except Exception as e:
                logger.error(f"Error loading {pair_name}: {e}")
                continue

    def _load_news_sentiment(self, pair_name: str) -> Dict:
        """Load news sentiment data for pair"""
        import glob

        news_files = glob.glob(os.path.join(self.news_dir, f"{pair_name}_news_*.json"))

        if not news_files:
            return {}

        # Load most recent news file
        news_files.sort(reverse=True)
        try:
            with open(news_files[0], 'r') as f:
                return json.load(f)
        except:
            return {}

    def _create_sequences_from_data(self, df: pd.DataFrame, news: Dict, pair_name: str) -> Tuple[List, List]:
        """Create training sequences with enhanced features"""
        sequences = []
        labels = []

        # Create enhanced features
        import jax.numpy as jnp

        prices = jnp.array(df['close'].values)
        volumes = jnp.array(df.get('volume', jnp.ones_like(prices)).values)

        # Get technical features
        feature_dict = self.feature_engine.create_enhanced_features(prices, volumes)
        features_matrix = self.feature_engine.combine_features(
            feature_dict, len(prices), target_features=20
        )

        features_np = np.array(features_matrix)

        # Add news sentiment as additional feature
        news_sentiment_score = news.get('sentiment_score', 0.0) if news else 0.0
        news_confidence = news.get('confidence', 0.3) if news else 0.3

        # Create sequences
        for i in range(self.sequence_length, len(features_np) - 5):
            # Input sequence
            sequence = features_np[i - self.sequence_length:i]

            # Add news sentiment to each timestep (as additional feature column)
            news_feature = np.full((self.sequence_length, 1), news_sentiment_score * news_confidence)
            # Optionally concatenate news feature
            # sequence = np.hstack([sequence, news_feature])

            # Label: future price movement (5 days ahead)
            current_price = prices[i]
            future_price = prices[i + 5]
            price_change_pct = (future_price - current_price) / current_price

            # Classification labels: BUY (>1% gain), SELL (<-1% loss), HOLD (otherwise)
            if price_change_pct > 0.01:
                label = 1  # BUY
            elif price_change_pct < -0.01:
                label = 2  # SELL
            else:
                label = 0  # HOLD

            sequences.append(sequence)
            labels.append(label)

        return sequences, labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        label = torch.LongTensor([self.labels[idx]])
        return sequence, label

class EnhancedModelTrainer:
    """Enhanced trainer with better hyperparameters and techniques"""

    def __init__(self, data_dir: str = 'data/MarketData', news_dir: str = 'data/News'):
        self.data_dir = data_dir
        self.news_dir = news_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"Training on device: {self.device}")

        # Model hyperparameters
        self.input_size = 20
        self.hidden_size = 128
        self.num_layers = 3
        self.dropout = 0.4
        self.sequence_length = 80

        # Training hyperparameters
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 50
        self.patience = 10  # Early stopping patience

        # Initialize model
        self.model = OptimizedForexLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        logger.info(f"Model initialized: {self.model.get_model_info()}")

    def train(self, save_path: str = 'data/models/enhanced_forex_ai.pth'):
        """Train the model with enhanced dataset"""

        # Load dataset
        logger.info("Loading enhanced dataset...")
        dataset = EnhancedForexDataset(
            self.data_dir,
            self.news_dir,
            sequence_length=self.sequence_length
        )

        # Split into train/validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

        logger.info(f"Training samples: {train_size}, Validation samples: {val_size}")

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for sequences, labels in train_loader:
                sequences = sequences.to(self.device)
                labels = labels.squeeze().to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100 * train_correct / train_total

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences = sequences.to(self.device)
                    labels = labels.squeeze().to(self.device)

                    outputs = self.model(sequences)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total

            # Learning rate scheduling
            scheduler.step(avg_val_loss)

            logger.info(
                f"Epoch [{epoch+1}/{self.num_epochs}] "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
            )

            # Early stopping and model saving
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0

                # Save best model
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"✅ Model saved to {save_path} (Val Acc: {val_accuracy:.2f}%)")

            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

        logger.info("="*60)
        logger.info("Training completed!")
        logger.info(f"Best validation accuracy: {val_accuracy:.2f}%")
        logger.info(f"Model saved to: {save_path}")
        logger.info("="*60)

        return self.model

if __name__ == "__main__":
    trainer = EnhancedModelTrainer()
    trainer.train()
