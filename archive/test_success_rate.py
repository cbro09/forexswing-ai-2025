#!/usr/bin/env python3
"""
Comprehensive AI Success Rate Testing
Test your ForexSwing AI across multiple scenarios and timeframes
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import jax.numpy as jnp
import joblib
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

sys.path.append('src')
from indicators.jax_indicators import jax_rsi, jax_sma, jax_macd, jax_ema

# Same model architecture
class StreamlinedForexLSTM(nn.Module):
    """Final AI model architecture"""
    
    def __init__(self, input_size=15, hidden_size=96, num_layers=2, dropout=0.25):
        super(StreamlinedForexLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # Bidirectional LSTM
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
        
        # Classification layers
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

class ForexAISuccessRateTester:
    """Comprehensive success rate testing for your Forex AI"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = 60
        
        # Load the AI model
        self.model = StreamlinedForexLSTM(
            input_size=15,
            hidden_size=96,
            num_layers=2,
            dropout=0.25
        ).to(self.device)
        
        # Try different model paths
        model_paths = [
            "models/real_market_ai.pth",
            "models/final_forex_lstm.pth",
            "models/improved_forex_lstm_best.pth"
        ]
        
        self.model_type = "unknown"
        for path in model_paths:
            if os.path.exists(path):
                self.model.load_state_dict(torch.load(path, map_location=self.device))
                self.model.eval()
                self.model_type = path.split('/')[-1].replace('.pth', '')
                print(f"AI Model loaded: {path}")
                break
        
        # Load scaler
        scaler_paths = [
            "models/real_market_scaler.pkl",
            "models/final_feature_scaler.pkl",
            "models/improved_feature_scaler.pkl"
        ]
        
        for path in scaler_paths:
            if os.path.exists(path):
                self.scaler = joblib.load(path)
                print(f"Feature scaler loaded: {path}")
                break
        
        print(f"ForexSwing AI Ready for Success Rate Testing!")
    
    def create_features(self, data):
        """Create the same 15 features used in training"""
        
        prices = jnp.array(data['close'].values)
        volumes = jnp.array(data['volume'].values)
        
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
        
        return np.array(features)
    
    def predict(self, data):
        """Make AI predictions"""
        
        if len(data) < self.sequence_length:
            return None, None
        
        features = self.create_features(data)
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
    
    def test_success_rate_comprehensive(self, test_days_ahead=[1, 3, 5, 10, 15]):
        """Test success rate across multiple timeframes"""
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE SUCCESS RATE TESTING")
        print("=" * 80)
        
        data_dir = "data/real_market"
        overall_results = {}
        
        for filename in os.listdir(data_dir):
            if filename.endswith('_real_daily.feather'):
                pair_name = filename.replace('_real_daily.feather', '').replace('_', '/')
                
                print(f"\nTesting {pair_name}")
                print("-" * 50)
                
                # Load data
                df = pd.read_feather(os.path.join(data_dir, filename))
                if 'Date' in df.columns:
                    df = df.set_index('Date')
                
                # Test different periods
                test_periods = {
                    "Recent 100 days": df.tail(150).head(100),
                    "Mid period": df.iloc[-400:-200],
                    "Earlier period": df.iloc[-600:-400]
                }
                
                pair_results = {}
                
                for period_name, test_data in test_periods.items():
                    if len(test_data) < self.sequence_length + 20:
                        continue
                    
                    print(f"\n  {period_name}:")
                    
                    # Get AI predictions
                    predictions, confidences = self.predict(test_data)
                    
                    if predictions is None:
                        continue
                    
                    # Test different forward-looking periods
                    timeframe_results = {}
                    
                    for days_ahead in test_days_ahead:
                        correct_predictions = 0
                        total_predictions = 0
                        profit_loss = []
                        
                        for i in range(len(predictions)):
                            # Find corresponding data index
                            data_idx = i + self.sequence_length - 1
                            future_idx = data_idx + days_ahead
                            
                            if future_idx >= len(test_data):
                                continue
                            
                            current_price = test_data.iloc[data_idx]['close']
                            future_price = test_data.iloc[future_idx]['close']
                            actual_return = (future_price - current_price) / current_price
                            
                            ai_prediction = predictions[i]
                            confidence = confidences[i]
                            
                            # Define success criteria
                            prediction_correct = False
                            trade_profit = 0
                            
                            if ai_prediction == 1:  # BUY signal
                                if actual_return > 0.002:  # Went up >0.2%
                                    prediction_correct = True
                                trade_profit = actual_return
                            elif ai_prediction == 2:  # SELL signal
                                if actual_return < -0.002:  # Went down <-0.2%
                                    prediction_correct = True
                                trade_profit = -actual_return  # Short profit
                            else:  # HOLD signal
                                if abs(actual_return) <= 0.005:  # Stayed flat
                                    prediction_correct = True
                                trade_profit = 0
                            
                            if prediction_correct:
                                correct_predictions += 1
                            
                            total_predictions += 1
                            profit_loss.append(trade_profit)
                        
                        if total_predictions > 0:
                            success_rate = correct_predictions / total_predictions * 100
                            avg_profit = np.mean(profit_loss) * 100
                            
                            timeframe_results[days_ahead] = {
                                'success_rate': success_rate,
                                'total_trades': total_predictions,
                                'correct_trades': correct_predictions,
                                'avg_profit_pct': avg_profit,
                                'cumulative_return': np.sum(profit_loss) * 100
                            }
                            
                            print(f"    {days_ahead} days ahead: "
                                  f"{success_rate:.1f}% success "
                                  f"({correct_predictions}/{total_predictions}) "
                                  f"| Avg: {avg_profit:+.2f}% "
                                  f"| Total: {np.sum(profit_loss)*100:+.1f}%")
                    
                    pair_results[period_name] = timeframe_results
                
                overall_results[pair_name] = pair_results
        
        return overall_results
    
    def test_trading_strategy_success(self):
        """Test as a real trading strategy"""
        
        print("\n" + "=" * 80)
        print("TRADING STRATEGY SUCCESS RATE")
        print("=" * 80)
        
        data_dir = "data/real_market"
        strategy_results = {}
        
        for filename in os.listdir(data_dir):
            if filename.endswith('_real_daily.feather'):
                pair_name = filename.replace('_real_daily.feather', '').replace('_', '/')
                
                print(f"\n{pair_name} Trading Strategy Test")
                print("-" * 40)
                
                # Load data
                df = pd.read_feather(os.path.join(data_dir, filename))
                if 'Date' in df.columns:
                    df = df.set_index('Date')
                
                # Use recent data for trading test
                test_data = df.tail(300)
                
                # Get AI predictions
                predictions, confidences = self.predict(test_data)
                
                if predictions is None:
                    continue
                
                # Simulate trading strategy
                initial_balance = 10000  # $10,000 starting capital
                balance = initial_balance
                trades = []
                positions = []
                
                for i in range(len(predictions) - 5):  # Leave buffer for exit
                    data_idx = i + self.sequence_length - 1
                    
                    if data_idx + 5 >= len(test_data):
                        break
                    
                    current_price = test_data.iloc[data_idx]['close']
                    ai_prediction = predictions[i]
                    confidence = confidences[i]
                    
                    # Only trade with high confidence (>80%)
                    if confidence < 0.8:
                        continue
                    
                    # Trade size based on confidence
                    risk_pct = min(confidence * 0.1, 0.05)  # Max 5% risk
                    trade_size = balance * risk_pct
                    
                    if ai_prediction == 1:  # BUY
                        # Buy and hold for 5 days
                        exit_price = test_data.iloc[data_idx + 5]['close']
                        profit_pct = (exit_price - current_price) / current_price
                        profit = trade_size * profit_pct
                        
                        balance += profit
                        
                        trades.append({
                            'type': 'BUY',
                            'entry_price': current_price,
                            'exit_price': exit_price,
                            'profit_pct': profit_pct * 100,
                            'profit_usd': profit,
                            'confidence': confidence
                        })
                    
                    elif ai_prediction == 2:  # SELL (Short)
                        # Short and close after 5 days
                        exit_price = test_data.iloc[data_idx + 5]['close']
                        profit_pct = (current_price - exit_price) / current_price
                        profit = trade_size * profit_pct
                        
                        balance += profit
                        
                        trades.append({
                            'type': 'SELL',
                            'entry_price': current_price,
                            'exit_price': exit_price,
                            'profit_pct': profit_pct * 100,
                            'profit_usd': profit,
                            'confidence': confidence
                        })
                
                # Calculate strategy metrics
                if trades:
                    total_return = (balance - initial_balance) / initial_balance * 100
                    winning_trades = [t for t in trades if t['profit_pct'] > 0]
                    losing_trades = [t for t in trades if t['profit_pct'] <= 0]
                    
                    win_rate = len(winning_trades) / len(trades) * 100
                    avg_win = np.mean([t['profit_pct'] for t in winning_trades]) if winning_trades else 0
                    avg_loss = np.mean([t['profit_pct'] for t in losing_trades]) if losing_trades else 0
                    
                    print(f"  Total trades: {len(trades)}")
                    print(f"  Win rate: {win_rate:.1f}%")
                    print(f"  Total return: {total_return:+.2f}%")
                    print(f"  Average win: {avg_win:+.2f}%")
                    print(f"  Average loss: {avg_loss:+.2f}%")
                    print(f"  Final balance: ${balance:,.2f}")
                    
                    strategy_results[pair_name] = {
                        'total_trades': len(trades),
                        'win_rate': win_rate,
                        'total_return': total_return,
                        'avg_win': avg_win,
                        'avg_loss': avg_loss,
                        'final_balance': balance,
                        'trades': trades
                    }
        
        return strategy_results
    
    def generate_success_report(self, comprehensive_results, strategy_results):
        """Generate comprehensive success rate report"""
        
        print("\n" + "=" * 80)
        print("FOREXSWING AI SUCCESS RATE REPORT")
        print("=" * 80)
        
        print(f"\nModel: {self.model_type}")
        print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Overall prediction accuracy
        print(f"\nPREDICTION ACCURACY SUMMARY:")
        print("-" * 50)
        
        all_success_rates = []
        for pair, periods in comprehensive_results.items():
            for period, timeframes in periods.items():
                for days, metrics in timeframes.items():
                    all_success_rates.append(metrics['success_rate'])
        
        if all_success_rates:
            avg_success = np.mean(all_success_rates)
            print(f"  Average Success Rate: {avg_success:.1f}%")
            print(f"  Best Performance: {max(all_success_rates):.1f}%")
            print(f"  Worst Performance: {min(all_success_rates):.1f}%")
            print(f"  Random Baseline: 33.3%")
            
            if avg_success > 33.3:
                print(f"  SUCCESS: AI BEATS RANDOM by {avg_success - 33.3:.1f}%")
            else:
                print(f"  WARNING: AI needs improvement ({33.3 - avg_success:.1f}% below random)")
        
        # Trading strategy performance
        print(f"\nTRADING STRATEGY PERFORMANCE:")
        print("-" * 50)
        
        if strategy_results:
            total_trades = sum([r['total_trades'] for r in strategy_results.values()])
            avg_win_rate = np.mean([r['win_rate'] for r in strategy_results.values()])
            avg_return = np.mean([r['total_return'] for r in strategy_results.values()])
            
            print(f"  Total trades executed: {total_trades}")
            print(f"  Average win rate: {avg_win_rate:.1f}%")
            print(f"  Average return: {avg_return:+.2f}%")
            
            profitable_pairs = [pair for pair, r in strategy_results.items() if r['total_return'] > 0]
            print(f"  Profitable pairs: {len(profitable_pairs)}/{len(strategy_results)}")
            
            if avg_return > 0:
                print(f"  SUCCESS: PROFITABLE STRATEGY!")
            else:
                print(f"  WARNING: Strategy needs optimization")
        
        # Best performing timeframes
        print(f"\nBEST TIMEFRAMES:")
        print("-" * 50)
        
        timeframe_performance = {}
        for pair, periods in comprehensive_results.items():
            for period, timeframes in periods.items():
                for days, metrics in timeframes.items():
                    if days not in timeframe_performance:
                        timeframe_performance[days] = []
                    timeframe_performance[days].append(metrics['success_rate'])
        
        for days in sorted(timeframe_performance.keys()):
            avg_perf = np.mean(timeframe_performance[days])
            print(f"  {days} days ahead: {avg_perf:.1f}% success rate")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        print("-" * 50)
        
        if avg_success > 40:
            print("  SUCCESS: AI shows strong predictive capability")
            print("  READY: For live trading consideration")
            print("  FOCUS: Risk management and position sizing")
        elif avg_success > 30:
            print("  PROMISE: AI shows potential but needs optimization")
            print("  CONSIDER: Ensemble methods or feature engineering")
            print("  TEST: Different prediction thresholds")
        else:
            print("  IMPROVE: AI needs significant enhancement")
            print("  RETRAIN: Consider different training data")
            print("  EXPERIMENT: Model architecture changes")
        
        print(f"\nTesting Complete! Your AI's success rate has been measured.")

def main():
    """Run comprehensive success rate testing"""
    
    print("FOREXSWING AI SUCCESS RATE TESTER")
    print("Professional-Grade Performance Validation")
    print("=" * 80)
    
    # Initialize tester
    tester = ForexAISuccessRateTester()
    
    # Run comprehensive tests
    print("\nRunning comprehensive prediction accuracy tests...")
    comprehensive_results = tester.test_success_rate_comprehensive()
    
    print("\nRunning trading strategy simulation...")
    strategy_results = tester.test_trading_strategy_success()
    
    # Generate final report
    tester.generate_success_report(comprehensive_results, strategy_results)
    
    print(f"\nYour ForexSwing AI has been thoroughly tested!")
    print(f"Ready for optimization and deployment!")

if __name__ == "__main__":
    main()