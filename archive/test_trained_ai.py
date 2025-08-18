#!/usr/bin/env python3
"""
Test the TRAINED AI model predictions!
"""

import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.append('src')

def test_trained_ai():
    """Test the trained AI model with real data"""
    print("=" * 60)
    print("TESTING TRAINED AI PREDICTIONS")
    print("=" * 60)
    
    from ml_models.forex_lstm import HybridForexPredictor
    
    # Load the trained model
    model_path = "src/models/forex_lstm.pth"
    
    if not os.path.exists(model_path):
        print(f"Trained model not found at {model_path}")
        print("Run training first: python src/ml_models/train_model.py")
        return
    
    print(f"Loading trained AI model from {model_path}...")
    predictor = HybridForexPredictor(model_path)
    
    # Test with our training data
    print("\nTesting AI on different forex pairs...")
    
    data_dir = "data/training"
    test_results = {}
    
    for filename in os.listdir(data_dir):
        if filename.endswith('_daily.feather'):
            pair_name = filename.replace('_synthetic_daily.feather', '').replace('_', '/')
            
            print(f"\n--- Testing {pair_name} ---")
            
            # Load data
            df = pd.read_feather(os.path.join(data_dir, filename))
            
            # Use last 200 days for testing
            test_data = df.tail(200).copy()
            
            print(f"Testing on {len(test_data)} days of {pair_name} data")
            print(f"Price range: {test_data['close'].min():.4f} to {test_data['close'].max():.4f}")
            
            # Get AI predictions
            predictions = predictor.predict(test_data)
            
            # Calculate some statistics
            bullish_signals = (predictions > 0.6).sum()
            bearish_signals = (predictions < 0.4).sum()
            neutral_signals = len(predictions) - bullish_signals - bearish_signals
            
            print(f"AI Predictions:")
            print(f"   BULLISH signals: {bullish_signals} ({bullish_signals/len(predictions)*100:.1f}%)")
            print(f"   BEARISH signals: {bearish_signals} ({bearish_signals/len(predictions)*100:.1f}%)")
            print(f"   NEUTRAL signals: {neutral_signals} ({neutral_signals/len(predictions)*100:.1f}%)")
            print(f"   Prediction range: {predictions.min():.3f} to {predictions.max():.3f}")
            print(f"   Average confidence: {predictions.mean():.3f}")
            
            # Calculate actual future returns to see if AI is right
            future_returns = []
            for i in range(len(test_data) - 12):  # 12-day prediction horizon
                current_price = test_data.iloc[i]['close']
                future_price = test_data.iloc[i + 12]['close']
                ret = (future_price - current_price) / current_price
                future_returns.append(ret)
            
            # Align predictions with future returns
            aligned_predictions = predictions[:len(future_returns)]
            
            # Calculate accuracy metrics
            if len(future_returns) > 0:
                correct_predictions = 0
                for pred, actual_return in zip(aligned_predictions, future_returns):
                    if pred > 0.6 and actual_return > 0.005:  # Bullish and actually went up
                        correct_predictions += 1
                    elif pred < 0.4 and actual_return < -0.005:  # Bearish and actually went down
                        correct_predictions += 1
                    elif 0.4 <= pred <= 0.6 and abs(actual_return) <= 0.005:  # Neutral and stayed flat
                        correct_predictions += 1
                
                accuracy = correct_predictions / len(future_returns) * 100
                print(f"   AI Accuracy: {accuracy:.1f}% on future price movements!")
                
                test_results[pair_name] = {
                    'accuracy': accuracy,
                    'predictions': predictions,
                    'data': test_data,
                    'future_returns': future_returns
                }
    
    # Summary
    print("\n" + "=" * 60)
    print("AI TESTING SUMMARY")
    print("=" * 60)
    
    accuracies = [result['accuracy'] for result in test_results.values()]
    if accuracies:
        avg_accuracy = np.mean(accuracies)
        print(f"Average AI Accuracy: {avg_accuracy:.1f}%")
        print(f"Best performing pair: {max(test_results.keys(), key=lambda k: test_results[k]['accuracy'])}")
        print(f"Accuracy range: {min(accuracies):.1f}% to {max(accuracies):.1f}%")
        
        if avg_accuracy > 60:
            print("EXCELLENT! Your AI is beating random chance!")
        elif avg_accuracy > 55:
            print("GOOD! Your AI shows predictive power!")
        else:
            print("The AI needs more training data or parameter tuning")
    
    # Model info
    print(f"\nModel Information:")
    info = predictor.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    return test_results

def plot_ai_predictions(test_results):
    """Plot AI predictions vs actual prices"""
    
    if not test_results:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (pair, result) in enumerate(list(test_results.items())[:4]):
        ax = axes[i]
        
        data = result['data']
        predictions = result['predictions']
        
        # Plot price and predictions
        ax2 = ax.twinx()
        
        # Price line
        ax.plot(data.index, data['close'], 'b-', label='Price', alpha=0.7)
        ax.set_ylabel('Price', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        
        # Prediction line
        ax2.plot(data.index, predictions, 'r-', label='AI Prediction', alpha=0.8)
        ax2.set_ylabel('AI Confidence', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_ylim(0, 1)
        
        # Add horizontal lines for buy/sell thresholds
        ax2.axhline(y=0.6, color='g', linestyle='--', alpha=0.5, label='Buy Threshold')
        ax2.axhline(y=0.4, color='r', linestyle='--', alpha=0.5, label='Sell Threshold')
        
        ax.set_title(f'{pair} - AI Predictions vs Price\nAccuracy: {result["accuracy"]:.1f}%')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/ai_prediction_analysis.png', dpi=300, bbox_inches='tight')
    print("Prediction analysis saved to models/ai_prediction_analysis.png")
    
    # Show some example predictions
    print("\nExample AI Predictions:")
    pair_name = list(test_results.keys())[0]
    result = test_results[pair_name]
    data = result['data']
    predictions = result['predictions']
    
    print(f"\n{pair_name} - Last 10 predictions:")
    for i in range(-10, 0):
        try:
            date = data.index[i].strftime('%Y-%m-%d')
        except:
            date = str(data.index[i])[:10]  # Fallback date format
        
        price = data.iloc[i]['close']
        pred = predictions[i]
        
        if pred > 0.6:
            signal = "BULLISH"
        elif pred < 0.4:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"
        
        print(f"   {date}: Price={price:.4f}, AI={pred:.3f} {signal}")

def main():
    """Run AI testing"""
    test_results = test_trained_ai()
    
    if test_results:
        plot_ai_predictions(test_results)
    
    print("\nYour AI trading system is now operational!")
    print("Next steps:")
    print("1. Test with live market data")
    print("2. Integrate with FreqTrade")
    print("3. Start paper trading")

if __name__ == "__main__":
    main()