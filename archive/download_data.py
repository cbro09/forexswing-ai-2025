#!/usr/bin/env python3
"""
Download Historical Forex Data for Training
Uses free APIs to get real market data
"""

import pandas as pd
import numpy as np
import requests
import time
import os
from datetime import datetime, timedelta
import json

class ForexDataDownloader:
    def __init__(self):
        self.data_dir = "data/training"
        os.makedirs(self.data_dir, exist_ok=True)
        print("Forex Data Downloader initialized")
    
    def download_alpha_vantage_data(self, symbol="EUR/USD", outputsize="full"):
        """
        Download data from Alpha Vantage (free tier: 25 calls/day)
        """
        # Free API key - replace with your own for higher limits
        api_key = "demo"  # This is a demo key with limited data
        
        # Convert forex symbol
        from_symbol = symbol.split('/')[0]
        to_symbol = symbol.split('/')[1]
        
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "FX_DAILY",
            "from_symbol": from_symbol,
            "to_symbol": to_symbol,
            "outputsize": outputsize,
            "apikey": api_key
        }
        
        print(f"Downloading {symbol} data from Alpha Vantage...")
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if "Error Message" in data:
                print(f"Error: {data['Error Message']}")
                return None
            
            if "Note" in data:
                print(f"API Limit: {data['Note']}")
                return None
            
            if "Time Series (Daily)" not in data:
                print("No time series data found")
                return None
            
            # Convert to DataFrame
            time_series = data["Time Series (Daily)"]
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Clean column names
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            df = df.sort_index()  # Oldest to newest
            
            print(f"Downloaded {len(df)} days of {symbol} data")
            return df
            
        except Exception as e:
            print(f"Error downloading from Alpha Vantage: {e}")
            return None
    
    def generate_synthetic_realistic_data(self, symbol="EUR/USD", days=1000):
        """
        Generate realistic synthetic forex data for training
        Based on actual forex market characteristics
        """
        print(f"Generating {days} days of realistic {symbol} data...")
        
        # Forex pair characteristics
        forex_params = {
            "EUR/USD": {"base": 1.1000, "volatility": 0.008, "trend": 0.0001},
            "GBP/USD": {"base": 1.3000, "volatility": 0.012, "trend": -0.0002},
            "USD/JPY": {"base": 110.00, "volatility": 0.006, "trend": 0.0001},
            "AUD/USD": {"base": 0.7500, "volatility": 0.010, "trend": 0.0001},
        }
        
        params = forex_params.get(symbol, forex_params["EUR/USD"])
        
        # Generate realistic price movements
        np.random.seed(42)  # Reproducible data
        
        # Daily returns with realistic characteristics
        returns = np.random.normal(
            params["trend"],  # Small trending bias
            params["volatility"],  # Realistic volatility
            days
        )
        
        # Add some market regime changes (trending vs ranging)
        regime_changes = np.random.choice([0, 1], days, p=[0.7, 0.3])  # 30% trending days
        returns[regime_changes == 1] *= 2  # Higher volatility on trending days
        
        # Add some correlation to previous days (momentum/mean reversion)
        for i in range(1, len(returns)):
            if np.random.random() < 0.3:  # 30% momentum
                returns[i] += returns[i-1] * 0.1
            elif np.random.random() < 0.2:  # 20% mean reversion
                returns[i] -= returns[i-1] * 0.1
        
        # Generate price series
        prices = [params["base"]]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        prices = np.array(prices[1:])  # Remove initial price
        
        # Generate OHLC data
        df_data = []
        for i, close in enumerate(prices):
            # Random intraday movement
            daily_range = abs(np.random.normal(0, params["volatility"] * 0.5))
            
            high = close + daily_range * np.random.uniform(0.3, 1.0)
            low = close - daily_range * np.random.uniform(0.3, 1.0)
            
            # Ensure high > low and close is between them
            high = max(high, close)
            low = min(low, close)
            
            # Open is previous close with some gap
            if i == 0:
                open_price = close
            else:
                gap = np.random.normal(0, params["volatility"] * 0.2)
                open_price = prices[i-1] * (1 + gap)
                open_price = max(min(open_price, high), low)  # Keep within range
            
            # Volume (realistic for forex - in units, not actual volume)
            volume = np.random.randint(10000, 100000)
            
            df_data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        # Create DataFrame
        start_date = datetime.now() - timedelta(days=days)
        dates = pd.date_range(start=start_date, periods=days, freq='D')
        
        df = pd.DataFrame(df_data, index=dates)
        
        print(f"Generated {len(df)} days of {symbol} data")
        print(f"Price range: {df['close'].min():.4f} to {df['close'].max():.4f}")
        print(f"Average daily return: {df['close'].pct_change().mean():.6f}")
        print(f"Daily volatility: {df['close'].pct_change().std():.6f}")
        
        return df
    
    def download_yahoo_finance_data(self, symbol="EURUSD=X", period="2y"):
        """
        Download from Yahoo Finance (free, no API key needed)
        """
        try:
            import yfinance as yf
            print(f"Downloading {symbol} from Yahoo Finance...")
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval="1d")
            
            if df.empty:
                print("No data received from Yahoo Finance")
                return None
            
            # Clean up the data
            df.columns = df.columns.str.lower()
            df = df.drop(['dividends', 'stock splits'], axis=1, errors='ignore')
            
            print(f"Downloaded {len(df)} days of {symbol} data")
            return df
            
        except ImportError:
            print("yfinance not installed. Install with: pip install yfinance")
            return None
        except Exception as e:
            print(f"Error downloading from Yahoo Finance: {e}")
            return None
    
    def save_data(self, df, symbol, source="synthetic"):
        """Save data to feather format for fast loading"""
        if df is None or df.empty:
            print("No data to save")
            return
        
        filename = f"{symbol.replace('/', '_')}_{source}_daily.feather"
        filepath = os.path.join(self.data_dir, filename)
        
        # Reset index to save date as column
        df_save = df.reset_index()
        df_save.rename(columns={'index': 'date'}, inplace=True)
        
        try:
            df_save.to_feather(filepath)
            print(f"Saved data to: {filepath}")
            
            # Also save as CSV for inspection
            csv_path = filepath.replace('.feather', '.csv')
            df_save.to_csv(csv_path, index=False)
            print(f"Also saved as CSV: {csv_path}")
            
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def download_all_pairs(self):
        """Download data for multiple forex pairs"""
        pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
        yahoo_symbols = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"]
        
        print("Starting forex data download...")
        
        for i, pair in enumerate(pairs):
            print(f"\n--- Downloading {pair} ---")
            
            # Try Yahoo Finance first
            yahoo_symbol = yahoo_symbols[i]
            df = self.download_yahoo_finance_data(yahoo_symbol)
            
            if df is not None and len(df) > 100:
                self.save_data(df, pair, "yahoo")
                print(f"Yahoo Finance data saved for {pair}")
            else:
                print(f"Yahoo Finance failed for {pair}, generating synthetic data")
                df = self.generate_synthetic_realistic_data(pair, days=800)
                self.save_data(df, pair, "synthetic")
            
            time.sleep(1)  # Be nice to APIs
        
        print("\n--- Data download complete! ---")
        self.show_summary()
    
    def show_summary(self):
        """Show summary of downloaded data"""
        print("\nData Summary:")
        print("=" * 50)
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.feather'):
                filepath = os.path.join(self.data_dir, filename)
                try:
                    df = pd.read_feather(filepath)
                    print(f"{filename}:")
                    print(f"  Records: {len(df)}")
                    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
                    print(f"  Price range: {df['close'].min():.4f} to {df['close'].max():.4f}")
                except Exception as e:
                    print(f"  Error reading {filename}: {e}")

def main():
    """Download forex data for training"""
    print("ForexSwing AI 2025 - Data Downloader")
    print("=" * 40)
    
    # Install yfinance if not present
    try:
        import yfinance
    except ImportError:
        print("Installing yfinance for data download...")
        os.system("pip install yfinance")
    
    downloader = ForexDataDownloader()
    downloader.download_all_pairs()
    
    print("\nNext step: Run training with:")
    print("python src/ml_models/train_model.py")

if __name__ == "__main__":
    main()