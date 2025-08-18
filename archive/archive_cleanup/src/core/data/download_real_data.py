#!/usr/bin/env python3
"""
Download REAL Forex Market Data
Multiple sources for maximum accuracy training
"""

import pandas as pd
import numpy as np
import requests
import time
import os
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class RealForexDataDownloader:
    def __init__(self):
        self.data_dir = "data/real_market"
        os.makedirs(self.data_dir, exist_ok=True)
        print("Real Market Data Downloader initialized")
        print("Target: Professional-grade training data")
    
    def download_yahoo_forex(self, symbols, period="5y", interval="1d"):
        """Download real forex data from Yahoo Finance"""
        
        print(f"\nDownloading REAL forex data from Yahoo Finance...")
        print(f"Period: {period}, Interval: {interval}")
        
        yahoo_symbols = {
            "EUR/USD": "EURUSD=X",
            "GBP/USD": "GBPUSD=X", 
            "USD/JPY": "USDJPY=X",
            "AUD/USD": "AUDUSD=X",
            "USD/CHF": "USDCHF=X",
            "USD/CAD": "USDCAD=X",
            "NZD/USD": "NZDUSD=X"
        }
        
        real_data = {}
        
        for pair_name, yahoo_symbol in yahoo_symbols.items():
            if pair_name in symbols:
                print(f"\nDownloading {pair_name} ({yahoo_symbol})...")
                
                try:
                    ticker = yf.Ticker(yahoo_symbol)
                    df = ticker.history(period=period, interval=interval)
                    
                    if not df.empty:
                        # Clean data
                        df.columns = df.columns.str.lower()
                        df = df.drop(['dividends', 'stock splits'], axis=1, errors='ignore')
                        
                        # Remove any NaN values
                        df = df.dropna()
                        
                        print(f"  Downloaded {len(df)} days of data")
                        print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
                        print(f"  Price range: {df['close'].min():.4f} to {df['close'].max():.4f}")
                        
                        # Save data
                        filename = f"{pair_name.replace('/', '_')}_real_daily.feather"
                        filepath = os.path.join(self.data_dir, filename)
                        
                        df_save = df.reset_index()
                        df_save.to_feather(filepath)
                        
                        # Also save as CSV for inspection
                        csv_path = filepath.replace('.feather', '.csv')
                        df_save.to_csv(csv_path, index=False)
                        
                        print(f"  Saved: {filename}")
                        
                        real_data[pair_name] = df
                        
                    else:
                        print(f"  No data received for {pair_name}")
                        
                except Exception as e:
                    print(f"  Error downloading {pair_name}: {e}")
                
                time.sleep(1)  # Be nice to Yahoo Finance
        
        return real_data
    
    def download_alpha_vantage_data(self, api_key=None):
        """Download from Alpha Vantage (if API key provided)"""
        
        if not api_key:
            print("No Alpha Vantage API key provided, skipping...")
            return {}
        
        print(f"\nDownloading from Alpha Vantage...")
        
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
        av_data = {}
        
        for symbol in symbols:
            try:
                url = f"https://www.alphavantage.co/query"
                params = {
                    "function": "FX_DAILY",
                    "from_symbol": symbol[:3],
                    "to_symbol": symbol[3:],
                    "outputsize": "full",
                    "apikey": api_key
                }
                
                print(f"Downloading {symbol}...")
                response = requests.get(url, params=params)
                data = response.json()
                
                if "Time Series (Daily)" in data:
                    time_series = data["Time Series (Daily)"]
                    df = pd.DataFrame.from_dict(time_series, orient='index')
                    
                    df.columns = ['open', 'high', 'low', 'close', 'volume']
                    df.index = pd.to_datetime(df.index)
                    df = df.astype(float).sort_index()
                    
                    pair_name = f"{symbol[:3]}/{symbol[3:]}"
                    av_data[pair_name] = df
                    
                    print(f"  Downloaded {len(df)} days for {pair_name}")
                else:
                    print(f"  API limit or error for {symbol}")
                
                time.sleep(12)  # Alpha Vantage rate limit
                
            except Exception as e:
                print(f"Error downloading {symbol}: {e}")
        
        return av_data
    
    def analyze_real_data(self):
        """Analyze the downloaded real market data"""
        
        print("\n" + "=" * 60)
        print("ANALYZING REAL MARKET DATA")
        print("=" * 60)
        
        if not os.path.exists(self.data_dir):
            print("No real data directory found!")
            return
        
        analysis_results = {}
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('_real_daily.feather'):
                pair_name = filename.replace('_real_daily.feather', '').replace('_', '/')
                
                print(f"\n--- Analyzing {pair_name} ---")
                
                df = pd.read_feather(os.path.join(self.data_dir, filename))
                
                if 'date' in df.columns:
                    df = df.set_index('date')
                elif 'Date' in df.columns:
                    df = df.set_index('Date')
                
                # Calculate returns
                returns = df['close'].pct_change().dropna()
                
                # Basic statistics
                print(f"Data points: {len(df)}")
                print(f"Date range: {df.index[0]} to {df.index[-1]}")
                print(f"Price range: {df['close'].min():.4f} to {df['close'].max():.4f}")
                
                # Return analysis
                print(f"Daily return stats:")
                print(f"  Mean return: {returns.mean():.6f} ({returns.mean()*252:.3f} annualized)")
                print(f"  Volatility: {returns.std():.6f} ({returns.std()*np.sqrt(252):.3f} annualized)")
                print(f"  Skewness: {returns.skew():.3f}")
                print(f"  Kurtosis: {returns.kurtosis():.3f}")
                
                # Market regime analysis
                positive_days = (returns > 0.005).sum()
                negative_days = (returns < -0.005).sum()
                neutral_days = len(returns) - positive_days - negative_days
                
                print(f"Market behavior:")
                print(f"  Strong up days (>0.5%): {positive_days} ({positive_days/len(returns)*100:.1f}%)")
                print(f"  Strong down days (<-0.5%): {negative_days} ({negative_days/len(returns)*100:.1f}%)")
                print(f"  Neutral days: {neutral_days} ({neutral_days/len(returns)*100:.1f}%)")
                
                # Volatility clustering
                rolling_vol = returns.rolling(20).std()
                high_vol_periods = (rolling_vol > rolling_vol.quantile(0.8)).sum()
                
                print(f"  High volatility periods: {high_vol_periods} ({high_vol_periods/len(rolling_vol)*100:.1f}%)")
                
                # Trending analysis
                sma_20 = df['close'].rolling(20).mean()
                sma_50 = df['close'].rolling(50).mean()
                
                uptrend_days = (df['close'] > sma_20).sum()
                strong_uptrend = ((df['close'] > sma_20) & (sma_20 > sma_50)).sum()
                
                print(f"  Uptrend days: {uptrend_days} ({uptrend_days/len(df)*100:.1f}%)")
                print(f"  Strong uptrend days: {strong_uptrend} ({strong_uptrend/len(df)*100:.1f}%)")
                
                analysis_results[pair_name] = {
                    'data_points': len(df),
                    'mean_return': returns.mean(),
                    'volatility': returns.std(),
                    'positive_days_pct': positive_days/len(returns)*100,
                    'negative_days_pct': negative_days/len(returns)*100,
                    'high_vol_pct': high_vol_periods/len(rolling_vol)*100 if len(rolling_vol) > 0 else 0,
                    'uptrend_pct': uptrend_days/len(df)*100
                }
        
        # Summary comparison
        print(f"\n" + "=" * 60)
        print("REAL MARKET DATA SUMMARY")
        print("=" * 60)
        
        if analysis_results:
            avg_vol = np.mean([r['volatility'] for r in analysis_results.values()])
            avg_pos = np.mean([r['positive_days_pct'] for r in analysis_results.values()])
            avg_neg = np.mean([r['negative_days_pct'] for r in analysis_results.values()])
            
            print(f"Average characteristics across pairs:")
            print(f"  Daily volatility: {avg_vol:.4f}")
            print(f"  Strong positive days: {avg_pos:.1f}%")
            print(f"  Strong negative days: {avg_neg:.1f}%")
            print(f"  Market balance: {abs(avg_pos - avg_neg):.1f}% bias")
            
            print(f"\nData quality assessment:")
            total_points = sum(r['data_points'] for r in analysis_results.values())
            print(f"  Total data points: {total_points}")
            print(f"  Average per pair: {total_points/len(analysis_results):.0f}")
            
            if avg_vol > 0.008:
                print("  HIGH VOLATILITY: Good for ML training")
            else:
                print("  MODERATE VOLATILITY: Stable but may need more features")
            
            if abs(avg_pos - avg_neg) < 5:
                print("  BALANCED MARKETS: Excellent for unbiased training")
            else:
                print("  DIRECTIONAL BIAS: May need data balancing")
        
        return analysis_results
    
    def compare_synthetic_vs_real(self):
        """Compare synthetic vs real data characteristics"""
        
        print("\n" + "=" * 60)
        print("SYNTHETIC vs REAL DATA COMPARISON")
        print("=" * 60)
        
        # Load synthetic data for comparison
        synthetic_dir = "data/balanced_training"
        real_dir = "data/real_market"
        
        if not os.path.exists(synthetic_dir) or not os.path.exists(real_dir):
            print("Missing data directories for comparison")
            return
        
        comparison = {}
        
        # Common pairs to compare
        pairs_to_compare = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
        
        for pair in pairs_to_compare:
            print(f"\n--- {pair} Comparison ---")
            
            # Load synthetic data
            synthetic_file = f"{pair.replace('/', '_')}_balanced_daily.feather"
            synthetic_path = os.path.join(synthetic_dir, synthetic_file)
            
            # Load real data
            real_file = f"{pair.replace('/', '_')}_real_daily.feather"
            real_path = os.path.join(real_dir, real_file)
            
            if os.path.exists(synthetic_path) and os.path.exists(real_path):
                
                # Synthetic data
                synthetic_df = pd.read_feather(synthetic_path)
                if 'date' in synthetic_df.columns:
                    synthetic_df = synthetic_df.set_index('date')
                synthetic_returns = synthetic_df['close'].pct_change().dropna()
                
                # Real data
                real_df = pd.read_feather(real_path)
                if 'date' in real_df.columns:
                    real_df = real_df.set_index('date')
                real_returns = real_df['close'].pct_change().dropna()
                
                # Compare statistics
                print(f"Data points:")
                print(f"  Synthetic: {len(synthetic_df)}")
                print(f"  Real: {len(real_df)}")
                
                print(f"Volatility:")
                print(f"  Synthetic: {synthetic_returns.std():.4f}")
                print(f"  Real: {real_returns.std():.4f}")
                
                print(f"Mean return:")
                print(f"  Synthetic: {synthetic_returns.mean():.6f}")
                print(f"  Real: {real_returns.mean():.6f}")
                
                # Distribution comparison
                synthetic_pos = (synthetic_returns > 0.005).sum() / len(synthetic_returns) * 100
                real_pos = (real_returns > 0.005).sum() / len(real_returns) * 100
                
                print(f"Strong positive days:")
                print(f"  Synthetic: {synthetic_pos:.1f}%")
                print(f"  Real: {real_pos:.1f}%")
                
                synthetic_neg = (synthetic_returns < -0.005).sum() / len(synthetic_returns) * 100
                real_neg = (real_returns < -0.005).sum() / len(real_returns) * 100
                
                print(f"Strong negative days:")
                print(f"  Synthetic: {synthetic_neg:.1f}%")
                print(f"  Real: {real_neg:.1f}%")
                
                comparison[pair] = {
                    'vol_diff': abs(synthetic_returns.std() - real_returns.std()),
                    'return_diff': abs(synthetic_returns.mean() - real_returns.mean()),
                    'pos_diff': abs(synthetic_pos - real_pos),
                    'neg_diff': abs(synthetic_neg - real_neg)
                }
        
        # Overall assessment
        if comparison:
            print(f"\n" + "=" * 60)
            print("DOMAIN GAP ANALYSIS")
            print("=" * 60)
            
            avg_vol_diff = np.mean([c['vol_diff'] for c in comparison.values()])
            avg_pos_diff = np.mean([c['pos_diff'] for c in comparison.values()])
            avg_neg_diff = np.mean([c['neg_diff'] for c in comparison.values()])
            
            print(f"Average differences:")
            print(f"  Volatility gap: {avg_vol_diff:.4f}")
            print(f"  Positive day gap: {avg_pos_diff:.1f}%")
            print(f"  Negative day gap: {avg_neg_diff:.1f}%")
            
            if avg_vol_diff < 0.002 and avg_pos_diff < 5 and avg_neg_diff < 5:
                print("ASSESSMENT: Small domain gap - Retraining should give major boost!")
            elif avg_vol_diff < 0.005 and avg_pos_diff < 10 and avg_neg_diff < 10:
                print("ASSESSMENT: Moderate domain gap - Expect significant improvement")
            else:
                print("ASSESSMENT: Large domain gap - Need careful retraining approach")
        
        return comparison

def main():
    """Download and analyze real market data"""
    
    print("REAL FOREX MARKET DATA DOWNLOADER")
    print("Professional-Grade Training Data")
    print("=" * 50)
    
    downloader = RealForexDataDownloader()
    
    # Download from Yahoo Finance (free and reliable)
    symbols_to_download = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CHF", "USD/CAD", "NZD/USD"]
    
    print("Downloading 5 years of daily forex data...")
    real_data = downloader.download_yahoo_forex(symbols_to_download, period="5y", interval="1d")
    
    if real_data:
        print(f"\nSuccessfully downloaded {len(real_data)} currency pairs!")
        
        # Analyze the data
        analysis = downloader.analyze_real_data()
        
        # Compare with synthetic data
        comparison = downloader.compare_synthetic_vs_real()
        
        print(f"\n" + "=" * 50)
        print("REAL DATA DOWNLOAD COMPLETE!")
        print("=" * 50)
        print("Ready for AI retraining with professional market data!")
        print("\nNext steps:")
        print("1. Retrain AI model on real market data")
        print("2. Test improved model performance")
        print("3. Deploy to live trading")
        
    else:
        print("No data downloaded. Check internet connection and try again.")

if __name__ == "__main__":
    main()