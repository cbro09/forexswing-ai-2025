#!/usr/bin/env python3
"""
Automatic Historical Data Updater for ForexSwing AI
Updates CSV files with fresh data from Yahoo Finance to keep 500-day history current
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from typing import Dict, List
import time

class HistoricalDataUpdater:
    """Updates historical market data CSV files automatically"""
    
    def __init__(self, data_dir: str = "data/MarketData"):
        self.data_dir = data_dir
        self.setup_logging()
        
        # Currency pairs mapping to Yahoo Finance symbols
        self.pairs_mapping = {
            'EUR_USD': 'EURUSD=X',
            'GBP_USD': 'GBPUSD=X', 
            'USD_JPY': 'USDJPY=X',
            'USD_CHF': 'USDCHF=X',
            'AUD_USD': 'AUDUSD=X',
            'USD_CAD': 'USDCAD=X',
            'NZD_USD': 'NZDUSD=X'
        }
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.logger.info(f"HistoricalDataUpdater initialized for {len(self.pairs_mapping)} pairs")
    
    def setup_logging(self):
        """Setup logging for data update operations"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create console handler if not exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def get_csv_file_path(self, pair: str) -> str:
        """Get CSV file path for a currency pair"""
        return os.path.join(self.data_dir, f"{pair}_real_daily.csv")
    
    def check_data_freshness(self, pair: str) -> Dict:
        """Check if data needs updating"""
        csv_path = self.get_csv_file_path(pair)
        
        if not os.path.exists(csv_path):
            return {
                "needs_update": True,
                "reason": "file_not_exists",
                "last_date": None,
                "days_old": 0
            }
        
        try:
            # Read existing data
            df = pd.read_csv(csv_path)
            if df.empty:
                return {
                    "needs_update": True,
                    "reason": "empty_file",
                    "last_date": None,
                    "days_old": 0
                }
            
            # Get the last date with actual data (not empty rows)
            df_clean = df.dropna(subset=['Close'])
            if df_clean.empty:
                return {
                    "needs_update": True,
                    "reason": "no_valid_data",
                    "last_date": None,
                    "days_old": 0
                }
            
            last_date_str = df_clean['Date'].iloc[-1]
            last_date = pd.to_datetime(last_date_str).date()
            today = datetime.now().date()
            days_old = (today - last_date).days
            
            # Update if data is more than 1 day old
            needs_update = days_old > 1
            
            return {
                "needs_update": needs_update,
                "reason": f"data_{days_old}_days_old" if needs_update else "current",
                "last_date": last_date,
                "days_old": days_old,
                "total_rows": len(df),
                "valid_rows": len(df_clean)
            }
            
        except Exception as e:
            self.logger.error(f"Error checking {pair} data freshness: {e}")
            return {
                "needs_update": True,
                "reason": f"error_checking: {e}",
                "last_date": None,
                "days_old": 0
            }
    
    def fetch_fresh_data(self, pair: str, days: int = 600) -> pd.DataFrame:
        """Fetch fresh historical data from Yahoo Finance"""
        try:
            yahoo_symbol = self.pairs_mapping.get(pair)
            if not yahoo_symbol:
                self.logger.error(f"No Yahoo Finance symbol mapping for {pair}")
                return pd.DataFrame()
            
            self.logger.info(f"Fetching {days} days of data for {pair} ({yahoo_symbol})")
            
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(yahoo_symbol)
            
            # Get historical data (more than 500 days to ensure we have enough)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            hist = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if hist.empty:
                self.logger.warning(f"No data received for {pair}")
                return pd.DataFrame()
            
            # Clean and format data
            df = hist.reset_index()
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S+01:00')
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            # Select only needed columns
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Keep only the last 550 days (gives us buffer above 500)
            if len(df) > 550:
                df = df.tail(550).reset_index(drop=True)
            
            self.logger.info(f"Successfully fetched {len(df)} days of data for {pair}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {pair}: {e}")
            return pd.DataFrame()
    
    def update_csv_file(self, pair: str, fresh_df: pd.DataFrame, backup: bool = True) -> bool:
        """Update CSV file with fresh data"""
        try:
            csv_path = self.get_csv_file_path(pair)
            
            # Create backup if file exists and backup is requested
            if backup and os.path.exists(csv_path):
                backup_path = csv_path.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d")}.csv')
                try:
                    os.rename(csv_path, backup_path)
                    self.logger.info(f"Created backup: {backup_path}")
                except Exception as e:
                    self.logger.warning(f"Could not create backup: {e}")
            
            # Write fresh data
            fresh_df.to_csv(csv_path, index=False)
            
            self.logger.info(f"Updated {pair} data: {len(fresh_df)} rows written to {csv_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating CSV for {pair}: {e}")
            return False
    
    def update_single_pair(self, pair: str, force: bool = False) -> Dict:
        """Update data for a single currency pair"""
        start_time = time.time()
        
        self.logger.info(f"Checking {pair} for updates...")
        
        # Check if update is needed
        freshness = self.check_data_freshness(pair)
        
        if not force and not freshness["needs_update"]:
            self.logger.info(f"{pair} data is current ({freshness['reason']})")
            return {
                "pair": pair,
                "updated": False,
                "reason": freshness["reason"],
                "processing_time": f"{time.time() - start_time:.2f}s"
            }
        
        # Fetch fresh data
        self.logger.info(f"Updating {pair} data ({freshness['reason']})")
        fresh_df = self.fetch_fresh_data(pair)
        
        if fresh_df.empty:
            return {
                "pair": pair,
                "updated": False,
                "reason": "fetch_failed",
                "processing_time": f"{time.time() - start_time:.2f}s"
            }
        
        # Update CSV file
        success = self.update_csv_file(pair, fresh_df)
        
        processing_time = time.time() - start_time
        
        if success:
            self.logger.info(f"Successfully updated {pair} in {processing_time:.2f}s")
            return {
                "pair": pair,
                "updated": True,
                "rows_written": len(fresh_df),
                "processing_time": f"{processing_time:.2f}s",
                "date_range": f"{fresh_df['Date'].iloc[0]} to {fresh_df['Date'].iloc[-1]}"
            }
        else:
            return {
                "pair": pair,
                "updated": False,
                "reason": "write_failed",
                "processing_time": f"{processing_time:.2f}s"
            }
    
    def update_all_pairs(self, force: bool = False, delay_between_pairs: float = 2.0) -> Dict:
        """Update all currency pairs"""
        start_time = time.time()
        results = []
        
        self.logger.info(f"Starting bulk update for {len(self.pairs_mapping)} pairs (force={force})")
        
        for i, pair in enumerate(self.pairs_mapping.keys()):
            try:
                result = self.update_single_pair(pair, force=force)
                results.append(result)
                
                # Add delay between pairs to be respectful to Yahoo Finance API
                if i < len(self.pairs_mapping) - 1 and delay_between_pairs > 0:
                    time.sleep(delay_between_pairs)
                    
            except Exception as e:
                self.logger.error(f"Error updating {pair}: {e}")
                results.append({
                    "pair": pair,
                    "updated": False,
                    "reason": f"exception: {e}",
                    "processing_time": "0s"
                })
        
        # Summary
        total_time = time.time() - start_time
        updated_count = sum(1 for r in results if r["updated"])
        failed_count = len(results) - updated_count
        
        summary = {
            "total_pairs": len(results),
            "updated_pairs": updated_count,
            "failed_pairs": failed_count,
            "total_time": f"{total_time:.2f}s",
            "results": results
        }
        
        self.logger.info(f"Bulk update complete: {updated_count}/{len(results)} pairs updated in {total_time:.2f}s")
        
        return summary
    
    def get_status_report(self) -> Dict:
        """Get status report for all currency pairs"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "pairs": {}
        }
        
        for pair in self.pairs_mapping.keys():
            freshness = self.check_data_freshness(pair)
            report["pairs"][pair] = freshness
        
        return report

def test_data_updater():
    """Test the data updater functionality"""
    print("TESTING HISTORICAL DATA UPDATER")
    print("=" * 50)
    
    updater = HistoricalDataUpdater()
    
    # Test status report
    print("Current data status:")
    status = updater.get_status_report()
    
    for pair, info in status["pairs"].items():
        print(f"  {pair}: {info['reason']} (last: {info['last_date']}, {info['days_old']} days old)")
    
    # Test single pair update (EUR_USD)
    print(f"\nTesting single pair update (EUR_USD):")
    result = updater.update_single_pair("EUR_USD", force=False)
    print(f"  Result: {result}")
    
    print(f"\nData updater ready for automatic updates!")

if __name__ == "__main__":
    test_data_updater()