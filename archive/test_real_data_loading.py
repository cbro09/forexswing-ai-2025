#!/usr/bin/env python3
"""
Test real data loading to debug issues
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.append('src')

def test_real_data_loading():
    """Test loading real market data"""
    print("Testing real market data loading...")
    
    data_dir = "data/real_market"
    
    if not os.path.exists(data_dir):
        print("Real market data directory not found!")
        return
    
    all_data = []
    
    # Load all real market feather files
    for filename in os.listdir(data_dir):
        if filename.endswith('_real_daily.feather'):
            print(f"Loading {filename}...")
            
            try:
                df = pd.read_feather(os.path.join(data_dir, filename))
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {df.columns.tolist()}")
                print(f"  Date range: {df.index[0] if hasattr(df.index, 'min') else 'No index'}")
                
                # Set date index
                if 'date' in df.columns:
                    df = df.set_index('date')
                elif 'Date' in df.columns:
                    df = df.set_index('Date')
                
                # Extract pair name
                pair_name = filename.replace('_real_daily.feather', '').replace('_', '/')
                df['pair'] = pair_name
                all_data.append(df)
                
                print(f"  Successfully processed {pair_name}")
                
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
    
    if all_data:
        combined_data = pd.concat(all_data).sort_index()
        print(f"\nTotal combined dataset: {len(combined_data)} candles")
        print(f"Pairs: {combined_data['pair'].unique()}")
        print(f"Date range: {combined_data.index.min()} to {combined_data.index.max()}")
        return combined_data
    else:
        print("No data loaded!")
        return None

if __name__ == "__main__":
    result = test_real_data_loading()