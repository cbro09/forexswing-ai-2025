#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily Data Update Script for ForexSwing AI
- Fetches latest market data
- Maintains 600-day rolling window per pair
- Removes old data to save space
- Run this daily via cron/task scheduler
"""

import sys
import os
import io
from datetime import datetime, timedelta

# Force UTF-8 encoding for console output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from src.data_updater import HistoricalDataUpdater

def trim_to_rolling_window(file_path: str, days_to_keep: int = 600):
    """Keep only the most recent N days of data"""
    try:
        if not os.path.exists(file_path):
            return False

        df = pd.read_csv(file_path)
        if df.empty:
            return False

        # Parse dates and make timezone-aware for comparison
        df['Date'] = pd.to_datetime(df['Date'])

        # Calculate cutoff date (make timezone-aware to match data)
        cutoff_date = pd.Timestamp(datetime.now() - timedelta(days=days_to_keep))
        if df['Date'].dt.tz is not None:
            cutoff_date = cutoff_date.tz_localize('UTC').tz_convert(df['Date'].dt.tz)

        # Filter to keep only recent data
        df_filtered = df[df['Date'] >= cutoff_date]

        rows_before = len(df)
        rows_after = len(df_filtered)
        rows_removed = rows_before - rows_after

        if rows_removed > 0:
            # Save filtered data
            df_filtered.to_csv(file_path, index=False)
            print(f"  ✂️  Trimmed {rows_removed} old rows, kept {rows_after} days")
            return True
        else:
            print(f"  ✅ Already within {days_to_keep} day window ({rows_after} rows)")
            return False

    except Exception as e:
        print(f"  ❌ Error trimming data: {e}")
        return False

def main():
    print("=" * 70)
    print("🚀 DAILY DATA UPDATE - ROLLING 1-YEAR WINDOW")
    print("=" * 70)
    print(f"Update Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize updater
    updater = HistoricalDataUpdater()
    pairs = list(updater.pairs_mapping.keys())

    print(f"📊 Updating {len(pairs)} currency pairs...")
    print("-" * 70)

    # Step 1: Update with fresh data
    print("\n🔄 STEP 1: Fetching latest market data...")
    results = updater.update_all_pairs()

    successful_updates = sum(1 for r in results.values() if isinstance(r, dict) and r.get('success'))

    # Step 2: Trim to rolling window
    print("\n✂️  STEP 2: Trimming to 600-day rolling window...")
    for pair in pairs:
        csv_path = updater.get_csv_file_path(pair)
        print(f"  {pair}:")
        trim_to_rolling_window(csv_path, days_to_keep=600)

    # Summary
    print("\n" + "=" * 70)
    print("📈 UPDATE SUMMARY")
    print("=" * 70)

    for pair, result in results.items():
        if isinstance(result, dict):
            status = "✅" if result.get('success') else "❌"
            message = result.get('message', 'Unknown')

            # Get current row count
            csv_path = updater.get_csv_file_path(pair)
            try:
                df = pd.read_csv(csv_path)
                row_count = len(df)
                print(f"{status} {pair}: {message} ({row_count} days)")
            except:
                print(f"{status} {pair}: {message}")
        else:
            print(f"❌ {pair}: Update failed")

    print("-" * 70)
    print(f"✅ Successfully updated: {successful_updates}/{len(results)} pairs")
    print(f"📅 Data window: ~600 days (1.5-2 years rolling)")
    print(f"💾 Old data automatically removed to save space")
    print("=" * 70)

    print("\n💡 TIP: Add this script to your daily cron/task scheduler:")
    print("   Linux/Mac: 0 2 * * * python scripts/daily_data_update.py")
    print("   Windows: Use Task Scheduler to run daily at 2 AM")

if __name__ == "__main__":
    main()
