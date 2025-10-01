#!/usr/bin/env python3
"""
Quick script to update all forex market data to current prices
Run this daily to keep data fresh for AI analysis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_updater import HistoricalDataUpdater

def main():
    print("=" * 70)
    print("FOREXSWING AI - MARKET DATA UPDATER")
    print("=" * 70)

    updater = HistoricalDataUpdater()

    # Update all pairs
    pairs = list(updater.pairs_mapping.keys())

    print(f"\nUpdating {len(pairs)} currency pairs...")
    print("-" * 70)

    results = updater.update_all_pairs()

    print("\n" + "=" * 70)
    print("UPDATE SUMMARY")
    print("=" * 70)

    successful = sum(1 for r in results.values() if r.get('success'))
    failed = len(results) - successful

    for pair, result in results.items():
        status = "✅" if result.get('success') else "❌"
        print(f"{status} {pair}: {result.get('message', 'Unknown')}")

    print("-" * 70)
    print(f"Successful: {successful}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    print("=" * 70)

    if successful > 0:
        print("\n✅ Market data updated! AI models will use fresh prices.")
    else:
        print("\n❌ Update failed. Check your internet connection.")

if __name__ == "__main__":
    main()
