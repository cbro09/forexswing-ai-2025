#!/usr/bin/env python3
"""
ForexSwing AI 2025 - Strategy Runner
Run the enhanced forex trading strategy
"""

import sys
import os
sys.path.append('src')

def main():
    print("FOREXSWING AI 2025 - ENHANCED STRATEGY")
    print("=" * 50)
    
    try:
        from integrations.gemini.enhanced_strategy import EnhancedForexStrategy
        from core.data.download_real_data import download_latest_data
        
        print("Initializing enhanced strategy...")
        strategy = EnhancedForexStrategy("data/models/optimized_forex_ai.pth")
        
        print("Strategy ready!")
        print("Use strategy.get_trading_recommendation(dataframe, pair) for predictions")
        
        return strategy
        
    except Exception as e:
        print(f"Strategy initialization failed: {e}")
        print("Please ensure:")
        print("1. Model is trained: python train.py")
        print("2. Gemini CLI is setup: python scripts/deployment/setup_gemini.py")

if __name__ == "__main__":
    strategy = main()
