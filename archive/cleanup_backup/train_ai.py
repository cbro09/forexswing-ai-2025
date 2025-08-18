#!/usr/bin/env python3
'''
ForexSwing AI - Main Training Script
Train your AI on real forex market data
'''

import sys
import os
sys.path.append('scripts')

def main():
    print("FOREXSWING AI TRAINING")
    print("=" * 30)
    
    # Import training module
    try:
        from training.optimize_ai import main as train_optimized
        train_optimized()
    except ImportError:
        print("Training module not found. Please check installation.")
        print("Run: python scripts/training/optimize_ai.py")

if __name__ == "__main__":
    main()
