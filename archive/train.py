#!/usr/bin/env python3
"""
ForexSwing AI 2025 - Main Training Script
Train the enhanced dual AI system (LSTM + Gemini)
"""

import sys
import os
sys.path.append('src')

def main():
    print("FOREXSWING AI 2025 - TRAINING")
    print("=" * 40)
    
    # Import training module
    try:
        sys.path.append('scripts')
        from training.train_enhanced_ai import main as train_enhanced
        train_enhanced()
    except ImportError as e:
        print(f"Training module import failed: {e}")
        print("Please ensure all dependencies are installed.")
        print("Run: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
