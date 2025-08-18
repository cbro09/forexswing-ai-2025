#!/usr/bin/env python3
"""
ForexSwing AI 2025 - Main Testing Script
Test the enhanced dual AI system performance
"""

import sys
import os
sys.path.append('src')

def main():
    print("FOREXSWING AI 2025 - TESTING")
    print("=" * 40)
    
    # Import testing module
    try:
        sys.path.append('scripts')
        from testing.test_performance import main as test_performance
        test_performance()
    except ImportError as e:
        print(f"Testing module import failed: {e}")
        print("Please ensure the system is properly trained.")
        print("Run: python train.py")

if __name__ == "__main__":
    main()
