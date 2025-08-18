#!/usr/bin/env python3
'''
ForexSwing AI - Main Testing Script
Test your optimized AI performance
'''

import sys
import os
sys.path.append('scripts')

def main():
    print("FOREXSWING AI TESTING")
    print("=" * 30)
    
    # Import testing module
    try:
        from testing.test_optimized_simple import main as test_ai
        test_ai()
    except ImportError:
        print("Testing module not found. Please check installation.")
        print("Run: python scripts/testing/test_optimized_simple.py")

if __name__ == "__main__":
    main()
