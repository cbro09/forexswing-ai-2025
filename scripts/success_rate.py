#!/usr/bin/env python3
'''
ForexSwing AI - Success Rate Testing
Comprehensive AI performance analysis
'''

import sys
import os
sys.path.append('scripts')

def main():
    print("FOREXSWING AI SUCCESS RATE TEST")
    print("=" * 40)
    
    # Import success testing module
    try:
        from testing.simple_success_test import main as test_success
        test_success()
    except ImportError:
        print("Success testing module not found.")
        print("Run: python scripts/testing/simple_success_test.py")

if __name__ == "__main__":
    main()
