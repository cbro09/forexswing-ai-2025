#!/usr/bin/env python3
"""
Test Gemini CLI integration with ForexSwing AI
"""

import sys
import os
sys.path.append('core')

from integrations.gemini_data_interpreter import GeminiDataInterpreter
from integrations.enhanced_strategy import EnhancedForexStrategy
import pandas as pd
import numpy as np

def test_integration():
    print("GEMINI CLI INTEGRATION TEST")
    print("=" * 40)
    
    # Test 1: Basic Gemini availability
    interpreter = GeminiDataInterpreter()
    print(f"Gemini CLI Available: {interpreter.gemini_available}")
    
    if not interpreter.gemini_available:
        print("[ERROR] Gemini CLI not available")
        print("Run: python scripts/setup_gemini.py")
        return False
    
    # Test 2: Enhanced strategy
    print("\nTesting Enhanced Strategy...")
    strategy = EnhancedForexStrategy()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 1.1000,
        'volume': np.random.randint(1000, 5000, 100),
        'rsi': np.random.uniform(30, 70, 100),
        'macd': np.random.randn(100) * 0.0001,
        'macd_signal': np.random.randn(100) * 0.0001,
    })
    
    # Test recommendation
    rec = strategy.get_trading_recommendation(sample_data, "EUR/USD")
    
    print(f"[OK] Trading recommendation generated:")
    print(f"   Action: {rec['action']}")
    print(f"   Confidence: {rec['confidence']:.1%}")
    print(f"   Risk: {rec['risk_level']}")
    
    print("\n[COMPLETE] Integration test complete!")
    return True

if __name__ == "__main__":
    test_integration()