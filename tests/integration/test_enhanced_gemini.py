#!/usr/bin/env python3
"""
Test Enhanced Gemini Integration
"""

import pandas as pd
import numpy as np
import os
from enhanced_gemini_trading_system import EnhancedGeminiTradingSystem

def test_gemini_enhanced_analysis():
    """Test the enhanced Gemini analysis"""
    print("TESTING ENHANCED GEMINI ANALYSIS")
    print("=" * 50)
    
    # Initialize system
    system = EnhancedGeminiTradingSystem(initial_balance=10000.0)
    
    # Load sample data
    if os.path.exists("data/MarketData/EUR_USD_real_daily.csv"):
        data = pd.read_csv("data/MarketData/EUR_USD_real_daily.csv")
        sample_data = data.tail(100)
        
        print(f"Testing with {len(sample_data)} EUR/USD candles")
        
        # Run enhanced analysis
        analysis = system.analyze_market_with_gemini(sample_data, "EUR/USD")
        
        print(f"\nANALYSIS RESULTS:")
        print(f"LSTM: {analysis.lstm_action} ({analysis.lstm_confidence:.1%})")
        print(f"Gemini: {analysis.gemini_sentiment} ({analysis.gemini_confidence:.1%})")
        print(f"Final: {analysis.final_action} ({analysis.final_confidence:.1%})")
        print(f"Confidence Boost: {analysis.confidence_boost:+.1%}")
        print(f"Risk Level: {analysis.risk_level}")
        print(f"Trend: {analysis.trend_direction}")
        print(f"Volatility: {analysis.volatility_level}")
        
        # Test trading execution
        print(f"\nTEST TRADING EXECUTION:")
        market_data = {"EUR/USD": sample_data}
        results = system.execute_enhanced_trading(market_data)
        
        summary = results['account_summary']
        print(f"Account Return: {summary['return_pct']:+.2f}%")
        print(f"Open Positions: {summary['open_positions']}")
        
        return True
    else:
        print("No EUR/USD data available for test")
        return False

def quick_gemini_comparison():
    """Quick comparison of LSTM vs Gemini+LSTM"""
    print(f"\nQUICK COMPARISON TEST")
    print("=" * 50)
    
    from ForexBot import ForexBot
    
    # Load data
    if os.path.exists("data/MarketData/GBP_USD_real_daily.csv"):
        data = pd.read_csv("data/MarketData/GBP_USD_real_daily.csv")
        test_data = data.tail(100)
        
        # Test 1: Pure LSTM
        forex_bot = ForexBot()
        lstm_rec = forex_bot.get_final_recommendation(test_data, "GBP/USD")
        
        # Test 2: Enhanced with Gemini
        enhanced_system = EnhancedGeminiTradingSystem(initial_balance=5000.0)
        enhanced_analysis = enhanced_system.analyze_market_with_gemini(test_data, "GBP/USD")
        
        print(f"\nCOMPARISON RESULTS:")
        print(f"Pure LSTM:")
        print(f"  Action: {lstm_rec['action']}")
        print(f"  Confidence: {lstm_rec['confidence']:.1%}")
        
        print(f"\nEnhanced (LSTM + Gemini):")
        print(f"  Action: {enhanced_analysis.final_action}")
        print(f"  Confidence: {enhanced_analysis.final_confidence:.1%}")
        print(f"  Gemini Contribution: {enhanced_analysis.confidence_boost:+.1%}")
        
        # Show difference
        conf_improvement = enhanced_analysis.final_confidence - lstm_rec['confidence']
        print(f"\nIMPROVEMENT:")
        print(f"  Confidence Change: {conf_improvement:+.1%}")
        print(f"  Risk Assessment: {enhanced_analysis.risk_level}")
        print(f"  Gemini Reasoning: {enhanced_analysis.gemini_reasoning}")
        
        return True
    else:
        print("No GBP/USD data available")
        return False

if __name__ == "__main__":
    print("ENHANCED GEMINI INTEGRATION TEST")
    print("=" * 60)
    
    test1 = test_gemini_enhanced_analysis()
    test2 = quick_gemini_comparison()
    
    if test1 and test2:
        print(f"\n[SUCCESS] Enhanced Gemini integration working!")
    else:
        print(f"\n[PARTIAL] Some tests completed")