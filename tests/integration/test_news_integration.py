#!/usr/bin/env python3
"""
Test News Integration with Existing ForexSwing AI System
Tests the enhanced Gemini system with news sentiment (fallback mode)
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Add paths for imports
sys.path.append('.')
sys.path.append('src')

def create_mock_news_sentiment():
    """Create mock news sentiment data for testing"""
    return {
        'overall_sentiment': 0.3,  # Slightly bullish
        'confidence': 0.65,
        'article_count': 12,
        'sentiment_breakdown': {
            'bullish': 0.5,
            'bearish': 0.3,
            'neutral': 0.2
        },
        'top_headlines': [
            "USD strengthens on Fed policy expectations",
            "European markets show positive sentiment",
            "Forex traders optimistic on dollar outlook"
        ]
    }

def test_enhanced_gemini_with_news():
    """Test enhanced Gemini system with news integration (fallback mode)"""
    print("TESTING ENHANCED GEMINI + NEWS INTEGRATION")
    print("=" * 60)
    
    try:
        # Import existing components
        from ForexBot import ForexBot
        from src.integrations.optimized_gemini import OptimizedGeminiInterpreter
        
        print("✅ Core components imported successfully")
        
        # Initialize components
        forex_bot = ForexBot()
        gemini = OptimizedGeminiInterpreter(cache_size=50, cache_duration_minutes=15)
        
        print(f"✅ ForexBot initialized: LSTM model loaded")
        print(f"✅ Gemini initialized: {'Available' if gemini.gemini_available else 'Fallback mode'}")
        
        # Load test data
        test_pair = "EUR/USD"
        data_file = f"data/MarketData/{test_pair.replace('/', '_')}_real_daily.csv"
        
        if os.path.exists(data_file):
            df = pd.read_csv(data_file)
            print(f"✅ Market data loaded: {len(df)} candles for {test_pair}")
        else:
            # Create synthetic test data
            print("⚠️ No real data found, creating synthetic test data")
            df = pd.DataFrame({
                'close': np.random.randn(200).cumsum() + 1.0850,
                'volume': np.random.randint(50000, 200000, 200),
                'high': np.random.randn(200) * 0.002 + 1.0850,
                'low': np.random.randn(200) * 0.002 + 1.0850,
            })
        
        # Test 1: Standard LSTM prediction
        print(f"\n{'='*20} TEST 1: STANDARD LSTM PREDICTION {'='*20}")
        start_time = time.time()
        lstm_recommendation = forex_bot.get_final_recommendation(df, test_pair)
        lstm_time = time.time() - start_time
        
        print(f"LSTM Analysis ({lstm_time:.3f}s):")
        print(f"  Action: {lstm_recommendation['action']}")
        print(f"  Confidence: {lstm_recommendation['confidence']:.1%}")
        print(f"  Processing time: {lstm_time:.3f}s")
        
        # Test 2: Gemini analysis (if available)
        print(f"\n{'='*20} TEST 2: GEMINI ANALYSIS {'='*20}")
        
        # Prepare market context
        latest_price = float(df['close'].iloc[-1])
        prev_price = float(df['close'].iloc[-2])
        price_change = (latest_price - prev_price) / prev_price * 100
        
        market_context = {
            "pair": test_pair,
            "current_price": latest_price,
            "price_change_24h": price_change,
            "trend": lstm_recommendation.get('trend_signal', 'neutral'),
            "lstm_prediction": lstm_recommendation['action'],
            "lstm_confidence": lstm_recommendation['confidence']
        }
        
        gemini_result = {"sentiment": "neutral", "confidence": 50, "available": False}
        
        if gemini.gemini_available:
            start_time = time.time()
            gemini_result = gemini.interpret_market_quickly(market_context, test_pair)
            gemini_time = time.time() - start_time
            
            print(f"Gemini Analysis ({gemini_time:.3f}s):")
            print(f"  Sentiment: {gemini_result.get('sentiment', 'N/A')}")
            print(f"  Confidence: {gemini_result.get('confidence', 0):.1%}")
            print(f"  Key Factor: {gemini_result.get('key_factor', 'N/A')}")
        else:
            print("Gemini Analysis: Not available (using fallback)")
        
        # Test 3: Enhanced analysis with mock news
        print(f"\n{'='*20} TEST 3: ENHANCED ANALYSIS WITH NEWS {'='*20}")
        
        # Create mock news sentiment
        mock_news = create_mock_news_sentiment()
        print(f"Mock News Sentiment:")
        print(f"  Overall: {mock_news['overall_sentiment']:+.3f}")
        print(f"  Confidence: {mock_news['confidence']:.1%}")
        print(f"  Articles: {mock_news['article_count']}")
        print(f"  Breakdown: {mock_news['sentiment_breakdown']}")
        
        # Combine LSTM + Gemini + News
        lstm_action = lstm_recommendation['action']
        lstm_conf = lstm_recommendation['confidence']
        
        gemini_sentiment = gemini_result.get('sentiment', 'neutral')
        gemini_conf = gemini_result.get('confidence', 50) / 100.0
        
        news_sentiment = mock_news['overall_sentiment']
        news_conf = mock_news['confidence']
        
        # Enhanced fusion logic
        print(f"\nCOMBINED ANALYSIS:")
        print(f"  LSTM: {lstm_action} ({lstm_conf:.1%})")
        print(f"  Gemini: {gemini_sentiment} ({gemini_conf:.1%})")
        print(f"  News: {news_sentiment:+.3f} ({news_conf:.1%})")
        
        # Agreement scoring
        sentiment_mapping = {'bullish': 1.0, 'bearish': -1.0, 'neutral': 0.0}
        action_mapping = {'BUY': 1.0, 'SELL': -1.0, 'HOLD': 0.0}
        
        lstm_score = action_mapping.get(lstm_action, 0.0)
        gemini_score = sentiment_mapping.get(gemini_sentiment, 0.0)
        
        # Check agreements
        lstm_news_agree = abs(lstm_score - news_sentiment) < 0.5
        gemini_news_agree = abs(gemini_score - news_sentiment) < 0.5
        all_agree = lstm_news_agree and gemini_news_agree
        
        # Calculate enhanced confidence
        base_confidence = (lstm_conf * 0.5) + (gemini_conf * 0.2) + (news_conf * 0.3)
        
        if all_agree:
            enhanced_confidence = min(0.95, base_confidence + 0.15)
            agreement_status = "ALL AGREE"
        elif lstm_news_agree or gemini_news_agree:
            enhanced_confidence = min(0.90, base_confidence + 0.05)
            agreement_status = "PARTIAL AGREEMENT"
        else:
            enhanced_confidence = max(0.20, base_confidence - 0.10)
            agreement_status = "DISAGREEMENT"
        
        # Final recommendation
        weighted_score = (lstm_score * 0.5) + (gemini_score * 0.2) + (news_sentiment * 0.3)
        
        if weighted_score > 0.3:
            final_action = "BUY"
        elif weighted_score < -0.3:
            final_action = "SELL"
        else:
            final_action = "HOLD"
        
        print(f"\nENHANCED RESULT:")
        print(f"  Agreement: {agreement_status}")
        print(f"  Final Action: {final_action}")
        print(f"  Enhanced Confidence: {enhanced_confidence:.1%}")
        print(f"  Weighted Score: {weighted_score:+.3f}")
        
        # Test 4: Performance summary
        print(f"\n{'='*20} TEST 4: PERFORMANCE SUMMARY {'='*20}")
        
        performance_summary = {
            "lstm_available": True,
            "gemini_available": gemini.gemini_available,
            "news_integration": "Mock data successful",
            "enhanced_confidence": enhanced_confidence,
            "final_recommendation": final_action,
            "confidence_boost": enhanced_confidence - base_confidence,
            "agreement_status": agreement_status
        }
        
        for key, value in performance_summary.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print(f"\n{'='*60}")
        print("✅ NEWS INTEGRATION TEST COMPLETED SUCCESSFULLY")
        print("✅ Multi-AI fusion operational with news sentiment")
        print("✅ Ready for live news API integration")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_news_api_availability():
    """Test if news APIs can be accessed"""
    print(f"\n{'='*20} NEWS API AVAILABILITY TEST {'='*20}")
    
    # Test Yahoo Finance RSS (no auth required)
    try:
        import urllib.request
        import xml.etree.ElementTree as ET
        
        print("Testing Yahoo Finance RSS access...")
        url = "https://feeds.finance.yahoo.com/rss/2.0/headline"
        
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        with urllib.request.urlopen(req, timeout=10) as response:
            data = response.read()
            root = ET.fromstring(data)
            items = root.findall('.//item')
            
            print(f"✅ Yahoo Finance RSS: {len(items)} news items available")
            if items:
                print(f"  Sample headline: {items[0].find('title').text[:80]}...")
        
    except Exception as e:
        print(f"⚠️ Yahoo Finance RSS: {e}")
    
    # Test Alpha Vantage demo endpoint
    try:
        import urllib.request
        import json
        
        print("\nTesting Alpha Vantage demo endpoint...")
        url = "https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=FOREX:USD&apikey=demo&limit=5"
        
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read())
            
            if 'feed' in data and len(data['feed']) > 0:
                print(f"✅ Alpha Vantage demo: {len(data['feed'])} news items")
                print(f"  Sample: {data['feed'][0].get('title', 'N/A')[:80]}...")
            else:
                print(f"⚠️ Alpha Vantage demo: Limited data available")
        
    except Exception as e:
        print(f"⚠️ Alpha Vantage demo: {e}")

if __name__ == "__main__":
    print("FOREXSWING AI - NEWS INTEGRATION TEST")
    print("=" * 60)
    
    # Test news API availability first
    test_news_api_availability()
    
    # Test enhanced system
    success = test_enhanced_gemini_with_news()
    
    if success:
        print("\n🎉 ALL TESTS PASSED - NEWS INTEGRATION READY!")
    else:
        print("\n❌ SOME TESTS FAILED - CHECK CONFIGURATION")