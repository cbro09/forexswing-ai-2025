#!/usr/bin/env python3
"""
Live Gemini API Integration for ForexSwing AI
Uses Google Gemini API directly instead of CLI simulation
"""

import google.generativeai as genai
import json
import time
from datetime import datetime
from typing import Dict, Optional
import hashlib

class LiveGeminiAnalyzer:
    """
    Live Google Gemini API integration for forex analysis
    """
    
    def __init__(self, api_key: str = "AIzaSyBcSKWOjFghU3UItDVjNL62tWpGnn7I-bQ"):
        self.api_key = api_key
        self.model = None
        self.available = False
        
        # Response cache
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
        # Rate limiting
        self.last_call_time = 0
        self.min_call_interval = 4.0  # 15 requests/minute = 4 second intervals
        
        self._initialize_gemini()
    
    def _initialize_gemini(self):
        """Initialize Gemini API connection"""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Test connection
            test_response = self.model.generate_content("Test connection")
            if test_response:
                self.available = True
                print(f"✅ Live Gemini API connected successfully")
            else:
                raise Exception("No response from Gemini")
                
        except Exception as e:
            print(f"⚠️ Gemini API initialization failed: {e}")
            self.available = False
    
    def _generate_cache_key(self, prompt: str) -> str:
        """Generate cache key for prompt"""
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached response is valid"""
        if cache_key not in self.cache:
            return False
        
        cached_time = self.cache[cache_key].get('timestamp', 0)
        return time.time() - cached_time < self.cache_duration
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        
        if time_since_last < self.min_call_interval:
            sleep_time = self.min_call_interval - time_since_last
            print(f"🕒 Rate limiting: waiting {sleep_time:.1f}s")
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    def analyze_forex_market(self, market_data: Dict, pair: str) -> Dict:
        """
        Analyze forex market using live Gemini API
        """
        if not self.available:
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'reasoning': 'Gemini API not available',
                'source': 'fallback'
            }
        
        # Create analysis prompt
        prompt = self._create_forex_prompt(market_data, pair)
        
        # Check cache first
        cache_key = self._generate_cache_key(prompt)
        if self._is_cache_valid(cache_key):
            cached_result = self.cache[cache_key]['result']
            cached_result['cached'] = True
            return cached_result
        
        # Rate limiting
        self._rate_limit()
        
        try:
            start_time = time.time()
            
            # Call Gemini API
            response = self.model.generate_content(prompt)
            
            end_time = time.time()
            
            if response and response.text:
                # Parse response
                analysis = self._parse_gemini_response(response.text, pair)
                analysis['processing_time'] = f"{end_time - start_time:.2f}s"
                analysis['cached'] = False
                analysis['source'] = 'live_gemini_api'
                
                # Cache result
                self.cache[cache_key] = {
                    'result': analysis,
                    'timestamp': time.time()
                }
                
                return analysis
            else:
                raise Exception("Empty response from Gemini")
                
        except Exception as e:
            print(f"❌ Gemini API call failed: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'reasoning': f'API error: {str(e)}',
                'source': 'error_fallback'
            }
    
    def _create_forex_prompt(self, market_data: Dict, pair: str) -> str:
        """Create optimized prompt for Gemini forex analysis"""
        
        # Extract key data
        current_price = market_data.get('current_price', 0)
        price_change = market_data.get('price_change_24h', 0)
        trend = market_data.get('trend', 'neutral')
        volatility = market_data.get('volatility', 0)
        lstm_prediction = market_data.get('lstm_prediction', 'HOLD')
        lstm_confidence = market_data.get('lstm_confidence', 0.5)
        
        prompt = f"""You are a professional forex analyst. Analyze {pair} and provide sentiment.

MARKET DATA:
- Current Price: {current_price:.5f}
- 24h Change: {price_change:+.2f}%
- Trend: {trend}
- Volatility: {volatility:.4f}
- ML Model says: {lstm_prediction} ({lstm_confidence:.1%} confidence)

INSTRUCTIONS:
1. Consider current economic conditions for both currencies
2. Analyze the technical indicators provided
3. Factor in the ML model prediction but use your own judgment
4. Provide sentiment: bullish/bearish/neutral
5. Give confidence level 0-100
6. Explain your reasoning briefly

RESPOND IN JSON FORMAT:
{{
  "sentiment": "bullish/bearish/neutral",
  "confidence": 75,
  "reasoning": "Brief explanation of your analysis",
  "key_factors": ["factor1", "factor2", "factor3"],
  "agreement_with_ml": true/false
}}"""
        
        return prompt
    
    def _parse_gemini_response(self, response_text: str, pair: str) -> Dict:
        """Parse Gemini response into structured format"""
        try:
            # Try to extract JSON from response
            response_text = response_text.strip()
            
            # Find JSON block
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_text = response_text[start_idx:end_idx]
                parsed = json.loads(json_text)
                
                # Validate and normalize
                sentiment = parsed.get('sentiment', 'neutral').lower()
                confidence = max(0, min(100, parsed.get('confidence', 50))) / 100.0
                reasoning = parsed.get('reasoning', 'Analysis provided')
                key_factors = parsed.get('key_factors', [])
                agreement_with_ml = parsed.get('agreement_with_ml', False)
                
                return {
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'reasoning': reasoning,
                    'key_factors': key_factors,
                    'agreement_with_ml': agreement_with_ml,
                    'pair': pair,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            print(f"⚠️ Could not parse Gemini response: {e}")
            print(f"Raw response: {response_text[:200]}...")
            
            # Fallback parsing
            sentiment = 'neutral'
            confidence = 0.5
            
            # Simple keyword detection
            text_lower = response_text.lower()
            if any(word in text_lower for word in ['bullish', 'buy', 'positive', 'up']):
                sentiment = 'bullish'
                confidence = 0.6
            elif any(word in text_lower for word in ['bearish', 'sell', 'negative', 'down']):
                sentiment = 'bearish'
                confidence = 0.6
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'reasoning': 'Parsed from unstructured response',
                'key_factors': [],
                'agreement_with_ml': False,
                'pair': pair,
                'timestamp': datetime.now().isoformat(),
                'parse_error': str(e)
            }
    
    def quick_sentiment_check(self, pair: str, price_change: float) -> Dict:
        """Quick sentiment check for currency pair"""
        if not self.available:
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'reasoning': 'Gemini not available'
            }
        
        # Simple prompt for quick analysis
        prompt = f"""Quick forex sentiment for {pair}:
Price change: {price_change:+.2f}%

Respond with JSON: {{"sentiment": "bullish/bearish/neutral", "confidence": 60, "reason": "brief"}}"""
        
        cache_key = self._generate_cache_key(prompt)
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['result']
        
        self._rate_limit()
        
        try:
            response = self.model.generate_content(prompt)
            if response and response.text:
                return self._parse_gemini_response(response.text, pair)
        except Exception as e:
            print(f"Quick sentiment failed: {e}")
        
        return {
            'sentiment': 'neutral',
            'confidence': 0.5,
            'reasoning': 'Quick analysis failed'
        }
    
    def get_performance_stats(self) -> Dict:
        """Get Gemini API performance statistics"""
        return {
            'api_available': self.available,
            'api_key_configured': bool(self.api_key),
            'cache_entries': len(self.cache),
            'rate_limit_interval': self.min_call_interval,
            'model': 'gemini-1.5-flash' if self.available else 'unavailable'
        }

def test_live_gemini():
    """Test live Gemini API integration"""
    print("🧪 TESTING LIVE GEMINI API INTEGRATION")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = LiveGeminiAnalyzer()
    
    if not analyzer.available:
        print("❌ Gemini API not available - check API key")
        return False
    
    # Test market analysis
    test_market_data = {
        'current_price': 1.0875,
        'price_change_24h': 0.35,
        'trend': 'bullish',
        'volatility': 0.012,
        'lstm_prediction': 'BUY',
        'lstm_confidence': 0.62
    }
    
    print(f"\n📊 Testing market analysis for EUR/USD...")
    result = analyzer.analyze_forex_market(test_market_data, 'EUR/USD')
    
    print(f"Results:")
    print(f"  Sentiment: {result.get('sentiment', 'N/A')}")
    print(f"  Confidence: {result.get('confidence', 0):.1%}")
    print(f"  Reasoning: {result.get('reasoning', 'N/A')}")
    print(f"  Processing time: {result.get('processing_time', 'N/A')}")
    print(f"  Source: {result.get('source', 'N/A')}")
    
    # Test quick sentiment
    print(f"\n⚡ Testing quick sentiment check...")
    quick_result = analyzer.quick_sentiment_check('GBP/USD', -0.25)
    print(f"Quick sentiment: {quick_result}")
    
    # Performance stats
    stats = analyzer.get_performance_stats()
    print(f"\n📈 Performance stats: {stats}")
    
    print(f"\n✅ Live Gemini API test complete!")
    return True

if __name__ == "__main__":
    test_live_gemini()