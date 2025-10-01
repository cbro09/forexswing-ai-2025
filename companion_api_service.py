#!/usr/bin/env python3
"""
ForexSwing AI - Companion API Service
Lightweight API service for providing AI analysis to companion interfaces
"""

import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import sys
import os
import pandas as pd
import numpy as np

# Add paths for imports
sys.path.append('.')
sys.path.append('src')

# Import real AI components
from ForexBot import ForexBot
from src.integrations.live_gemini_api import LiveGeminiAnalyzer
from src.integrations.news_sentiment_analyzer import MultiSourceNewsAnalyzer

# Create wrapper class for news analyzer
class NewsSentimentAnalyzer:
    def __init__(self, alpha_vantage_key: str):
        self.news_analyzer = MultiSourceNewsAnalyzer(alpha_vantage_key)
    
    def get_sentiment_for_pair(self, pair: str) -> Dict:
        try:
            sentiment = self.news_analyzer.analyze_forex_sentiment(pair, hours_back=12)
            return {
                'sentiment_score': sentiment.overall_sentiment,
                'confidence': sentiment.confidence,
                'articles_count': sentiment.article_count,
                'latest_headlines': [article.title for article in sentiment.top_articles[:3]]
            }
        except Exception as e:
            return {
                'sentiment_score': 0.0,
                'confidence': 0.3,
                'articles_count': 0,
                'latest_headlines': [],
                'error': str(e)
            }

class CompanionAIAnalyzer:
    """
    Real AI analyzer for companion interfaces
    Provides real-time forex analysis using trained models and live data
    """
    
    def __init__(self, alpha_vantage_key: str = "OXGW647WZO8XTKA1"):
        self.alpha_vantage_key = alpha_vantage_key
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
        # Initialize real AI components
        print("Initializing real AI models...")
        try:
            self.forex_bot = ForexBot()
            print("✅ ForexBot (LSTM) loaded successfully")
        except Exception as e:
            print(f"❌ ForexBot failed to load: {e}")
            self.forex_bot = None
        
        try:
            self.gemini_analyzer = LiveGeminiAnalyzer()
            print("✅ Gemini API analyzer loaded successfully")
        except Exception as e:
            print(f"❌ Gemini analyzer failed to load: {e}")
            self.gemini_analyzer = None
        
        try:
            self.news_analyzer = NewsSentimentAnalyzer(alpha_vantage_key)
            print("✅ News sentiment analyzer loaded successfully")
        except Exception as e:
            print(f"❌ News analyzer failed to load: {e}")
            self.news_analyzer = None
        
        # Supported currency pairs
        self.supported_pairs = [
            'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 
            'AUD/USD', 'USD/CAD', 'NZD/USD'
        ]
        
        print("🤖 CompanionAIAnalyzer initialized with REAL AI models")
        print(f"📊 Supported pairs: {len(self.supported_pairs)}")
        print(f"🔑 Alpha Vantage key: {'Configured' if alpha_vantage_key else 'Missing'}")
    
    def get_quick_analysis(self, pair: str) -> Dict:
        """
        Get quick AI analysis for a currency pair
        Uses cached data and lightweight analysis
        """
        # Check cache first
        cache_key = f"{pair}_{int(time.time() // self.cache_duration)}"
        if cache_key in self.cache:
            analysis = self.cache[cache_key].copy()
            analysis['cached'] = True
            return analysis
        
        # Generate fresh analysis
        analysis = self._generate_analysis(pair)
        
        # Cache result
        self.cache[cache_key] = analysis
        
        # Clean old cache entries
        self._clean_cache()
        
        analysis['cached'] = False
        return analysis
    
    def _generate_analysis(self, pair: str) -> Dict:
        """Generate REAL AI analysis for currency pair"""
        start_time = time.time()
        
        try:
            # Step 1: Get real market data for the pair
            market_data = self._get_real_market_data(pair)
            if market_data is None:
                return self._get_fallback_analysis(pair, "No market data available")
            
            # Step 2: Run real LSTM analysis
            lstm_analysis = self._get_real_lstm_analysis(pair, market_data)
            
            # Step 3: Get real Gemini analysis
            gemini_analysis = self._get_real_gemini_analysis(pair, market_data)
            
            # Step 4: Get real news sentiment
            news_sentiment = self._get_real_news_sentiment(pair)
            
            # Step 5: Combine all real analyses
            final_analysis = self._combine_real_analyses(
                lstm_analysis, gemini_analysis, news_sentiment
            )
            
            # Step 6: Add metadata
            processing_time = time.time() - start_time
            final_analysis.update({
                'pair': pair,
                'timestamp': datetime.now().isoformat(),
                'processing_time': f"{processing_time:.3f}s",
                'data_sources': ['Real_LSTM', 'Live_Gemini', 'Live_News', 'Market_Data'],
                'models_used': {
                    'lstm': 'active' if lstm_analysis.get('success') else 'fallback',
                    'gemini': 'active' if gemini_analysis.get('success') else 'fallback',
                    'news': 'active' if news_sentiment.get('success') else 'fallback'
                }
            })
            
            return final_analysis
            
        except Exception as e:
            processing_time = time.time() - start_time
            return self._get_fallback_analysis(pair, f"Analysis error: {str(e)}", processing_time)
    
    def _get_real_market_data(self, pair: str) -> Optional[pd.DataFrame]:
        """Get real market data for the currency pair"""
        try:
            # Convert EUR/USD to EUR_USD format for file lookup
            file_pair = pair.replace('/', '_')
            file_path = f"data/MarketData/{file_pair}_real_daily.csv"
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # Return last 120 days for analysis
                return df.tail(120)
            else:
                print(f"⚠️ No market data file found for {pair} at {file_path}")
                return None
                
        except Exception as e:
            print(f"❌ Error loading market data for {pair}: {e}")
            return None
    
    def _get_real_news_sentiment(self, pair: str) -> Dict:
        """Get REAL news sentiment using news analyzer"""
        try:
            if self.news_analyzer is None:
                return {'success': False, 'sentiment': 0.0, 'confidence': 0.3, 'error': 'News analyzer not available'}
            
            # Get live news sentiment analysis
            analysis = self.news_analyzer.get_sentiment_for_pair(pair)
            
            return {
                'success': True,
                'sentiment': analysis.get('sentiment_score', 0.0),
                'confidence': analysis.get('confidence', 0.3),
                'articles_analyzed': analysis.get('articles_count', 0),
                'latest_headlines': analysis.get('latest_headlines', [])[:3],  # Top 3 headlines
                'source': 'Alpha_Vantage_News_API'
            }
            
        except Exception as e:
            print(f"❌ News sentiment error for {pair}: {e}")
            return {'success': False, 'sentiment': 0.0, 'confidence': 0.3, 'error': str(e)}
    
    def _get_real_lstm_analysis(self, pair: str, market_data: pd.DataFrame) -> Dict:
        """Get REAL LSTM analysis using the trained model"""
        try:
            if self.forex_bot is None:
                return {'success': False, 'action': 'HOLD', 'confidence': 0.3, 'error': 'LSTM model not loaded'}
            
            # Use real ForexBot to get analysis
            recommendation = self.forex_bot.get_final_recommendation(market_data, pair)
            
            return {
                'success': True,
                'action': recommendation['action'],
                'confidence': recommendation['confidence'],
                'trend_signal': recommendation.get('trend_signal', 'neutral'),
                'trend_strength': recommendation.get('trend_strength', '50%'),
                'processing_time': recommendation.get('processing_time', 'N/A'),
                'source': 'Real_LSTM_55.2%_Accuracy'
            }
            
        except Exception as e:
            print(f"❌ LSTM analysis error for {pair}: {e}")
            return {'success': False, 'action': 'HOLD', 'confidence': 0.3, 'error': str(e)}
    
    def _get_real_gemini_analysis(self, pair: str, market_data: pd.DataFrame) -> Dict:
        """Get REAL Gemini analysis using live API"""
        try:
            if self.gemini_analyzer is None:
                return {'success': False, 'sentiment': 'neutral', 'confidence': 0.3, 'error': 'Gemini API not available'}
            
            # Prepare market context for Gemini
            recent_prices = market_data['close'].tail(5).tolist()
            current_price = recent_prices[-1] if recent_prices else 0
            price_change = ((recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100) if len(recent_prices) >= 2 else 0
            
            market_context = {
                'pair': pair,
                'current_price': current_price,
                'recent_change': f"{price_change:+.2f}%",
                'recent_prices': recent_prices
            }
            
            # Get live Gemini analysis
            analysis = self.gemini_analyzer.analyze_forex_market(market_context, pair)
            
            return {
                'success': True,
                'sentiment': analysis.get('sentiment', 'neutral'),
                'confidence': analysis.get('confidence', 0.5),
                'reasoning': analysis.get('reasoning', 'Live Gemini analysis'),
                'source': 'Live_Gemini_API'
            }
            
        except Exception as e:
            print(f"❌ Gemini analysis error for {pair}: {e}")
            return {'success': False, 'sentiment': 'neutral', 'confidence': 0.3, 'error': str(e)}
    
    def _combine_real_analyses(self, lstm: Dict, gemini: Dict, news: Dict) -> Dict:
        """Combine all REAL AI analyses into final recommendation"""
        
        # Extract actual values from real analyses
        lstm_action = lstm.get('action', 'HOLD')
        lstm_confidence = lstm.get('confidence', 0.5)
        lstm_success = lstm.get('success', False)
        
        gemini_sentiment = gemini.get('sentiment', 'neutral')
        gemini_confidence = gemini.get('confidence', 0.5)
        gemini_success = gemini.get('success', False)
        
        news_sentiment = news.get('sentiment', 0.0)
        news_confidence = news.get('confidence', 0.3)
        news_success = news.get('success', False)
        
        # Convert to normalized scores for combination
        lstm_score = 0.6 if lstm_action == 'BUY' else -0.6 if lstm_action == 'SELL' else 0.0
        gemini_score = 0.6 if gemini_sentiment in ['bullish', 'positive'] else -0.6 if gemini_sentiment in ['bearish', 'negative'] else 0.0
        news_score = news_sentiment  # Already normalized
        
        # Dynamic weights based on success
        weights = []
        weights.append(0.6 if lstm_success else 0.2)  # LSTM gets higher weight if successful
        weights.append(0.3 if gemini_success else 0.1)  # Gemini
        weights.append(0.1 if news_success else 0.05)  # News
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]
        
        # Weighted combination
        combined_score = (lstm_score * weights[0] + 
                         gemini_score * weights[1] + 
                         news_score * weights[2])
        
        # Final action determination
        if combined_score > 0.3:
            final_action = 'BUY'
        elif combined_score < -0.3:
            final_action = 'SELL'
        else:
            final_action = 'HOLD'
        
        # Agreement analysis
        gemini_action = 'BUY' if gemini_sentiment in ['bullish', 'positive'] else 'SELL' if gemini_sentiment in ['bearish', 'negative'] else 'HOLD'
        news_action = 'BUY' if news_score > 0.1 else 'SELL' if news_score < -0.1 else 'HOLD'
        
        # Count agreements
        agreements = 0
        if lstm_action == final_action:
            agreements += 1
        if gemini_action == final_action:
            agreements += 1
        if news_action == final_action:
            agreements += 1
        
        # Base confidence (weighted by success)
        base_confidence = (lstm_confidence * weights[0] + 
                          gemini_confidence * weights[1] + 
                          news_confidence * weights[2])
        
        # Agreement bonus (higher for real data)
        agreement_bonus = agreements * 0.08 if agreements >= 2 else 0
        final_confidence = min(0.95, base_confidence + agreement_bonus)
        
        return {
            'action': final_action,
            'confidence': final_confidence,
            'score': combined_score,
            'agreements': agreements,
            'components': {
                'lstm': f"{lstm_action} {lstm_confidence:.0%}" + (" ✅" if lstm_success else " ❌"),
                'gemini': f"{gemini_sentiment} {gemini_confidence:.0%}" + (" ✅" if gemini_success else " ❌"),
                'news': f"{news_score:+.2f} ({news.get('articles_analyzed', 0)} articles)" + (" ✅" if news_success else " ❌")
            },
            'risk_level': 'HIGH' if final_confidence < 0.4 else 'LOW' if final_confidence > 0.75 else 'MEDIUM',
            'data_quality': f"{sum([lstm_success, gemini_success, news_success])}/3 models active"
        }
    
    def _get_fallback_analysis(self, pair: str, error: str, processing_time: float = 0.0) -> Dict:
        """Provide fallback analysis when real systems fail"""
        return {
            'pair': pair,
            'action': 'HOLD',
            'confidence': 0.25,
            'score': 0.0,
            'agreements': 0,
            'risk_level': 'HIGH',
            'components': {
                'lstm': 'UNAVAILABLE ❌',
                'gemini': 'UNAVAILABLE ❌', 
                'news': 'UNAVAILABLE ❌'
            },
            'data_quality': '0/3 models active',
            'error': error,
            'timestamp': datetime.now().isoformat(),
            'processing_time': f"{processing_time:.3f}s",
            'fallback': True,
            'data_sources': ['Fallback_Only']
        }
    
    def _clean_cache(self):
        """Remove old cache entries"""
        current_time = int(time.time() // self.cache_duration)
        keys_to_remove = []
        
        for key in self.cache:
            if key.endswith(f"_{current_time}") or key.endswith(f"_{current_time-1}"):
                continue  # Keep current and previous cache
            keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.cache[key]
    
    def get_supported_pairs(self) -> List[str]:
        """Get list of supported currency pairs"""
        return self.supported_pairs.copy()
    
    def get_system_status(self) -> Dict:
        """Get system status information"""
        return {
            'status': 'operational',
            'supported_pairs': len(self.supported_pairs),
            'cache_entries': len(self.cache),
            'last_update': datetime.now().isoformat(),
            'alpha_vantage_key': 'configured' if self.alpha_vantage_key else 'missing'
        }

class CompanionAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the companion API"""
    
    def __init__(self, *args, analyzer=None, **kwargs):
        self.analyzer = analyzer
        super().__init__(*args, **kwargs)
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.send_header('Access-Control-Max-Age', '86400')
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        try:
            # Parse URL and query parameters
            path = self.path.split('?')[0]
            query_params = {}
            
            if '?' in self.path:
                query_string = self.path.split('?')[1]
                for param in query_string.split('&'):
                    if '=' in param:
                        key, value = param.split('=', 1)
                        query_params[urllib.parse.unquote(key)] = urllib.parse.unquote(value)
            
            # Route requests
            if path == '/api/analyze':
                self._handle_analyze(query_params)
            elif path == '/api/pairs':
                self._handle_pairs()
            elif path == '/api/status':
                self._handle_status()
            elif path == '/':
                self._handle_root()
            else:
                self._send_error(404, 'Endpoint not found')
                
        except Exception as e:
            self._send_error(500, f'Internal error: {str(e)}')
    
    def _handle_analyze(self, params: Dict):
        """Handle analysis requests"""
        pair = params.get('pair', '').upper()
        
        if not pair:
            self._send_error(400, 'Missing pair parameter')
            return
        
        if pair not in self.analyzer.get_supported_pairs():
            self._send_error(400, f'Unsupported pair: {pair}')
            return
        
        # Get analysis
        start_time = time.time()
        analysis = self.analyzer.get_quick_analysis(pair)
        analysis['processing_time'] = f"{time.time() - start_time:.3f}s"
        
        self._send_json_response(analysis)
    
    def _handle_pairs(self):
        """Handle supported pairs request"""
        pairs = self.analyzer.get_supported_pairs()
        response = {
            'supported_pairs': pairs,
            'count': len(pairs),
            'timestamp': datetime.now().isoformat()
        }
        self._send_json_response(response)
    
    def _handle_status(self):
        """Handle status request"""
        status = self.analyzer.get_system_status()
        self._send_json_response(status)
    
    def _handle_root(self):
        """Handle root request with API documentation"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ForexSwing AI - Companion API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { color: #2c3e50; }
                .endpoint { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .method { color: #27ae60; font-weight: bold; }
                .url { color: #3498db; font-family: monospace; }
                .example { background: #2c3e50; color: white; padding: 10px; border-radius: 5px; font-family: monospace; }
                .status { color: #27ae60; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 ForexSwing AI - Companion API</h1>
                <p class="status">Status: Operational ✅</p>
                
                <h2>Available Endpoints:</h2>
                
                <div class="endpoint">
                    <span class="method">GET</span> <span class="url">/api/analyze?pair=EUR/USD</span>
                    <p>Get AI analysis for a currency pair</p>
                    <div class="example">
                        Example: curl "http://localhost:8080/api/analyze?pair=EUR/USD"
                    </div>
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> <span class="url">/api/pairs</span>
                    <p>Get list of supported currency pairs</p>
                    <div class="example">
                        Example: curl "http://localhost:8080/api/pairs"
                    </div>
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> <span class="url">/api/status</span>
                    <p>Get system status and health information</p>
                    <div class="example">
                        Example: curl "http://localhost:8080/api/status"
                    </div>
                </div>
                
                <h2>Example Response:</h2>
                <div class="example">
{
  "pair": "EUR/USD",
  "action": "BUY",
  "confidence": 0.67,
  "risk_level": "MEDIUM",
  "components": {
    "lstm": "BUY 62%",
    "gemini": "bullish 60%", 
    "news": "+0.20 (5 articles)"
  },
  "timestamp": "2025-01-19T..."
}
                </div>
                
                <p>🚀 Ready for companion interface integration!</p>
            </div>
        </body>
        </html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def _send_json_response(self, data: Dict):
        """Send JSON response with CORS headers"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.send_header('Access-Control-Max-Age', '86400')
        self.end_headers()
        
        json_data = json.dumps(data, indent=2)
        self.wfile.write(json_data.encode())
    
    def _send_error(self, code: int, message: str):
        """Send error response"""
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        error_data = {
            'error': message,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }
        
        json_data = json.dumps(error_data, indent=2)
        self.wfile.write(json_data.encode())
    
    def log_message(self, format, *args):
        """Override to reduce log noise"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {format % args}")

def create_handler_class(analyzer):
    """Create handler class with analyzer instance"""
    def handler(*args, **kwargs):
        CompanionAPIHandler(*args, analyzer=analyzer, **kwargs)
    return handler

def run_companion_api_server(port: int = 8080):
    """Run the companion API server"""
    print("🤖 FOREXSWING AI - COMPANION API SERVICE")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = CompanionAIAnalyzer()
    
    # Create HTTP server
    handler_class = create_handler_class(analyzer)
    
    with HTTPServer(('localhost', port), handler_class) as server:
        print(f"🚀 Companion API server starting on http://localhost:{port}")
        print(f"📊 Supporting {len(analyzer.get_supported_pairs())} currency pairs")
        print(f"🔗 Ready for companion interface integration")
        print(f"📖 API documentation: http://localhost:{port}")
        print("=" * 60)
        print("Press Ctrl+C to stop the server")
        
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user")
        except Exception as e:
            print(f"❌ Server error: {e}")

if __name__ == "__main__":
    import sys
    
    port = 8082
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Invalid port number. Using default 8082.")
    
    run_companion_api_server(port)