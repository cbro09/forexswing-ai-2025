#!/usr/bin/env python3
"""
Enhanced Gemini Integration with Live News Sentiment
Combines optimized Gemini CLI with multi-source news analysis for superior forex predictions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.integrations.optimized_gemini import OptimizedGeminiInterpreter
from src.integrations.news_sentiment_analyzer import MultiSourceNewsAnalyzer, ForexNewsSentiment
import json
import time
from datetime import datetime
from typing import Dict, Optional, Tuple
import pandas as pd

class EnhancedNewsGeminiInterpreter:
    """
    Enhanced Gemini interpreter that combines:
    1. Live news sentiment analysis from multiple sources
    2. Optimized Gemini CLI interpretation
    3. Intelligent fusion of technical + fundamental analysis
    """
    
    def __init__(self, 
                 cache_size: int = 100, 
                 cache_duration_minutes: int = 30,
                 alpha_vantage_key: str = None,
                 news_api_key: str = None):
        
        # Initialize base Gemini interpreter
        self.gemini = OptimizedGeminiInterpreter(cache_size, cache_duration_minutes)
        
        # Initialize news analyzer
        self.news_analyzer = MultiSourceNewsAnalyzer(
            alpha_vantage_key=alpha_vantage_key,
            news_api_key=news_api_key
        )
        
        # Configuration
        self.news_weight = 0.4  # Weight of news sentiment in final decision
        self.enable_news_analysis = True
        self.news_cache_minutes = 15  # Cache news for 15 minutes
        
        print(f"EnhancedNewsGeminiInterpreter initialized:")
        print(f"  - Gemini CLI: {'Available' if self.gemini.gemini_available else 'Not available'}")
        print(f"  - News Sources: Alpha Vantage, Yahoo Finance, NewsAPI")
        print(f"  - News Weight: {self.news_weight:.1%}")
    
    def interpret_market_with_news(self, price_data: Dict, pair: str, 
                                  include_news: bool = True) -> Dict:
        """
        Advanced market interpretation combining technical analysis and live news sentiment
        """
        start_time = time.time()
        
        print(f"\nENHANCED MARKET ANALYSIS FOR {pair}")
        print("-" * 50)
        
        # 1. Get live news sentiment
        news_sentiment = None
        if include_news and self.enable_news_analysis:
            try:
                print("Analyzing live news sentiment...")
                news_sentiment = self.news_analyzer.analyze_forex_sentiment(pair, hours_back=12)
                print(f"News sentiment: {news_sentiment.overall_sentiment:+.3f} (confidence: {news_sentiment.confidence:.1%})")
            except Exception as e:
                print(f"News analysis failed: {e}")
        
        # 2. Prepare enhanced context for Gemini
        enhanced_context = self._prepare_enhanced_context(price_data, pair, news_sentiment)
        
        # 3. Get Gemini analysis with news context
        gemini_result = self._call_gemini_with_news_context(enhanced_context, pair)
        
        # 4. Combine technical and fundamental analysis
        final_result = self._combine_technical_and_fundamental(
            gemini_result, news_sentiment, price_data
        )
        
        end_time = time.time()
        final_result['total_processing_time'] = f"{end_time - start_time:.2f}s"
        
        print(f"Enhanced analysis complete in {end_time - start_time:.2f}s")
        return final_result
    
    def _prepare_enhanced_context(self, price_data: Dict, pair: str, 
                                 news_sentiment: Optional[ForexNewsSentiment]) -> Dict:
        """Prepare comprehensive context including news for Gemini analysis"""
        
        context = {
            "pair": pair,
            "technical_data": {
                "price": f"{price_data.get('current_price', 0):.5f}",
                "change": f"{price_data.get('price_change_24h', 0):+.2f}%",
                "trend": price_data.get('trend', 'neutral'),
                "volatility": price_data.get('volatility', 0)
            }
        }
        
        # Add news sentiment if available
        if news_sentiment:
            context["news_analysis"] = {
                "sentiment": news_sentiment.overall_sentiment,
                "confidence": news_sentiment.confidence,
                "article_count": news_sentiment.article_count,
                "breakdown": news_sentiment.sentiment_breakdown,
                "top_headlines": [
                    article.title[:80] + "..." if len(article.title) > 80 else article.title
                    for article in news_sentiment.top_articles[:3]
                ]
            }
        
        return context
    
    def _call_gemini_with_news_context(self, context: Dict, pair: str) -> Dict:
        """Call Gemini with enhanced news context"""
        
        if not self.gemini.gemini_available:
            return {
                "sentiment": "neutral",
                "confidence": 50,
                "key_factor": "gemini_unavailable",
                "news_enhanced": False
            }
        
        try:
            # Create comprehensive prompt
            technical = context["technical_data"]
            news = context.get("news_analysis", {})
            
            if news:
                # Enhanced prompt with news
                prompt = f"""Advanced forex analysis for {pair}:

TECHNICAL DATA:
- Price: {technical['price']} ({technical['change']})
- Trend: {technical['trend']}
- Volatility: {technical.get('volatility', 'N/A')}

LIVE NEWS SENTIMENT:
- Overall sentiment: {news['sentiment']:+.3f}
- Confidence: {news['confidence']:.1%}
- Articles analyzed: {news['article_count']}
- Breakdown: Bullish {news['breakdown'].get('bullish', 0):.1%}, Bearish {news['breakdown'].get('bearish', 0):.1%}

TOP HEADLINES:
{chr(10).join(f"- {headline}" for headline in news.get('top_headlines', []))}

Considering both technical indicators AND current news sentiment, provide analysis in JSON:
{{"sentiment": "bullish/bearish/neutral", "confidence": 0-100, "key_factor": "primary driver", "news_impact": "high/medium/low", "trading_recommendation": "buy/sell/hold"}}"""
            else:
                # Standard prompt without news
                prompt = f"""Forex analysis for {pair}:
Price: {technical['price']} ({technical['change']})
Trend: {technical['trend']}

Respond with JSON: {{"sentiment": "bullish/bearish/neutral", "confidence": 0-100, "key_factor": "brief reason"}}"""
            
            response = self.gemini._call_gemini_cached(prompt, json.dumps(context))
            
            if response:
                try:
                    interpretation = json.loads(response)
                    interpretation["timestamp"] = datetime.now().isoformat()
                    interpretation["pair"] = pair
                    interpretation["news_enhanced"] = bool(news)
                    return interpretation
                except json.JSONDecodeError:
                    return {
                        "sentiment": "neutral",
                        "confidence": 50,
                        "key_factor": "parsing_error",
                        "raw_response": response[:100],
                        "timestamp": datetime.now().isoformat(),
                        "pair": pair,
                        "news_enhanced": bool(news)
                    }
            
            return {"error": "No response from Gemini", "news_enhanced": bool(news)}
            
        except Exception as e:
            return {"error": f"Gemini analysis failed: {e}", "news_enhanced": bool(news)}
    
    def _combine_technical_and_fundamental(self, gemini_result: Dict, 
                                          news_sentiment: Optional[ForexNewsSentiment],
                                          price_data: Dict) -> Dict:
        """Intelligently combine technical analysis (Gemini) with fundamental analysis (news)"""
        
        # Extract Gemini sentiment
        gemini_sentiment = gemini_result.get('sentiment', 'neutral')
        gemini_confidence = gemini_result.get('confidence', 50) / 100.0
        
        # Convert sentiments to numeric scores
        sentiment_mapping = {'bullish': 1.0, 'bearish': -1.0, 'neutral': 0.0}
        gemini_score = sentiment_mapping.get(gemini_sentiment, 0.0)
        
        # Start with Gemini analysis
        final_score = gemini_score
        final_confidence = gemini_confidence
        confidence_factors = ["Gemini technical analysis"]
        
        # Incorporate news sentiment if available
        if news_sentiment and news_sentiment.confidence > 0.1:
            news_score = news_sentiment.overall_sentiment
            news_conf = news_sentiment.confidence
            
            # Agreement/disagreement analysis
            agreement = abs(gemini_score - news_score) < 0.5
            
            if agreement:
                # News and technical agree - boost confidence
                final_score = (gemini_score * 0.6) + (news_score * 0.4)
                final_confidence = min(0.95, gemini_confidence + (news_conf * 0.2))
                confidence_factors.append(f"News agreement (sentiment: {news_score:+.2f})")
            else:
                # Disagreement - reduce confidence but still incorporate news
                final_score = (gemini_score * 0.7) + (news_score * 0.3)
                final_confidence = max(0.1, gemini_confidence - 0.1)
                confidence_factors.append(f"News disagreement (sentiment: {news_score:+.2f})")
            
            # High-impact news override
            if news_sentiment.article_count >= 5 and news_conf > 0.7:
                final_score = (gemini_score * 0.4) + (news_score * 0.6)
                confidence_factors.append("High-impact news override")
        
        # Convert back to categorical
        if final_score > 0.3:
            final_sentiment = "bullish"
            action_recommendation = "BUY"
        elif final_score < -0.3:
            final_sentiment = "bearish"
            action_recommendation = "SELL"
        else:
            final_sentiment = "neutral"
            action_recommendation = "HOLD"
        
        # Risk assessment
        price_volatility = price_data.get('volatility', 0)
        risk_level = "low"
        if price_volatility > 0.02:
            risk_level = "high"
        elif price_volatility > 0.01:
            risk_level = "medium"
        
        # Compile final result
        result = {
            "pair": gemini_result.get('pair', 'UNKNOWN'),
            "timestamp": datetime.now().isoformat(),
            
            # Final combined analysis
            "final_sentiment": final_sentiment,
            "final_confidence": final_confidence,
            "final_score": final_score,
            "action_recommendation": action_recommendation,
            "risk_level": risk_level,
            
            # Component analysis
            "gemini_analysis": {
                "sentiment": gemini_sentiment,
                "confidence": gemini_confidence,
                "key_factor": gemini_result.get('key_factor', 'N/A')
            },
            
            # News analysis
            "news_analysis": {
                "available": news_sentiment is not None,
                "sentiment_score": news_sentiment.overall_sentiment if news_sentiment else 0.0,
                "confidence": news_sentiment.confidence if news_sentiment else 0.0,
                "article_count": news_sentiment.article_count if news_sentiment else 0
            } if news_sentiment else {"available": False},
            
            # Metadata
            "confidence_factors": confidence_factors,
            "enhanced_with_news": news_sentiment is not None,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def validate_signal_with_news(self, signal_data: Dict, pair: str) -> Dict:
        """Enhanced signal validation incorporating news sentiment"""
        
        # Get base validation from Gemini
        base_validation = self.gemini.validate_signal_fast(signal_data)
        
        # Get current news sentiment
        try:
            news_sentiment = self.news_analyzer.analyze_forex_sentiment(pair, hours_back=6)
            
            # Adjust validation based on news
            ml_prediction = signal_data.get('ml_prediction', 0.5)
            news_score = news_sentiment.overall_sentiment
            
            # Check for alignment
            signal_direction = "bullish" if ml_prediction > 0.6 else "bearish" if ml_prediction < 0.4 else "neutral"
            news_direction = "bullish" if news_score > 0.2 else "bearish" if news_score < -0.2 else "neutral"
            
            alignment = signal_direction == news_direction
            
            # Enhanced validation
            enhanced_validation = {
                "signal_valid": base_validation.get("valid", True),
                "news_alignment": alignment,
                "news_sentiment": news_score,
                "news_confidence": news_sentiment.confidence,
                "enhanced_confidence": signal_data.get('ml_confidence', 0.5),
                "risk_adjustment": "none"
            }
            
            # Adjust confidence based on news
            base_conf = signal_data.get('ml_confidence', 0.5)
            if alignment and news_sentiment.confidence > 0.5:
                enhanced_validation["enhanced_confidence"] = min(0.95, base_conf * 1.15)
                enhanced_validation["risk_adjustment"] = "reduce"
            elif not alignment and news_sentiment.confidence > 0.6:
                enhanced_validation["enhanced_confidence"] = max(0.1, base_conf * 0.85)
                enhanced_validation["risk_adjustment"] = "increase"
            
            return enhanced_validation
            
        except Exception as e:
            print(f"News validation error: {e}")
            return base_validation
    
    def get_comprehensive_performance_stats(self) -> Dict:
        """Get performance statistics for all components"""
        
        return {
            "enhanced_interpreter": {
                "news_weight": self.news_weight,
                "news_enabled": self.enable_news_analysis
            },
            "gemini_stats": self.gemini.get_performance_stats(),
            "news_analyzer_available": hasattr(self.news_analyzer, 'cache'),
            "total_integrations": 3  # Gemini + News Sources
        }

def test_enhanced_news_gemini():
    """Test enhanced news-enabled Gemini interpreter"""
    print("TESTING ENHANCED NEWS-GEMINI INTEGRATION")
    print("=" * 60)
    
    # Initialize with demo keys (replace with your actual keys)
    interpreter = EnhancedNewsGeminiInterpreter(
        cache_size=50,
        cache_duration_minutes=15,
        alpha_vantage_key="OXGW647WZO8XTKA1",  # Live Alpha Vantage key
        news_api_key=None  # Replace with your NewsAPI key if available
    )
    
    # Test market data
    test_market_data = {
        "current_price": 1.0875,
        "price_change_24h": 0.25,
        "trend": "bullish",
        "volatility": 0.015
    }
    
    # Test enhanced analysis
    print(f"\nTesting enhanced market interpretation for EUR/USD...")
    result = interpreter.interpret_market_with_news(test_market_data, "EUR/USD")
    
    print(f"\nRESULTS:")
    print(f"Final Sentiment: {result.get('final_sentiment', 'N/A')}")
    print(f"Final Confidence: {result.get('final_confidence', 0):.1%}")
    print(f"Action: {result.get('action_recommendation', 'N/A')}")
    print(f"Enhanced with News: {result.get('enhanced_with_news', False)}")
    print(f"Processing Time: {result.get('total_processing_time', 'N/A')}")
    
    if result.get('news_analysis', {}).get('available'):
        news = result['news_analysis']
        print(f"\nNews Analysis:")
        print(f"  Sentiment Score: {news.get('sentiment_score', 0):+.3f}")
        print(f"  Confidence: {news.get('confidence', 0):.1%}")
        print(f"  Articles: {news.get('article_count', 0)}")
    
    # Test signal validation
    print(f"\n{'='*30} SIGNAL VALIDATION {'='*30}")
    test_signal = {
        'ml_prediction': 0.65,
        'ml_confidence': 0.58,
        'rsi': 45,
        'trend_direction': 'bullish'
    }
    
    validation = interpreter.validate_signal_with_news(test_signal, "EUR/USD")
    print(f"Signal Validation: {validation}")
    
    # Performance stats
    stats = interpreter.get_comprehensive_performance_stats()
    print(f"\nPerformance Stats: {stats}")

if __name__ == "__main__":
    test_enhanced_news_gemini()