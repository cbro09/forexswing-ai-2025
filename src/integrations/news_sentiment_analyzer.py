#!/usr/bin/env python3
"""
Multi-Source News Sentiment Analyzer for ForexSwing AI
Integrates Alpha Vantage, Yahoo Finance, and other news sources for live forex sentiment
"""

import requests
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import feedparser
import re
from urllib.parse import quote
import warnings
warnings.filterwarnings('ignore')

@dataclass
class NewsArticle:
    """Individual news article data"""
    title: str
    summary: str
    source: str
    published: str
    url: str
    sentiment_score: float
    relevance_score: float
    tickers: List[str]
    topics: List[str]

@dataclass
class ForexNewsSentiment:
    """Aggregated forex news sentiment for a currency pair"""
    pair: str
    overall_sentiment: float  # -1 (bearish) to +1 (bullish)
    confidence: float  # 0 to 1
    article_count: int
    top_articles: List[NewsArticle]
    sentiment_breakdown: Dict[str, float]
    timestamp: str

class MultiSourceNewsAnalyzer:
    """
    Multi-source news sentiment analyzer for forex trading
    Sources: Alpha Vantage, Yahoo Finance RSS, NewsAPI
    """
    
    def __init__(self, alpha_vantage_key: str = None, news_api_key: str = None):
        self.alpha_vantage_key = alpha_vantage_key or "OXGW647WZO8XTKA1"  # Live API key
        self.news_api_key = news_api_key
        
        # Source weights for sentiment aggregation
        self.source_weights = {
            'alpha_vantage': 0.4,
            'yahoo_finance': 0.3,
            'news_api': 0.2,
            'fed_economic': 0.1
        }
        
        # Currency pair mappings
        self.currency_mappings = {
            'EUR/USD': ['EUR', 'USD', 'EURO', 'DOLLAR', 'EURUSD'],
            'GBP/USD': ['GBP', 'USD', 'POUND', 'STERLING', 'GBPUSD'],
            'USD/JPY': ['USD', 'JPY', 'DOLLAR', 'YEN', 'USDJPY'],
            'USD/CHF': ['USD', 'CHF', 'DOLLAR', 'FRANC', 'USDCHF'],
            'AUD/USD': ['AUD', 'USD', 'AUSTRALIAN', 'DOLLAR', 'AUDUSD'],
            'USD/CAD': ['USD', 'CAD', 'DOLLAR', 'CANADIAN', 'USDCAD'],
            'NZD/USD': ['NZD', 'USD', 'ZEALAND', 'DOLLAR', 'NZDUSD']
        }
        
        # Cache for API responses
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
        print(f"MultiSourceNewsAnalyzer initialized:")
        print(f"  - Alpha Vantage: {'API Key Set' if alpha_vantage_key else 'Using Demo Key'}")
        print(f"  - NewsAPI: {'API Key Set' if news_api_key else 'Not Available'}")
        print(f"  - Yahoo Finance: RSS feeds (free)")
    
    def get_alpha_vantage_news(self, pair: str, hours_back: int = 24) -> List[NewsArticle]:
        """Get news from Alpha Vantage News & Sentiment API"""
        if not self.alpha_vantage_key or self.alpha_vantage_key == "demo":
            return []
        
        try:
            # Prepare forex ticker format
            currencies = self.currency_mappings.get(pair, [])
            if not currencies:
                return []
            
            # Alpha Vantage forex format: FOREX:USD
            forex_tickers = [f"FOREX:{curr}" for curr in currencies[:2]]
            tickers_param = ",".join(forex_tickers)
            
            # Time range
            time_from = (datetime.now() - timedelta(hours=hours_back)).strftime("%Y%m%dT%H%M")
            
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': tickers_param,
                'time_from': time_from,
                'limit': 50,
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            articles = []
            if 'feed' in data:
                for item in data['feed']:
                    # Extract sentiment score
                    sentiment_score = 0.0
                    if 'overall_sentiment_score' in item:
                        sentiment_score = float(item['overall_sentiment_score'])
                    
                    # Extract relevance score
                    relevance_score = 0.5
                    if 'ticker_sentiment' in item:
                        for ticker_data in item['ticker_sentiment']:
                            if any(ticker_data.get('ticker', '').endswith(curr) for curr in currencies):
                                relevance_score = max(relevance_score, float(ticker_data.get('relevance_score', 0.5)))
                    
                    article = NewsArticle(
                        title=item.get('title', '')[:200],
                        summary=item.get('summary', '')[:500],
                        source='Alpha Vantage',
                        published=item.get('time_published', ''),
                        url=item.get('url', ''),
                        sentiment_score=sentiment_score,
                        relevance_score=relevance_score,
                        tickers=currencies,
                        topics=item.get('topics', [])
                    )
                    articles.append(article)
            
            print(f"Alpha Vantage: Found {len(articles)} articles for {pair}")
            return articles
            
        except Exception as e:
            print(f"Alpha Vantage news error: {e}")
            return []
    
    def get_yahoo_finance_news(self, pair: str, hours_back: int = 24) -> List[NewsArticle]:
        """Get news from Yahoo Finance RSS feeds"""
        try:
            currencies = self.currency_mappings.get(pair, [])
            if not currencies:
                return []
            
            articles = []
            
            # Yahoo Finance RSS feeds
            rss_urls = [
                'https://feeds.finance.yahoo.com/rss/2.0/headline',
                'https://feeds.finance.yahoo.com/rss/2.0/topstories',
                'https://finance.yahoo.com/news/rssindex'
            ]
            
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            for rss_url in rss_urls:
                try:
                    feed = feedparser.parse(rss_url)
                    
                    for entry in feed.entries[:20]:  # Limit to recent articles
                        # Check if article mentions our currencies
                        text_content = f"{entry.get('title', '')} {entry.get('summary', '')}"
                        
                        # Simple relevance scoring
                        relevance_score = 0.0
                        for currency in currencies:
                            if currency.lower() in text_content.lower():
                                relevance_score += 0.2
                        
                        if relevance_score > 0.1:  # Only include relevant articles
                            # Simple sentiment scoring (you could enhance with NLP)
                            sentiment_score = self._simple_sentiment_analysis(text_content)
                            
                            # Check publication time
                            pub_time = entry.get('published_parsed')
                            if pub_time:
                                pub_datetime = datetime(*pub_time[:6])
                                if pub_datetime < cutoff_time:
                                    continue
                            
                            article = NewsArticle(
                                title=entry.get('title', '')[:200],
                                summary=entry.get('summary', '')[:500],
                                source='Yahoo Finance',
                                published=entry.get('published', ''),
                                url=entry.get('link', ''),
                                sentiment_score=sentiment_score,
                                relevance_score=min(relevance_score, 1.0),
                                tickers=currencies,
                                topics=['forex', 'finance']
                            )
                            articles.append(article)
                
                except Exception as e:
                    print(f"RSS feed error ({rss_url}): {e}")
                    continue
            
            print(f"Yahoo Finance: Found {len(articles)} articles for {pair}")
            return articles
            
        except Exception as e:
            print(f"Yahoo Finance news error: {e}")
            return []
    
    def get_newsapi_news(self, pair: str, hours_back: int = 24) -> List[NewsArticle]:
        """Get news from NewsAPI.org"""
        if not self.news_api_key:
            return []
        
        try:
            currencies = self.currency_mappings.get(pair, [])
            if not currencies:
                return []
            
            # Create search query
            currency_query = " OR ".join(currencies[:2])
            query = f"({currency_query}) AND (forex OR currency OR exchange rate)"
            
            from_time = (datetime.now() - timedelta(hours=hours_back)).isoformat()
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'from': from_time,
                'sortBy': 'publishedAt',
                'language': 'en',
                'pageSize': 20,
                'apiKey': self.news_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            articles = []
            if data.get('status') == 'ok' and 'articles' in data:
                for item in data['articles']:
                    # Calculate relevance and sentiment
                    text_content = f"{item.get('title', '')} {item.get('description', '')}"
                    relevance_score = self._calculate_relevance(text_content, currencies)
                    sentiment_score = self._simple_sentiment_analysis(text_content)
                    
                    if relevance_score > 0.2:
                        article = NewsArticle(
                            title=item.get('title', '')[:200],
                            summary=item.get('description', '')[:500],
                            source=item.get('source', {}).get('name', 'NewsAPI'),
                            published=item.get('publishedAt', ''),
                            url=item.get('url', ''),
                            sentiment_score=sentiment_score,
                            relevance_score=relevance_score,
                            tickers=currencies,
                            topics=['forex', 'currency']
                        )
                        articles.append(article)
            
            print(f"NewsAPI: Found {len(articles)} articles for {pair}")
            return articles
            
        except Exception as e:
            print(f"NewsAPI error: {e}")
            return []
    
    def _simple_sentiment_analysis(self, text: str) -> float:
        """Simple sentiment analysis using keyword matching"""
        positive_words = [
            'bullish', 'rise', 'gains', 'up', 'positive', 'strong', 'boost', 'surge',
            'rally', 'growth', 'increase', 'optimistic', 'outperform', 'buy'
        ]
        
        negative_words = [
            'bearish', 'fall', 'losses', 'down', 'negative', 'weak', 'decline', 'drop',
            'crash', 'recession', 'decrease', 'pessimistic', 'underperform', 'sell'
        ]
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        # Normalize to -1 to +1 range
        sentiment = (positive_count - negative_count) / max(10, total_words * 0.1)
        return max(-1.0, min(1.0, sentiment))
    
    def _calculate_relevance(self, text: str, currencies: List[str]) -> float:
        """Calculate how relevant the text is to the currency pair"""
        text_lower = text.lower()
        relevance = 0.0
        
        for currency in currencies:
            if currency.lower() in text_lower:
                relevance += 0.3
        
        # Bonus for forex-related terms
        forex_terms = ['forex', 'currency', 'exchange rate', 'central bank', 'monetary policy']
        for term in forex_terms:
            if term in text_lower:
                relevance += 0.1
        
        return min(1.0, relevance)
    
    def analyze_forex_sentiment(self, pair: str, hours_back: int = 24) -> ForexNewsSentiment:
        """
        Comprehensive forex sentiment analysis from multiple sources
        """
        print(f"\nANALYZING NEWS SENTIMENT FOR {pair}")
        print("-" * 50)
        
        # Check cache first
        cache_key = f"{pair}_{hours_back}_{int(time.time() // self.cache_duration)}"
        if cache_key in self.cache:
            print("Using cached news data")
            return self.cache[cache_key]
        
        all_articles = []
        
        # Gather news from all sources
        start_time = time.time()
        
        # Alpha Vantage
        alpha_articles = self.get_alpha_vantage_news(pair, hours_back)
        all_articles.extend(alpha_articles)
        
        # Yahoo Finance
        yahoo_articles = self.get_yahoo_finance_news(pair, hours_back)
        all_articles.extend(yahoo_articles)
        
        # NewsAPI
        newsapi_articles = self.get_newsapi_news(pair, hours_back)
        all_articles.extend(newsapi_articles)
        
        end_time = time.time()
        
        print(f"Gathered {len(all_articles)} articles in {end_time - start_time:.1f}s")
        
        if not all_articles:
            return ForexNewsSentiment(
                pair=pair,
                overall_sentiment=0.0,
                confidence=0.0,
                article_count=0,
                top_articles=[],
                sentiment_breakdown={'neutral': 1.0},
                timestamp=datetime.now().isoformat()
            )
        
        # Aggregate sentiment
        sentiment_scores = []
        confidence_weights = []
        
        for article in all_articles:
            # Weight by relevance and source reliability
            source_weight = self.source_weights.get(article.source.lower().replace(' ', '_'), 0.1)
            article_weight = article.relevance_score * source_weight
            
            sentiment_scores.append(article.sentiment_score)
            confidence_weights.append(article_weight)
        
        # Calculate weighted average sentiment
        if sentiment_scores:
            overall_sentiment = np.average(sentiment_scores, weights=confidence_weights)
            confidence = min(1.0, sum(confidence_weights) / len(sentiment_scores))
        else:
            overall_sentiment = 0.0
            confidence = 0.0
        
        # Sentiment breakdown
        positive_articles = sum(1 for score in sentiment_scores if score > 0.1)
        negative_articles = sum(1 for score in sentiment_scores if score < -0.1)
        neutral_articles = len(sentiment_scores) - positive_articles - negative_articles
        
        total = len(sentiment_scores)
        sentiment_breakdown = {
            'bullish': positive_articles / total if total > 0 else 0,
            'bearish': negative_articles / total if total > 0 else 0,
            'neutral': neutral_articles / total if total > 0 else 1
        }
        
        # Top articles (by relevance score)
        top_articles = sorted(all_articles, key=lambda x: x.relevance_score, reverse=True)[:5]
        
        # Create result
        result = ForexNewsSentiment(
            pair=pair,
            overall_sentiment=overall_sentiment,
            confidence=confidence,
            article_count=len(all_articles),
            top_articles=top_articles,
            sentiment_breakdown=sentiment_breakdown,
            timestamp=datetime.now().isoformat()
        )
        
        # Cache result
        self.cache[cache_key] = result
        
        print(f"Overall Sentiment: {overall_sentiment:+.3f} (confidence: {confidence:.1%})")
        print(f"Breakdown: {sentiment_breakdown}")
        
        return result
    
    def get_enhanced_market_context(self, pair: str) -> Dict:
        """
        Get enhanced market context including news sentiment for Gemini analysis
        """
        news_sentiment = self.analyze_forex_sentiment(pair, hours_back=12)
        
        return {
            'news_sentiment': news_sentiment.overall_sentiment,
            'news_confidence': news_sentiment.confidence,
            'news_article_count': news_sentiment.article_count,
            'sentiment_breakdown': news_sentiment.sentiment_breakdown,
            'top_news_summary': [
                f"{article.title[:100]}..." for article in news_sentiment.top_articles[:3]
            ],
            'news_enhanced': True,
            'timestamp': news_sentiment.timestamp
        }

def test_news_sentiment_analyzer():
    """Test the news sentiment analyzer"""
    print("TESTING MULTI-SOURCE NEWS SENTIMENT ANALYZER")
    print("=" * 60)
    
    # Initialize analyzer (you can add your API keys here)
    analyzer = MultiSourceNewsAnalyzer(
        alpha_vantage_key="demo",  # Replace with your Alpha Vantage key
        news_api_key=None  # Replace with your NewsAPI key if available
    )
    
    # Test pairs
    test_pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY']
    
    for pair in test_pairs:
        print(f"\n{'='*20} {pair} {'='*20}")
        
        # Get sentiment analysis
        sentiment = analyzer.analyze_forex_sentiment(pair, hours_back=24)
        
        print(f"Sentiment Score: {sentiment.overall_sentiment:+.3f}")
        print(f"Confidence: {sentiment.confidence:.1%}")
        print(f"Articles Analyzed: {sentiment.article_count}")
        print(f"Breakdown: {sentiment.sentiment_breakdown}")
        
        if sentiment.top_articles:
            print("\nTop Article:")
            top_article = sentiment.top_articles[0]
            print(f"  Title: {top_article.title}")
            print(f"  Source: {top_article.source}")
            print(f"  Sentiment: {top_article.sentiment_score:+.3f}")
            print(f"  Relevance: {top_article.relevance_score:.1%}")
        
        # Test enhanced context
        context = analyzer.get_enhanced_market_context(pair)
        print(f"\nEnhanced Context Keys: {list(context.keys())}")
        
        time.sleep(1)  # Rate limiting
    
    print(f"\n{'='*60}")
    print("NEWS SENTIMENT ANALYSIS COMPLETE")

if __name__ == "__main__":
    test_news_sentiment_analyzer()