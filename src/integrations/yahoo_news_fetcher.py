#!/usr/bin/env python3
"""
Yahoo Finance News Fetcher for ForexSwing AI
Free RSS feed - no API key required
"""

import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List
import re

class YahooNewsAnalyzer:
    """Fetch and analyze forex news from Yahoo Finance RSS"""

    def __init__(self):
        self.base_url = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"

        # Forex symbols for Yahoo Finance
        self.forex_symbols = {
            'EUR/USD': 'EURUSD=X',
            'GBP/USD': 'GBPUSD=X',
            'USD/JPY': 'USDJPY=X',
            'USD/CHF': 'USDCHF=X',
            'AUD/USD': 'AUDUSD=X',
            'USD/CAD': 'USDCAD=X',
            'NZD/USD': 'NZDUSD=X'
        }

        # Sentiment keywords
        self.bullish_words = [
            'rise', 'rises', 'gain', 'gains', 'surge', 'surges', 'rally', 'rallies',
            'strengthen', 'strengthens', 'boost', 'jump', 'climbs', 'soar', 'advance',
            'bullish', 'optimistic', 'positive', 'recovery', 'growth', 'strong'
        ]

        self.bearish_words = [
            'fall', 'falls', 'drop', 'drops', 'plunge', 'plunges', 'decline', 'declines',
            'weaken', 'weakens', 'slide', 'slump', 'sink', 'tumble', 'retreat',
            'bearish', 'pessimistic', 'negative', 'recession', 'weak', 'concern', 'fears'
        ]

    def fetch_news(self, pair: str, max_articles: int = 10) -> List[Dict]:
        """Fetch latest news headlines for a forex pair"""
        symbol = self.forex_symbols.get(pair)
        if not symbol:
            return []

        try:
            url = self.base_url.format(symbol=symbol)
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

            with urllib.request.urlopen(req, timeout=10) as response:
                data = response.read()
                root = ET.fromstring(data)

                articles = []
                items = root.findall('.//item')[:max_articles]

                for item in items:
                    title_elem = item.find('title')
                    desc_elem = item.find('description')
                    date_elem = item.find('pubDate')

                    if title_elem is not None:
                        article = {
                            'title': title_elem.text or '',
                            'description': desc_elem.text if desc_elem is not None else '',
                            'date': date_elem.text if date_elem is not None else '',
                            'source': 'Yahoo Finance'
                        }
                        articles.append(article)

                return articles

        except Exception as e:
            print(f"Error fetching Yahoo Finance news for {pair}: {e}")
            return []

    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using keyword matching
        Returns: -1.0 (very bearish) to +1.0 (very bullish)
        """
        text_lower = text.lower()

        bullish_count = sum(1 for word in self.bullish_words if word in text_lower)
        bearish_count = sum(1 for word in self.bearish_words if word in text_lower)

        if bullish_count == 0 and bearish_count == 0:
            return 0.0

        # Calculate sentiment score
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0

        sentiment = (bullish_count - bearish_count) / total

        return max(-1.0, min(1.0, sentiment))

    def get_news_sentiment(self, pair: str) -> Dict:
        """Get aggregated news sentiment for a pair"""
        articles = self.fetch_news(pair)

        if not articles:
            return {
                'success': False,
                'sentiment': 0.0,
                'confidence': 0.0,
                'article_count': 0,
                'top_headlines': []
            }

        # Analyze sentiment for each article
        sentiments = []
        headlines = []

        for article in articles:
            text = f"{article['title']} {article['description']}"
            sentiment = self.analyze_sentiment(text)
            sentiments.append(sentiment)
            headlines.append(article['title'])

        # Calculate overall sentiment
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

        # Confidence based on number of articles and agreement
        confidence = min(0.9, len(articles) / 10) * 0.7  # Max 0.63 confidence

        return {
            'success': True,
            'sentiment': avg_sentiment,
            'confidence': confidence,
            'article_count': len(articles),
            'top_headlines': headlines[:5],
            'source': 'Yahoo_Finance_RSS'
        }

# Quick test
if __name__ == "__main__":
    analyzer = YahooNewsAnalyzer()

    print("Testing Yahoo Finance News Fetcher...")
    print("=" * 70)

    for pair in ['EUR/USD', 'GBP/USD']:
        print(f"\nFetching news for {pair}...")
        result = analyzer.get_news_sentiment(pair)

        if result['success']:
            print(f"  Sentiment: {result['sentiment']:+.3f}")
            print(f"  Confidence: {result['confidence']:.1%}")
            print(f"  Articles: {result['article_count']}")
            print(f"  Top headlines:")
            for i, headline in enumerate(result['top_headlines'][:3], 1):
                print(f"    {i}. {headline[:80]}...")
        else:
            print(f"  No news found")

    print("\n" + "=" * 70)
