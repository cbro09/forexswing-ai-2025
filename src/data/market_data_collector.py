#!/usr/bin/env python3
"""
Enhanced Market Data Collector with News Integration
Automatically collects forex data, news, and sentiment analysis
Runs continuously on VM for better predictions
"""

import os
import sys
import time
import schedule
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import logging
import yfinance as yf
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.timeseries import TimeSeries
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/data_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MarketDataCollector')

class EnhancedMarketDataCollector:
    """
    Enhanced data collector that gathers:
    1. Real-time forex price data
    2. News articles and sentiment
    3. Economic indicators
    4. Technical indicators
    """

    def __init__(self, alpha_vantage_key: str = None):
        self.alpha_vantage_key = alpha_vantage_key or os.getenv('ALPHA_VANTAGE_KEY', 'OXGW647WZO8XTKA1')
        self.fx = ForeignExchange(key=self.alpha_vantage_key, output_format='pandas')
        self.ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')

        # Currency pairs to track
        self.pairs = [
            ('EUR', 'USD'),
            ('GBP', 'USD'),
            ('USD', 'JPY'),
            ('AUD', 'USD'),
            ('USD', 'CAD'),
            ('NZD', 'USD'),
            ('USD', 'CHF')
        ]

        # Data storage paths
        self.data_dir = 'data/MarketData'
        self.news_dir = 'data/News'
        self.indicators_dir = 'data/TechnicalIndicators'

        # Create directories
        for directory in [self.data_dir, self.news_dir, self.indicators_dir]:
            os.makedirs(directory, exist_ok=True)

        logger.info(f"EnhancedMarketDataCollector initialized")
        logger.info(f"Tracking {len(self.pairs)} currency pairs")

    def collect_forex_data(self, from_currency: str, to_currency: str):
        """Collect forex data from multiple sources"""
        pair_name = f"{from_currency}_{to_currency}"

        try:
            logger.info(f"Collecting data for {pair_name}...")

            # Method 1: Alpha Vantage (primary source)
            try:
                data_daily, meta_data = self.fx.get_currency_exchange_daily(
                    from_symbol=from_currency,
                    to_symbol=to_currency,
                    outputsize='full'
                )

                # Rename columns to standard format
                data_daily.columns = ['open', 'high', 'low', 'close']
                data_daily['volume'] = 0  # FX doesn't have volume in traditional sense
                data_daily.index.name = 'date'

                # Save data
                output_path = os.path.join(self.data_dir, f"{pair_name}_real_daily.csv")
                data_daily.to_csv(output_path)
                logger.info(f"✅ Saved {len(data_daily)} days of data for {pair_name}")

                return data_daily

            except Exception as e:
                logger.warning(f"Alpha Vantage failed for {pair_name}: {e}")

                # Method 2: Yahoo Finance (fallback)
                ticker_symbol = f"{from_currency}{to_currency}=X"
                yf_data = yf.download(ticker_symbol, period="2y", interval="1d", progress=False)

                if not yf_data.empty:
                    # Standardize column names
                    yf_data.columns = [col.lower() for col in yf_data.columns]
                    yf_data['volume'] = yf_data.get('volume', 0)

                    output_path = os.path.join(self.data_dir, f"{pair_name}_real_daily.csv")
                    yf_data.to_csv(output_path)
                    logger.info(f"✅ Saved {len(yf_data)} days of YF data for {pair_name}")
                    return yf_data

                logger.error(f"❌ Failed to collect data for {pair_name}")
                return None

        except Exception as e:
            logger.error(f"Error collecting forex data for {pair_name}: {e}")
            return None

    def collect_intraday_data(self, from_currency: str, to_currency: str, interval: str = '60min'):
        """Collect intraday forex data for real-time analysis"""
        pair_name = f"{from_currency}_{to_currency}"

        try:
            logger.info(f"Collecting intraday data for {pair_name}...")

            data_intraday, meta_data = self.fx.get_currency_exchange_intraday(
                from_symbol=from_currency,
                to_symbol=to_currency,
                interval=interval,
                outputsize='full'
            )

            # Rename columns
            data_intraday.columns = ['open', 'high', 'low', 'close']
            data_intraday['volume'] = 0
            data_intraday.index.name = 'timestamp'

            # Save intraday data
            output_path = os.path.join(self.data_dir, f"{pair_name}_intraday_{interval}.csv")
            data_intraday.to_csv(output_path)
            logger.info(f"✅ Saved {len(data_intraday)} intraday records for {pair_name}")

            return data_intraday

        except Exception as e:
            logger.warning(f"Intraday collection failed for {pair_name}: {e}")
            return None

    def collect_news_sentiment(self, pair: str):
        """Collect news and sentiment for currency pair"""
        try:
            from src.integrations.news_sentiment_analyzer import MultiSourceNewsAnalyzer

            news_analyzer = MultiSourceNewsAnalyzer(alpha_vantage_key=self.alpha_vantage_key)
            sentiment = news_analyzer.analyze_forex_sentiment(pair, hours_back=24)

            # Save news sentiment data
            news_data = {
                'timestamp': datetime.now().isoformat(),
                'pair': pair,
                'sentiment_score': sentiment.overall_sentiment,
                'confidence': sentiment.confidence,
                'article_count': sentiment.article_count,
                'sentiment_breakdown': sentiment.sentiment_breakdown,
                'top_articles': [
                    {
                        'title': article.title,
                        'source': article.source,
                        'timestamp': article.timestamp.isoformat() if article.timestamp else None,
                        'sentiment': article.sentiment_score
                    }
                    for article in sentiment.top_articles[:10]
                ]
            }

            # Save to file
            date_str = datetime.now().strftime('%Y%m%d')
            output_path = os.path.join(self.news_dir, f"{pair.replace('/', '_')}_news_{date_str}.json")

            with open(output_path, 'w') as f:
                json.dump(news_data, f, indent=2)

            logger.info(f"✅ Saved news sentiment for {pair} ({sentiment.article_count} articles)")
            return news_data

        except Exception as e:
            logger.error(f"Error collecting news for {pair}: {e}")
            return None

    def calculate_technical_indicators(self, data: pd.DataFrame, pair_name: str):
        """Calculate and save enhanced technical indicators"""
        try:
            indicators = {}

            # Price-based indicators
            indicators['sma_10'] = data['close'].rolling(window=10).mean()
            indicators['sma_20'] = data['close'].rolling(window=20).mean()
            indicators['sma_50'] = data['close'].rolling(window=50).mean()
            indicators['sma_200'] = data['close'].rolling(window=200).mean()

            # EMA
            indicators['ema_12'] = data['close'].ewm(span=12).mean()
            indicators['ema_26'] = data['close'].ewm(span=26).mean()

            # MACD
            indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
            indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
            indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']

            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi_14'] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            indicators['bb_middle'] = data['close'].rolling(window=20).mean()
            bb_std = data['close'].rolling(window=20).std()
            indicators['bb_upper'] = indicators['bb_middle'] + (bb_std * 2)
            indicators['bb_lower'] = indicators['bb_middle'] - (bb_std * 2)

            # ATR (Average True Range)
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            indicators['atr_14'] = true_range.rolling(14).mean()

            # Volatility
            indicators['volatility_20'] = data['close'].pct_change().rolling(20).std()

            # Create DataFrame
            indicators_df = pd.DataFrame(indicators, index=data.index)

            # Save indicators
            output_path = os.path.join(self.indicators_dir, f"{pair_name}_indicators.csv")
            indicators_df.to_csv(output_path)
            logger.info(f"✅ Calculated {len(indicators)} technical indicators for {pair_name}")

            return indicators_df

        except Exception as e:
            logger.error(f"Error calculating indicators for {pair_name}: {e}")
            return None

    def run_full_collection(self):
        """Run complete data collection for all pairs"""
        logger.info("="*60)
        logger.info("Starting full market data collection")
        logger.info("="*60)

        for from_curr, to_curr in self.pairs:
            pair_name = f"{from_curr}_{to_curr}"
            pair_display = f"{from_curr}/{to_curr}"

            try:
                # 1. Collect daily data
                daily_data = self.collect_forex_data(from_curr, to_curr)

                # 2. Collect intraday data (hourly)
                # intraday_data = self.collect_intraday_data(from_curr, to_curr, '60min')

                # 3. Calculate technical indicators
                if daily_data is not None:
                    self.calculate_technical_indicators(daily_data, pair_name)

                # 4. Collect news sentiment
                self.collect_news_sentiment(pair_display)

                # Rate limiting to avoid API throttling
                time.sleep(15)  # Wait 15 seconds between pairs

            except Exception as e:
                logger.error(f"Error processing {pair_name}: {e}")
                continue

        logger.info("="*60)
        logger.info("Full collection completed")
        logger.info("="*60)

    def run_quick_update(self):
        """Quick update - only intraday and news"""
        logger.info("Running quick market update...")

        for from_curr, to_curr in self.pairs:
            pair_display = f"{from_curr}/{to_curr}"

            try:
                # Only collect news for quick updates
                self.collect_news_sentiment(pair_display)
                time.sleep(10)

            except Exception as e:
                logger.error(f"Quick update error for {pair_display}: {e}")
                continue

        logger.info("Quick update completed")

def schedule_data_collection():
    """Schedule automated data collection"""
    collector = EnhancedMarketDataCollector()

    # Schedule full collection once per day at 00:00 UTC
    schedule.every().day.at("00:00").do(collector.run_full_collection)

    # Schedule quick updates every 4 hours
    schedule.every(4).hours.do(collector.run_quick_update)

    logger.info("Data collection scheduled:")
    logger.info("  - Full collection: Daily at 00:00 UTC")
    logger.info("  - Quick updates: Every 4 hours")

    # Run initial collection
    logger.info("Running initial data collection...")
    collector.run_full_collection()

    # Keep running
    logger.info("Starting scheduled data collection loop...")
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    logger.info("EnhancedMarketDataCollector starting...")

    # Check if running in one-shot mode
    if len(sys.argv) > 1 and sys.argv[1] == '--once':
        collector = EnhancedMarketDataCollector()
        collector.run_full_collection()
    else:
        # Run in continuous mode
        schedule_data_collection()
