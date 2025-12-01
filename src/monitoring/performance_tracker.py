#!/usr/bin/env python3
"""
Performance Tracker - Monitors signal accuracy over time
Validates generated trading signals against actual market movement
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/performance_tracker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('PerformanceTracker')

class PerformanceTracker:
    """
    Tracks trading signal performance over time
    Validates signals against actual market movement
    """

    def __init__(self, api_url="http://localhost:8082"):
        self.api_url = api_url
        self.signals_db = "data/logs/signals_history.json"
        self.metrics_db = "data/logs/performance_metrics.json"

        # Create directories if needed
        os.makedirs("data/logs", exist_ok=True)

        logger.info("PerformanceTracker initialized")

    def log_signal(self, pair, signal_data):
        """
        Log a generated signal for future validation

        Args:
            pair: Currency pair (e.g., "EUR/USD")
            signal_data: Dict with signal details
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "pair": pair,
            "signal": signal_data.get('action', 'HOLD'),
            "confidence": signal_data.get('confidence', 0.5),
            "price_at_signal": self.get_current_price(pair),
            "components": signal_data.get('components', {}),
            "validation_time": (datetime.now() + timedelta(hours=4)).isoformat(),
            "validated": False,
            "correct": None,
            "actual_change": None
        }

        # Append to JSON log
        signals = self.load_signals()
        signals.append(record)

        with open(self.signals_db, 'w') as f:
            json.dump(signals, f, indent=2)

        logger.info(f"Logged signal: {pair} {record['signal']} @ {record['confidence']:.1%}")

    def validate_signals(self):
        """
        Check past signals against actual market movement
        Returns accuracy percentage
        """
        signals = self.load_signals()
        newly_validated = 0

        for i, signal in enumerate(signals):
            # Skip already validated signals
            if signal.get('validated', False):
                continue

            validation_time = datetime.fromisoformat(signal['validation_time'])

            # If validation time has passed, check actual result
            if datetime.now() >= validation_time:
                pair = signal['pair']

                # Get current price
                try:
                    current_price = self.get_current_price(pair)

                    # Calculate actual movement
                    price_change = (current_price - signal['price_at_signal']) / signal['price_at_signal']

                    # Determine if signal was correct
                    if signal['signal'] == 'BUY':
                        correct = price_change > 0.001  # >0.1% gain
                    elif signal['signal'] == 'SELL':
                        correct = price_change < -0.001  # >0.1% loss
                    else:  # HOLD
                        correct = abs(price_change) < 0.005  # <0.5% movement

                    signal['validated'] = True
                    signal['correct'] = correct
                    signal['actual_change'] = float(price_change)
                    signal['validation_timestamp'] = datetime.now().isoformat()

                    newly_validated += 1

                    logger.info(
                        f"Validated: {pair} {signal['signal']} "
                        f"{'✅' if correct else '❌'} "
                        f"({price_change:+.2%})"
                    )

                except Exception as e:
                    logger.error(f"Error validating signal for {pair}: {e}")
                    continue

        # Save updated signals
        if newly_validated > 0:
            with open(self.signals_db, 'w') as f:
                json.dump(signals, f, indent=2)

            logger.info(f"Validated {newly_validated} signals")

        # Calculate and return accuracy
        return self.calculate_accuracy(signals)

    def calculate_accuracy(self, signals=None, days=7):
        """
        Calculate model accuracy over recent signals

        Args:
            signals: List of signal records (or load from file)
            days: Number of days to include

        Returns:
            Dict with accuracy metrics
        """
        if signals is None:
            signals = self.load_signals()

        # Filter to validated signals in last N days
        cutoff = datetime.now() - timedelta(days=days)
        recent = [
            s for s in signals
            if s.get('validated')
            and datetime.fromisoformat(s['timestamp']) > cutoff
        ]

        if len(recent) < 5:  # Need at least 5 signals
            logger.warning(f"Insufficient validated signals: {len(recent)} (need 5+)")
            return None

        # Overall accuracy
        correct = sum(1 for s in recent if s.get('correct', False))
        accuracy = correct / len(recent)

        # Per-signal-type accuracy
        buy_signals = [s for s in recent if s['signal'] == 'BUY']
        sell_signals = [s for s in recent if s['signal'] == 'SELL']
        hold_signals = [s for s in recent if s['signal'] == 'HOLD']

        buy_accuracy = (sum(1 for s in buy_signals if s.get('correct')) / len(buy_signals)) if buy_signals else 0
        sell_accuracy = (sum(1 for s in sell_signals if s.get('correct')) / len(sell_signals)) if sell_signals else 0
        hold_accuracy = (sum(1 for s in hold_signals if s.get('correct')) / len(hold_signals)) if hold_signals else 0

        metrics = {
            'timestamp': datetime.now().isoformat(),
            'period_days': days,
            'total_signals': len(recent),
            'correct_signals': correct,
            'overall_accuracy': accuracy,
            'buy_accuracy': buy_accuracy,
            'sell_accuracy': sell_accuracy,
            'hold_accuracy': hold_accuracy,
            'signal_distribution': {
                'BUY': len(buy_signals),
                'SELL': len(sell_signals),
                'HOLD': len(hold_signals)
            }
        }

        # Save metrics
        self.save_metrics(metrics)

        logger.info(
            f"Accuracy ({days}d): {accuracy:.1%} "
            f"({correct}/{len(recent)} signals)"
        )

        return metrics

    def get_current_price(self, pair):
        """Get current price for a currency pair"""
        try:
            # Try to get from market data file
            pair_file = pair.replace('/', '_')
            file_path = f"data/MarketData/{pair_file}_real_daily.csv"

            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                return float(df['close'].iloc[-1])

        except Exception as e:
            logger.error(f"Error getting price for {pair}: {e}")

        return None

    def load_signals(self):
        """Load signal history from JSON file"""
        try:
            with open(self.signals_db, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("No signals history file found, creating new one")
            return []
        except json.JSONDecodeError:
            logger.error("Corrupted signals file, starting fresh")
            return []

    def save_metrics(self, metrics):
        """Save performance metrics"""
        try:
            # Load existing metrics
            try:
                with open(self.metrics_db, 'r') as f:
                    all_metrics = json.load(f)
            except FileNotFoundError:
                all_metrics = []

            # Append new metrics
            all_metrics.append(metrics)

            # Keep only last 90 days
            cutoff = datetime.now() - timedelta(days=90)
            all_metrics = [
                m for m in all_metrics
                if datetime.fromisoformat(m['timestamp']) > cutoff
            ]

            # Save
            with open(self.metrics_db, 'w') as f:
                json.dump(all_metrics, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

    def run_continuous(self, check_interval=3600):
        """
        Run continuous monitoring loop

        Args:
            check_interval: Seconds between checks (default 1 hour)
        """
        logger.info("=" * 60)
        logger.info("Performance Tracker - Continuous Monitoring Started")
        logger.info("=" * 60)
        logger.info(f"Check interval: {check_interval}s ({check_interval/3600:.1f}h)")

        while True:
            try:
                logger.info(f"\n[{datetime.now()}] Running validation check...")

                # Validate past signals
                metrics = self.validate_signals()

                if metrics:
                    accuracy = metrics['overall_accuracy']
                    logger.info(f"Current accuracy: {accuracy:.1%} ({metrics['total_signals']} signals)")

                    # Alert if accuracy drops
                    if accuracy < 0.50:
                        logger.warning("⚠️  WARNING: Accuracy dropped below 50%!")
                        logger.warning("   Model may need retraining")

                # Sleep until next check
                logger.info(f"Next check in {check_interval/3600:.1f} hours...")
                time.sleep(check_interval)

            except KeyboardInterrupt:
                logger.info("\n👋 Performance tracker stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(300)  # Wait 5 minutes on error

if __name__ == "__main__":
    tracker = PerformanceTracker()

    # Run continuous monitoring (check every hour)
    tracker.run_continuous(check_interval=3600)
