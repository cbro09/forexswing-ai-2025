#!/usr/bin/env python3
"""
Auto-Retrainer - Automatically retrains model when performance drops
Enables continuous self-improvement of the trading bot
"""

import subprocess
import json
import time
import os
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/auto_retrainer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AutoRetrainer')

class AutoRetrainer:
    """
    Automatic model retraining system
    Monitors performance and triggers retraining when needed
    """

    def __init__(self):
        self.signals_db = "data/logs/signals_history.json"
        self.retrain_log = "data/logs/retrain_history.json"

        # Thresholds
        self.accuracy_threshold = 0.50  # Retrain if accuracy < 50%
        self.days_between_retrain = 30  # Max 30 days between retrains
        self.min_signals_required = 20  # Need at least 20 validated signals

        # Track last retrain
        self.last_retrain = self.get_last_retrain_time()

        logger.info("AutoRetrainer initialized")
        logger.info(f"  Accuracy threshold: {self.accuracy_threshold:.1%}")
        logger.info(f"  Max days between retrains: {self.days_between_retrain}")
        logger.info(f"  Last retrain: {self.last_retrain or 'Never'}")

    def should_retrain(self):
        """
        Check if retraining is needed

        Returns:
            (bool, str): (should_retrain, reason)
        """

        # Reason 1: Accuracy dropped below threshold
        accuracy = self.get_recent_accuracy(days=7)

        if accuracy is not None:
            if accuracy < self.accuracy_threshold:
                logger.warning(
                    f"Accuracy dropped: {accuracy:.1%} < {self.accuracy_threshold:.1%}"
                )
                return True, f"low_accuracy_{accuracy:.1%}"

        # Reason 2: Been too long since last retrain
        days_since_retrain = self.get_days_since_retrain()

        if days_since_retrain > self.days_between_retrain:
            logger.info(
                f"{days_since_retrain} days since last retrain (max {self.days_between_retrain})"
            )
            return True, "scheduled_retrain"

        # Reason 3: Sufficient new data collected (weekly check)
        if days_since_retrain > 7:
            data_quality = self.check_data_quality()
            if data_quality['sufficient_new_data']:
                logger.info("Sufficient new data collected for improvement")
                return True, "new_data_available"

        return False, None

    def retrain_model(self):
        """
        Trigger model retraining

        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("\n" + "=" * 60)
        logger.info("🤖 STARTING AUTOMATIC MODEL RETRAINING")
        logger.info("=" * 60)
        logger.info(f"Time: {datetime.now()}")

        try:
            # Check if training is already running
            if self.is_training_running():
                logger.warning("Training already in progress, skipping")
                return False

            # Pre-retraining checks
            if not self.pre_retrain_checks():
                logger.error("Pre-retrain checks failed")
                return False

            # Run training container
            logger.info("Executing: docker-compose --profile training run --rm model-trainer")

            result = subprocess.run(
                ['docker-compose', '--profile', 'training', 'run', '--rm', 'model-trainer'],
                cwd='/opt/forexswing-ai-2025',  # Adjust path as needed
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )

            if result.returncode == 0:
                logger.info("✅ Model retraining completed successfully")
                logger.info(f"Training output (last 500 chars):\n{result.stdout[-500:]}")

                # Verify new model exists
                if os.path.exists('data/models/enhanced_forex_ai.pth'):
                    logger.info("✅ New model file verified")

                    # Restart API to load new model
                    logger.info("Restarting API service...")
                    restart_result = subprocess.run(
                        ['docker-compose', 'restart', 'forexbot'],
                        cwd='/opt/forexswing-ai-2025',
                        capture_output=True,
                        text=True,
                        timeout=60
                    )

                    if restart_result.returncode == 0:
                        logger.info("✅ API service restarted with new model")
                    else:
                        logger.error(f"Failed to restart API: {restart_result.stderr}")

                    # Log successful retrain
                    self.log_retrain(success=True, output=result.stdout[-1000:])
                    self.last_retrain = datetime.now()

                    return True
                else:
                    logger.error("New model file not found after training")
                    return False

            else:
                logger.error(f"❌ Model retraining failed (exit code {result.returncode})")
                logger.error(f"Error output:\n{result.stderr}")
                self.log_retrain(success=False, error=result.stderr)
                return False

        except subprocess.TimeoutExpired:
            logger.error("❌ Training timeout (>2 hours)")
            return False
        except Exception as e:
            logger.error(f"❌ Retraining error: {e}")
            return False

    def pre_retrain_checks(self):
        """
        Perform checks before retraining

        Returns:
            bool: True if all checks pass
        """
        logger.info("Running pre-retrain checks...")

        # Check 1: Sufficient market data
        data_files = [
            'data/MarketData/EUR_USD_real_daily.csv',
            'data/MarketData/GBP_USD_real_daily.csv',
            'data/MarketData/USD_JPY_real_daily.csv',
        ]

        for file in data_files:
            if not os.path.exists(file):
                logger.error(f"Missing data file: {file}")
                return False

            # Check file size (should be >100KB for 2 years of data)
            size = os.path.getsize(file)
            if size < 100000:
                logger.error(f"Data file too small: {file} ({size} bytes)")
                return False

        logger.info("✅ Market data files verified")

        # Check 2: Disk space
        stat = os.statvfs('data/')
        free_space = stat.f_bavail * stat.f_frsize
        free_gb = free_space / (1024**3)

        if free_gb < 1.0:
            logger.error(f"Insufficient disk space: {free_gb:.2f}GB free")
            return False

        logger.info(f"✅ Disk space: {free_gb:.1f}GB free")

        # Check 3: Docker is running
        try:
            result = subprocess.run(
                ['docker', 'ps'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                logger.error("Docker is not running")
                return False
        except Exception as e:
            logger.error(f"Docker check failed: {e}")
            return False

        logger.info("✅ Docker is running")

        logger.info("All pre-retrain checks passed")
        return True

    def is_training_running(self):
        """Check if training is already in progress"""
        try:
            result = subprocess.run(
                ['docker', 'ps', '--filter', 'name=model-trainer', '--format', '{{.Names}}'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return 'model-trainer' in result.stdout
        except:
            return False

    def get_recent_accuracy(self, days=7):
        """
        Get accuracy over last N days

        Args:
            days: Number of days to look back

        Returns:
            float: Accuracy (0-1) or None if insufficient data
        """
        try:
            with open(self.signals_db, 'r') as f:
                signals = json.load(f)

            # Filter to validated signals in last N days
            cutoff = datetime.now() - timedelta(days=days)
            recent = [
                s for s in signals
                if s.get('validated')
                and datetime.fromisoformat(s['timestamp']) > cutoff
            ]

            if len(recent) < self.min_signals_required:
                logger.info(f"Insufficient validated signals: {len(recent)} (need {self.min_signals_required})")
                return None

            correct = sum(1 for s in recent if s.get('correct', False))
            accuracy = correct / len(recent)

            logger.info(f"Recent accuracy ({days}d): {accuracy:.1%} ({correct}/{len(recent)} signals)")
            return accuracy

        except FileNotFoundError:
            logger.warning("No signals history file found")
            return None
        except Exception as e:
            logger.error(f"Error calculating accuracy: {e}")
            return None

    def get_days_since_retrain(self):
        """Get days since last retrain"""
        if self.last_retrain:
            if isinstance(self.last_retrain, str):
                self.last_retrain = datetime.fromisoformat(self.last_retrain)
            return (datetime.now() - self.last_retrain).days
        return 999  # Never trained

    def get_last_retrain_time(self):
        """Load last retrain time from log"""
        try:
            with open(self.retrain_log, 'r') as f:
                history = json.load(f)

            # Find most recent successful retrain
            successful = [r for r in history if r.get('success')]
            if successful:
                return datetime.fromisoformat(successful[-1]['timestamp'])

        except FileNotFoundError:
            pass
        except Exception as e:
            logger.error(f"Error loading retrain history: {e}")

        return None

    def check_data_quality(self):
        """
        Check if sufficient new data has been collected

        Returns:
            dict: Data quality metrics
        """
        try:
            # Check most recent data file modification time
            data_file = 'data/MarketData/EUR_USD_real_daily.csv'

            if os.path.exists(data_file):
                modified_time = datetime.fromtimestamp(os.path.getmtime(data_file))
                days_since_update = (datetime.now() - modified_time).days

                return {
                    'sufficient_new_data': days_since_update <= 1,  # Updated in last day
                    'days_since_update': days_since_update
                }

        except Exception as e:
            logger.error(f"Error checking data quality: {e}")

        return {'sufficient_new_data': False, 'days_since_update': 999}

    def log_retrain(self, success, output=None, error=None):
        """Log retraining event"""
        try:
            # Load existing history
            try:
                with open(self.retrain_log, 'r') as f:
                    history = json.load(f)
            except FileNotFoundError:
                history = []

            # Add new entry
            entry = {
                'timestamp': datetime.now().isoformat(),
                'success': success,
                'output': output[-500:] if output else None,  # Last 500 chars
                'error': error[-500:] if error else None
            }

            history.append(entry)

            # Keep only last 50 entries
            history = history[-50:]

            # Save
            with open(self.retrain_log, 'w') as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            logger.error(f"Error logging retrain: {e}")

    def run(self, check_interval=21600):
        """
        Main loop - checks every N seconds

        Args:
            check_interval: Seconds between checks (default 6 hours)
        """
        logger.info("\n" + "=" * 60)
        logger.info("🤖 AUTO-RETRAINER STARTED")
        logger.info("=" * 60)
        logger.info(f"Check interval: {check_interval}s ({check_interval/3600:.1f}h)")

        while True:
            try:
                logger.info(f"\n[{datetime.now()}] Checking if retraining needed...")

                should_retrain, reason = self.should_retrain()

                if should_retrain:
                    logger.info(f"🔄 Retraining triggered: {reason}")
                    success = self.retrain_model()

                    if success:
                        logger.info("✅ Model improvement cycle complete")
                        logger.info("Bot will now use improved model")
                    else:
                        logger.warning("⚠️  Retraining failed, will retry at next check")

                else:
                    logger.info("✓ No retraining needed at this time")

                # Sleep until next check
                logger.info(f"Next check in {check_interval/3600:.1f} hours...")
                time.sleep(check_interval)

            except KeyboardInterrupt:
                logger.info("\n👋 Auto-retrainer stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in auto-retrainer loop: {e}")
                time.sleep(3600)  # Wait 1 hour on error

if __name__ == "__main__":
    retrainer = AutoRetrainer()

    # Run with checks every 6 hours
    retrainer.run(check_interval=21600)
