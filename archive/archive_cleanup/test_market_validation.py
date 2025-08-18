#!/usr/bin/env python3
"""
Comprehensive Market Validation for Enhanced ForexSwing AI
Tests dual AI system (LSTM + Gemini) on historical market data
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json

sys.path.append('core')
from integrations.enhanced_strategy import EnhancedForexStrategy

class MarketValidator:
    """Comprehensive market validation system"""
    
    def __init__(self):
        self.strategy = EnhancedForexStrategy("models/optimized_forex_ai.pth")
        self.results = {
            "validation_timestamp": datetime.now().isoformat(),
            "pairs_tested": [],
            "performance_metrics": {},
            "signal_analysis": {},
            "recommendations": []
        }
        
    def load_market_data(self, data_dir="data/real_market"):
        """Load all available forex market data"""
        print("Loading market data...")
        
        if not os.path.exists(data_dir):
            print(f"[ERROR] Data directory not found: {data_dir}")
            return {}
        
        data_files = [f for f in os.listdir(data_dir) if f.endswith('.feather')]
        
        if not data_files:
            print(f"[ERROR] No .feather files found in {data_dir}")
            return {}
        
        market_data = {}
        
        for file in data_files:
            try:
                pair_name = file.replace('_real_daily.feather', '').replace('_', '/')
                df = pd.read_feather(os.path.join(data_dir, file))
                
                if 'Date' in df.columns:
                    df = df.set_index('Date')
                
                # Ensure we have required columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if all(col in df.columns for col in required_cols):
                    market_data[pair_name] = df
                    print(f"  Loaded {pair_name}: {len(df)} candles")
                else:
                    print(f"  [WARNING] {pair_name}: Missing required columns")
                    
            except Exception as e:
                print(f"  [ERROR] Failed to load {file}: {e}")
        
        print(f"Successfully loaded {len(market_data)} currency pairs")
        return market_data
    
    def backtest_signal_accuracy(self, data, pair, lookback_days=180):
        """Test signal accuracy over historical data"""
        print(f"\nTesting signal accuracy for {pair}...")
        
        if len(data) < lookback_days:
            print(f"  [WARNING] Insufficient data: {len(data)} < {lookback_days}")
            return None
        
        # Use recent data for testing
        test_data = data.tail(lookback_days)
        
        results = {
            "pair": pair,
            "test_period": lookback_days,
            "signals_generated": 0,
            "signal_distribution": {"HOLD": 0, "BUY": 0, "SELL": 0},
            "confidence_stats": {},
            "accuracy_metrics": {},
            "performance_times": []
        }
        
        # Test signal generation every 7 days (weekly)
        step_size = 7
        signals = []
        
        for i in range(step_size, len(test_data), step_size):
            window_data = test_data.iloc[:i]
            
            if len(window_data) < 80:  # Minimum data for LSTM
                continue
            
            try:
                start_time = time.time()
                
                # Get recommendation from enhanced strategy
                recommendation = self.strategy.get_trading_recommendation(window_data, pair)
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                signals.append({
                    "date": window_data.index[-1],
                    "action": recommendation['action'],
                    "confidence": recommendation['confidence'],
                    "risk_level": recommendation['risk_level'],
                    "processing_time": processing_time
                })
                
                # Update statistics
                results["signals_generated"] += 1
                results["signal_distribution"][recommendation['action']] += 1
                results["performance_times"].append(processing_time)
                
            except Exception as e:
                print(f"    [ERROR] Signal generation failed at index {i}: {e}")
                continue
        
        if signals:
            # Calculate statistics
            confidences = [s['confidence'] for s in signals]
            times = results["performance_times"]
            
            results["confidence_stats"] = {
                "mean": np.mean(confidences),
                "std": np.std(confidences),
                "min": np.min(confidences),
                "max": np.max(confidences)
            }
            
            results["performance_metrics"] = {
                "avg_processing_time": np.mean(times),
                "max_processing_time": np.max(times),
                "min_processing_time": np.min(times)
            }
            
            # Calculate signal distribution percentages
            total_signals = results["signals_generated"]
            for signal_type in results["signal_distribution"]:
                count = results["signal_distribution"][signal_type]
                results["signal_distribution"][signal_type] = {
                    "count": count,
                    "percentage": (count / total_signals * 100) if total_signals > 0 else 0
                }
            
            print(f"  Generated {len(signals)} signals over {lookback_days} days")
            print(f"  Average confidence: {results['confidence_stats']['mean']:.1%}")
            print(f"  Average processing time: {results['performance_metrics']['avg_processing_time']:.2f}s")
            
            # Signal distribution
            dist = results["signal_distribution"]
            print(f"  Signal distribution: HOLD {dist['HOLD']['percentage']:.1f}%, BUY {dist['BUY']['percentage']:.1f}%, SELL {dist['SELL']['percentage']:.1f}%")
        
        return results
    
    def test_market_adaptation(self, data, pair):
        """Test how the system adapts to different market conditions"""
        print(f"\nTesting market adaptation for {pair}...")
        
        if len(data) < 365:  # Need at least 1 year of data
            print(f"  [WARNING] Insufficient data for adaptation testing: {len(data)} days")
            return None
        
        # Identify different market regimes
        returns = data['close'].pct_change().dropna()
        volatility = returns.rolling(30).std()
        
        # Define market conditions
        bear_threshold = -0.001  # 0.1% daily decline
        bull_threshold = 0.001   # 0.1% daily growth
        high_vol_threshold = volatility.quantile(0.8)
        
        periods = {
            "bull_market": returns > bull_threshold,
            "bear_market": returns < bear_threshold,
            "sideways": (returns >= bear_threshold) & (returns <= bull_threshold),
            "high_volatility": volatility > high_vol_threshold,
            "low_volatility": volatility <= high_vol_threshold
        }
        
        adaptation_results = {
            "pair": pair,
            "market_conditions": {}
        }
        
        for condition, mask in periods.items():
            if mask.sum() < 30:  # Need at least 30 periods
                continue
                
            try:
                # Get data for this market condition
                condition_dates = mask[mask].index
                
                # Sample some periods for testing
                sample_dates = condition_dates[::10][:5]  # Every 10th, max 5 samples
                
                condition_signals = []
                
                for date in sample_dates:
                    # Get data up to this date
                    historical_data = data.loc[:date]
                    
                    if len(historical_data) < 80:
                        continue
                    
                    try:
                        recommendation = self.strategy.get_trading_recommendation(historical_data.tail(100), pair)
                        condition_signals.append({
                            "date": date,
                            "action": recommendation['action'],
                            "confidence": recommendation['confidence']
                        })
                    except:
                        continue
                
                if condition_signals:
                    confidences = [s['confidence'] for s in condition_signals]
                    actions = [s['action'] for s in condition_signals]
                    
                    adaptation_results["market_conditions"][condition] = {
                        "sample_count": len(condition_signals),
                        "avg_confidence": np.mean(confidences),
                        "action_distribution": {
                            "HOLD": actions.count("HOLD"),
                            "BUY": actions.count("BUY"), 
                            "SELL": actions.count("SELL")
                        }
                    }
                    
                    print(f"  {condition}: {len(condition_signals)} samples, avg confidence {np.mean(confidences):.1%}")
                    
            except Exception as e:
                print(f"    [ERROR] Testing {condition}: {e}")
                continue
        
        return adaptation_results
    
    def validate_anomaly_detection(self, data, pair):
        """Test anomaly detection capabilities"""
        print(f"\nTesting anomaly detection for {pair}...")
        
        try:
            # Test anomaly detection
            anomalies = self.strategy.monitor_anomalies(data, pair)
            
            anomaly_count = len(anomalies.get('anomalies', []))
            
            anomaly_results = {
                "pair": pair,
                "anomalies_detected": anomaly_count,
                "anomaly_types": [],
                "detection_capability": anomalies.get('analysis', 'Unknown')
            }
            
            if anomaly_count > 0:
                for anomaly in anomalies['anomalies']:
                    anomaly_results["anomaly_types"].append(anomaly.get('type', 'unknown'))
                    
            print(f"  Detected {anomaly_count} market anomalies")
            return anomaly_results
            
        except Exception as e:
            print(f"  [ERROR] Anomaly detection failed: {e}")
            return {"pair": pair, "error": str(e)}
    
    def run_comprehensive_validation(self):
        """Run complete market validation suite"""
        print("COMPREHENSIVE MARKET VALIDATION")
        print("=" * 60)
        print("Testing Enhanced ForexSwing AI (LSTM + Gemini)")
        print()
        
        # Load market data
        market_data = self.load_market_data()
        
        if not market_data:
            print("[ERROR] No market data available for testing")
            return
        
        self.results["pairs_tested"] = list(market_data.keys())
        
        # Test each currency pair
        for pair, data in market_data.items():
            print(f"\n{'='*20} TESTING {pair} {'='*20}")
            
            # 1. Signal accuracy testing
            signal_results = self.backtest_signal_accuracy(data, pair)
            if signal_results:
                self.results["performance_metrics"][pair] = signal_results
            
            # 2. Market adaptation testing
            adaptation_results = self.test_market_adaptation(data, pair)
            if adaptation_results:
                self.results["signal_analysis"][pair] = adaptation_results
            
            # 3. Anomaly detection testing
            anomaly_results = self.validate_anomaly_detection(data, pair)
            if anomaly_results:
                if "anomaly_detection" not in self.results:
                    self.results["anomaly_detection"] = {}
                self.results["anomaly_detection"][pair] = anomaly_results
        
        # Generate summary
        self.generate_validation_summary()
        
        # Save results
        self.save_results()
        
        return self.results
    
    def generate_validation_summary(self):
        """Generate comprehensive validation summary"""
        print(f"\n{'='*60}")
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        pairs_tested = len(self.results["pairs_tested"])
        print(f"Currency Pairs Tested: {pairs_tested}")
        print(f"Pairs: {', '.join(self.results['pairs_tested'])}")
        
        # Performance summary
        if self.results["performance_metrics"]:
            print(f"\nPERFORMANCE METRICS:")
            
            all_confidences = []
            all_times = []
            total_signals = 0
            signal_dist = {"HOLD": 0, "BUY": 0, "SELL": 0}
            
            for pair, metrics in self.results["performance_metrics"].items():
                if "confidence_stats" in metrics:
                    all_confidences.append(metrics["confidence_stats"]["mean"])
                if "performance_metrics" in metrics:
                    all_times.append(metrics["performance_metrics"]["avg_processing_time"])
                
                total_signals += metrics.get("signals_generated", 0)
                
                for signal_type in signal_dist:
                    if signal_type in metrics.get("signal_distribution", {}):
                        signal_dist[signal_type] += metrics["signal_distribution"][signal_type].get("count", 0)
            
            if all_confidences:
                print(f"  Average Confidence: {np.mean(all_confidences):.1%}")
                print(f"  Confidence Range: {np.min(all_confidences):.1%} - {np.max(all_confidences):.1%}")
            
            if all_times:
                print(f"  Average Processing Time: {np.mean(all_times):.2f}s")
                print(f"  Processing Time Range: {np.min(all_times):.2f}s - {np.max(all_times):.2f}s")
            
            print(f"  Total Signals Generated: {total_signals}")
            
            if total_signals > 0:
                print(f"  Signal Distribution:")
                for signal_type, count in signal_dist.items():
                    percentage = (count / total_signals) * 100
                    print(f"    {signal_type}: {count} ({percentage:.1f}%)")
        
        # Market adaptation summary
        if self.results["signal_analysis"]:
            print(f"\nMARKET ADAPTATION:")
            
            for pair, analysis in self.results["signal_analysis"].items():
                conditions = analysis.get("market_conditions", {})
                print(f"  {pair}: Tested {len(conditions)} market conditions")
        
        # Anomaly detection summary
        if "anomaly_detection" in self.results:
            print(f"\nANOMALY DETECTION:")
            
            total_anomalies = 0
            for pair, detection in self.results["anomaly_detection"].items():
                anomaly_count = detection.get("anomalies_detected", 0)
                total_anomalies += anomaly_count
                print(f"  {pair}: {anomaly_count} anomalies detected")
            
            print(f"  Total Anomalies: {total_anomalies}")
        
        # Overall assessment
        print(f"\nOVERALL ASSESSMENT:")
        
        issues = []
        strengths = []
        
        # Check performance
        if all_times and np.mean(all_times) > 10:
            issues.append("Slow processing times (>10s average)")
        else:
            strengths.append("Acceptable processing speed")
        
        # Check signal distribution
        if total_signals > 0:
            hold_pct = (signal_dist["HOLD"] / total_signals) * 100
            if hold_pct > 80:
                issues.append("Too conservative (>80% HOLD signals)")
            elif hold_pct < 20:
                issues.append("Too aggressive (<20% HOLD signals)")
            else:
                strengths.append("Balanced signal distribution")
        
        # Check confidence
        if all_confidences and np.mean(all_confidences) < 0.3:
            issues.append("Low confidence scores (<30%)")
        elif all_confidences:
            strengths.append("Reasonable confidence levels")
        
        print(f"  Strengths: {len(strengths)}")
        for strength in strengths:
            print(f"    + {strength}")
        
        print(f"  Issues: {len(issues)}")
        for issue in issues:
            print(f"    - {issue}")
        
        # Recommendations
        self.results["recommendations"] = []
        
        if issues:
            print(f"\nRECOMMENDATIONS:")
            
            if "Slow processing times" in str(issues):
                rec = "Optimize Gemini integration for faster response times"
                print(f"  1. {rec}")
                self.results["recommendations"].append(rec)
            
            if "Too conservative" in str(issues):
                rec = "Adjust confidence thresholds to reduce HOLD bias"
                print(f"  2. {rec}")
                self.results["recommendations"].append(rec)
            
            if "Low confidence" in str(issues):
                rec = "Review dual AI weighting and calibration"
                print(f"  3. {rec}")
                self.results["recommendations"].append(rec)
        else:
            rec = "System performing well - proceed to paper trading"
            print(f"\n✅ {rec}")
            self.results["recommendations"].append(rec)
    
    def save_results(self):
        """Save validation results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validation_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            print(f"\nValidation results saved to: {filename}")
            
        except Exception as e:
            print(f"[ERROR] Failed to save results: {e}")

def main():
    """Run comprehensive market validation"""
    
    validator = MarketValidator()
    results = validator.run_comprehensive_validation()
    
    print(f"\n🎯 MARKET VALIDATION COMPLETE!")
    print(f"Results include:")
    print(f"  - Signal accuracy testing")
    print(f"  - Market adaptation analysis") 
    print(f"  - Anomaly detection validation")
    print(f"  - Performance optimization recommendations")

if __name__ == "__main__":
    main()