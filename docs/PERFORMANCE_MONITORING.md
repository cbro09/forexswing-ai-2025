# Performance Monitoring & Analytics Framework

> **Objective**: Comprehensive performance tracking, analysis, and optimization for ForexSwing AI trading system

---

## 📊 **Performance Monitoring Architecture**

### **Real-Time Performance Dashboard**
```python
class RealTimePerformanceDashboard:
    def __init__(self):
        self.metrics = {
            'daily': {},
            'weekly': {},
            'monthly': {},
            'all_time': {}
        }
        self.alerts = []
        self.performance_thresholds = {
            'min_win_rate': 0.50,
            'max_drawdown': 0.15,
            'min_profit_factor': 1.2,
            'min_sharpe_ratio': 1.0
        }
    
    def update_performance_metrics(self, new_trade_data):
        """Update all performance metrics with new trade data"""
        
        # Calculate metrics for all timeframes
        for timeframe in ['daily', 'weekly', 'monthly', 'all_time']:
            trades = self._filter_trades_by_timeframe(new_trade_data, timeframe)
            self.metrics[timeframe] = self._calculate_metrics(trades)
        
        # Check for performance alerts
        self._check_performance_alerts()
        
        return self.get_dashboard_summary()
    
    def _calculate_metrics(self, trades):
        """Calculate comprehensive performance metrics"""
        if not trades:
            return self._empty_metrics()
            
        df = pd.DataFrame(trades)
        
        # Basic Performance
        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])
        
        # Profitability Metrics
        total_pnl = df['pnl'].sum()
        gross_profit = df[df['pnl'] > 0]['pnl'].sum()
        gross_loss = df[df['pnl'] < 0]['pnl'].sum()
        
        # Performance Ratios
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = abs(gross_loss / losing_trades) if losing_trades > 0 else 0
        profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float('inf')
        
        # Risk Metrics
        returns = df['pnl'] / df['account_balance']  # Assuming account_balance is tracked
        max_drawdown = self._calculate_max_drawdown(df)
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        
        # Consistency Metrics
        consecutive_wins = self._max_consecutive_wins(df)
        consecutive_losses = self._max_consecutive_losses(df)
        largest_win = df['pnl'].max()
        largest_loss = df['pnl'].min()
        
        return {
            # Basic Stats
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            
            # Profitability
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            
            # Ratios
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': (win_rate * avg_win) - ((1 - win_rate) * avg_loss),
            
            # Risk Metrics
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'volatility': returns.std() * np.sqrt(252) if len(returns) > 1 else 0,
            
            # Consistency
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            
            # Additional Metrics
            'recovery_factor': total_pnl / abs(max_drawdown) if max_drawdown != 0 else float('inf'),
            'calmar_ratio': (returns.mean() * 252) / abs(max_drawdown) if max_drawdown != 0 else float('inf'),
            'sterling_ratio': (returns.mean() * 252) / returns.std() if returns.std() != 0 else 0
        }
```

### **ML Model Performance Tracking**
```python
class MLModelPerformanceTracker:
    def __init__(self):
        self.prediction_history = []
        self.feature_importance_tracker = {}
        self.model_versions = {}
        self.confidence_calibration = {}
        
    def track_prediction(self, prediction_data):
        """Track ML model prediction and actual outcome"""
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'symbol': prediction_data['symbol'],
            'predicted_action': prediction_data['action'],
            'predicted_confidence': prediction_data['confidence'],
            'actual_outcome': None,  # To be filled when outcome is known
            'market_conditions': prediction_data.get('market_conditions', {}),
            'model_version': prediction_data.get('model_version', 'v1.0')
        })
    
    def update_prediction_outcome(self, prediction_id, actual_outcome):
        """Update prediction with actual market outcome"""
        if prediction_id < len(self.prediction_history):
            self.prediction_history[prediction_id]['actual_outcome'] = actual_outcome
            
    def calculate_ml_performance_metrics(self, lookback_days=30):
        """Calculate ML-specific performance metrics"""
        
        # Filter recent predictions with known outcomes
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_predictions = [
            p for p in self.prediction_history 
            if p['timestamp'] > cutoff_date and p['actual_outcome'] is not None
        ]
        
        if not recent_predictions:
            return None
            
        # Calculate accuracy metrics
        total_predictions = len(recent_predictions)
        correct_predictions = sum(1 for p in recent_predictions 
                                 if p['predicted_action'] == p['actual_outcome']['action'])
        
        accuracy = correct_predictions / total_predictions
        
        # Confidence calibration analysis
        confidence_buckets = {}
        for p in recent_predictions:
            conf_bucket = round(p['predicted_confidence'], 1)  # Round to nearest 0.1
            if conf_bucket not in confidence_buckets:
                confidence_buckets[conf_bucket] = {'total': 0, 'correct': 0}
            
            confidence_buckets[conf_bucket]['total'] += 1
            if p['predicted_action'] == p['actual_outcome']['action']:
                confidence_buckets[conf_bucket]['correct'] += 1
        
        # Calculate calibration for each confidence level
        calibration_data = {}
        for conf, data in confidence_buckets.items():
            calibration_data[conf] = {
                'predicted_accuracy': conf,
                'actual_accuracy': data['correct'] / data['total'],
                'sample_size': data['total'],
                'calibration_error': abs(conf - (data['correct'] / data['total']))
            }
        
        # Feature importance tracking (if available)
        feature_performance = self._analyze_feature_importance(recent_predictions)
        
        return {
            'overall_accuracy': accuracy,
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'confidence_calibration': calibration_data,
            'feature_performance': feature_performance,
            'model_drift_score': self._calculate_model_drift(),
            'prediction_distribution': self._analyze_prediction_distribution(recent_predictions)
        }
    
    def _calculate_model_drift(self):
        """Detect if model performance is degrading over time"""
        if len(self.prediction_history) < 100:
            return 0  # Not enough data
        
        # Compare recent performance to historical baseline
        recent_accuracy = self._calculate_accuracy(self.prediction_history[-50:])
        historical_accuracy = self._calculate_accuracy(self.prediction_history[-200:-50])
        
        drift_score = (historical_accuracy - recent_accuracy) / historical_accuracy
        return max(0, drift_score)  # Only positive drift (degradation)
```

### **Advanced Analytics Engine**
```python
class AdvancedAnalyticsEngine:
    def __init__(self):
        self.analytics_modules = {
            'market_regime_analysis': MarketRegimeAnalyzer(),
            'correlation_analysis': CorrelationAnalyzer(),
            'seasonality_analysis': SeasonalityAnalyzer(),
            'drawdown_analysis': DrawdownAnalyzer(),
            'trade_clustering': TradeClusteringAnalyzer()
        }
    
    def run_comprehensive_analysis(self, trade_data, market_data):
        """Run all analytics modules"""
        
        analysis_results = {}
        
        for module_name, analyzer in self.analytics_modules.items():
            try:
                analysis_results[module_name] = analyzer.analyze(trade_data, market_data)
            except Exception as e:
                analysis_results[module_name] = {'error': str(e)}
        
        # Generate insights and recommendations
        insights = self._generate_insights(analysis_results)
        recommendations = self._generate_recommendations(analysis_results)
        
        return {
            'detailed_analysis': analysis_results,
            'key_insights': insights,
            'optimization_recommendations': recommendations,
            'analysis_timestamp': datetime.now()
        }

class MarketRegimeAnalyzer:
    """Analyze performance across different market conditions"""
    
    def analyze(self, trade_data, market_data):
        # Classify market regimes
        regimes = self._classify_market_regimes(market_data)
        
        # Group trades by market regime
        regime_performance = {}
        for regime in ['trending_up', 'trending_down', 'ranging', 'volatile']:
            regime_trades = [trade for trade in trade_data 
                           if self._get_trade_regime(trade, regimes) == regime]
            
            if regime_trades:
                regime_performance[regime] = {
                    'trade_count': len(regime_trades),
                    'win_rate': len([t for t in regime_trades if t['pnl'] > 0]) / len(regime_trades),
                    'avg_pnl': np.mean([t['pnl'] for t in regime_trades]),
                    'total_pnl': sum([t['pnl'] for t in regime_trades])
                }
        
        return regime_performance
    
    def _classify_market_regimes(self, market_data):
        """Classify market conditions"""
        df = pd.DataFrame(market_data)
        
        # Calculate trend and volatility indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['volatility'] = df['close'].rolling(20).std()
        
        regimes = []
        for i in range(len(df)):
            if i < 50:  # Not enough data
                regimes.append('unknown')
                continue
            
            trend_strength = (df['sma_20'].iloc[i] - df['sma_50'].iloc[i]) / df['sma_50'].iloc[i]
            volatility_pct = df['volatility'].iloc[i] / df['close'].iloc[i]
            
            if volatility_pct > 0.02:  # High volatility threshold
                regime = 'volatile'
            elif trend_strength > 0.005:  # Strong uptrend
                regime = 'trending_up'
            elif trend_strength < -0.005:  # Strong downtrend
                regime = 'trending_down'
            else:
                regime = 'ranging'
            
            regimes.append(regime)
        
        return regimes

class DrawdownAnalyzer:
    """Analyze drawdown patterns and recovery times"""
    
    def analyze(self, trade_data, market_data):
        # Calculate equity curve
        equity_curve = []
        running_balance = 10000  # Starting balance
        
        for trade in sorted(trade_data, key=lambda x: x['timestamp']):
            running_balance += trade['pnl']
            equity_curve.append({
                'timestamp': trade['timestamp'],
                'balance': running_balance,
                'trade_pnl': trade['pnl']
            })
        
        # Identify drawdown periods
        drawdowns = self._identify_drawdown_periods(equity_curve)
        
        # Analyze drawdown characteristics
        drawdown_analysis = {
            'total_drawdowns': len(drawdowns),
            'avg_drawdown_duration': np.mean([dd['duration_days'] for dd in drawdowns]) if drawdowns else 0,
            'max_drawdown_duration': max([dd['duration_days'] for dd in drawdowns]) if drawdowns else 0,
            'avg_recovery_time': np.mean([dd['recovery_days'] for dd in drawdowns if dd['recovered']]) if drawdowns else 0,
            'recovery_rate': len([dd for dd in drawdowns if dd['recovered']]) / len(drawdowns) if drawdowns else 1,
            'drawdown_frequency': len(drawdowns) / max(1, len(equity_curve) / 30),  # Drawdowns per month
            'deepest_drawdown': min([dd['max_drawdown_pct'] for dd in drawdowns]) if drawdowns else 0
        }
        
        return drawdown_analysis
```

### **Performance Optimization Recommendations**
```python
class PerformanceOptimizer:
    def __init__(self):
        self.optimization_strategies = {
            'threshold_optimization': ThresholdOptimizer(),
            'position_sizing_optimization': PositionSizingOptimizer(),
            'market_timing_optimization': MarketTimingOptimizer(),
            'feature_optimization': FeatureOptimizer()
        }
    
    def generate_optimization_recommendations(self, performance_data):
        """Generate specific recommendations for improvement"""
        
        recommendations = []
        
        # Analyze current performance weaknesses
        weaknesses = self._identify_performance_weaknesses(performance_data)
        
        for weakness in weaknesses:
            if weakness['type'] == 'low_win_rate':
                recommendations.extend(self._recommend_win_rate_improvements(weakness))
            elif weakness['type'] == 'high_drawdown':
                recommendations.extend(self._recommend_drawdown_reduction(weakness))
            elif weakness['type'] == 'poor_risk_reward':
                recommendations.extend(self._recommend_risk_reward_improvements(weakness))
            elif weakness['type'] == 'inconsistent_performance':
                recommendations.extend(self._recommend_consistency_improvements(weakness))
        
        # Prioritize recommendations by potential impact
        prioritized_recommendations = self._prioritize_recommendations(recommendations)
        
        return prioritized_recommendations
    
    def _recommend_win_rate_improvements(self, weakness_data):
        """Recommend specific actions to improve win rate"""
        recommendations = []
        
        if weakness_data['current_win_rate'] < 0.45:
            recommendations.append({
                'type': 'threshold_adjustment',
                'description': 'Increase confidence thresholds to reduce false signals',
                'expected_impact': 'Win rate +5-8%',
                'implementation_difficulty': 'Easy',
                'specific_actions': [
                    'Increase BUY threshold from 0.25 to 0.35',
                    'Increase SELL threshold from 0.20 to 0.30',
                    'Add trend confirmation requirement'
                ],
                'risk': 'May reduce trading frequency',
                'priority': 'High'
            })
        
        if weakness_data.get('poor_market_timing', False):
            recommendations.append({
                'type': 'market_timing',
                'description': 'Avoid trading during high-volatility periods',
                'expected_impact': 'Win rate +3-5%',
                'implementation_difficulty': 'Medium',
                'specific_actions': [
                    'Skip trades 1 hour before/after major news releases',
                    'Reduce position sizes during high volatility periods',
                    'Implement market regime detection'
                ],
                'risk': 'May miss some profitable opportunities',
                'priority': 'Medium'
            })
        
        return recommendations
```

### **Automated Performance Alerts**
```python
class PerformanceAlertSystem:
    def __init__(self):
        self.alert_rules = {
            'critical': [
                {'metric': 'daily_loss', 'threshold': -0.05, 'message': 'Daily loss exceeds 5%'},
                {'metric': 'drawdown', 'threshold': 0.15, 'message': 'Drawdown exceeds 15%'},
                {'metric': 'consecutive_losses', 'threshold': 7, 'message': '7 consecutive losses'}
            ],
            'warning': [
                {'metric': 'win_rate_7d', 'threshold': 0.40, 'message': '7-day win rate below 40%'},
                {'metric': 'profit_factor_30d', 'threshold': 1.0, 'message': '30-day profit factor below 1.0'},
                {'metric': 'sharpe_ratio_30d', 'threshold': 0.5, 'message': '30-day Sharpe ratio below 0.5'}
            ],
            'info': [
                {'metric': 'new_equity_high', 'threshold': None, 'message': 'New equity high reached'},
                {'metric': 'model_accuracy_change', 'threshold': 0.05, 'message': 'Model accuracy changed >5%'}
            ]
        }
        self.alert_history = []
        self.notification_channels = ['email', 'sms', 'dashboard']
    
    def check_alerts(self, current_metrics):
        """Check all alert conditions"""
        new_alerts = []
        
        for severity, rules in self.alert_rules.items():
            for rule in rules:
                if self._should_trigger_alert(rule, current_metrics):
                    alert = self._create_alert(rule, current_metrics, severity)
                    new_alerts.append(alert)
                    self._send_notifications(alert)
        
        self.alert_history.extend(new_alerts)
        return new_alerts
    
    def _create_alert(self, rule, metrics, severity):
        """Create alert object"""
        return {
            'timestamp': datetime.now(),
            'severity': severity,
            'metric': rule['metric'],
            'threshold': rule['threshold'],
            'current_value': metrics.get(rule['metric']),
            'message': rule['message'],
            'recommendation': self._get_alert_recommendation(rule['metric'], severity)
        }
    
    def _get_alert_recommendation(self, metric, severity):
        """Get specific recommendation for each alert type"""
        recommendations = {
            'daily_loss': 'Stop trading immediately and review risk management',
            'drawdown': 'Reduce position sizes and review strategy',
            'consecutive_losses': 'Take a break and analyze recent trades',
            'win_rate_7d': 'Review recent trades for pattern of failures',
            'profit_factor_30d': 'Analyze risk/reward ratio and adjust targets'
        }
        
        return recommendations.get(metric, 'Review performance and consider adjustments')
```

## 📱 **Performance Dashboard Interface**

### **Real-Time Dashboard Layout**
```python
class PerformanceDashboard:
    def generate_dashboard_layout(self):
        """Generate dashboard structure"""
        return {
            'overview_panel': {
                'today_pnl': {'value': 0, 'color': 'green'},
                'week_pnl': {'value': 0, 'color': 'green'},
                'month_pnl': {'value': 0, 'color': 'green'},
                'total_pnl': {'value': 0, 'color': 'green'},
                'current_drawdown': {'value': 0, 'color': 'yellow'},
                'active_positions': {'value': 0, 'color': 'blue'}
            },
            
            'performance_metrics': {
                'win_rate': {'current': 0.55, 'target': 0.52, 'trend': 'up'},
                'profit_factor': {'current': 1.3, 'target': 1.2, 'trend': 'stable'},
                'sharpe_ratio': {'current': 1.1, 'target': 1.0, 'trend': 'up'},
                'max_drawdown': {'current': 0.08, 'target': 0.15, 'trend': 'down'}
            },
            
            'charts': {
                'equity_curve': self._generate_equity_curve_data(),
                'daily_pnl': self._generate_daily_pnl_data(),
                'drawdown_chart': self._generate_drawdown_data(),
                'win_rate_trend': self._generate_win_rate_trend()
            },
            
            'recent_trades': self._get_recent_trades_summary(),
            'active_alerts': self._get_active_alerts(),
            'model_performance': self._get_ml_performance_summary()
        }
```

## 🎯 **Implementation Priority**

### **Phase 1: Core Monitoring (Week 1-2)**
- Real-time performance tracking
- Basic metrics calculation
- Simple dashboard interface

### **Phase 2: Advanced Analytics (Week 3-4)**
- ML model performance tracking
- Market regime analysis
- Drawdown analysis

### **Phase 3: Optimization & Alerts (Week 5-6)**
- Performance optimization recommendations
- Automated alert system
- Advanced dashboard features

---

This comprehensive monitoring system will provide deep insights into your trading performance and help optimize the ForexSwing AI system continuously.