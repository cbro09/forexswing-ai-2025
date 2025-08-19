# Risk Management Framework - Professional Trading System

> **Mission**: Protect capital while maximizing risk-adjusted returns through systematic risk control

---

## 🎯 **Core Risk Management Principles**

### **The 4 Pillars of Trading Risk Management**

#### **1. Capital Preservation** ⭐⭐⭐
- **Never lose more than you can afford**
- **Protect trading capital at all costs**
- **Survival is more important than profits**

#### **2. Position Sizing** ⭐⭐⭐
- **Risk a fixed percentage per trade**
- **Scale position size with account balance**
- **Account for volatility in position sizing**

#### **3. Diversification** ⭐⭐
- **Don't put all eggs in one basket**
- **Spread risk across time and instruments**
- **Avoid correlated positions**

#### **4. Systematic Approach** ⭐⭐
- **Follow rules, not emotions**
- **Consistent application of risk rules**
- **Continuous monitoring and adjustment**

---

## 📊 **Risk Management Hierarchy**

### **Level 1: Account-Level Risk Management**

#### **Maximum Account Drawdown**
```python
class AccountRiskManager:
    def __init__(self, max_account_drawdown=0.20):  # 20% max drawdown
        self.max_account_drawdown = max_account_drawdown
        self.peak_equity = 0
        self.current_equity = 0
        
    def update_equity(self, current_equity):
        self.current_equity = current_equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            
    def get_current_drawdown(self):
        if self.peak_equity == 0:
            return 0
        return (self.peak_equity - self.current_equity) / self.peak_equity
    
    def is_drawdown_limit_exceeded(self):
        return self.get_current_drawdown() > self.max_account_drawdown
    
    def get_max_additional_risk(self):
        """How much more can we risk before hitting drawdown limit"""
        current_dd = self.get_current_drawdown()
        remaining_dd = self.max_account_drawdown - current_dd
        return remaining_dd * self.peak_equity
```

#### **Daily Risk Limits**
```python
class DailyRiskManager:
    def __init__(self, max_daily_loss=0.03):  # 3% max daily loss
        self.max_daily_loss = max_daily_loss
        self.daily_pnl = 0
        self.trading_day_start = None
        self.reset_daily_pnl()
        
    def reset_daily_pnl(self):
        """Reset daily P&L at start of trading day"""
        from datetime import datetime
        current_time = datetime.now()
        
        # Reset if new trading day (after 17:00 EST when forex markets close)
        if (self.trading_day_start is None or 
            current_time.date() > self.trading_day_start.date()):
            self.daily_pnl = 0
            self.trading_day_start = current_time
    
    def add_trade_pnl(self, pnl):
        """Add trade P&L to daily total"""
        self.reset_daily_pnl()  # Check if new day
        self.daily_pnl += pnl
    
    def can_take_more_risk(self, account_balance):
        """Check if more risk can be taken today"""
        max_loss_amount = account_balance * self.max_daily_loss
        return self.daily_pnl > -max_loss_amount
    
    def remaining_daily_risk(self, account_balance):
        """How much more can we risk today"""
        max_loss_amount = account_balance * self.max_daily_loss
        used_risk = max(0, -self.daily_pnl)  # Only count losses
        return max(0, max_loss_amount - used_risk)
```

### **Level 2: Position-Level Risk Management**

#### **Position Sizing (Kelly Criterion + Fixed Fractional)**
```python
class PositionSizingManager:
    def __init__(self, base_risk_per_trade=0.01, max_risk_per_trade=0.02):
        self.base_risk_per_trade = base_risk_per_trade  # 1% base risk
        self.max_risk_per_trade = max_risk_per_trade    # 2% max risk
        
    def calculate_position_size(self, account_balance, stop_loss_pips, 
                               pip_value, confidence=0.5, win_rate=0.5):
        """
        Calculate position size using multiple methods
        
        Args:
            account_balance: Current account balance
            stop_loss_pips: Stop loss in pips
            pip_value: Value per pip for the instrument
            confidence: AI confidence (0.5-1.0)
            win_rate: Historical win rate (0-1.0)
        """
        
        # Method 1: Fixed Fractional (Conservative baseline)
        risk_amount_fixed = account_balance * self.base_risk_per_trade
        position_size_fixed = risk_amount_fixed / (stop_loss_pips * pip_value)
        
        # Method 2: Confidence-based scaling
        confidence_multiplier = self._confidence_to_multiplier(confidence)
        risk_amount_confidence = account_balance * self.base_risk_per_trade * confidence_multiplier
        position_size_confidence = risk_amount_confidence / (stop_loss_pips * pip_value)
        
        # Method 3: Kelly Criterion (if we have good win rate data)
        if win_rate > 0 and win_rate < 1:
            kelly_fraction = self._calculate_kelly_fraction(win_rate)
            kelly_fraction = min(kelly_fraction, self.max_risk_per_trade)  # Cap at max risk
            risk_amount_kelly = account_balance * kelly_fraction
            position_size_kelly = risk_amount_kelly / (stop_loss_pips * pip_value)
        else:
            position_size_kelly = position_size_fixed
        
        # Use the most conservative of the three methods
        final_position_size = min(position_size_fixed, position_size_confidence, position_size_kelly)
        
        return {
            'position_size': final_position_size,
            'risk_amount': final_position_size * stop_loss_pips * pip_value,
            'risk_percentage': (final_position_size * stop_loss_pips * pip_value) / account_balance,
            'method_used': self._determine_method_used(position_size_fixed, position_size_confidence, position_size_kelly),
            'calculations': {
                'fixed_fractional': position_size_fixed,
                'confidence_based': position_size_confidence,
                'kelly_criterion': position_size_kelly
            }
        }
    
    def _confidence_to_multiplier(self, confidence):
        """Convert AI confidence to risk multiplier"""
        if confidence < 0.6:
            return 0.5  # Reduce risk for low confidence
        elif confidence < 0.7:
            return 0.8
        elif confidence < 0.8:
            return 1.0  # Normal risk
        elif confidence < 0.9:
            return 1.2
        else:
            return 1.5  # Increase risk for very high confidence (capped by max_risk)
    
    def _calculate_kelly_fraction(self, win_rate, avg_win=1.0, avg_loss=1.0):
        """Calculate Kelly Criterion fraction"""
        # Kelly = (bp - q) / b
        # where b = odds received (avg_win/avg_loss), p = win probability, q = loss probability
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        return max(0, kelly_fraction)  # Never negative
    
    def _determine_method_used(self, fixed, confidence, kelly):
        """Determine which method gave the final result"""
        final = min(fixed, confidence, kelly)
        if final == fixed:
            return "fixed_fractional"
        elif final == confidence:
            return "confidence_based"
        else:
            return "kelly_criterion"
```

#### **Stop Loss Management**
```python
class StopLossManager:
    def __init__(self):
        self.stop_loss_methods = {
            'fixed_pips': self._fixed_pip_stop,
            'atr_based': self._atr_based_stop,
            'support_resistance': self._support_resistance_stop,
            'volatility_adjusted': self._volatility_adjusted_stop
        }
    
    def calculate_stop_loss(self, entry_price, direction, market_data, method='volatility_adjusted'):
        """Calculate stop loss using specified method"""
        return self.stop_loss_methods[method](entry_price, direction, market_data)
    
    def _fixed_pip_stop(self, entry_price, direction, market_data, pips=20):
        """Fixed pip-based stop loss"""
        pip_size = 0.0001  # For major pairs
        if direction == 'BUY':
            return entry_price - (pips * pip_size)
        else:  # SELL
            return entry_price + (pips * pip_size)
    
    def _atr_based_stop(self, entry_price, direction, market_data, atr_multiplier=2.0):
        """ATR-based dynamic stop loss"""
        # Calculate ATR (Average True Range)
        high = market_data['high'].tail(20)
        low = market_data['low'].tail(20)
        close = market_data['close'].tail(20)
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.mean()
        
        stop_distance = atr * atr_multiplier
        
        if direction == 'BUY':
            return entry_price - stop_distance
        else:  # SELL
            return entry_price + stop_distance
    
    def _volatility_adjusted_stop(self, entry_price, direction, market_data, base_pips=20, vol_multiplier=1.5):
        """Volatility-adjusted stop loss"""
        # Calculate recent volatility
        returns = market_data['close'].pct_change().tail(20)
        volatility = returns.std()
        
        # Adjust base stop loss by volatility
        pip_size = 0.0001
        volatility_adjustment = volatility * vol_multiplier * 10000  # Convert to pips
        adjusted_pips = base_pips + volatility_adjustment
        
        if direction == 'BUY':
            return entry_price - (adjusted_pips * pip_size)
        else:  # SELL
            return entry_price + (adjusted_pips * pip_size)
```

### **Level 3: Portfolio-Level Risk Management**

#### **Correlation Management**
```python
class CorrelationManager:
    def __init__(self, max_correlated_positions=2, correlation_threshold=0.7):
        self.max_correlated_positions = max_correlated_positions
        self.correlation_threshold = correlation_threshold
        
        # Currency correlation matrix (simplified)
        self.correlations = {
            ('EUR/USD', 'GBP/USD'): 0.75,
            ('EUR/USD', 'AUD/USD'): 0.68,
            ('USD/JPY', 'USD/CHF'): 0.82,
            ('GBP/USD', 'EUR/GBP'): -0.85,
            # Add more correlations...
        }
    
    def check_correlation_risk(self, new_symbol, current_positions):
        """Check if new position would violate correlation limits"""
        correlated_count = 0
        
        for position in current_positions:
            correlation = self._get_correlation(new_symbol, position['symbol'])
            if abs(correlation) > self.correlation_threshold:
                correlated_count += 1
        
        return correlated_count < self.max_correlated_positions
    
    def _get_correlation(self, symbol1, symbol2):
        """Get correlation between two symbols"""
        pair = (symbol1, symbol2)
        reverse_pair = (symbol2, symbol1)
        
        if pair in self.correlations:
            return self.correlations[pair]
        elif reverse_pair in self.correlations:
            return self.correlations[reverse_pair]
        else:
            return 0.0  # Assume no correlation if not in matrix
```

#### **Heat Map Risk Monitoring**
```python
class RiskHeatMapMonitor:
    def __init__(self):
        self.risk_categories = {
            'position_size': {'low': 0.005, 'medium': 0.01, 'high': 0.02},
            'correlation': {'low': 0.3, 'medium': 0.6, 'high': 0.8},
            'drawdown': {'low': 0.05, 'medium': 0.10, 'high': 0.15},
            'volatility': {'low': 0.01, 'medium': 0.02, 'high': 0.04}
        }
    
    def generate_risk_heat_map(self, portfolio_state):
        """Generate risk heat map for current portfolio"""
        heat_map = {}
        
        for category, thresholds in self.risk_categories.items():
            current_value = self._get_current_value(category, portfolio_state)
            risk_level = self._categorize_risk(current_value, thresholds)
            
            heat_map[category] = {
                'value': current_value,
                'risk_level': risk_level,
                'color': self._get_risk_color(risk_level)
            }
        
        return heat_map
    
    def _categorize_risk(self, value, thresholds):
        """Categorize risk level"""
        if value <= thresholds['low']:
            return 'low'
        elif value <= thresholds['medium']:
            return 'medium'
        else:
            return 'high'
    
    def _get_risk_color(self, risk_level):
        """Get color code for risk level"""
        colors = {'low': 'green', 'medium': 'yellow', 'high': 'red'}
        return colors[risk_level]
```

---

## 🚨 **Emergency Risk Controls**

### **Circuit Breakers**
```python
class CircuitBreakerSystem:
    def __init__(self):
        self.breakers = {
            'daily_loss': {'threshold': 0.05, 'duration_minutes': 1440},  # 5% daily loss = 24h break
            'rapid_loss': {'threshold': 0.02, 'duration_minutes': 60},    # 2% rapid loss = 1h break
            'consecutive_losses': {'threshold': 5, 'duration_minutes': 120}, # 5 losses = 2h break
            'system_error': {'threshold': 3, 'duration_minutes': 30}      # 3 errors = 30min break
        }
        self.breaker_states = {name: {'active': False, 'activated_at': None} 
                              for name in self.breakers.keys()}
    
    def check_breakers(self, trading_state):
        """Check if any circuit breakers should be triggered"""
        for breaker_name, config in self.breakers.items():
            if self._should_trigger_breaker(breaker_name, config, trading_state):
                self._activate_breaker(breaker_name, config)
        
        return self._get_active_breakers()
    
    def _should_trigger_breaker(self, breaker_name, config, trading_state):
        """Check if specific breaker should trigger"""
        if breaker_name == 'daily_loss':
            return trading_state['daily_pnl_pct'] <= -config['threshold']
        elif breaker_name == 'rapid_loss':
            return trading_state['hour_pnl_pct'] <= -config['threshold']
        elif breaker_name == 'consecutive_losses':
            return trading_state['consecutive_losses'] >= config['threshold']
        elif breaker_name == 'system_error':
            return trading_state['recent_errors'] >= config['threshold']
        
        return False
    
    def _activate_breaker(self, breaker_name, config):
        """Activate circuit breaker"""
        self.breaker_states[breaker_name] = {
            'active': True,
            'activated_at': datetime.now(),
            'duration_minutes': config['duration_minutes']
        }
        
        print(f"🚨 CIRCUIT BREAKER ACTIVATED: {breaker_name}")
        print(f"Trading suspended for {config['duration_minutes']} minutes")
    
    def is_trading_allowed(self):
        """Check if trading is allowed (no active breakers)"""
        current_time = datetime.now()
        
        for breaker_name, state in self.breaker_states.items():
            if state['active']:
                # Check if breaker should be deactivated
                duration = current_time - state['activated_at']
                if duration.total_seconds() < state['duration_minutes'] * 60:
                    return False  # Still active
                else:
                    # Deactivate expired breaker
                    self.breaker_states[breaker_name]['active'] = False
        
        return True  # No active breakers
```

---

## 📈 **Risk Metrics & Monitoring**

### **Key Risk Metrics**
```python
class RiskMetricsCalculator:
    def __init__(self):
        self.metrics = {}
    
    def calculate_comprehensive_metrics(self, trade_history, current_positions):
        """Calculate comprehensive risk metrics"""
        
        # Convert trade history to DataFrame for easier calculations
        df = pd.DataFrame(trade_history)
        
        metrics = {}
        
        # Performance Metrics
        metrics['total_return'] = df['pnl'].sum()
        metrics['win_rate'] = len(df[df['pnl'] > 0]) / len(df) if len(df) > 0 else 0
        metrics['avg_win'] = df[df['pnl'] > 0]['pnl'].mean() if len(df[df['pnl'] > 0]) > 0 else 0
        metrics['avg_loss'] = df[df['pnl'] < 0]['pnl'].mean() if len(df[df['pnl'] < 0]) > 0 else 0
        metrics['profit_factor'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else float('inf')
        
        # Risk Metrics
        metrics['max_drawdown'] = self._calculate_max_drawdown(df)
        metrics['var_95'] = self._calculate_var(df['pnl'], 0.95)
        metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(df['pnl'])
        metrics['sortino_ratio'] = self._calculate_sortino_ratio(df['pnl'])
        
        # Current Risk
        metrics['current_exposure'] = sum(pos['size'] * pos['current_price'] for pos in current_positions)
        metrics['current_risk'] = sum(abs(pos['unrealized_pnl']) for pos in current_positions)
        
        return metrics
    
    def _calculate_max_drawdown(self, trades_df):
        """Calculate maximum drawdown"""
        cumulative_pnl = trades_df['pnl'].cumsum()
        peak = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - peak) / peak
        return drawdown.min()
    
    def _calculate_var(self, returns, confidence_level):
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def _calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """Calculate Sharpe ratio"""
        if returns.std() == 0:
            return 0
        return (returns.mean() * 252 - risk_free_rate) / (returns.std() * np.sqrt(252))
    
    def _calculate_sortino_ratio(self, returns, risk_free_rate=0.02):
        """Calculate Sortino ratio (downside deviation only)"""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf')
        
        downside_std = downside_returns.std() * np.sqrt(252)
        return (returns.mean() * 252 - risk_free_rate) / downside_std
```

---

## 🎯 **Risk Management Integration with ForexBot**

### **Enhanced ForexBot with Risk Management**
```python
class RiskManagedForexBot(ForexBot):
    def __init__(self):
        super().__init__()
        
        # Initialize risk management components
        self.account_risk_manager = AccountRiskManager(max_account_drawdown=0.20)
        self.daily_risk_manager = DailyRiskManager(max_daily_loss=0.03)
        self.position_sizing_manager = PositionSizingManager()
        self.stop_loss_manager = StopLossManager()
        self.correlation_manager = CorrelationManager()
        self.circuit_breaker = CircuitBreakerSystem()
        self.risk_metrics = RiskMetricsCalculator()
        
        # Trading state
        self.current_positions = []
        self.trade_history = []
        self.account_balance = 10000  # Initial balance
        
    def enhanced_trading_decision(self, market_data, symbol):
        """Make trading decision with comprehensive risk management"""
        
        # Step 1: Get base ML recommendation
        base_recommendation = self.get_final_recommendation(market_data, symbol)
        
        # Step 2: Check circuit breakers
        trading_state = self._get_current_trading_state()
        active_breakers = self.circuit_breaker.check_breakers(trading_state)
        
        if active_breakers:
            return {
                **base_recommendation,
                'action': 'HOLD',
                'risk_management_override': True,
                'reason': f'Circuit breakers active: {active_breakers}'
            }
        
        # Step 3: Check account-level risk
        if self.account_risk_manager.is_drawdown_limit_exceeded():
            return {
                **base_recommendation,
                'action': 'HOLD',
                'risk_management_override': True,
                'reason': 'Maximum account drawdown exceeded'
            }
        
        # Step 4: Check daily risk limits
        if not self.daily_risk_manager.can_take_more_risk(self.account_balance):
            return {
                **base_recommendation,
                'action': 'HOLD',
                'risk_management_override': True,
                'reason': 'Daily risk limit exceeded'
            }
        
        # Step 5: Check correlation limits
        if not self.correlation_manager.check_correlation_risk(symbol, self.current_positions):
            return {
                **base_recommendation,
                'action': 'HOLD',
                'risk_management_override': True,
                'reason': 'Too many correlated positions'
            }
        
        # Step 6: Calculate position size if trade is approved
        if base_recommendation['action'] in ['BUY', 'SELL']:
            current_price = market_data['close'].iloc[-1]
            stop_loss_price = self.stop_loss_manager.calculate_stop_loss(
                current_price, base_recommendation['action'], market_data
            )
            stop_loss_pips = abs(current_price - stop_loss_price) / 0.0001
            
            position_info = self.position_sizing_manager.calculate_position_size(
                account_balance=self.account_balance,
                stop_loss_pips=stop_loss_pips,
                pip_value=10,  # $10 per pip for standard lot
                confidence=base_recommendation['confidence'],
                win_rate=self._get_recent_win_rate()
            )
            
            # Add risk management info to recommendation
            enhanced_recommendation = {
                **base_recommendation,
                'position_size': position_info['position_size'],
                'risk_amount': position_info['risk_amount'],
                'risk_percentage': position_info['risk_percentage'],
                'stop_loss_price': stop_loss_price,
                'stop_loss_pips': stop_loss_pips,
                'risk_management_method': position_info['method_used'],
                'risk_management_override': False
            }
            
            return enhanced_recommendation
        
        return base_recommendation
    
    def _get_current_trading_state(self):
        """Get current trading state for risk assessment"""
        return {
            'daily_pnl_pct': self.daily_risk_manager.daily_pnl / self.account_balance,
            'hour_pnl_pct': self._calculate_hourly_pnl() / self.account_balance,
            'consecutive_losses': self._count_consecutive_losses(),
            'recent_errors': self._count_recent_errors(),
            'current_positions_count': len(self.current_positions),
            'total_exposure': sum(pos.get('exposure', 0) for pos in self.current_positions)
        }
    
    def _get_recent_win_rate(self):
        """Calculate win rate from recent trades"""
        if len(self.trade_history) < 10:
            return 0.5  # Default assumption
        
        recent_trades = self.trade_history[-30:]  # Last 30 trades
        wins = sum(1 for trade in recent_trades if trade.get('pnl', 0) > 0)
        return wins / len(recent_trades)
```

---

## 🎯 **Implementation Priority**

### **Phase 1: Core Risk Management (Week 1-2)**
1. Implement PositionSizingManager
2. Implement StopLossManager
3. Implement DailyRiskManager
4. Basic integration with ForexBot

### **Phase 2: Advanced Risk Controls (Week 3-4)**
1. Implement AccountRiskManager
2. Implement CircuitBreakerSystem
3. Implement CorrelationManager
4. Full integration testing

### **Phase 3: Monitoring & Metrics (Week 5-6)**
1. Implement RiskMetricsCalculator
2. Implement RiskHeatMapMonitor
3. Create risk dashboard
4. Performance validation

---

*Risk management is the foundation of profitable trading. Without proper risk control, even the best trading system will eventually fail.*

**Next Update**: After live trading validation
**Target**: Phase 4 completion (Q4 2025)