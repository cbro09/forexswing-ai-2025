# user_data/strategies/ForexSwingAI2025.py
"""
Advanced Forex Strategy with JAX-accelerated indicators + PyTorch LSTM predictions
Combines 60K+ calc/sec performance with neural network intelligence
"""

# import talib.abstract as ta  # Commented out for now due to Windows compilation issues
# from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, CategoricalParameter
# import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas as pd
import numpy as np
from functools import reduce
import sys
import os

# Import our custom modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from jax_indicators import jax_rsi, jax_sma, jax_macd, jax_ema
from ml_models.forex_lstm import HybridForexPredictor

class ForexSwingAI2025(IStrategy):
    """
    Hybrid Strategy: JAX Speed + Neural Network Intelligence
    """
    
    INTERFACE_VERSION = 3
    
    # ===== STRATEGY PARAMETERS (Auto-optimizable) =====
    
    # ML Parameters
    ml_threshold_long = DecimalParameter(0.55, 0.75, default=0.65, space="buy", optimize=True)
    ml_confidence_min = DecimalParameter(0.60, 0.85, default=0.70, space="buy", optimize=True)
    ml_weight = DecimalParameter(0.3, 0.7, default=0.5, space="buy", optimize=True)
    
    # Technical Parameters  
    rsi_period = IntParameter(10, 25, default=14, space="buy", optimize=True)
    rsi_oversold = IntParameter(25, 40, default=30, space="buy", optimize=True)
    rsi_overbought = IntParameter(75, 90, default=85, space="sell", optimize=True)
    
    sma_fast = IntParameter(15, 25, default=20, space="buy", optimize=True)
    sma_slow = IntParameter(45, 55, default=50, space="buy", optimize=True)
    
    # Risk Management
    use_ml_filter = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    
    # ===== STRATEGY SETTINGS =====
    
    # ROI table (optimized for forex swing trading)
    minimal_roi = {
        "0": 0.06,    # 6% max profit
        "40": 0.03,   # 3% after 40 minutes  
        "120": 0.01,  # 1% after 2 hours
        "720": 0     # Break even after 12 hours
    }
    
    # Stop loss
    stoploss = -0.025  # 2.5% stop loss
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True
    
    # Timeframe
    timeframe = '4h'
    
    # Startup candle count  
    startup_candle_count: int = 200
    
    # Order types
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        # Initialize ML predictor
        try:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                'models', 
                'forex_lstm.pth'
            )
            self.ml_predictor = HybridForexPredictor(model_path)
            self.ml_enabled = True
            print(f"✅ ML Predictor initialized: {self.ml_predictor.get_model_info()}")
        except Exception as e:
            print(f"⚠️ ML Predictor failed to initialize: {e}")
            print("📊 Strategy will use technical indicators only")
            self.ml_enabled = False
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Create indicators using JAX acceleration + ML predictions
        """
        
        # ===== JAX-ACCELERATED INDICATORS (60K+ calc/sec) =====
        
        try:
            import jax.numpy as jnp
            
            # Convert to JAX arrays for lightning-fast computation
            close_prices = jnp.array(dataframe['close'].values)
            volume_data = jnp.array(dataframe['volume'].values)
            
            print(f"🚀 Processing {len(close_prices)} candles with JAX acceleration...")
            
            # Ultra-fast technical indicators
            rsi_values = jax_rsi(close_prices, self.rsi_period.value)
            sma_fast_values = jax_sma(close_prices, self.sma_fast.value)
            sma_slow_values = jax_sma(close_prices, self.sma_slow.value)
            ema_20_values = jax_ema(close_prices, 20)
            
            # MACD
            macd_line, macd_signal, macd_histogram = jax_macd(close_prices)
            
            # Convert back to numpy for pandas compatibility
            dataframe['rsi'] = np.array(rsi_values)
            dataframe['sma_fast'] = np.array(sma_fast_values)
            dataframe['sma_slow'] = np.array(sma_slow_values)
            dataframe['ema_20'] = np.array(ema_20_values)
            dataframe['macd'] = np.array(macd_line)
            dataframe['macd_signal'] = np.array(macd_signal)
            dataframe['macd_histogram'] = np.array(macd_histogram)
            
            print("✅ JAX indicators computed successfully")
            
        except Exception as e:
            print(f"⚠️ JAX computation failed: {e}")
            print("📊 Falling back to traditional indicators...")
            
            # Fallback to traditional indicators
            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
            dataframe['sma_fast'] = ta.SMA(dataframe, timeperiod=self.sma_fast.value)
            dataframe['sma_slow'] = ta.SMA(dataframe, timeperiod=self.sma_slow.value)
            dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
            
            # MACD
            macd = ta.MACD(dataframe)
            dataframe['macd'] = macd['macd']
            dataframe['macd_signal'] = macd['macdsignal']
            dataframe['macd_histogram'] = macd['macdhist']
        
        # ===== ADDITIONAL TECHNICAL INDICATORS =====
        
        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lower'] = bollinger['lower']
        dataframe['bb_middle'] = bollinger['mid']  
        dataframe['bb_upper'] = bollinger['upper']
        dataframe['bb_percent'] = (dataframe['close'] - dataframe['bb_lower']) / (dataframe['bb_upper'] - dataframe['bb_lower'])
        
        # Volume analysis
        dataframe['volume_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        
        # Volatility
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['volatility'] = dataframe['close'].rolling(20).std()
        
        # ===== NEURAL NETWORK PREDICTIONS =====
        
        if self.ml_enabled and self.use_ml_filter.value:
            try:
                print("🤖 Generating ML predictions...")
                
                # Get ML predictions
                ml_signals = self.ml_predictor.predict(dataframe)
                dataframe['ml_signal'] = ml_signals
                
                # Calculate confidence (distance from neutral 0.5)
                dataframe['ml_confidence'] = np.abs(ml_signals - 0.5) * 2
                
                # ML trend detection
                dataframe['ml_trend'] = np.where(ml_signals > 0.5, 1, -1)
                dataframe['ml_strength'] = np.abs(ml_signals - 0.5) * 2
                
                print(f"✅ ML predictions generated - Range: {ml_signals.min():.3f} to {ml_signals.max():.3f}")
                
            except Exception as e:
                print(f"❌ ML prediction failed: {e}")
                # Fill with neutral values
                dataframe['ml_signal'] = 0.5
                dataframe['ml_confidence'] = 0.0
                dataframe['ml_trend'] = 0
                dataframe['ml_strength'] = 0.0
        else:
            # No ML - fill with neutral values
            dataframe['ml_signal'] = 0.5
            dataframe['ml_confidence'] = 0.0
            dataframe['ml_trend'] = 0
            dataframe['ml_strength'] = 0.0
        
        # ===== COMBINED SIGNALS =====
        
        # Trend strength combination
        dataframe['trend_alignment'] = np.where(
            (dataframe['close'] > dataframe['sma_fast']) & 
            (dataframe['sma_fast'] > dataframe['sma_slow']), 1, 0
        )
        
        # Multi-timeframe momentum
        dataframe['momentum_score'] = (
            np.where(dataframe['rsi'] < 50, 1, 0) +
            np.where(dataframe['macd'] > dataframe['macd_signal'], 1, 0) +
            np.where(dataframe['close'] > dataframe['ema_20'], 1, 0) +
            np.where(dataframe['ml_trend'] > 0, 1, 0)
        )
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        ULTRA-SIMPLE TEST: Just ML + basic trend
        """
        
        conditions = []
        
        # Only 3 conditions - very liberal
        conditions.append(dataframe['close'] > dataframe['sma_slow'])  # Basic uptrend
        conditions.append(dataframe['ml_signal'] > 0.6)  # ML bullish (lowered from 0.65)
        conditions.append(dataframe['ml_trend'] > 0)     # ML positive
        
        # ===== COMBINE ALL CONDITIONS =====
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'
            ] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Define exit signals
        """
        
        conditions = []
        
        # RSI overbought
        conditions.append(dataframe['rsi'] > self.rsi_overbought.value)
        
        # MACD bearish crossover
        conditions.append(dataframe['macd'] < dataframe['macd_signal'])
        
        # ML bearish signal (if enabled)
        if self.ml_enabled and self.use_ml_filter.value:
            conditions.append(dataframe['ml_signal'] < (1 - self.ml_threshold_long.value))
        
        # Bollinger Band rejection
        conditions.append(dataframe['bb_percent'] > 0.8)
        
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),  # OR logic for exits
                'exit_long'
            ] = 1
        
        return dataframe
    
    def custom_stake_amount(self, pair: str, current_time, current_rate: float, 
                          proposed_stake: float, min_stake: float, max_stake: float,
                          leverage: float, entry_tag, side: str, **kwargs) -> float:
        """
        Dynamic position sizing based on ML confidence
        """
        
        if self.ml_enabled and hasattr(self, 'ml_confidence_current'):
            # Scale position size by ML confidence
            confidence_multiplier = 0.5 + (self.ml_confidence_current * 0.5)  # 0.5x to 1.0x
            adjusted_stake = proposed_stake * confidence_multiplier
            return max(min_stake, min(adjusted_stake, max_stake))
        
        return proposed_stake