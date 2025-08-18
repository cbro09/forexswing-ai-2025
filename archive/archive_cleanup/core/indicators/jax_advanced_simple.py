import jax.numpy as jnp
import numpy as np
from jax import jit

@jit
def jax_bollinger_bands(prices, window=20, std_dev=2):
    """JAX Bollinger Bands - simplified"""
    sma = jnp.convolve(prices, jnp.ones(window)/window, mode='same')
    
    # Simplified rolling std using convolution
    squared_prices = prices ** 2
    rolling_mean_sq = jnp.convolve(squared_prices, jnp.ones(window)/window, mode='same')
    rolling_std = jnp.sqrt(jnp.maximum(rolling_mean_sq - sma**2, 0))
    
    upper_band = sma + (rolling_std * std_dev)
    lower_band = sma - (rolling_std * std_dev)
    
    # Avoid division by zero
    bb_width = upper_band - lower_band
    bb_percent = jnp.where(
        bb_width == 0,
        0.5,  # Neutral when no volatility
        (prices - lower_band) / bb_width
    )
    
    return upper_band, sma, lower_band, bb_percent

def simple_stochastic(high, low, close, period=14):
    """Simplified stochastic oscillator"""
    k_values = np.zeros(len(close))
    
    for i in range(len(close)):
        if i < period - 1:
            k_values[i] = 50.0
        else:
            period_high = np.max(high[i-period+1:i+1])
            period_low = np.min(low[i-period+1:i+1])
            
            if period_high == period_low:
                k_values[i] = 50.0
            else:
                k_values[i] = 100 * (close[i] - period_low) / (period_high - period_low)
    
    # Simple moving average for %D
    d_values = np.convolve(k_values, np.ones(3)/3, mode='same')
    
    return jnp.array(k_values), jnp.array(d_values)

def simple_williams_r(high, low, close, period=14):
    """Simplified Williams %R"""
    wr_values = np.zeros(len(close))
    
    for i in range(len(close)):
        if i < period - 1:
            wr_values[i] = -50.0
        else:
            period_high = np.max(high[i-period+1:i+1])
            period_low = np.min(low[i-period+1:i+1])
            
            if period_high == period_low:
                wr_values[i] = -50.0
            else:
                wr_values[i] = -100 * (period_high - close[i]) / (period_high - period_low)
    
    return jnp.array(wr_values)

def simple_atr(high, low, close, period=14):
    """Simplified ATR"""
    tr_values = np.zeros(len(close))
    
    for i in range(len(close)):
        if i == 0:
            tr_values[i] = high[i] - low[i]
        else:
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr_values[i] = max(tr1, tr2, tr3)
    
    # Simple moving average
    atr = np.convolve(tr_values, np.ones(period)/period, mode='same')
    
    return jnp.array(atr)

@jit
def jax_momentum(prices, period=10):
    """JAX Price Momentum"""
    momentum = jnp.concatenate([
        jnp.zeros(period),
        prices[period:] - prices[:-period]
    ])
    
    return momentum

@jit
def jax_rate_of_change(prices, period=12):
    """JAX Rate of Change"""
    roc = jnp.concatenate([
        jnp.zeros(period),
        ((prices[period:] - prices[:-period]) / jnp.maximum(prices[:-period], 1e-8)) * 100
    ])
    
    return roc