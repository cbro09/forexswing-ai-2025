
import jax.numpy as jnp
from jax import jit

@jit
def jax_bollinger_bands(prices, window=20, std_dev=2):
    """JAX Bollinger Bands"""
    sma = jnp.convolve(prices, jnp.ones(window)/window, mode='same')
    
    # Calculate rolling standard deviation
    rolling_std = jnp.array([
        jnp.std(prices[max(0, i-window+1):i+1]) if i >= window-1 else 0.0
        for i in range(len(prices))
    ])
    
    upper_band = sma + (rolling_std * std_dev)
    lower_band = sma - (rolling_std * std_dev)
    bb_percent = (prices - lower_band) / (upper_band - lower_band)
    
    return upper_band, sma, lower_band, bb_percent

@jit
def jax_stochastic_oscillator(high, low, close, k_period=14, d_period=3):
    """JAX Stochastic Oscillator"""
    
    # Calculate %K
    k_values = []
    for i in range(len(close)):
        if i < k_period - 1:
            k_values.append(50.0)  # Default value
        else:
            period_high = jnp.max(high[i-k_period+1:i+1])
            period_low = jnp.min(low[i-k_period+1:i+1])
            
            if period_high == period_low:
                k_val = 50.0
            else:
                k_val = 100 * (close[i] - period_low) / (period_high - period_low)
            
            k_values.append(k_val)
    
    k_values = jnp.array(k_values)
    
    # Calculate %D (SMA of %K)
    d_values = jnp.convolve(k_values, jnp.ones(d_period)/d_period, mode='same')
    
    return k_values, d_values

@jit
def jax_williams_r(high, low, close, period=14):
    """JAX Williams %R"""
    
    wr_values = []
    for i in range(len(close)):
        if i < period - 1:
            wr_values.append(-50.0)  # Default value
        else:
            period_high = jnp.max(high[i-period+1:i+1])
            period_low = jnp.min(low[i-period+1:i+1])
            
            if period_high == period_low:
                wr_val = -50.0
            else:
                wr_val = -100 * (period_high - close[i]) / (period_high - period_low)
            
            wr_values.append(wr_val)
    
    return jnp.array(wr_values)

@jit
def jax_atr(high, low, close, period=14):
    """JAX Average True Range"""
    
    # Calculate True Range
    tr_values = []
    for i in range(len(close)):
        if i == 0:
            tr = high[i] - low[i]
        else:
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr = max(tr1, tr2, tr3)
        
        tr_values.append(tr)
    
    tr_values = jnp.array(tr_values)
    
    # Calculate ATR (SMA of TR)
    atr = jnp.convolve(tr_values, jnp.ones(period)/period, mode='same')
    
    return atr

@jit
def jax_momentum(prices, period=10):
    """JAX Price Momentum"""
    momentum = jnp.concatenate([
        jnp.zeros(period),  # Pad beginning
        prices[period:] - prices[:-period]
    ])
    
    return momentum

@jit
def jax_rate_of_change(prices, period=12):
    """JAX Rate of Change"""
    roc = jnp.concatenate([
        jnp.zeros(period),  # Pad beginning
        ((prices[period:] - prices[:-period]) / prices[:-period]) * 100
    ])
    
    return roc
