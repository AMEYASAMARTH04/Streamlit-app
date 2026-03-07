import pandas as pd
import ta

def add_indicators(df):
    # Ensure we're working with Series objects
    close = pd.Series(df['Close'].values.flatten(), index=df.index)
    high = pd.Series(df['High'].values.flatten(), index=df.index)
    low = pd.Series(df['Low'].values.flatten(), index=df.index)
    volume = pd.Series(df['Volume'].values.flatten(), index=df.index)

    # Momentum Indicators
    df['RSI'] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    df['Stoch_K'] = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14).stoch()
    df['Stoch_D'] = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14).stoch_signal()
    df['Momentum'] = close.pct_change(periods=10)

    # Trend Indicators
    df['SMA_20'] = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
    df['SMA_50'] = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
    df['EMA_20'] = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
    
    macd = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    # Volatility Indicators
    df['ATR'] = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()

    # Volume Indicators
    df['Volume_SMA_20'] = volume.rolling(window=20).mean()

    # Return Indicators
    df['Daily_Return'] = close.pct_change()

    # Clean up
    df.dropna(inplace=True)
    
    return df