"""
Module: features.py
Responsibility: Vectorized technical indicator, volatility, and sector relative strength calculation.
Implements FR-2.1 through FR-2.6 without external indicator dependencies.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Computes intraday VWAP resetting at the start of every trading day.
    Requires columns: ['high', 'low', 'close', 'volume', 'date']
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    tp_vol = typical_price * df['volume']
    
    cum_tp_vol = tp_vol.groupby(df['date']).cumsum()
    cum_vol = df['volume'].groupby(df['date']).cumsum()
        
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    return vwap


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Computes Relative Strength Index (RSI) using standard Wilder's smoothing.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Computes Average True Range (ATR).
    """
    high = df['high']
    low = df['low']
    close_prev = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr


def compute_rvol(df: pd.DataFrame, rolling_days: int = 20) -> pd.Series:
    """
    Computes Relative Volume (RVOL) compared to the rolling mean volume
    for the exact same time-of-day bucket.
    """
    # Group by time of day (e.g. '09:15', '09:30')
    time_str = df['timestamp'].dt.strftime('%H:%M')
    
    # Calculate rolling mean for each time bucket
    rvol_series = pd.Series(index=df.index, dtype=float)
    for t in time_str.unique():
        mask = (time_str == t)
        bucket_vol = df.loc[mask, 'volume']
        # Rolling mean of previous N occurrences of this exact time slot
        rolling_mean = bucket_vol.rolling(window=rolling_days, min_periods=1).mean().shift(1)
        # Avoid division by zero
        rvol = bucket_vol / rolling_mean.replace(0, np.nan)
        rvol_series.loc[mask] = rvol.fillna(1.0)
        
    return rvol_series


def compute_opening_range(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Identifies the 9:15 - 9:30 AM Opening Range High and Low for each day.
    Returns (orb_high, orb_low) series aligned with df.
    """
    df_temp = df.copy()
    time_str = df_temp['timestamp'].dt.strftime('%H:%M')
    is_orb_candle = (time_str == '09:15')
    
    # Extract ORB high and low
    orb_high_map = df_temp[is_orb_candle].groupby('date')['high'].max()
    orb_low_map = df_temp[is_orb_candle].groupby('date')['low'].min()
    
    orb_high = df_temp['date'].map(orb_high_map)
    orb_low = df_temp['date'].map(orb_low_map)
    
    return orb_high, orb_low


def compute_features(stock_df: pd.DataFrame, sector_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Master feature engineering pipeline.
    Produces clean feature matrix X for machine learning inference and training.
    """
    df = stock_df.copy().sort_values('timestamp').reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)
    
    # 1. VWAP & VWAP Distance (FR-2.1)
    df['vwap'] = compute_vwap(df)
    df['vwap_dist_pct'] = ((df['close'] - df['vwap']) / df['vwap']) * 100.0
    
    # 2. Opening Range Context (FR-2.2)
    orb_high, orb_low = compute_opening_range(df)
    df['orb_high'] = orb_high
    df['orb_low'] = orb_low
    df['is_above_orb_high'] = (df['close'] > df['orb_high']).astype(int)
    df['is_below_orb_low'] = (df['close'] < df['orb_low']).astype(int)
    df['orb_high_dist_pct'] = ((df['close'] - df['orb_high']) / df['orb_high']) * 100.0
    
    # 3. Momentum & Volatility (FR-2.3)
    df['rsi_14'] = compute_rsi(df['close'], period=14)
    df['atr_14'] = compute_atr(df, period=14)
    df['atr_norm_pct'] = (df['atr_14'] / df['close']) * 100.0
    
    ema_9 = df['close'].ewm(span=9, adjust=False).mean()
    ema_21 = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_spread_pct'] = ((ema_9 - ema_21) / ema_21) * 100.0
    
    # 4. Relative Volume (RVOL) (FR-2.4)
    df['rvol'] = compute_rvol(df, rolling_days=20)
    
    # 5. Sector Relative Strength (FR-2.5)
    df['return_15m'] = df['close'].pct_change() * 100.0
    if sector_df is not None and not sector_df.empty:
        sec = sector_df.copy().sort_values('timestamp').reset_index(drop=True)
        sec['timestamp'] = pd.to_datetime(sec['timestamp'])
        sec['sector_return_15m'] = sec['close'].pct_change() * 100.0
        # Merge on timestamp
        merged = pd.merge(df, sec[['timestamp', 'sector_return_15m']], on='timestamp', how='left')
        df['sector_return_15m'] = merged['sector_return_15m'].fillna(0.0)
        df['rel_strength'] = df['return_15m'] - df['sector_return_15m']
    else:
        df['sector_return_15m'] = 0.0
        df['rel_strength'] = 0.0
        
    # 6. Cyclical Time-of-Day Encoding (FR-2.6)
    minutes_since_open = (df['timestamp'].dt.hour - 9) * 60 + (df['timestamp'].dt.minute - 15)
    minutes_since_open = minutes_since_open.clip(lower=0, upper=375)
    df['time_sin'] = np.sin(2 * np.pi * minutes_since_open / 375.0)
    df['time_cos'] = np.cos(2 * np.pi * minutes_since_open / 375.0)

    # Candle price action ratios
    body = (df['close'] - df['open']).abs()
    candle_range = (df['high'] - df['low']).replace(0, np.nan)
    df['body_to_range'] = (body / candle_range).fillna(0.0)
    df['upper_wick_ratio'] = ((df['high'] - df[['open', 'close']].max(axis=1)) / candle_range).fillna(0.0)
    df['lower_wick_ratio'] = (((df[['open', 'close']].min(axis=1)) - df['low']) / candle_range).fillna(0.0)

    return df
