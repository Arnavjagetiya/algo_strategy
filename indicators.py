"""
indicators.py
-----------------------------------------------------------------------------
Hand-rolled technical indicators in pandas/numpy only. No TA-Lib dependency
on purpose -- TA-Lib needs a C library compiled first, which is exactly the
kind of environment friction worth avoiding, and building these yourself
means you actually understand what each one is doing rather than trusting
a black box.

All functions expect a DatetimeIndex'd DataFrame of 5-min OHLCV bars for a
SINGLE symbol, already sorted by time.
-----------------------------------------------------------------------------
"""
import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's ATR. alpha = 1/period is mathematically Wilder's smoothing method."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index -- measures trend STRENGTH (not direction).
    Used here purely as the regime switch between the trend and mean-reversion
    sub-strategies."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(high, low, close)
    atr_smooth = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / atr_smooth
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / atr_smooth

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def session_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """VWAP that resets every trading day (grouped by the bar's own calendar date)."""
    typical_price = (high + low + close) / 3
    pv = typical_price * volume
    day = pd.Series(high.index.date, index=high.index)
    cum_pv = pv.groupby(day).cumsum()
    cum_vol = volume.groupby(day).cumsum()
    return cum_pv / cum_vol.replace(0, np.nan)


def relative_volume(volume: pd.Series, lookback: int = 20) -> pd.Series:
    """Current bar's volume vs. its own rolling average -- flags participation surges."""
    avg_vol = volume.rolling(lookback).mean()
    return volume / avg_vol.replace(0, np.nan)


def opening_range(df: pd.DataFrame, minutes: int = 15):
    """Returns per-day (high, low) of the first `minutes` of each session,
    broadcast across every bar in that session -- used for breakout confirmation."""
    day = pd.Series(df.index.date, index=df.index)
    session_start = df.groupby(day).apply(lambda g: g.index[0])
    or_high, or_low = {}, {}
    for d, start in session_start.items():
        window = df.loc[start: start + pd.Timedelta(minutes=minutes)]
        or_high[d] = window["high"].max()
        or_low[d] = window["low"].min()
    return day.map(or_high), day.map(or_low)
