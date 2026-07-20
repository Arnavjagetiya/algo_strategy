"""
signals.py
-----------------------------------------------------------------------------
Regime-adaptive signal generation. This is the "brain" of the strategy.

Design rationale: a naive strategy that mixes trend-following and mean-
reversion signals into one blended score fights itself -- trend-following
wants to buy strength, mean-reversion wants to buy weakness, and averaging
them together produces a mushy signal that's good at neither. Instead this
detects which REGIME the stock is currently in (via ADX, a trend-strength
indicator) and only runs the sub-strategy that fits that regime:

    ADX >= 25  -> TRENDING  -> trend/breakout sub-strategy
    ADX <  20  -> RANGING   -> mean-reversion sub-strategy
    otherwise  -> NEUTRAL   -> no new entries (ambiguous regime, sit out)

Everything here computes signals as of bar t using only information
available AT bar t's close. backtest_engine.py is responsible for shifting
these by one bar before treating them as executable (i.e. a signal seen at
bar t's close is only actionable at bar t+1's open) -- that discipline is
what keeps this backtest free of look-ahead bias.
-----------------------------------------------------------------------------
"""
import numpy as np
import pandas as pd

import config
import indicators as ind


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attaches every indicator this strategy needs to a copy of the bar data."""
    out = df.copy()
    out["ema_fast"] = ind.ema(out["close"], config.EMA_FAST)
    out["ema_slow"] = ind.ema(out["close"], config.EMA_SLOW)
    out["atr"] = ind.atr(out["high"], out["low"], out["close"])
    out["adx"] = ind.adx(out["high"], out["low"], out["close"], config.ADX_PERIOD)
    out["rsi"] = ind.rsi(out["close"], config.RSI_PERIOD)
    out["bb_upper"], out["bb_mid"], out["bb_lower"] = ind.bollinger_bands(
        out["close"], config.BOLLINGER_PERIOD, config.BOLLINGER_STD)
    out["vwap"] = ind.session_vwap(out["high"], out["low"], out["close"], out["volume"])
    out["rel_vol"] = ind.relative_volume(out["volume"])
    out["or_high"], out["or_low"] = ind.opening_range(out, config.OPENING_RANGE_MINUTES)
    return out


def classify_regime(row) -> str:
    if pd.isna(row["adx"]):
        return "NEUTRAL"
    if row["adx"] >= config.ADX_TRENDING_THRESHOLD:
        return "TRENDING"
    if row["adx"] < config.ADX_RANGING_THRESHOLD:
        return "RANGING"
    return "NEUTRAL"


def trend_long_signal(row, index_row) -> bool:
    """Breakout continuation, long only. Requires FOUR independent
    confirmations to line up -- this is what makes it a multi-factor
    signal rather than a single-indicator crossover:
      1. EMA alignment (short-term trend is up)
      2. Price breaking above the opening range (breakout, not just drift)
      3. Volume confirms real participation, not a thin/false move
      4. The broader index isn't fighting you (simple regime-alignment filter)
    """
    if pd.isna(row["or_high"]) or pd.isna(row["ema_fast"]):
        return False
    ema_aligned = row["ema_fast"] > row["ema_slow"]
    breakout = row["close"] > row["or_high"]
    volume_confirms = row["rel_vol"] >= config.REL_VOLUME_TREND_MIN
    index_supportive = index_row is None or index_row["ema_fast"] > index_row["ema_slow"]
    return bool(ema_aligned and breakout and volume_confirms and index_supportive)


def mean_reversion_long_signal(row) -> bool:
    """Oversold-bounce, long only:
      1. RSI confirms oversold conditions
      2. Price is at/through the lower Bollinger Band (statistically stretched)
      3. Volume confirms this looks like capitulation, not just drift down
    """
    if pd.isna(row["bb_lower"]) or pd.isna(row["rsi"]):
        return False
    oversold = row["rsi"] < config.RSI_OVERSOLD
    band_touch = row["close"] <= row["bb_lower"]
    volume_confirms = row["rel_vol"] >= config.REL_VOLUME_MEANREV_MIN
    return bool(oversold and band_touch and volume_confirms)


def generate_entry_signals(df: pd.DataFrame, index_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Returns the feature DataFrame with two extra columns:
      - regime: 'TRENDING' / 'RANGING' / 'NEUTRAL'
      - raw_signal: True if this bar's CLOSE triggers an entry condition
                     (still needs the 1-bar shift + risk/portfolio checks
                     that backtest_engine.py applies before it becomes a trade)
    """
    feats = compute_features(df)
    feats["regime"] = feats.apply(classify_regime, axis=1)

    signals = []
    for ts, row in feats.iterrows():
        if row["regime"] == "TRENDING":
            idx_row = index_df.loc[ts] if index_df is not None and ts in index_df.index else None
            signals.append(trend_long_signal(row, idx_row))
        elif row["regime"] == "RANGING":
            signals.append(mean_reversion_long_signal(row))
        else:
            signals.append(False)
    feats["raw_signal"] = signals
    return feats
