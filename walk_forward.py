"""
walk_forward.py
-----------------------------------------------------------------------------
Walk-forward validation: instead of trusting a single backtest over your
whole dataset (which tells you how well the strategy fits THAT data, not
how it performs on data it hasn't seen), split time into sequential folds
and roll forward -- tune/inspect on fold i, evaluate out-of-sample on fold
i+1, slide forward, repeat.

This module provides the SPLITTING mechanism and a runner that executes
the existing backtest_engine on each fold. It does not do parameter
optimization itself (that's a separate, bigger project -- a naive grid
search here would just reintroduce overfitting through the back door).
Use this first to check whether your CURRENT fixed parameters hold up
out-of-sample before you ever start tuning them.
-----------------------------------------------------------------------------
"""
import pandas as pd

import backtest_engine
import metrics


def make_folds(all_dates: list, train_days: int, test_days: int, step_days: int = None):
    """
    Returns a list of (train_dates, test_dates) tuples, sliding forward by
    step_days each time (defaults to test_days, i.e. non-overlapping test
    windows).
    """
    step_days = step_days or test_days
    all_dates = sorted(set(all_dates))
    folds = []
    start = 0
    while start + train_days + test_days <= len(all_dates):
        train_dates = all_dates[start: start + train_days]
        test_dates = all_dates[start + train_days: start + train_days + test_days]
        folds.append((train_dates, test_dates))
        start += step_days
    return folds


def _slice_by_dates(data: dict, dates: set) -> dict:
    return {sym: df[pd.Series(df.index.date, index=df.index).isin(dates)] for sym, df in data.items()}


def run_walk_forward(data: dict, starting_capital: float, train_days: int = 20,
                      test_days: int = 10, index_symbol: str = None) -> list:
    """
    Runs the backtest on each fold's TEST window only (fresh starting_capital
    per fold, since the point is comparing out-of-sample fold quality, not
    compounding one continuous account across folds).

    Indicators are computed ONCE on the full history via
    backtest_engine.compute_all_features(), and only the resulting FEATURES
    are sliced per fold -- not the raw OHLCV. Slicing raw data down to a
    10-day window before computing ADX/RSI/Bollinger starves them of the
    14-20 bars of lookback they need to produce a single non-NaN value,
    which silently produces zero signals for the entire fold. That was a
    real bug here: every fold showed 0 trades even though the same data,
    backtested whole, produced real ones.
    """
    feats = backtest_engine.compute_all_features(data, index_symbol=index_symbol)

    any_df = next(iter(feats.values()))
    all_dates = sorted(set(any_df.index.date))
    folds = make_folds(all_dates, train_days, test_days)

    results = []
    for i, (train_dates, test_dates) in enumerate(folds):
        test_feats = _slice_by_dates(feats, set(test_dates))
        test_feats = {s: d for s, d in test_feats.items() if len(d) > 0}
        if not test_feats:
            continue
        bt = backtest_engine.run_backtest(starting_capital=starting_capital, precomputed_feats=test_feats)
        m = metrics.compute_metrics(bt["equity_curve"], bt["trade_log"], starting_capital)
        m["fold"] = i
        m["test_start"] = str(test_dates[0])
        m["test_end"] = str(test_dates[-1])
        results.append(m)
    return results
