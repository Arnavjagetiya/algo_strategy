"""
data_loader.py
-----------------------------------------------------------------------------
Data layer for BACKTESTING. Deliberately decoupled from the Superr API --
see broker_superr.py for the live-execution side, which is a separate
concern for a later phase.

Two ways to get real data in here when you're ready:

  1. nsepython (NSE-native: handles the exchange's own historical/intraday
     endpoints, no API key needed):
         pip install nsepython
         from nsepython import nse_history  # check current nsepython docs --
         # its function names shift as NSE's own endpoints change.

  2. yfinance (easiest, but general-purpose -- append ".NS" to the symbol,
     and sanity-check for gaps/holes before trusting it for backtesting):
         pip install yfinance
         import yfinance as yf
         yf.download("RELIANCE.NS", interval="5m", period="60d")

Either way, the contract this module needs to satisfy is simple: return a
DataFrame per symbol with columns [open, high, low, close, volume] and a
DatetimeIndex of 5-min bars during NSE market hours (9:15-15:30 IST).
Write a small adapter that reshapes whichever source you pick into that
shape, and nothing downstream (indicators, signals, backtest_engine) needs
to change.

For now, `generate_synthetic_data()` below produces data with that exact
shape so the rest of the pipeline can be built and tested end-to-end before
you've wired up a real source.
-----------------------------------------------------------------------------
"""
import numpy as np
import pandas as pd

import config


def _session_index(start_date: str, n_days: int, bar_minutes: int) -> pd.DatetimeIndex:
    """Builds a DatetimeIndex covering NSE market hours (9:15-15:30 IST)
    for n_days trading days (Mon-Fri only, no holiday calendar -- good
    enough for synthetic testing, swap in a real NSE holiday list for
    production backtests)."""
    days = pd.bdate_range(start=start_date, periods=n_days)
    all_bars = []
    for d in days:
        day_start = d + pd.Timedelta(hours=9, minutes=15)
        day_end = d + pd.Timedelta(hours=15, minutes=30)
        all_bars.append(pd.date_range(day_start, day_end, freq=f"{bar_minutes}min"))
    return pd.DatetimeIndex(np.concatenate(all_bars))


def load_real_data_yfinance(symbols, days=55) -> dict:
    """
    Pulls real 5-min NSE bars via yfinance. Yahoo only keeps 5-min data for
    the trailing ~60 days, so `days` can't go much above that -- ask for
    more and yfinance just silently gives you less.
    Needs: pip install yfinance
    """
    import yfinance as yf
    data = {}
    for sym in symbols:
        df = yf.download(f"{sym}.NS", period=f"{days}d", interval="5m",
                          progress=False, auto_adjust=True)
        if df.empty:
            print(f"WARNING: no data returned for {sym}, skipping")
            continue
        if isinstance(df.columns, pd.MultiIndex):     # yfinance sometimes returns multi-index columns
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        df.index = df.index.tz_localize(None)          # drop timezone so it matches the rest of the pipeline
        data[sym] = df
    return data


def generate_synthetic_data(symbols, start_date="2026-01-01", n_days=120,
                             bar_minutes=5, seed=42) -> dict:
    """
    Generates plausible-looking synthetic 5-min OHLCV bars for pipeline
    testing ONLY. This is NOT real market data and any backtest metrics
    produced on it are meaningless as performance estimates -- its only
    job is to prove the mechanics (signals, risk, costs, backtest loop)
    run correctly end to end before you point it at real data.

    Includes deliberate regime shifts (alternating trending/choppy stretches)
    so both sub-strategies actually get exercised during the demo run.
    """
    rng = np.random.default_rng(seed)
    index = _session_index(start_date, n_days, bar_minutes)
    bars_per_day = len(index) // n_days

    data = {}
    for i, sym in enumerate(symbols):
        price0 = 500 + i * 350          # spread starting prices out
        n = len(index)

        # regime schedule: alternate ~15-trading-day trending / choppy blocks
        block = bars_per_day * 15
        regime = np.array([1 if (k // block) % 2 == 0 else 0 for k in range(n)])  # 1=trend, 0=chop

        drift = np.where(regime == 1, 0.00012, 0.0)         # small upward drift while "trending"
        noise_scale = np.where(regime == 1, 0.0009, 0.0016)  # choppier regime is noisier, not trendier
        noise = rng.normal(loc=drift, scale=noise_scale, size=n)

        log_returns = noise
        log_price = np.log(price0) + np.cumsum(log_returns)
        close = np.exp(log_price)

        # build OHLC around each close with a small random intrabar range
        rng_frac = np.abs(rng.normal(0.0009, 0.0004, size=n))
        high = close * (1 + rng_frac)
        low = close * (1 - rng_frac)
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        # keep OHLC internally consistent
        high = np.maximum.reduce([high, open_, close])
        low = np.minimum.reduce([low, open_, close])

        base_vol = 80_000 + i * 5_000
        volume = rng.lognormal(mean=np.log(base_vol), sigma=0.5, size=n).astype(int)
        # occasional volume surges, more likely at the start of trending blocks
        surge_prob = np.where(regime == 1, 0.03, 0.01)
        surge = rng.random(n) < surge_prob
        volume = np.where(surge, (volume * rng.uniform(2, 4, n)).astype(int), volume)

        df = pd.DataFrame({
            "open": open_, "high": high, "low": low, "close": close, "volume": volume,
        }, index=index)
        data[sym] = df

    return data


def apply_liquidity_filter(data: dict) -> dict:
    """Drops symbols that fail the ADV / ATR% band configured in config.py.

    Deliberately screens ATR% on DAILY-resampled bars, not the raw 5-min
    bars used for signal generation -- 5-min-bar ATR is naturally a much
    smaller percentage of price than day-to-day ATR, and comparing it
    against daily-scale thresholds would (and, in an earlier version of
    this function, did) filter out every symbol regardless of how it
    actually trades."""
    import indicators as ind
    kept = {}
    for sym, df in data.items():
        adv = (df["close"] * df["volume"]).groupby(df.index.date).sum().mean()

        daily = df.resample("1D").agg({"open": "first", "high": "max",
                                        "low": "min", "close": "last"}).dropna()
        atr_pct = (ind.atr(daily["high"], daily["low"], daily["close"]) / daily["close"] * 100).mean()

        if adv >= config.MIN_ADV_INR and config.ATR_PCT_MIN <= atr_pct <= config.ATR_PCT_MAX:
            kept[sym] = df
    return kept
