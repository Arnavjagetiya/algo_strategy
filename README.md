# Intraday-Cash + 5-Day-Carry Strategy

Long-only, regime-adaptive NSE cash-equity strategy. Trend-following when
ADX says the stock is trending, mean-reversion when it isn't, no new
entries when the regime is ambiguous. Positions square off intraday by
default; a position is only allowed to carry past the same day if its
trend thesis is still structurally intact at square-off time, and even
then it's force-closed at 5 trading days regardless of P&L.

Backtesting only, for now -- no derivatives, no live orders. See
`config.py` for why (and for the reasoning behind every other parameter
in this project).

## Project layout

| File | Responsibility |
|---|---|
| `config.py` | Every tunable parameter. Change behavior here, not by editing logic elsewhere. |
| `indicators.py` | EMA, ATR, RSI, ADX, Bollinger Bands, VWAP, relative volume, opening range -- hand-rolled, no TA-Lib. |
| `signals.py` | Regime classification (ADX) + the trend and mean-reversion entry rules. |
| `risk_management.py` | Position sizing (risk-based AND notional-capped), stop/target/scale-out/time-stop exit logic, portfolio-level caps. |
| `costs.py` | Indian cash-equity transaction costs: brokerage, STT (intraday vs. delivery rate), GST, stamp duty, slippage. |
| `data_loader.py` | Data layer. Ships with a synthetic data generator for pipeline testing; documents how to swap in real NSE data. |
| `backtest_engine.py` | The event loop: iterates every symbol, every bar, in true chronological order, applying signals -> risk checks -> costs. One-bar execution lag is enforced here to avoid look-ahead bias. |
| `metrics.py` | Sharpe, Sortino, max drawdown, Calmar, win rate, profit factor, and a Monte Carlo trade-reshuffling robustness check. |
| `walk_forward.py` | Splits history into sequential train/test folds and reruns the backtest per fold, so you can see whether results hold up out-of-sample or are concentrated in one lucky stretch. |
| `broker_superr.py` | Stoxkart Superr API client -- for the LIVE/PAPER phase later. Not used anywhere in the backtest. |
| `run_backtest.py` | Entry point. Run this. |

## Running it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 run_backtest.py
```

This runs on synthetic data out of the box, purely to prove the pipeline
is mechanically correct (it is -- see the debugging notes below). **The
performance numbers from that run mean nothing** -- it's a random walk
with no real structure to find an edge in.

## Moving to real data

Replace the `data_loader.generate_synthetic_data(...)` call in
`run_backtest.py` with a loader that returns the same shape: a dict of
`{symbol: DataFrame}`, each DataFrame indexed by 5-min timestamps during
NSE market hours (9:15-15:30 IST) with columns `[open, high, low, close,
volume]`. Two starting points, both documented in `data_loader.py`:

- **nsepython** -- NSE-native, no API key needed.
- **yfinance** -- easiest, append `.NS` to the symbol; sanity-check for
  gaps before trusting it.

Nothing else in the project needs to change once your loader returns that
shape.

## Known simplifications (read before trusting a result)

- **No NSE holiday calendar.** Synthetic data uses `pd.bdate_range`
  (weekdays only). Swap in a real trading-day calendar with real data.
- **Regime labels (TRENDING/RANGING) are a lens on the price series for
  signal purposes, not a property the synthetic price GENERATOR itself
  enforces** -- the underlying synthetic prices are a plain random walk
  with varying noise, not genuinely mean-reverting during "RANGING"
  blocks. This is part of why synthetic performance looks worse and more
  volatile than you should expect on real, less erratic index/stock
  behavior.
- **Fill assumption**: stops/targets fill at the exact stop/target price,
  not at the bar's actual low/high or next tick -- optimistic during a
  large gap-through, conservative otherwise. A `SLIPPAGE_BPS` estimate in
  `costs.py` is the only adjustment for this; tune it against your own
  fill data once live.
- **STT and stamp duty rates in `costs.py`** are current as of when this
  was written but are set by the government and occasionally revised in
  Union Budgets -- confirm before trusting a live P&L number.
- **`broker_superr.py`'s historical-data method is not implemented** --
  Stoxkart's public docs didn't have a confirmed historical-candle REST
  path at the time this was written. Check the "API Documentation"
  section at developers.stoxkart.com directly; backtesting doesn't need
  it anyway (see `data_loader.py`).

## Debugging notes (kept on purpose)

Two real bugs came up building this, both instructive enough to leave
documented rather than silently fix and hide:

1. **Liquidity filter screened 5-min-bar ATR% against thresholds scaled
   for daily bars**, filtering out every symbol regardless of how it
   actually traded. Fixed by resampling to daily bars specifically for
   that screen (see `apply_liquidity_filter` in `data_loader.py`).
2. **Risk-based position sizing alone demanded more capital than the
   account had.** A tight stop relative to price means a large share
   count is needed to accumulate meaningful rupee risk -- and that share
   count's total value can exceed the account. Fixed by adding a second,
   independent notional-exposure cap and taking whichever constraint
   gives the smaller quantity (see `position_size` in
   `risk_management.py`).
