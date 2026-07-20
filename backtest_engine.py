"""
backtest_engine.py
-----------------------------------------------------------------------------
The event loop that ties signals + risk_management + costs together bar by
bar, across the whole universe simultaneously (not symbol-by-symbol) because
portfolio-level rules -- MAX_CONCURRENT_POSITIONS, portfolio heat, the daily
loss kill switch -- only make sense evaluated across all open positions at
once.

Look-ahead discipline (read this before changing anything):
  A signal computed from bar t's CLOSE is only acted on at bar t+1's OPEN.
  That one-bar shift is what separates a real backtest from a fantasy one --
  it's implemented by reading `prev_bar['raw_signal']` when deciding whether
  to enter on the CURRENT bar, never `bar['raw_signal']`.
-----------------------------------------------------------------------------
"""
import pandas as pd

import config
import costs
import risk_management as rm
import signals as sig


def _build_trading_day_index(df: pd.DataFrame) -> dict:
    """Maps each calendar date present in df to a 0-based sequential trading-day
    number, so 'holding period in trading days' can be computed by subtraction
    instead of approximated from calendar days (which weekends would distort)."""
    dates = sorted(set(df.index.date))
    return {d: i for i, d in enumerate(dates)}


def compute_all_features(raw_data: dict, index_symbol: str = None) -> dict:
    """
    Runs signal generation once per symbol on its FULL history. Deliberately
    separated from run_backtest() so callers that need to slice by date
    range afterward (walk_forward.py) can compute indicators against the
    complete series first, THEN slice -- slicing the raw OHLCV down to a
    short window before computing indicators starves ADX/RSI/Bollinger of
    the 14-20 bars of lookback they need, so they stay NaN for the entire
    window and nothing ever signals. That was a real bug: every single
    walk-forward fold showed 0 trades even though the same data, run whole,
    produced real trades -- because each fold was recomputing indicators
    from scratch on a slice too short to warm them up.
    """
    index_df = None
    feats = {}
    for sym, df in raw_data.items():
        f = sig.generate_entry_signals(df)
        feats[sym] = f
        if sym == index_symbol:
            index_df = f

    if index_symbol is not None:
        tradeable = {s: d for s, d in raw_data.items() if s != index_symbol}
        feats = {s: sig.generate_entry_signals(d, index_df=index_df) for s, d in tradeable.items()}
    return feats


def run_backtest(raw_data: dict = None, starting_capital: float = None,
                  index_symbol: str = None, precomputed_feats: dict = None) -> dict:
    """
    raw_data: {symbol: OHLCV DataFrame}. index_symbol, if given, must also be
    a key in raw_data and is used only as the broader-market regime filter
    for the trend sub-strategy (not itself traded).

    precomputed_feats: pass this INSTEAD of raw_data when you've already run
    compute_all_features() yourself (e.g. walk_forward.py does this, slicing
    the fully-warmed-up features down to a fold's date range afterward).
    Returns {'equity_curve': pd.Series, 'trade_log': list[dict]}.
    """
    starting_capital = starting_capital or config.STARTING_CAPITAL_INR
    if precomputed_feats is not None:
        feats = precomputed_feats
    else:
        feats = compute_all_features(raw_data, index_symbol=index_symbol)

    day_idx = {sym: _build_trading_day_index(df) for sym, df in feats.items()}

    all_ts = sorted(set().union(*[df.index for df in feats.values()]))

    cash = starting_capital
    equity_for_sizing = starting_capital     # updated once per bar; used for RISK_PER_TRADE_PCT sizing
    open_positions: dict[str, rm.Position] = {}
    trade_log = []
    equity_points = []

    current_day = None
    day_realized_pnl = 0.0

    for ts in all_ts:
        if current_day is None or ts.date() != current_day:
            current_day = ts.date()
            day_realized_pnl = 0.0

        ts_time = ts.strftime("%H:%M")

        # ---- 1. Exits (stop / target / scale-out / regime-flip / time-stop) ----
        for symbol in list(open_positions.keys()):
            df = feats[symbol]
            if ts not in df.index:
                continue
            bar = df.loc[ts]
            position = open_positions[symbol]
            held_days = day_idx[symbol][ts.date()] - day_idx[symbol][position.entry_time.date()]

            reason, exit_price = rm.check_exit(position, bar, bar["regime"], held_days)

            if reason == "scale_out":
                sell_qty = max(
                    1,
                    int(position.quantity * config.PARTIAL_SCALE_OUT_FRACTION),
                )

                if sell_qty >= position.quantity:
                    sell_qty = position.quantity - 1

                if sell_qty <= 0:
                    continue
                same_day = ts.date() == position.entry_time.date()
                proceeds = exit_price * sell_qty - costs.exit_cost(exit_price, sell_qty, same_day)
                cash += proceeds
                day_realized_pnl += proceeds - position.entry_price * sell_qty
                position.total_exit_cash_in += proceeds
                position.quantity -= sell_qty
                position.stop_price = position.entry_price   # move stop to breakeven
                position.scaled_out = True
                trade_log.append({
                    "event": "partial", "symbol": symbol, "time": ts, "price": exit_price,
                    "quantity": sell_qty, "reason": "scale_out",
                })
                continue

            if reason in ("stop", "target", "regime_flip", "time_stop", "adx_exit"):
                same_day = ts.date() == position.entry_time.date()
                proceeds = exit_price * position.quantity - costs.exit_cost(exit_price, position.quantity, same_day)
                cash += proceeds
                position.total_exit_cash_in += proceeds
                pnl = position.total_exit_cash_in - position.total_entry_cash_out
                day_realized_pnl += pnl
                trade_log.append({
                    "event": "close", "symbol": symbol, "entry_time": position.entry_time,
                    "exit_time": ts, "entry_price": position.entry_price, "exit_price": exit_price,
                    "quantity": position.original_quantity, "reason": reason, "pnl": pnl,
                    "pnl_pct": pnl / position.total_entry_cash_out * 100 if position.total_entry_cash_out else 0.0,
                    "holding_days": held_days, "entry_regime": position.entry_regime,
                })
                del open_positions[symbol]

        # ---- 1b. End-of-day square-off for positions not eligible to carry ----
        # >= rather than == on purpose: real feeds don't always have a bar
        # landing exactly on 15:15 (half days, missed prints, timestamp
        # drift). This fires on every bar from 15:15 onward, but a position
        # is removed from open_positions the moment it's closed, so it's
        # only ever actually squared off once -- this just makes sure a
        # missing 15:15 bar specifically doesn't let a position slip through
        # uncaught for the rest of the day.
        if ts_time >= config.INTRADAY_SQUARE_OFF_TIME:
            for symbol in list(open_positions.keys()):
                df = feats[symbol]
                if ts not in df.index:
                    continue
                bar = df.loc[ts]
                position = open_positions[symbol]
                carry_eligible = (
                    position.entry_regime == "TRENDING"
                    and bar["regime"] == "TRENDING"
                    and bar["ema_fast"] > bar["ema_slow"]
                )
                if carry_eligible:
                    continue
                exit_price = bar["close"]
                same_day = ts.date() == position.entry_time.date()
                proceeds = exit_price * position.quantity - costs.exit_cost(exit_price, position.quantity, same_day)
                cash += proceeds
                position.total_exit_cash_in += proceeds
                pnl = position.total_exit_cash_in - position.total_entry_cash_out
                held_days = day_idx[symbol][ts.date()] - day_idx[symbol][position.entry_time.date()]
                day_realized_pnl += pnl
                trade_log.append({
                    "event": "close", "symbol": symbol, "entry_time": position.entry_time,
                    "exit_time": ts, "entry_price": position.entry_price, "exit_price": exit_price,
                    "quantity": position.original_quantity, "reason": "eod_squareoff", "pnl": pnl,
                    "pnl_pct": pnl / position.total_entry_cash_out * 100 if position.total_entry_cash_out else 0.0,
                    "holding_days": held_days, "entry_regime": position.entry_regime,
                })
                del open_positions[symbol]

        # ---- 2. New entries (signal must have fired on the PREVIOUS bar) ----
        if ts_time < config.INTRADAY_SQUARE_OFF_TIME:
            for symbol, df in feats.items():
                if symbol in open_positions or ts not in df.index:
                    continue
                loc = df.index.get_loc(ts)
                if loc == 0:
                    continue
                prev_bar = df.iloc[loc - 1]
                if not prev_bar["raw_signal"]:
                    continue
                if not rm.can_open_new_position(list(open_positions.values()), equity_for_sizing, day_realized_pnl):
                    continue

                bar = df.loc[ts]
                entry_price = bar["open"]
                if pd.isna(prev_bar["atr"]):
                    continue
                qty, stop, target, risk_per_share = rm.position_size(equity_for_sizing, entry_price, prev_bar["atr"])
                if qty <= 0:
                    continue
                entry_cash_out = entry_price * qty + costs.entry_cost(entry_price, qty)
                if entry_cash_out > cash:      # can't afford it -- skip rather than over-leverage
                    continue
                cash -= entry_cash_out

                open_positions[symbol] = rm.Position(
                    symbol=symbol,
                    entry_time=ts,
                    entry_price=entry_price,
                    quantity=qty,
                    stop_price=stop,
                    target_price=target,
                    initial_risk_per_share=risk_per_share,
                    entry_regime=prev_bar["regime"],
                    original_quantity=qty,
                    total_entry_cash_out=entry_cash_out,
                    total_exit_cash_in=0.0,
                )

        # ---- 3. Mark-to-market equity for the curve, and next bar's sizing basis ----
        position_value = 0.0

        for symbol, position in open_positions.items():
            df = feats[symbol]

            if ts not in df.index:
                continue

            position_value += (
                df.loc[ts, "close"] * position.quantity
            )

        equity_for_sizing = cash + position_value
        equity_points.append((ts, equity_for_sizing))

    equity_curve = pd.Series({t: v for t, v in equity_points}).sort_index()
    return {"equity_curve": equity_curve, "trade_log": trade_log}
