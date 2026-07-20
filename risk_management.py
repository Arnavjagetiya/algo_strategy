"""
risk_management.py
-----------------------------------------------------------------------------
Everything about HOW MUCH to trade and WHEN to get out, independent of
what triggered the entry. Deliberately separated from signals.py: a good
signal with bad risk management still loses money over time, and this
separation lets you swap signal logic later without touching risk logic.
-----------------------------------------------------------------------------
"""
from dataclasses import dataclass, field
import pandas as pd

import config


@dataclass
class Position:
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    quantity: int                          # remaining (live) quantity -- shrinks on partial exits
    stop_price: float
    target_price: float
    initial_risk_per_share: float
    scaled_out: bool = False
    entry_regime: str = ""
    original_quantity: int = 0             # quantity at entry, fixed -- used for trade-log sizing
    total_entry_cash_out: float = 0.0      # cash spent opening the position (incl. costs)
    total_exit_cash_in: float = 0.0        # cash recovered across all partial + final exits


def position_size(capital: float, entry_price: float, atr_value: float):
    """
    Volatility-adjusted position sizing with TWO independent constraints,
    taking whichever gives the smaller quantity:

      1. Risk-based: every trade risks the same rupee amount
         (RISK_PER_TRADE_PCT of capital), sized inversely to the stop
         distance -- a quiet stock gets a bigger position, a wild one a
         smaller one, so each trade carries comparable risk.
      2. Notional-based: no single position's total value exceeds
         MAX_POSITION_NOTIONAL_PCT of capital, full stop.

    Constraint 1 alone is not enough: when the stop is tight relative to
    price (small ATR relative to entry price), risk-based sizing alone can
    demand a share quantity whose total value exceeds your entire account
    -- technically "correct" risk in rupee terms, but requiring leverage
    you don't have. Constraint 2 catches that case.
    """
    risk_amount = capital * config.RISK_PER_TRADE_PCT
    stop_distance = config.ATR_STOP_MULT * atr_value
    if stop_distance <= 0 or entry_price <= 0:
        return 0, 0.0, 0.0, 0.0

    risk_based_qty = int(risk_amount // stop_distance)
    max_notional = capital * config.MAX_POSITION_NOTIONAL_PCT
    notional_based_qty = int(max_notional // entry_price)
    quantity = max(0, min(risk_based_qty, notional_based_qty))

    stop_price = entry_price - stop_distance
    target_price = entry_price + config.REWARD_RISK_RATIO * stop_distance
    return quantity, stop_price, target_price, stop_distance


def check_portfolio_heat(open_positions: list, capital: float) -> float:
    """Sum of open risk (entry - stop, times quantity, for every open
    position) as a fraction of capital -- this is what MAX_PORTFOLIO_HEAT_PCT
    caps, independent of how many individual positions that risk is spread
    across."""
    total_risk = sum(
        (p.entry_price - p.stop_price) * p.quantity for p in open_positions
    )
    return total_risk / capital if capital > 0 else 0.0


def can_open_new_position(open_positions: list, capital: float, day_pnl: float) -> bool:
    if len(open_positions) >= config.MAX_CONCURRENT_POSITIONS:
        return False
    if check_portfolio_heat(open_positions, capital) >= config.MAX_PORTFOLIO_HEAT_PCT:
        return False
    if day_pnl <= -config.DAILY_LOSS_LIMIT_PCT * capital:   # kill switch
        return False
    return True


def check_exit(position: Position, bar: pd.Series, current_regime: str, bars_held_days: int):
    """
    Returns one of:
      ("stop", exit_price)          -- stop-loss hit
      ("target", exit_price)        -- full target hit
      ("scale_out", exit_price)     -- partial profit-take, stop moves to breakeven
      ("regime_flip", exit_price)   -- the thesis this trade was entered on no longer holds
      ("time_stop", exit_price)     -- hit the max holding period (your weekly cap)
      (None, None)                  -- keep holding
    Checked in this priority order every bar for every open position.
    """
    low, high, close = bar["low"], bar["high"], bar["close"]

    if low <= position.stop_price:
        return "stop", position.stop_price

    if not position.scaled_out:
        r_target = position.entry_price + config.PARTIAL_SCALE_OUT_R * position.initial_risk_per_share
        if high >= r_target:
            return "scale_out", r_target

    if high >= position.target_price:
        return "target", position.target_price

    if position.entry_regime == "TRENDING" and current_regime == "RANGING":
        return "regime_flip", close

    # bars_held_days is 0 on the ENTRY day itself (day_idx[now] - day_idx[entry]).
    # So "this position has touched MAX_HOLDING_DAYS distinct trading days,
    # counting the entry day" means bars_held_days == MAX_HOLDING_DAYS - 1,
    # not MAX_HOLDING_DAYS. The old `>= MAX_HOLDING_DAYS` let a position
    # survive into a 6th distinct day before forcing it closed when the cap
    # was meant to be 5 -- confirmed by the real-data run showing exactly
    # that: "Max holding period actually used: 6 trading days (cap is 5)".
    if bars_held_days >= config.MAX_HOLDING_DAYS - 1:
        return "time_stop", close

    # Defensive backstop, independent of the trading-day-index arithmetic
    # above: real feeds can have gaps (a stock-specific halt, a data quality
    # hole, a holiday that doesn't line up across symbols) that a clean
    # index-based day count can be fooled by. This catches those cases even
    # if the logic above somehow doesn't. It should essentially never be the
    # one that fires -- if it does, that's worth investigating separately.
    calendar_days_held = (bar.name.date() - position.entry_time.date()).days
    if calendar_days_held >= 8:
        return "time_stop", close

    return None, None
