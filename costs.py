"""
costs.py
-----------------------------------------------------------------------------
Realistic Indian cash-equity transaction costs. This is the single most
common way retail backtests lie to you: a strategy that looks profitable
ignoring costs can be a net loser once brokerage, taxes, and slippage are
included -- especially for a higher-frequency intraday strategy where costs
are paid on every round trip. This module is deliberately explicit rather
than a single "fudge factor" percentage.

STT is intraday-rate if the position was opened and closed the same day,
and switches to the higher delivery-rate (both legs) the moment a position
survives past its entry day -- because your strategy explicitly allows
carrying a position up to 5 trading days, that distinction meaningfully
changes the total cost of any carried trade and needs to be modeled, not
approximated away.
-----------------------------------------------------------------------------
"""
import config


def entry_cost(fill_price: float, quantity: float) -> float:
    """Costs on the BUY leg: brokerage, exchange charges, GST, stamp duty, slippage."""
    turnover = fill_price * quantity
    brokerage = config.BROKERAGE_PER_ORDER_INR
    exchange_charge = turnover * config.EXCHANGE_TXN_CHARGE_PCT
    gst = (brokerage + exchange_charge) * config.GST_PCT
    stamp_duty = turnover * config.STAMP_DUTY_BUY_PCT
    slippage = turnover * (config.SLIPPAGE_BPS / 10_000)
    return brokerage + exchange_charge + gst + stamp_duty + slippage


def exit_cost(fill_price: float, quantity: float, same_day: bool) -> float:
    """Costs on the SELL leg: brokerage, STT (rate depends on same_day),
    exchange charges, GST, slippage. No stamp duty on the sell leg (buyer pays)."""
    turnover = fill_price * quantity
    brokerage = config.BROKERAGE_PER_ORDER_INR
    stt_rate = config.STT_INTRADAY_SELL_PCT if same_day else config.STT_DELIVERY_PCT
    stt = turnover * stt_rate
    exchange_charge = turnover * config.EXCHANGE_TXN_CHARGE_PCT
    gst = (brokerage + exchange_charge) * config.GST_PCT
    slippage = turnover * (config.SLIPPAGE_BPS / 10_000)
    return brokerage + stt + exchange_charge + gst + slippage


def round_trip_cost(entry_price: float, exit_price: float, quantity: float, same_day: bool) -> float:
    return entry_cost(entry_price, quantity) + exit_cost(exit_price, quantity, same_day)
