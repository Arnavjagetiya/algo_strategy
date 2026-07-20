"""
live_controls.py
-----------------------------------------------------------------------------
Safety layer for live/paper trading. Nothing in here talks to the Superr
API -- it's pure control logic, which makes it easy to test and to trust
on its own merits before it's anywhere near a real order.

KILL SWITCH -- two independent mechanisms, on purpose:

  1. INSTANT: press Ctrl+C in the terminal running run_live.py, or just
     close the terminal window. This kills the Python process immediately.
     Nothing more can be sent, because nothing is running anymore. This is
     the one to use if something looks badly wrong -- don't look for a
     button, just Ctrl+C.

  2. GRACEFUL: create an empty file named exactly `KILL_SWITCH` in the
     algo_strategy/ folder (Finder/Explorer works fine, or
     `touch KILL_SWITCH` in the terminal). run_live.py checks for this file
     before every single new order. If present, it stops opening new
     positions but keeps running and keeps managing any positions already
     open (so your existing stops/targets/time-stops still work) --
     use this when you want to stop NEW activity but let what's already
     running finish safely rather than yanking the process. Delete the
     file (or run clear_kill_switch()) to resume.

Ctrl+C is the one to trust in an emergency. The file-based switch is for
"stop starting new things, let existing things finish."
-----------------------------------------------------------------------------
"""
import csv
import datetime as dt
import os

KILL_SWITCH_FILE = "KILL_SWITCH"
LOG_FILE = "live_trade_log.csv"


def kill_switch_active() -> bool:
    return os.path.exists(KILL_SWITCH_FILE)


def activate_kill_switch(reason: str = "manual"):
    with open(KILL_SWITCH_FILE, "w") as f:
        f.write(f"Activated: {dt.datetime.now().isoformat()}\nReason: {reason}\n")
    print(f"KILL SWITCH ACTIVATED ({reason}). No new positions will open. "
          f"Existing positions are still managed. Delete the file '{KILL_SWITCH_FILE}' to resume.")


def clear_kill_switch():
    if os.path.exists(KILL_SWITCH_FILE):
        os.remove(KILL_SWITCH_FILE)
        print("Kill switch cleared. New positions can open again.")


class DailyLimits:
    """
    Tracks today's deployed capital, position count, and realized P&L
    against hard caps you set. check_before_entry() must be called -- and
    must return (True, ...) -- before every single order this system sends.
    Resets automatically at the first check of a new calendar day.
    """

    def __init__(self, max_daily_capital_inr: float, max_positions_per_day: int,
                 max_shares_per_trade: int, daily_loss_limit_inr: float):
        self.max_daily_capital_inr = max_daily_capital_inr
        self.max_positions_per_day = max_positions_per_day
        self.max_shares_per_trade = max_shares_per_trade
        self.daily_loss_limit_inr = daily_loss_limit_inr
        self._reset_for_new_day()

    def _reset_for_new_day(self):
        self.date = dt.date.today()
        self.capital_deployed_today = 0.0
        self.positions_opened_today = 0
        self.realized_pnl_today = 0.0

    def _roll_if_new_day(self):
        if dt.date.today() != self.date:
            self._reset_for_new_day()

    def check_before_entry(self, notional: float, quantity: int):
        """Returns (allowed: bool, reason: str). ALWAYS check the bool --
        never place an order just because this didn't raise an exception."""
        self._roll_if_new_day()
        if kill_switch_active():
            return False, "KILL_SWITCH file is present"
        if self.realized_pnl_today <= -self.daily_loss_limit_inr:
            return False, f"Daily loss limit hit (Rs {self.realized_pnl_today:,.2f} realized today)"
        if self.positions_opened_today >= self.max_positions_per_day:
            return False, f"Max new positions for today reached ({self.max_positions_per_day})"
        if quantity > self.max_shares_per_trade:
            return False, f"Quantity {quantity} exceeds MAX_SHARES_PER_TRADE ({self.max_shares_per_trade})"
        if self.capital_deployed_today + notional > self.max_daily_capital_inr:
            return False, (f"Would push today's deployed capital to Rs {self.capital_deployed_today + notional:,.0f}, "
                            f"over the Rs {self.max_daily_capital_inr:,.0f} daily cap")
        return True, "ok"

    def record_entry(self, notional: float):
        self.capital_deployed_today += notional
        self.positions_opened_today += 1

    def record_exit_pnl(self, pnl: float):
        self.realized_pnl_today += pnl

    def status(self) -> dict:
        self._roll_if_new_day()
        return {
            "date": str(self.date),
            "capital_deployed_today": round(self.capital_deployed_today, 2),
            "capital_remaining_today": round(self.max_daily_capital_inr - self.capital_deployed_today, 2),
            "positions_opened_today": self.positions_opened_today,
            "positions_remaining_today": self.max_positions_per_day - self.positions_opened_today,
            "realized_pnl_today": round(self.realized_pnl_today, 2),
            "distance_to_daily_loss_limit": round(self.daily_loss_limit_inr + self.realized_pnl_today, 2),
            "kill_switch_active": kill_switch_active(),
        }


_csv_fields = ["timestamp", "event", "symbol", "reason", "quantity", "price", "notional", "mode", "detail"]


def log_event(event: dict):
    """Every signal, skip, and order goes through here: printed to the
    console immediately AND appended to live_trade_log.csv, so you always
    have both a live view and a durable record to open in Excel/Numbers
    afterward. This is the 'see what all trades are being made' piece."""
    row = {k: event.get(k, "") for k in _csv_fields}
    row["timestamp"] = dt.datetime.now().isoformat(timespec="seconds")
    write_header = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[{row['timestamp']}] {row['event']:>10} | {row['symbol']:<10} | {row['reason']}")
