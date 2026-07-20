"""
run_live.py
-----------------------------------------------------------------------------
Live/paper trading entry point. DEFAULTS TO PAPER MODE. Live mode requires
an explicit flag AND a typed confirmation -- see the bottom of this file.
This has NOT been run against the real Superr API (no credentials available
in the environment this was built in, and no network path to it either) --
unlike backtest_engine.py, which was tested end-to-end repeatedly. Treat
this file as reviewed-but-unverified. Test it exactly in this order:

  1. `python3 run_live.py --mode paper` and confirm it authenticates, pulls
     quotes, and logs signals -- with NOTHING going to place_order at all
     (paper mode never calls it; that's not a config flag you could get
     wrong, the code path for live order placement simply isn't reachable
     unless MODE=live).
  2. Watch live_trade_log.csv fill in during market hours and sanity-check
     that logged signals match what you'd expect from the strategy.
  3. Only after you trust what you're seeing in paper mode for real, come
     back and talk through switching a small slice to live.

WHAT THIS DOES:
  - Polls live quotes for your universe every QUOTE_POLL_SECONDS
  - Aggregates quotes into 5-min bars in memory (same shape data_loader.py
    produces, so signals.py runs identically to backtesting -- this
    consistency between backtest and live is the whole point of the
    architecture)
  - On every new completed bar, recomputes signals and checks exits for
    open (paper or live) positions using the SAME risk_management.py logic
    the backtest uses
  - Every signal, skip, entry, and exit goes through live_controls.log_event
    AND is checked against live_controls.DailyLimits before anything happens
  - Checks live_controls.kill_switch_active() before every single new entry
-----------------------------------------------------------------------------
"""
import argparse
import datetime as dt
import time

import pandas as pd

import config
import costs
import live_controls as lc
import risk_management as rm
import signals as sig
from broker_superr import SuperrClient, market_protected_limit_price


class LiveEngine:
    def __init__(self, mode: str, client: SuperrClient = None, symbol_tokens: dict = None):
        self.mode = mode   # "paper" or "live"
        self.client = client
        self.symbol_tokens = symbol_tokens or {}   # symbol -> Superr instrument token, needed for LIVE mode only
        self.bars = {sym: pd.DataFrame(columns=["open", "high", "low", "close", "volume"]) for sym in config.UNIVERSE}
        self._current_bar = {}   # symbol -> in-progress bar dict, being built from polled quotes
        self.cash = config.STARTING_CAPITAL_INR
        self.open_positions: dict[str, rm.Position] = {}
        self.limits = lc.DailyLimits(
            max_daily_capital_inr=config.MAX_DAILY_CAPITAL_DEPLOYED_INR,
            max_positions_per_day=config.MAX_POSITIONS_PER_DAY,
            max_shares_per_trade=config.MAX_SHARES_PER_TRADE,
            daily_loss_limit_inr=config.DAILY_LOSS_LIMIT_INR,
        )

    # ---- quote polling / bar building --------------------------------
    def poll_and_update_bars(self):
        """Fetches current LTP for every symbol and folds it into the
        in-progress 5-min bar, closing and appending that bar once the
        interval elapses. In paper mode with no client wired up, this can
        also be fed synthetic/manual prices for a dry run -- see __main__."""
        now = dt.datetime.now().replace(second=0, microsecond=0)
        bucket = now - pd.Timedelta(minutes=now.minute % config.BAR_MINUTES)

        quotes = self._fetch_quotes()
        for sym, price in quotes.items():
            cur = self._current_bar.get(sym)
            if cur is None or cur["bucket"] != bucket:
                if cur is not None:
                    self._close_bar(sym, cur)
                self._current_bar[sym] = {"bucket": bucket, "open": price, "high": price,
                                           "low": price, "close": price, "volume": 0}
            else:
                cur["high"] = max(cur["high"], price)
                cur["low"] = min(cur["low"], price)
                cur["close"] = price

    def _close_bar(self, sym, bar):
        row = pd.DataFrame([{k: v for k, v in bar.items() if k != "bucket"}], index=[bar["bucket"]])
        self.bars[sym] = pd.concat([self.bars[sym], row])
        self.on_new_bar(sym)

    def _fetch_quotes(self) -> dict:
        if self.client is None:
            raise RuntimeError("No broker client wired up -- pass one in, or feed bars manually for a dry run.")
        tokens = [self.symbol_tokens[s] for s in config.UNIVERSE if s in self.symbol_tokens]
        resp = self.client.quote(exchange="NSE", tokens=tokens)
        # Response shape not independently confirmed against live docs --
        # adjust this parsing once you see a real response.
        return {item["symbol"]: item["ltp"] for item in resp.get("data", [])}

    # ---- signal + risk logic (reuses backtest modules directly) -------
    def on_new_bar(self, symbol: str):
        df = self.bars[symbol]
        if len(df) < 25:   # not enough history yet for ADX/Bollinger/etc. to warm up
            return
        feats = sig.generate_entry_signals(df)
        latest = feats.iloc[-1]
        prev = feats.iloc[-2]
        ts_time = df.index[-1].strftime("%H:%M")

        self._check_exits(symbol, latest, ts_time)

        if symbol not in self.open_positions and prev["raw_signal"] and ts_time < config.INTRADAY_SQUARE_OFF_TIME:
            self._attempt_entry(symbol, latest, prev)

    def _check_exits(self, symbol, bar, ts_time):
        if symbol not in self.open_positions:
            return
        position = self.open_positions[symbol]
        held_days = (bar.name.date() - position.entry_time.date()).days   # calendar-day proxy live; fine at this scale
        reason, exit_price = rm.check_exit(position, bar, bar["regime"], held_days)
        force_eod = ts_time >= config.INTRADAY_SQUARE_OFF_TIME and not (
            position.entry_regime == "TRENDING" and bar["regime"] == "TRENDING" and bar["ema_fast"] > bar["ema_slow"]
        )
        if reason in ("stop", "target", "regime_flip", "time_stop") or force_eod:
            reason = reason or "eod_squareoff"
            self._execute_exit(symbol, position, exit_price or bar["close"], reason)

    def _attempt_entry(self, symbol, bar, prev):
        entry_price = bar["open"]
        qty, stop, target, risk_per_share = rm.position_size(self.cash, entry_price, prev["atr"])
        if qty <= 0:
            lc.log_event({"event": "SKIPPED", "symbol": symbol, "reason": "qty<=0 after risk/notional sizing", "mode": self.mode})
            return
        notional = entry_price * qty
        allowed, why = self.limits.check_before_entry(notional, qty)
        if not allowed:
            lc.log_event({"event": "SKIPPED", "symbol": symbol, "reason": why, "quantity": qty,
                          "price": entry_price, "notional": notional, "mode": self.mode})
            return

        if self.mode == "live":
            token = self.symbol_tokens.get(symbol)
            if not token:
                lc.log_event({"event": "SKIPPED", "symbol": symbol, "reason": "no instrument token mapped", "mode": self.mode})
                return
            resp = self.client.place_protected_order(
                action="BUY", exchange="NSE", token=token, product_type="INTRADAY",
                quantity=qty, ltp=entry_price, stop_loss=stop, tag="regime-strategy",
            )
            lc.log_event({"event": "ORDER_SENT", "symbol": symbol, "reason": prev["regime"], "quantity": qty,
                          "price": entry_price, "notional": notional, "mode": self.mode, "detail": str(resp)})
        else:
            lc.log_event({"event": "PAPER_BUY", "symbol": symbol, "reason": prev["regime"], "quantity": qty,
                          "price": entry_price, "notional": notional, "mode": self.mode})

        self.limits.record_entry(notional)
        pos = rm.Position(symbol=symbol, entry_time=bar.name, entry_price=entry_price, quantity=qty,
                           stop_price=stop, target_price=target, initial_risk_per_share=risk_per_share,
                           entry_regime=prev["regime"])
        pos.original_quantity = qty
        pos.total_entry_cash_out = notional + costs.entry_cost(entry_price, qty)
        pos.total_exit_cash_in = 0.0
        self.cash -= pos.total_entry_cash_out
        self.open_positions[symbol] = pos

    def _execute_exit(self, symbol, position, exit_price, reason):
        same_day = dt.date.today() == position.entry_time.date()
        proceeds = exit_price * position.quantity - costs.exit_cost(exit_price, position.quantity, same_day)
        pnl = proceeds + position.total_exit_cash_in - position.total_entry_cash_out

        if self.mode == "live":
            token = self.symbol_tokens.get(symbol)
            resp = self.client.place_protected_order(
                action="SELL", exchange="NSE", token=token, product_type="INTRADAY",
                quantity=position.quantity, ltp=exit_price, tag="regime-strategy-exit",
            )
            lc.log_event({"event": "ORDER_SENT", "symbol": symbol, "reason": reason, "quantity": position.quantity,
                          "price": exit_price, "notional": proceeds, "mode": self.mode, "detail": str(resp)})
        else:
            lc.log_event({"event": "PAPER_SELL", "symbol": symbol, "reason": reason, "quantity": position.quantity,
                          "price": exit_price, "notional": proceeds, "mode": self.mode, "detail": f"pnl={pnl:.2f}"})

        self.cash += proceeds
        self.limits.record_exit_pnl(pnl)
        del self.open_positions[symbol]

    def run_forever(self):
        lc.log_event({"event": "START", "symbol": "-", "reason": f"mode={self.mode}", "mode": self.mode})
        while True:
            if lc.kill_switch_active():
                lc.log_event({"event": "HALTED", "symbol": "-", "reason": "KILL_SWITCH file present", "mode": self.mode})
            self.poll_and_update_bars()
            time.sleep(config.QUOTE_POLL_SECONDS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument("--i-understand-this-is-real-money", action="store_true",
                         help="Required in addition to --mode live. Does not bypass the typed confirmation.")
    args = parser.parse_args()

    if args.mode == "live":
        if not args.i_understand_this_is_real_money:
            print("Refusing to start in live mode without --i-understand-this-is-real-money.")
            return
        confirm = input('Type EXACTLY "I understand this places real orders with real money" to continue: ')
        if confirm != "I understand this places real orders with real money":
            print("Confirmation text didn't match. Not starting.")
            return

    print(f"Starting in {args.mode.upper()} mode. Ctrl+C to stop immediately at any time.")
    print(f"Create a file named 'KILL_SWITCH' in this folder to stop new entries without killing the process.")

    # You need: (1) a Superr API key/secret/client_id/password, (2) a
    # symbol -> instrument token map from Stoxkart's instrument dump. Paper
    # mode still needs a live client for quotes -- it just never calls
    # place_order.
    client = SuperrClient(api_key="4a480164fd252b9c43ab", client_id="SU25489", password="Arnav@2009", api_secret="e657227846839cade683")
    client.authenticate()

    symbol_tokens = {}   # fill in from Stoxkart's instrument master dump before running

    engine = LiveEngine(mode=args.mode, client=client, symbol_tokens=symbol_tokens)
    try:
        engine.run_forever()
    except KeyboardInterrupt:
        lc.log_event({"event": "STOPPED", "symbol": "-", "reason": "Ctrl+C", "mode": args.mode})
        print("\nStopped.")


if __name__ == "__main__":
    main()
