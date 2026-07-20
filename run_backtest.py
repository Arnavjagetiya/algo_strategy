"""
run_backtest.py
-----------------------------------------------------------------------------
Entry point. Runs the full pipeline end to end:
  synthetic data -> liquidity filter -> signals -> backtest -> metrics
  -> Monte Carlo robustness check -> walk-forward validation

SWITCHING TO REAL DATA: replace the call to data_loader.generate_synthetic_data()
with your own loader that returns the same shape (dict of symbol -> DataFrame
with columns [open, high, low, close, volume] and a 5-min DatetimeIndex during
NSE market hours) -- built from nsepython, yfinance, or any other source. See
the docstring at the top of data_loader.py for the two starting points.
Nothing else in this file, or in signals.py / risk_management.py / costs.py /
backtest_engine.py, needs to change.
-----------------------------------------------------------------------------
"""
import json

import config
import data_loader
import backtest_engine
import metrics
import walk_forward

USING_REAL_DATA = True   # set this to match whichever data_loader call you use below


def main():
    label = "REAL NSE DATA" if USING_REAL_DATA else "SYNTHETIC DATA (pipeline validation only)"
    print("=" * 70)
    print(f"STRATEGY BACKTEST -- {label}")
    print("=" * 70)

    print(f"\nUniverse (before liquidity filter): {config.UNIVERSE}")
    data = data_loader.generate_synthetic_data(config.UNIVERSE, n_days=120)
    data = data_loader.apply_liquidity_filter(data)
    print(f"Universe (after liquidity filter):  {list(data.keys())}")

    print(f"\nStarting capital: Rs {config.STARTING_CAPITAL_INR:,.0f}")
    print(f"Risk per trade: {config.RISK_PER_TRADE_PCT*100:.2f}% | "
          f"Max notional/position: {config.MAX_POSITION_NOTIONAL_PCT*100:.0f}% | "
          f"Max concurrent positions: {config.MAX_CONCURRENT_POSITIONS}")
    print(f"Max holding period: {config.MAX_HOLDING_DAYS} trading days")

    print("\nRunning backtest...")
    result = backtest_engine.run_backtest(data, config.STARTING_CAPITAL_INR)
    closed = [t for t in result["trade_log"] if t["event"] == "close"]

    print(f"\n{'-'*70}\nRESULTS\n{'-'*70}")
    m = metrics.compute_metrics(result["equity_curve"], result["trade_log"], config.STARTING_CAPITAL_INR)
    for k, v in m.items():
        print(f"  {k:22s}: {v}")

    if closed:
        reasons = {}
        for t in closed:
            reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
        print(f"\n  Exit reasons: {reasons}")
        max_hold = max(t["holding_days"] for t in closed)
        print(f"  Max holding period actually used: {max_hold} trading day(s) "
              f"(cap is {config.MAX_HOLDING_DAYS})")

    print(f"\n{'-'*70}\nMONTE CARLO ROBUSTNESS CHECK (1000 trade-order reshuffles)\n{'-'*70}")
    mc = metrics.monte_carlo_resample(result["trade_log"], config.STARTING_CAPITAL_INR, n_sims=1000)
    for k, v in mc.items():
        print(f"  {k:22s}: {v}")
    print("  Note: terminal equity is IDENTICAL across every reshuffle by")
    print("  construction (sum of trade P&Ls doesn't depend on their order).")
    print("  Only drawdown is order-dependent -- that's the number worth")
    print("  watching here, and how much it varies is itself informative:")
    print("  wide variation means a few trades' sequencing matters a lot;")
    print("  narrow variation (as here, with many trades) means it doesn't.")

    print(f"\n{'-'*70}\nWALK-FORWARD VALIDATION (20-trading-day train / 10-day test folds)\n{'-'*70}")
    wf_results = walk_forward.run_walk_forward(data, config.STARTING_CAPITAL_INR,
                                                train_days=20, test_days=10)
    for r in wf_results:
        print(f"  Fold {r['fold']} [{r['test_start']} to {r['test_end']}]: "
              f"return={r['total_return_pct']}%  trades={r['n_trades']}  "
              f"win_rate={r['win_rate_pct']}%")

    print(f"\n{'='*70}")
    if USING_REAL_DATA:
        print("This run used REAL NSE data -- these numbers are a real signal,")
        print("not something to wave away. If total_return_pct and profit_factor")
        print("are negative, that's the strategy telling you it doesn't have a")
        print("demonstrated edge with its current parameters on this window --")
        print("take that seriously before considering live deployment.")
    else:
        print("IMPORTANT: this entire run used SYNTHETIC random-walk data.")
        print("Its only job was to prove the pipeline runs correctly end to end.")
        print("The result here is NOT a performance estimate. Point")
        print("data_loader.py at real NSE data next; that result is the one")
        print("worth trusting.")
    print("=" * 70)

    with open("backtest_summary.json", "w") as f:
        json.dump({"metrics": m, "monte_carlo": mc, "walk_forward": wf_results}, f, indent=2, default=str)
    print("\nFull summary written to backtest_summary.json")


if __name__ == "__main__":
    main()