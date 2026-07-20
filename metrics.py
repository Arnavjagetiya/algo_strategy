"""
metrics.py
-----------------------------------------------------------------------------
Performance metrics for evaluating a backtest, plus a lightweight Monte
Carlo robustness check. A single equity curve is one sample path -- Monte
Carlo resampling of the trade sequence gives you a distribution of possible
outcomes instead of trusting the one path you happened to get, which
matters more the fewer trades you have.
-----------------------------------------------------------------------------
"""
import numpy as np
import pandas as pd


BARS_PER_DAY = 75          # ~6h15m session / 5-min bars
TRADING_DAYS_PER_YEAR = 252


def compute_metrics(equity_curve: pd.Series, trade_log: list, starting_capital: float) -> dict:
    equity_curve = equity_curve.sort_index()
    daily_equity = equity_curve.resample("1D").last().dropna()
    daily_returns = daily_equity.pct_change().dropna()

    total_return_pct = (equity_curve.iloc[-1] / starting_capital - 1) * 100
    n_days = max((equity_curve.index[-1] - equity_curve.index[0]).days, 1)
    cagr_pct = ((equity_curve.iloc[-1] / starting_capital) ** (365 / n_days) - 1) * 100 if equity_curve.iloc[-1] > 0 else float("nan")

    sharpe = float("nan")
    sortino = float("nan")
    if daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
    downside = daily_returns[daily_returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (daily_returns.mean() / downside.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_dd_pct = drawdown.min() * 100

    calmar = (cagr_pct / abs(max_dd_pct)) if max_dd_pct != 0 else float("nan")

    closed_trades = [t for t in trade_log if t["event"] == "close"]
    n_trades = len(closed_trades)
    wins = [t for t in closed_trades if t["pnl"] > 0]
    losses = [t for t in closed_trades if t["pnl"] <= 0]
    win_rate = len(wins) / n_trades * 100 if n_trades else float("nan")
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0.0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0.0
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    expectancy = (
        (win_rate / 100.0) * avg_win
        + (1 - win_rate / 100.0) * avg_loss
    ) if n_trades else 0.0
    avg_holding = (
        np.mean([t["holding_days"] for t in closed_trades])
        if closed_trades
        else 0.0
    )
    largest_win = max(
        (t["pnl"] for t in closed_trades),
        default=0.0
    )

    largest_loss = min(
        (t["pnl"] for t in closed_trades),
        default=0.0
    )

    max_consecutive_losses = 0
    current_losses = 0

    for trade in closed_trades:

        if trade["pnl"] <= 0:
            current_losses += 1
            max_consecutive_losses = max(
                max_consecutive_losses,
                current_losses,
            )
        else:
            current_losses = 0

    net_profit = equity_curve.iloc[-1] - starting_capital

    recovery_factor = (
        net_profit / abs(max_dd_pct)
        if max_dd_pct != 0
        else float("nan")
    )

    return {
        "total_return_pct": round(total_return_pct, 2),
        "cagr_pct": round(cagr_pct, 2) if not np.isnan(cagr_pct) else None,
        "sharpe_ratio": round(sharpe, 2) if not np.isnan(sharpe) else None,
        "sortino_ratio": round(sortino, 2) if not np.isnan(sortino) else None,
        "max_drawdown_pct": round(max_dd_pct, 2),
        "calmar_ratio": round(calmar, 2) if not np.isnan(calmar) else None,
        "n_trades": n_trades,
        "win_rate_pct": round(win_rate, 2) if not np.isnan(win_rate) else None,
        "avg_win_inr": round(avg_win, 2),
        "avg_loss_inr": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf (no losing trades)",
        "expectancy_inr": round(expectancy, 2),

        "average_holding_days": round(avg_holding, 2),

        "largest_win_inr": round(largest_win, 2),

        "largest_loss_inr": round(largest_loss, 2),

        "max_consecutive_losses": max_consecutive_losses,

        "recovery_factor": round(recovery_factor, 2)
        if not np.isnan(recovery_factor)
        else None,
    }


def monte_carlo_resample(trade_log: list, starting_capital: float, n_sims: int = 1000, seed: int = 7) -> dict:
    """
    Shuffles the ORDER of completed trades (not their outcomes) n_sims times
    and replays the resulting equity path. A strategy whose profitability
    depends on lucky trade ordering will show a wide, unstable spread of
    terminal equity here even though the individual trade outcomes never
    changed -- that instability is the warning sign, not the average.
    """
    closed_trades = [t for t in trade_log if t["event"] == "close"]
    pnls = np.array([t["pnl"] for t in closed_trades])
    if len(pnls) == 0:
        return {"note": "no closed trades to resample"}

    rng = np.random.default_rng(seed)
    terminal_equities = []
    max_drawdowns = []
    for _ in range(n_sims):
        shuffled = rng.permutation(pnls)
        path = starting_capital + np.cumsum(shuffled)
        terminal_equities.append(path[-1])
        running_max = np.maximum.accumulate(path)
        dd = (path - running_max) / running_max
        max_drawdowns.append(dd.min() * 100)

    terminal_equities = np.array(terminal_equities)
    return {
        "n_simulations": n_sims,
        "median_terminal_equity": round(float(np.median(terminal_equities)), 2),
        "p5_terminal_equity": round(float(np.percentile(terminal_equities, 5)), 2),
        "p95_terminal_equity": round(float(np.percentile(terminal_equities, 95)), 2),
        "median_max_drawdown_pct": round(float(np.median(max_drawdowns)), 2),
        "worst_case_max_drawdown_pct": round(float(np.percentile(max_drawdowns, 5)), 2),
    }
