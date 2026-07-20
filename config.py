"""
config.py
-----------------------------------------------------------------------------
Central configuration for the intraday-cash + max-5-day-carry strategy.
Every tunable knob lives here so nothing is buried inside logic files —
when you re-run backtests with different parameters, this is the only
file you should need to touch.
-----------------------------------------------------------------------------
"""

# ---- Universe -----------------------------------------------------------
# Start small and liquid. Superr instrument "tokens" are not the same as
# ticker symbols -- map symbol -> token using the broker's daily instrument
# dump before you wire this to live/paper data.
UNIVERSE = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
            "SBIN", "ITC", "LT", "AXISBANK", "KOTAKBANK"]

MIN_ADV_INR = 20_00_00_000     # min. ~20 crore average daily traded value (liquidity filter)
ATR_PCT_MIN = 0.4              # skip stocks that barely move (dead / illiquid)
ATR_PCT_MAX = 4.0              # skip stocks that are too explosive / circuit-prone

# ---- Regime detection -----------------------------------------------------
ADX_PERIOD = 14
ADX_TRENDING_THRESHOLD = 30    # ADX >= this  -> trending regime -> trend sub-strategy
ADX_RANGING_THRESHOLD = 18     # ADX <  this  -> ranging regime  -> mean-reversion sub-strategy
# Between the two thresholds -> NEUTRAL regime -> no new entries (ambiguous, sit out)

# ---- Trend sub-strategy (used only in TRENDING regime) --------------------
EMA_FAST = 10
EMA_SLOW = 30
OPENING_RANGE_MINUTES = 15     # first 15 minutes of the session defines the opening range
REL_VOLUME_TREND_MIN = 2.0     # need 1.5x the recent average volume to confirm a breakout

# ---- Mean-reversion sub-strategy (used only in RANGING regime) ------------
RSI_PERIOD = 14
RSI_OVERSOLD = 25
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0
REL_VOLUME_MEANREV_MIN = 1.5

# ---- Direction ---------------------------------------------------------
# Long-only by design: an intraday SHORT in the cash segment must be squared
# off the same day (no overnight short without derivatives/SLB access), which
# would conflict directly with the up-to-5-day carry rule below. Going short
# would need F&O or a stock-lending desk -- explicitly out of scope since you
# said no derivatives.
ALLOW_SHORTS = False

# ---- Risk management --------------------------------------------------
RISK_PER_TRADE_PCT = 0.0040    # 0.40% of capital risked per trade (moderate/medium-aggressive)
ATR_STOP_MULT = 1.8            # stop distance = 1.5x ATR(14)
REWARD_RISK_RATIO = 999.0        # target distance = 2x the stop distance
MAX_POSITION_NOTIONAL_PCT = 0.15   # no single position's total value exceeds 20% of capital,
                                    # regardless of what pure risk-based sizing would otherwise buy
                                    # (with MAX_CONCURRENT_POSITIONS=4, this caps total deployment
                                    # at 80% of capital even in the worst case, leaving a buffer)
PARTIAL_SCALE_OUT_R = 2.0      # take partial profit once price has moved 1x risk in your favor
PARTIAL_SCALE_OUT_FRACTION = 0.40   # ...scale out 50% there, move stop to breakeven on the rest

MAX_CONCURRENT_POSITIONS = 3       # keeps this monitorable for one person
MAX_PORTFOLIO_HEAT_PCT = 0.025      # sum of *open risk* across all positions, capped at 3% of capital
DAILY_LOSS_LIMIT_PCT = 0.015        # kill switch: stop opening new trades after -2% for the day

# ---- Holding period ------------------------------------------------------
INTRADAY_SQUARE_OFF_TIME = "15:15"   # square off pure-intraday signals by 3:15pm (avoid the close auction)
MAX_HOLDING_DAYS = 4                  # hard time-stop -- implements your "max weekly holding" rule

# ---- Transaction costs (India, cash equity segment) ------------------------
# These are current at time of writing but STT/stamp duty are set by the
# government and do get revised in Union Budgets -- sanity-check before
# trusting a live P&L number.
BROKERAGE_PER_ORDER_INR = 20.0
STT_INTRADAY_SELL_PCT = 0.00025    # 0.025%, sell leg only, if squared off same day
STT_DELIVERY_PCT = 0.001           # 0.1%, both legs, once a position is carried past a day
EXCHANGE_TXN_CHARGE_PCT = 0.0000297
GST_PCT = 0.18                     # on (brokerage + exchange txn charge)
STAMP_DUTY_BUY_PCT = 0.00015       # 0.015%, buy leg only
SLIPPAGE_BPS = 5                   # 0.05% assumed slippage per fill -- tune this against real fills later

# ---- Capital -------------------------------------------------------------
STARTING_CAPITAL_INR = 5_00_000    # example capital base -- change to your own before running

# ---- Bar interval ----------------------------------------------------
BAR_MINUTES = 5     # 5-minute bars: enough signal, manageable data volume, less noise than 1-min

# ---- Live/paper trading safety controls (used by live_controls.py, NOT the backtest) ----
MAX_DAILY_CAPITAL_DEPLOYED_INR = 30_000   # hard cap on total NEW capital committed in a single day
MAX_POSITIONS_PER_DAY = 5                 # hard cap on number of NEW positions opened in a single day
MAX_SHARES_PER_TRADE = 200                # hard cap on quantity for any single order, regardless of sizing math
DAILY_LOSS_LIMIT_INR = 3_000              # kill switch: stop opening new positions after this much realized loss today
QUOTE_POLL_SECONDS = 30                   # how often run_live.py polls for live quotes
