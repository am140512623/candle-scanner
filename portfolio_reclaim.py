"""
Portfolio equity sim -- GRAB -> REVERSE CANDLE -> RECLAIM, fixed-% risk, compounding.

The backtest (backtest_reclaim.py) measures SIGNAL QUALITY: every signal simulated
independently, trades overlapping freely, results in R. That cannot tell you what an
account does, because a real account has finite cash, can't take every signal at
once, and feels drawdown. This answers the account question instead.

Reuses portfolio_lower_touch's engine verbatim (fixed-% risk, capped concurrency, no
leverage, cash settled in time order) and only swaps in the reclaim pattern, so the
two strategies' numbers are directly comparable.

Sizing:
    risk_$ = RISK_PCT * current equity
    shares = risk_$ / (entry - stop)      -> a stop-out loses exactly risk_$
    cost   = shares * entry               -> capped to cash on hand (NO leverage)
Up to MAX_OPEN positions at once; signals arriving when full are SKIPPED, which is
the honest constraint -- the raw backtest silently takes all of them.

Also reports MAX DRAWDOWN, the number that actually decides whether a strategy is
tradeable, which the R-multiple backtest cannot show.

The pattern here is the PURE one (no Bollinger gate) -- arm "reclaim" in
backtest_reclaim, the version the pattern was actually described as.

THIS IS A HISTORICAL SIMULATION, NOT A FORECAST. See the notes printed at the end.

Run:  python portfolio_reclaim.py
Env:  START=10000  RISK_PCT=2  MAX_OPEN=50  RR=2.0  BT_DAILY_PERIOD=max  BT_LIMIT=0
      ARM=reclaim  (or reclaim_green / reclaim_lower / grab to compare)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import scan_all as s
import scan_crypto as sc
import backtest_reclaim as bt
import compound_lower_touch as cp
import portfolio_lower_touch as pl

ARM   = os.environ.get("ARM", "reclaim")
START = float(os.environ.get("START", "10000"))


def collect_trades(base_data):
    """Every daily signal for ARM across all tickers, with its simulated exit."""
    trades = []
    for t, df in base_data.items():
        try:
            d = df.iloc[:-1]                      # drop still-forming candle
            if len(d) < bt.MIN_BARS + 2:
                continue
            for entry_idx, entry, stop, _grab_i in bt.find_signals(d)[ARM]:
                exit_ts, exit_price, result = cp.simulate_exit(d, entry_idx, entry, stop)
                trades.append({
                    "ticker": t,
                    "entry_ts": pd.Timestamp(d.index[entry_idx]),
                    "exit_ts": pd.Timestamp(exit_ts),
                    "entry": entry, "stop": stop,
                    "exit_price": exit_price,
                    "result": result,
                })
        except Exception:
            continue
    trades.sort(key=lambda r: r["entry_ts"])
    return trades


def drawdown(curve):
    """(max_dd_pct, peak_$, trough_$, trough_ts) from an equity curve."""
    peak = -1.0
    worst = 0.0
    at = (0.0, 0.0, None)
    for ts, eq in curve:
        peak = max(peak, eq)
        if peak > 0:
            dd = (peak - eq) / peak
            if dd > worst:
                worst, at = dd, (peak, eq, ts)
    return worst * 100, at[0], at[1], at[2]


def main():
    print("Building S&P 500 universe...")
    tickers = s.get_sp500()
    limit = int(os.environ.get("BT_LIMIT", "0"))
    if limit > 0:
        tickers = tickers[:limit]
    print(f"  {len(tickers)} tickers")

    print("Downloading daily candles...")
    base = sc.download_base(tickers, "1d", os.environ.get("BT_DAILY_PERIOD", "max"))
    print(f"  got data for {len(base)}/{len(tickers)} tickers")

    print(f"Finding '{ARM}' signals + simulating exits...")
    trades = collect_trades(base)
    print(f"  {len(trades)} total signals across all tickers")

    pl.START = START                              # the engine reads this at call time
    r = pl.run_portfolio(trades)

    years = (r["last"] - r["first"]).days / 365.25 if r["first"] else 0
    cagr = (r["final"] / START) ** (1 / years) - 1 if years > 0 and r["final"] > 0 else 0
    wr = 100 * r["wins"] / (r["wins"] + r["losses"]) if (r["wins"] + r["losses"]) else 0
    dd, dd_peak, dd_trough, dd_ts = drawdown(r["curve"])

    xs = [p[0] for p in r["curve"]]
    ys = [p[1] for p in r["curve"]]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(xs, ys, color="steelblue", lw=1.3)
    ax.axhline(START, color="gray", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.set_title(f"Grab -> reverse candle -> reclaim  --  ${START:,.0f} start, "
                 f"{pl.RISK_PCT*100:g}% risk/trade, max {pl.MAX_OPEN} open, {bt.RR:g}:1")
    ax.set_ylabel("Equity $ (log scale)")
    ax.grid(True, which="both", alpha=0.25)
    png = "portfolio_reclaim_equity.png"
    fig.tight_layout()
    fig.savefig(png, dpi=120)

    line = "=" * 72
    print("\n" + line)
    print(f"PORTFOLIO SIM ({ARM}) -- ${START:,.0f} start, {pl.RISK_PCT*100:g}% risk/trade, "
          f"up to {pl.MAX_OPEN} open")
    print(line)
    print(f"  Signals available     : {len(trades)}")
    print(f"  Trades taken          : {r['taken']}")
    print(f"  Skipped (max open)    : {r['skipped_cap']}")
    print(f"  Skipped (no cash)     : {r['skipped_cash']}")
    print(f"  Peak positions open   : {r['max_open']}")
    print(f"  Wins / Losses         : {r['wins']} / {r['losses']}  ({wr:.1f}% win rate)")
    if r["first"]:
        print(f"  Date range            : {r['first'].date()} -> {r['last'].date()}"
              f"  ({years:.0f} years)")
    print("-" * 72)
    print(f"  FINAL BALANCE         : $ {r['final']:,.2f}")
    print(f"  Total return          : {(r['final']/START-1)*100:,.0f}%   "
          f"({r['final']/START:,.1f}x)")
    print(f"  CAGR (annualised)     : {cagr*100:.1f}% / year")
    print(f"  MAX DRAWDOWN          : -{dd:.1f}%   "
          f"(${dd_peak:,.0f} -> ${dd_trough:,.0f}"
          + (f", {pd.Timestamp(dd_ts).date()})" if dd_ts is not None else ")"))
    print(line)
    print(f"\nEquity curve -> {png}")
    print("""
READ THIS BEFORE BELIEVING THE NUMBER ABOVE
  * HISTORICAL SIMULATION, NOT A FORECAST. It is what these rules would have done
    on past data, not what they will do on your money.
  * SURVIVORSHIP BIAS: the universe is TODAY's S&P 500 walked backwards. Companies
    that died are absent, which flatters every long-only result. This is the single
    biggest reason to discount the final balance.
  * NO COSTS: no spread, slippage, commission, or tax are charged.
  * The CAGR assumes you took every signal for the whole period, through the max
    drawdown above, without changing size or skipping trades.
""")


if __name__ == "__main__":
    main()
