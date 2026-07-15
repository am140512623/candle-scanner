"""
Portfolio equity sim -- Lower-Band Touch grab, fixed-% risk, MANY positions at once.

The realistic sizing the all-in sim was missing:
  * Start with $100.
  * Risk a FIXED %% of current equity on each trade (default 2%).
        risk_$ = RISK_PCT * equity
        shares = risk_$ / (entry - stop)     (so a stop-out loses exactly risk_$)
        cost   = shares * entry              (dollars deployed)
  * Hold up to MAX_OPEN positions AT THE SAME TIME (default 50). A signal that
    appears while 50 are already open is skipped.
  * NO LEVERAGE: if a position would cost more cash than you have, it's capped to
    the cash on hand (or skipped if you're fully deployed).
  * Compounding: realized wins/losses feed back into equity, so size grows with
    the account.

Daily frame only (the reliable one). Every S&P 500 ticker's signals are merged
into one chronological event stream; exits are settled in time order so cash frees
up for new trades.  Win = +RR risk_$, loss = -1 risk_$, unresolved = closed at the
last available price (mark-to-market).

Run:  python portfolio_lower_touch.py
Env:  RISK_PCT=2   MAX_OPEN=50   RR=2.0   START=100   BT_DAILY_PERIOD=max
"""

import heapq
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import scan_all as s
import scan_crypto as sc
import compound_lower_touch as cp

RISK_PCT = float(os.environ.get("RISK_PCT", "2")) / 100.0
MAX_OPEN = int(os.environ.get("MAX_OPEN", "50"))
START    = float(os.environ.get("START", "100"))


def run_portfolio(trades):
    """Event-driven fixed-% risk sim with capped concurrency and no leverage."""
    trades = sorted(trades, key=lambda r: r["entry_ts"])
    cash = START
    deployed = 0.0                     # cost basis of open positions
    open_heap = []                     # (exit_ts, seq, shares, exit_price, cost, result)
    seq = 0
    taken = wins = losses = 0
    skipped_cap = skipped_cash = 0
    max_open = 0
    curve = []                         # (timestamp, equity)

    def settle(until_ts):
        nonlocal cash, deployed
        while open_heap and open_heap[0][0] <= until_ts:
            ex_ts, _, shares, ex_price, cost, _res = heapq.heappop(open_heap)
            cash += shares * ex_price
            deployed -= cost
            curve.append((ex_ts, cash + deployed))

    for tr in trades:
        et = tr["entry_ts"]
        settle(et)                                     # free cash from closed trades
        if len(open_heap) >= MAX_OPEN:
            skipped_cap += 1
            continue
        if cash <= 0:
            skipped_cash += 1
            continue
        risk_frac = (tr["entry"] - tr["stop"]) / tr["entry"]
        if risk_frac <= 0:
            continue
        equity = cash + deployed
        cost = (RISK_PCT * equity) / risk_frac         # $ to deploy for the intended risk
        if cost > cash:
            cost = cash                                # no leverage -> cap to cash
        shares = cost / tr["entry"]
        cash -= cost
        deployed += cost
        heapq.heappush(open_heap,
                       (tr["exit_ts"], seq, shares, tr["exit_price"], cost, tr["result"]))
        seq += 1
        taken += 1
        if tr["result"] == "win":
            wins += 1
        elif tr["result"] == "loss":
            losses += 1
        max_open = max(max_open, len(open_heap))
        curve.append((et, cash + deployed))

    settle(pd.Timestamp.max)                            # close everything left open
    final = cash                                       # deployed is 0 now

    curve.sort(key=lambda p: p[0])
    return {
        "final": final, "taken": taken, "wins": wins, "losses": losses,
        "skipped_cap": skipped_cap, "skipped_cash": skipped_cash,
        "max_open": max_open, "curve": curve,
        "first": trades[0]["entry_ts"] if trades else None,
        "last": curve[-1][0] if curve else None,
    }


def main():
    print("Building S&P 500 universe...")
    tickers = s.get_sp500()
    limit = int(os.environ.get("BT_LIMIT", "0"))
    if limit > 0:
        tickers = tickers[:limit]
    print(f"  {len(tickers)} tickers")

    print("Downloading daily candles (max)...")
    base = sc.download_base(tickers, "1d", os.environ.get("BT_DAILY_PERIOD", "max"))
    print(f"  got data for {len(base)}/{len(tickers)} tickers")

    print("Finding signals + simulating exits...")
    trades = cp.collect_trades(base)
    print(f"  {len(trades)} total signals across all tickers")

    r = run_portfolio(trades)

    years = (r["last"] - r["first"]).days / 365.25 if r["first"] else 0
    cagr = (r["final"] / START) ** (1 / years) - 1 if years > 0 and r["final"] > 0 else 0
    wr = 100 * r["wins"] / (r["wins"] + r["losses"]) if (r["wins"] + r["losses"]) else 0

    # Equity curve PNG
    xs = [p[0] for p in r["curve"]]
    ys = [p[1] for p in r["curve"]]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(xs, ys, color="steelblue", lw=1.3)
    ax.axhline(START, color="gray", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.set_title(f"Lower-Band Touch grab -- ${START:.0f} start, "
                 f"{RISK_PCT*100:g}% risk/trade, max {MAX_OPEN} open ({int(os.environ.get('RR_DISP', '2'))}:1)")
    ax.set_ylabel("Equity $ (log scale)")
    ax.grid(True, which="both", alpha=0.25)
    png = "portfolio_lower_touch_equity.png"
    fig.tight_layout()
    fig.savefig(png, dpi=120)

    print("\n" + "=" * 72)
    print(f"PORTFOLIO SIM -- ${START:.0f} start, {RISK_PCT*100:g}% risk/trade, "
          f"up to {MAX_OPEN} open at once")
    print("=" * 72)
    print(f"  Signals available     : {len(trades)}")
    print(f"  Trades taken          : {r['taken']}")
    print(f"  Skipped (50 full)     : {r['skipped_cap']}")
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
    print("=" * 72)
    print(f"\nEquity curve -> {png}")


if __name__ == "__main__":
    main()
