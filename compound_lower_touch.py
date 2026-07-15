"""
Compounding equity sim -- Lower-Band Touch grab, ONE position at a time, ALL-IN.

Rules the user asked for:
  * Start with $100.
  * Put the WHOLE balance into each trade (all-in, long).
  * Only ONE trade at a time: while money is tied up in an open position, any new
    signal that appears is SKIPPED (cash isn't available).
  * When flat again, take the next signal in time order and compound -- a win
    grows the balance, a loss shrinks it, and the next trade uses the new balance.

This is the DAILY frame only (the reliable one; intraday Yahoo history is too
short / resamples badly for stocks). Signals from every S&P 500 ticker are merged
into ONE chronological stream, because you only have one pot of cash.

Trade mechanics (same pattern + 2:1 target as the backtest):
  entry  = signal candle's close
  stop   = the grab's swept low   target = entry + RR*(entry-stop)
  exit   = stop or target, whichever price reaches first (a bar covering BOTH is
           scored at the stop -- conservative). If neither is ever hit, the trade
           is closed at the last available close (mark-to-market).
  all-in multiplier = exit_price / entry_price.

Writes an equity curve CSV + a PNG, and prints the final balance.

Run:  python compound_lower_touch.py
Env:  RR=2.0   START=100   BT_DAILY_PERIOD=max   BT_LIMIT=0
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scan_all as s
import scan_crypto as sc
import backtest_lower_touch as bt

RR    = float(os.environ.get("RR", "2.0"))
START = float(os.environ.get("START", "100"))


def simulate_exit(df, entry_idx, entry, stop):
    """Walk forward; return (exit_ts, exit_price, result). Conservative: a bar that
    touches both stop and target is treated as a stop (loss)."""
    target = entry + RR * (entry - stop)
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    for j in range(entry_idx + 1, len(df)):
        if l[j] <= stop:
            return df.index[j], stop, "loss"
        if h[j] >= target:
            return df.index[j], target, "win"
    # Never resolved -> close at the last bar (mark-to-market).
    return df.index[-1], float(df["Close"].iloc[-1]), "open"


def collect_trades(base_data):
    """Every daily signal across all tickers as dicts, sorted by entry time."""
    trades = []
    for t, df in base_data.items():
        try:
            d = df.iloc[:-1]                      # drop still-forming candle
            if len(d) < bt.MIN_BARS:
                continue
            for entry_idx, entry, stop in bt.find_signals(d):
                exit_ts, exit_price, result = simulate_exit(d, entry_idx, entry, stop)
                trades.append({
                    "ticker": t,
                    "entry_ts": pd.Timestamp(d.index[entry_idx]),
                    "exit_ts": pd.Timestamp(exit_ts),
                    "entry": entry, "stop": stop,
                    "exit_price": exit_price,
                    "mult": exit_price / entry,
                    "result": result,
                })
        except Exception:
            continue
    trades.sort(key=lambda r: r["entry_ts"])
    return trades


def run_compound(trades):
    """Walk the chronological stream, one position at a time, all-in, compounding.
    Returns (taken_trades, equity_points)."""
    balance = START
    free_at = pd.Timestamp.min                     # cash is free from the start
    taken = []
    equity = [(None, balance)]
    for tr in trades:
        if tr["entry_ts"] < free_at:
            continue                               # still in a trade -> skip signal
        balance *= tr["mult"]
        free_at = tr["exit_ts"]
        rec = dict(tr)
        rec["balance_after"] = balance
        taken.append(rec)
        equity.append((tr["exit_ts"], balance))
    return taken, equity


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
    trades = collect_trades(base)
    print(f"  {len(trades)} total signals across all tickers")

    taken, equity = run_compound(trades)

    wins = sum(1 for t in taken if t["result"] == "win")
    losses = sum(1 for t in taken if t["result"] == "loss")
    opens = sum(1 for t in taken if t["result"] == "open")
    final = taken[-1]["balance_after"] if taken else START

    # Per-trade CSV
    tr_csv = "compound_lower_touch_trades.csv"
    pd.DataFrame([{
        "ticker": t["ticker"],
        "entry_ts": t["entry_ts"].date(), "exit_ts": t["exit_ts"].date(),
        "entry": round(t["entry"], 4), "stop": round(t["stop"], 4),
        "exit_price": round(t["exit_price"], 4),
        "pct_move": round((t["mult"] - 1) * 100, 2),
        "result": t["result"],
        "balance_after": round(t["balance_after"], 2),
    } for t in taken]).to_csv(tr_csv, index=False)

    # Equity curve PNG
    xs = [p[0] for p in equity if p[0] is not None]
    ys = [p[1] for p in equity if p[0] is not None]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(xs, ys, color="seagreen", lw=1.4)
    ax.axhline(START, color="gray", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.set_title(f"Lower-Band Touch grab -- $100 start, all-in, compounded ({RR:g}:1)")
    ax.set_ylabel("Balance $ (log scale)")
    ax.grid(True, which="both", alpha=0.25)
    png = "compound_lower_touch_equity.png"
    fig.tight_layout()
    fig.savefig(png, dpi=120)

    print("\n" + "=" * 70)
    print(f"COMPOUNDING SIM -- $ {START:.0f} start, all-in, one position at a time")
    print("=" * 70)
    print(f"  Signals available     : {len(trades)}")
    print(f"  Trades actually taken : {len(taken)}   "
          f"(rest skipped -- cash was in a trade)")
    print(f"  Wins / Losses / Open  : {wins} / {losses} / {opens}")
    if taken:
        wr = 100 * wins / (wins + losses) if (wins + losses) else 0
        print(f"  Win rate (taken)      : {wr:.1f}%")
        print(f"  Date range            : {taken[0]['entry_ts'].date()} "
              f"-> {taken[-1]['exit_ts'].date()}")
    print("-" * 70)
    print(f"  FINAL BALANCE         : $ {final:,.2f}")
    print(f"  Total return          : {(final/START - 1)*100:,.1f}%   "
          f"({final/START:,.2f}x)")
    print("=" * 70)
    print(f"\nEquity curve -> {png}")
    print(f"Per-trade detail -> {tr_csv}")


if __name__ == "__main__":
    main()
