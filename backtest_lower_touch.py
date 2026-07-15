"""
Batch backtest -- Lower-Band Touch -> Liquidity Grab (LONG), fixed 2:1 TP/SL.

Unlike the live scanners (which only check the JUST-CLOSED candle), this walks the
ENTIRE history of every S&P 500 name and fires the pattern wherever it occurred in
the past, then simulates a fixed reward:risk trade and reports aggregate results.

PATTERN  (twin of bb_lower_touch_grab_strategy.pine):
    STAGE 1 -- bullish "swallow" liquidity grab (the SAME `_swallow` engine every
        bot uses): a green trigger candle whose body covers the prior green
        body(s) and whose low sweeps below their low(s). 2- or 3-candle layout.
    STAGE 2 -- the trigger candle's range TOUCHES the LOWER Bollinger band:
        low <= lower AND high >= lower  (the bottom line passes through the bar).

TRADE:
    ENTRY  = close of the signal bar.
    STOP   = the grab candle's swept low (the lower of the swept lows on a
             3-candle grab), optionally padded by STOP_PAD_PCT.
    TARGET = entry + RR * (entry - stop).   RR defaults to 2.0 (a 2:1 trade).
    Exit is decided by walking forward bar-by-bar: whichever of stop/target the
    price reaches first. If a single bar's range covers BOTH, it's scored as a
    LOSS (conservative -- we can't know the intrabar order of that bar's H/L).

Each signal is simulated independently (a signal-QUALITY test, like score_signals.py
-- trades may overlap; this is not an equity-curve/position sim). Because every
trade risks 1R to make RR, results are reported in R multiples:
    win  = +RR      loss = -1R
    net R, expectancy (R per trade), and win rate -- overall and per timeframe.

Run (PowerShell):
    python backtest_lower_touch.py
Env knobs:
    RR=2.0                 reward:risk target
    STOP_PAD_PCT=0.0       extra stop distance below the swept low, in %
    BT_FRAMES=4H,6H,8H,12H,1D   which frames to test (subset of the catalog)
    BT_DAILY_PERIOD=max    history for the 1d base (e.g. 5y, 10y, max)
    BT_LIMIT=0             cap the universe to the first N tickers (0 = all; for a quick smoke test)
    BT_OUT=backtest_lower_touch_trades.csv   per-trade CSV (blank = don't write)
"""

import csv
import os

import numpy as np
import pandas as pd

import scan_all as s
import scan_crypto as sc

# --------------------------- pattern parameters ---------------------------
BB_LEN  = int(os.environ.get("BB_LEN", "20"))
BB_MULT = float(os.environ.get("BB_MULT", "2.0"))
MIN_BARS = BB_LEN + 5                      # warm-up for the band + swallow lookback

RR           = float(os.environ.get("RR", "2.0"))
STOP_PAD_PCT = float(os.environ.get("STOP_PAD_PCT", "0.0")) / 100.0

# --------------------------- timeframes ---------------------------
# 4H/6H/8H/12H are built from Yahoo's 1h candles (capped at ~60 days of history),
# so those frames only cover the recent ~2 months. 1D is built from daily candles
# and covers full history -- it will carry the bulk of the sample.
RESAMPLE_ORIGIN = "epoch"
FRAMES_CATALOG = [
    {"label": "4H",  "base": "1h", "rkw": {"rule": "4h",  "origin": RESAMPLE_ORIGIN}},
    {"label": "6H",  "base": "1h", "rkw": {"rule": "6h",  "origin": RESAMPLE_ORIGIN}},
    {"label": "8H",  "base": "1h", "rkw": {"rule": "8h",  "origin": RESAMPLE_ORIGIN}},
    {"label": "12H", "base": "1h", "rkw": {"rule": "12h", "origin": RESAMPLE_ORIGIN}},
    {"label": "1D",  "base": "1d", "rkw": None},
]
BASE_PERIODS = {"1h": "60d", "1d": os.environ.get("BT_DAILY_PERIOD", "max")}


def selected_frames():
    raw = os.environ.get("BT_FRAMES")
    if not raw:
        return list(FRAMES_CATALOG)
    want = {x.strip().upper() for x in raw.split(",") if x.strip()}
    return [tf for tf in FRAMES_CATALOG if tf["label"] in want]


def _bollinger(close, length=BB_LEN, mult=BB_MULT):
    """Bollinger Bands matching TradingView (ta.sma + population stdev, ddof=0)."""
    basis = close.rolling(length).mean()
    dev = close.rolling(length).std(ddof=0)
    return basis, basis + mult * dev, basis - mult * dev


def find_signals(df):
    """Every bar in `df` where the Lower-Band Touch grab fires.
    Returns a list of (entry_idx, entry_close, stop_level). Vectorised where cheap;
    the swallow/touch checks read only the last 1-2 bars so they stay O(1) per bar.
    The `_swallow` colour/body/sweep rules are reproduced here bar-by-bar so the
    match is identical to scan_all._swallow evaluated on df.iloc[:i+1]."""
    if df is None or len(df) < MIN_BARS:
        return []

    o = df["Open"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    _, _, lower = _bollinger(df["Close"])
    lower = lower.to_numpy(dtype=float)

    out = []
    for i in range(MIN_BARS - 1, len(df)):
        if np.isnan(lower[i]) or np.isnan(o[i]) or np.isnan(c[i]):
            continue
        if not (c[i] > o[i]):                      # trigger must be green
            continue
        body_top = max(o[i], c[i])

        green1 = c[i - 1] > o[i - 1]
        ref_top1 = max(o[i - 1], c[i - 1])
        swallow2 = green1 and body_top >= ref_top1 and l[i] <= l[i - 1]

        swallow3 = False
        if i >= 2:
            green2 = c[i - 2] > o[i - 2]
            ref_top2 = max(o[i - 2], c[i - 2])
            swallow3 = (green1 and green2
                        and body_top >= max(ref_top1, ref_top2)
                        and l[i] <= min(l[i - 1], l[i - 2]))

        if not (swallow2 or swallow3):
            continue
        # Lower-band touch: the bottom line passes through the trigger candle.
        if not (l[i] <= lower[i] and h[i] >= lower[i]):
            continue

        grab_low = min(l[i], l[i - 1], l[i - 2]) if swallow3 else min(l[i], l[i - 1])
        stop = grab_low * (1.0 - STOP_PAD_PCT)
        if stop <= 0 or stop >= c[i]:              # need positive risk
            continue
        out.append((i, c[i], stop))
    return out


def simulate(df, entry_idx, entry, stop):
    """Walk forward from entry_idx+1; return 'win' | 'loss' | 'open'.
    Whichever of target/stop is reached first wins; a bar covering both = loss."""
    target = entry + RR * (entry - stop)
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    for j in range(entry_idx + 1, len(df)):
        hit_stop = l[j] <= stop
        hit_tgt = h[j] >= target
        if hit_stop:                # conservative: stop resolves first on a both-bar
            return "loss"
        if hit_tgt:
            return "win"
    return "open"


def backtest_frame(tf, base_data):
    """Run one timeframe over all tickers. Returns (rows, counts)."""
    rows = []
    counts = {"win": 0, "loss": 0, "open": 0}
    for t, base_df in base_data.items():
        try:
            d = sc.resample_ohlc(base_df, tf["rkw"]) if tf["rkw"] else base_df
            d = d.iloc[:-1]                        # drop the still-forming candle
            if len(d) < MIN_BARS:
                continue
            for entry_idx, entry, stop in find_signals(d):
                res = simulate(d, entry_idx, entry, stop)
                counts[res] += 1
                rows.append({
                    "ticker": t, "timeframe": tf["label"],
                    "entry_date": str(d.index[entry_idx]),
                    "entry": round(entry, 4), "stop": round(stop, 4),
                    "target": round(entry + RR * (entry - stop), 4),
                    "risk_pct": round((entry - stop) / entry * 100, 2),
                    "result": res,
                })
        except Exception:
            continue
    return rows, counts


def report(all_counts, by_tf):
    """Print the aggregate table. Wins pay +RR, losses cost -1R."""
    def block(name, counts):
        w, ls, op = counts["win"], counts["loss"], counts["open"]
        resolved = w + ls
        wr = 100 * w / resolved if resolved else 0
        net_r = w * RR - ls
        exp = net_r / resolved if resolved else 0
        print(f"  {name:>5}: {resolved:>5} trades | win {wr:5.1f}%  "
              f"({w} W / {ls} L) | net {net_r:+8.1f}R | exp {exp:+.3f}R/trade"
              + (f"  ({op} still open)" if op else ""))

    print("\n" + "=" * 78)
    print(f"LOWER-BAND TOUCH GRAB  --  S&P 500  --  fixed {RR:g}:1 TP/SL")
    print("=" * 78)
    print("Per timeframe:")
    for label in [tf["label"] for tf in selected_frames()]:
        if label in by_tf:
            block(label, by_tf[label])
    print("-" * 78)
    block("ALL", all_counts)
    resolved = all_counts["win"] + all_counts["loss"]
    if resolved:
        wr = 100 * all_counts["win"] / resolved
        net_r = all_counts["win"] * RR - all_counts["loss"]
        print("-" * 78)
        print(f"Break-even win rate for {RR:g}:1 is {100 / (1 + RR):.1f}%.  "
              f"You got {wr:.1f}%  ->  {'EDGE' if net_r > 0 else 'no edge'} "
              f"(net {net_r:+.1f}R over {resolved} trades).")
    print("=" * 78)


def main():
    print("Building S&P 500 universe...")
    tickers = s.get_sp500()
    limit = int(os.environ.get("BT_LIMIT", "0"))
    if limit > 0:
        tickers = tickers[:limit]
    print(f"  {len(tickers)} tickers")

    frames = selected_frames()
    print("Frames: " + ", ".join(tf["label"] for tf in frames))
    print("NOTE: 4H/6H/8H/12H come from Yahoo 1h candles (~60 days of history only);"
          " 1D has full history and carries most of the sample.\n")

    needed_bases = {tf["base"] for tf in frames}
    bases = {}
    for b in needed_bases:
        print(f"Downloading {b} candles ({BASE_PERIODS[b]})...")
        bases[b] = sc.download_base(tickers, b, BASE_PERIODS[b])
        print(f"  got data for {len(bases[b])}/{len(tickers)} tickers")

    all_counts = {"win": 0, "loss": 0, "open": 0}
    by_tf = {}
    all_rows = []
    for tf in frames:
        rows, counts = backtest_frame(tf, bases[tf["base"]])
        by_tf[tf["label"]] = counts
        for k in all_counts:
            all_counts[k] += counts[k]
        all_rows.extend(rows)
        print(f"[{tf['label']}] {counts['win']}W / {counts['loss']}L / "
              f"{counts['open']} open  ({len(rows)} signals)")

    report(all_counts, by_tf)

    out = os.environ.get("BT_OUT", "backtest_lower_touch_trades.csv")
    if out and all_rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nPer-trade detail -> {out} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
