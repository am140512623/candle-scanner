"""
Batch backtest -- GRAB -> OPPOSITE CANDLE -> RECLAIM (LONG), fixed 2:1 TP/SL.

Tests the pattern added on top of the two Bollinger bots (see bb_reclaim.py), and
-- the point of this script -- tests it AGAINST the plain grab it is built on, so
you can see whether waiting for the reclaim actually earns its keep.

Four arms are simulated on the SAME bars, same universe, same stop convention:

    ① grab_upper       plain upper-band breakout grab      (baseline)
    ② grab_lower       plain lower-band touch grab         (baseline)
    ③ reclaim_upper    ① + opposite candle + reclaim
    ④ reclaim_lower    ② + opposite candle + reclaim

③ is a strict SUBSET of ①'s setups (same grabs, minus the ones that never printed
an opposite candle or never reclaimed), and ④ of ②. So comparing ③ vs ① and ④ vs ②
answers the real question: does the reclaim filter improve expectancy, or does it
just throw away trades and arrive late?

TRADE (identical rules for every arm, so the comparison is like-for-like):
    ENTRY  = close of the SIGNAL bar. For ①② that's the grab candle; for ③④ it's
             the candle that reclaimed the opposite candle's open -- which is the
             whole point: the reclaim enters LATER and therefore worse.
    STOP   = STOP_MODE=grab     -> the grab candle's swept low (default; matches
                                   backtest_lower_touch.py so numbers line up)
             STOP_MODE=pullback -> the lowest low from the opposite candle to the
                                   signal bar (tighter; ③④ only)
    TARGET = entry + RR * (entry - stop).
    Exit walks forward bar-by-bar; whichever of stop/target is reached first wins.
    A bar covering BOTH is scored a LOSS (we can't know intrabar order).

Each signal is simulated independently (a signal-QUALITY test, like
backtest_lower_touch.py) -- trades may overlap; this is not an equity-curve sim.
Results are in R multiples: win = +RR, loss = -1R.

The `_swallow` / band / reclaim rules are reproduced bar-by-bar here rather than
imported, exactly as backtest_lower_touch.py does, so walking full history stays
fast. test_backtest_reclaim.py asserts this agrees with the live detector.

Run (PowerShell):
    python backtest_reclaim.py
Env knobs:
    RR=2.0                  reward:risk target
    STOP_MODE=grab          grab | pullback
    STOP_PAD_PCT=0.0        extra stop distance below the low, in %
    MAX_WAIT=10             candles the reclaim may take (matches bb_reclaim)
    BT_FRAMES=1D            which frames to test
    BT_DAILY_PERIOD=max     history for the 1d base
    BT_LIMIT=0              cap the universe to the first N tickers (0 = all)
    BT_OUT=backtest_reclaim_trades.csv   per-trade CSV (blank = don't write)
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
MIN_BARS = BB_LEN + 5

RR           = float(os.environ.get("RR", "2.0"))
STOP_PAD_PCT = float(os.environ.get("STOP_PAD_PCT", "0.0")) / 100.0
STOP_MODE    = os.environ.get("STOP_MODE", "grab").strip().lower()
MAX_WAIT     = int(os.environ.get("MAX_WAIT", "10"))

RESAMPLE_ORIGIN = "epoch"
FRAMES_CATALOG = [
    {"label": "4H",  "base": "1h", "rkw": {"rule": "4h",  "origin": RESAMPLE_ORIGIN}},
    {"label": "6H",  "base": "1h", "rkw": {"rule": "6h",  "origin": RESAMPLE_ORIGIN}},
    {"label": "8H",  "base": "1h", "rkw": {"rule": "8h",  "origin": RESAMPLE_ORIGIN}},
    {"label": "12H", "base": "1h", "rkw": {"rule": "12h", "origin": RESAMPLE_ORIGIN}},
    {"label": "1D",  "base": "1d", "rkw": None},
]
BASE_PERIODS = {"1h": "60d", "1d": os.environ.get("BT_DAILY_PERIOD", "max")}

ARMS = ["grab_upper", "grab_lower", "reclaim_upper", "reclaim_lower"]


def selected_frames():
    raw = os.environ.get("BT_FRAMES")
    if not raw:
        return list(FRAMES_CATALOG)
    want = {x.strip().upper() for x in raw.split(",") if x.strip()}
    return [tf for tf in FRAMES_CATALOG if tf["label"] in want]


def _bollinger(close, length=BB_LEN, mult=BB_MULT):
    basis = close.rolling(length).mean()
    dev = close.rolling(length).std(ddof=0)
    return basis, basis + mult * dev, basis - mult * dev


def _grabs(df):
    """Every bar where the bullish swallow fires, as {i: swept_low}. Mirrors
    scan_all._swallow evaluated on df.iloc[:i+1]."""
    o = df["Open"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)

    out = {}
    for i in range(MIN_BARS - 1, len(df)):
        if np.isnan(o[i]) or np.isnan(c[i]) or not (c[i] > o[i]):
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
        out[i] = min(l[i], l[i - 1], l[i - 2]) if swallow3 else min(l[i], l[i - 1])
    return out


def find_signals(df):
    """All four arms in one pass over `df`.
    Returns {arm: [(signal_idx, entry, stop, grab_idx)]}."""
    res = {a: [] for a in ARMS}
    if df is None or len(df) < MIN_BARS + 2:
        return res

    o = df["Open"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    _, upper, lower = _bollinger(df["Close"])
    upper = upper.to_numpy(dtype=float)
    lower = lower.to_numpy(dtype=float)

    def _add(arm, sig_i, entry, low_for_stop, grab_i):
        stop = low_for_stop * (1.0 - STOP_PAD_PCT)
        if stop <= 0 or stop >= entry:          # need positive risk
            return
        res[arm].append((sig_i, entry, stop, grab_i))

    for g, swept_low in _grabs(df).items():
        if np.isnan(upper[g]) or np.isnan(lower[g]):
            continue
        # Each bot's band gate, on the grab candle.
        gates = {
            "upper": o[g] < upper[g] and c[g] > upper[g],
            "lower": l[g] <= lower[g] and h[g] >= lower[g],
        }
        for band, ok in gates.items():
            if not ok:
                continue
            # --- baseline arm: take the grab itself ---
            _add(f"grab_{band}", g, c[g], swept_low, g)

            # --- reclaim arm: opposite candle right after, then first close above
            #     its open, within MAX_WAIT bars ---
            r = g + 1
            if r >= len(df) or not (c[r] < o[r]):
                continue                        # next candle isn't opposite
            level = o[r]
            for j in range(r + 1, min(r + MAX_WAIT, len(df) - 1) + 1):
                if c[j] > level:                # FIRST close above -> the signal
                    low_for_stop = (min(l[r:j + 1]) if STOP_MODE == "pullback"
                                    else swept_low)
                    _add(f"reclaim_{band}", j, c[j], low_for_stop, g)
                    break
    return res


def simulate(df, entry_idx, entry, stop):
    """Walk forward from entry_idx+1; 'win' | 'loss' | 'open'."""
    target = entry + RR * (entry - stop)
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    for j in range(entry_idx + 1, len(df)):
        if l[j] <= stop:            # conservative: stop resolves first on a both-bar
            return "loss"
        if h[j] >= target:
            return "win"
    return "open"


def backtest_frame(tf, base_data):
    rows = []
    counts = {a: {"win": 0, "loss": 0, "open": 0} for a in ARMS}
    for t, base_df in base_data.items():
        try:
            d = sc.resample_ohlc(base_df, tf["rkw"]) if tf["rkw"] else base_df
            d = d.iloc[:-1]
            if len(d) < MIN_BARS + 2:
                continue
            found = find_signals(d)
            for arm, sigs in found.items():
                for sig_i, entry, stop, grab_i in sigs:
                    res = simulate(d, sig_i, entry, stop)
                    counts[arm][res] += 1
                    rows.append({
                        "arm": arm, "ticker": t, "timeframe": tf["label"],
                        "grab_date": str(d.index[grab_i]),
                        "entry_date": str(d.index[sig_i]),
                        "bars_waited": sig_i - grab_i,
                        "entry": round(entry, 4), "stop": round(stop, 4),
                        "target": round(entry + RR * (entry - stop), 4),
                        "risk_pct": round((entry - stop) / entry * 100, 2),
                        "result": res,
                    })
        except Exception:
            continue
    return rows, counts


def _stats(counts):
    w, ls, op = counts["win"], counts["loss"], counts["open"]
    resolved = w + ls
    return {
        "w": w, "l": ls, "open": op, "resolved": resolved,
        "wr": 100 * w / resolved if resolved else 0.0,
        "net_r": w * RR - ls,
        "exp": (w * RR - ls) / resolved if resolved else 0.0,
    }


def report(by_arm, by_tf):
    line = "=" * 92
    print("\n" + line)
    # ASCII only: the Windows console is cp1252 and dies on "→".
    print(f"GRAB -> OPPOSITE -> RECLAIM  vs  the plain grab  --  fixed {RR:g}:1 TP/SL"
          f"  --  stop: {STOP_MODE}")
    print(line)
    print(f"{'arm':<16}{'trades':>8}{'win%':>8}{'W':>7}{'L':>7}"
          f"{'net R':>10}{'exp R/trade':>14}{'open':>7}")
    print("-" * 92)
    for arm in ARMS:
        st = _stats(by_arm[arm])
        print(f"{arm:<16}{st['resolved']:>8}{st['wr']:>7.1f}%{st['w']:>7}{st['l']:>7}"
              f"{st['net_r']:>+10.1f}{st['exp']:>+14.3f}{st['open']:>7}")

    be = 100 / (1 + RR)
    print("-" * 92)
    print(f"Break-even win rate at {RR:g}:1 is {be:.1f}%.")

    # The comparison this whole script exists for.
    print("\nDoes waiting for the reclaim help?  (reclaim vs the grab it filters)")
    for band in ("upper", "lower"):
        b = _stats(by_arm[f"grab_{band}"])
        r = _stats(by_arm[f"reclaim_{band}"])
        if not b["resolved"] or not r["resolved"]:
            print(f"  {band}: not enough resolved trades to compare.")
            continue
        d_exp = r["exp"] - b["exp"]
        kept = 100 * r["resolved"] / b["resolved"]
        verdict = ("BETTER" if d_exp > 0.02 else
                   "WORSE" if d_exp < -0.02 else "about the same")
        print(f"  {band:<6}: exp {b['exp']:+.3f}R -> {r['exp']:+.3f}R "
              f"({d_exp:+.3f}R, {verdict}) | win {b['wr']:.1f}% -> {r['wr']:.1f}% "
              f"| keeps {kept:.0f}% of the setups ({r['resolved']} of {b['resolved']})")

    print("\nPer timeframe:")
    for label in [tf["label"] for tf in selected_frames()]:
        if label not in by_tf:
            continue
        print(f"  [{label}]")
        for arm in ARMS:
            st = _stats(by_tf[label][arm])
            if not st["resolved"] and not st["open"]:
                continue
            print(f"    {arm:<16}{st['resolved']:>6} trades | win {st['wr']:5.1f}% "
                  f"| net {st['net_r']:+8.1f}R | exp {st['exp']:+.3f}R")
    print(line)


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

    bases = {}
    for b in {tf["base"] for tf in frames}:
        print(f"Downloading {b} candles ({BASE_PERIODS[b]})...")
        bases[b] = sc.download_base(tickers, b, BASE_PERIODS[b])
        print(f"  got data for {len(bases[b])}/{len(tickers)} tickers")

    by_arm = {a: {"win": 0, "loss": 0, "open": 0} for a in ARMS}
    by_tf = {}
    all_rows = []
    for tf in frames:
        rows, counts = backtest_frame(tf, bases[tf["base"]])
        by_tf[tf["label"]] = counts
        for a in ARMS:
            for k in by_arm[a]:
                by_arm[a][k] += counts[a][k]
        all_rows.extend(rows)
        summary = "  ".join(
            f"{a}={counts[a]['win']}W/{counts[a]['loss']}L" for a in ARMS)
        print(f"[{tf['label']}] {summary}")

    report(by_arm, by_tf)

    out = os.environ.get("BT_OUT", "backtest_reclaim_trades.csv")
    if out and all_rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nPer-trade detail -> {out} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
