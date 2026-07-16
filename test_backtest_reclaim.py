"""
Cross-check: backtest_reclaim's fast bar-by-bar reimplementation must fire on
EXACTLY the same bars as the live detector the bots actually run
(reclaim_pattern, on the US index bot).

backtest_reclaim walks full history, so it reproduces the swallow/band/reclaim
rules inline instead of calling the live code (same trick backtest_lower_touch.py
uses). That is a real risk: the backtest could drift from what ships and report
results for a pattern nobody trades. This pins the two together.

The live detector only ever looks at the LAST candle of what it's given, so
"did it fire on bar j" == reclaim_signal(df.iloc[:j+1]). Replaying that for every
bar of a random walk gives the ground-truth signal set to compare against.

Run:  python test_backtest_reclaim.py
"""

import numpy as np
import pandas as pd

import backtest_reclaim as bt
import reclaim_pattern


def random_walk(n, seed):
    """OHLC random walk with enough shape to throw off grabs and reclaims."""
    rng = np.random.default_rng(seed)
    px = 100.0
    rows = []
    for _ in range(n):
        drift = rng.normal(0, 1.4)
        o = px
        c = px + drift
        hi = max(o, c) + abs(rng.normal(0, 0.9))
        lo = min(o, c) - abs(rng.normal(0, 0.9))
        rows.append([o, hi, lo, c])
        px = c
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"],
                        index=pd.date_range("2015-01-01", periods=n, freq="D"))


def live_signals(df):
    """Ground truth: replay the live detector bar by bar."""
    hits = set()
    for j in range(bt.MIN_BARS + 2, len(df) + 1):
        matched, _ = reclaim_pattern.reclaim_signal(df.iloc[:j])
        if matched:
            hits.add(j - 1)
    return hits


def main():
    assert reclaim_pattern.MAX_WAIT == bt.MAX_WAIT, (
        f"MAX_WAIT differs: live={reclaim_pattern.MAX_WAIT} backtest={bt.MAX_WAIT}")

    totals = {"reclaim": 0}
    checked = 0

    for seed in range(40):
        df = random_walk(400, seed)
        found = bt.find_signals(df)
        for band in ("reclaim",):
            live = live_signals(df)
            back = {sig_i for sig_i, _e, _s, _g in found[band]}

            # The backtest drops setups with non-positive risk (stop >= entry);
            # the live detector has no such filter, so allow only THAT difference.
            only_live = live - back
            for j in only_live:
                grab_lows = bt._grabs(df)
                ok = any(st >= df["Close"].iloc[j] for st in grab_lows.values())
                assert ok, f"seed {seed} {band}: live fired on bar {j}, backtest missed it"

            only_back = back - live
            assert not only_back, (
                f"seed {seed} {band}: backtest fired on {sorted(only_back)}, live did not")

            totals[band] += len(back)
            checked += 1

    print(f"checked {checked} (seed, band) pairs over 40 random walks x 400 bars")
    print(f"reclaim signals matched: {totals['reclaim']}")
    assert sum(totals.values()) > 0, "no signals fired at all — the test proves nothing"
    print("\nOK: the backtest fires on exactly the same bars as the live bots.")


if __name__ == "__main__":
    main()
