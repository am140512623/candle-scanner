"""
Manual scanner -- run it by hand to get the latest pattern matches as chart PNGs.

It reproduces the SAME coverage as all the scheduled bots, in this one file:
  - US stocks, every cap tier >= $250M    (weekly + monthly)
  - Crypto top 1000                        (weekly + monthly)
  - Crypto top 300                         (intraday: 6H, 8H, 12H, 1D, 2D, 3D, 4D)

Unlike the bots it does NOT log to signals.csv and does NOT send Telegram/email,
and it ignores the "candle just closed" freshness gate -- it simply checks each
ticker's most recent CLOSED candle on every timeframe. Every match is saved as a
candlestick PNG into a fresh, timestamped folder under the charts/ directory,
which opens in File Explorer when the run finishes.

    python scan_now.py                # everything (slow: thousands of tickers)
    python scan_now.py --only stocks  # just stocks   (or: crypto)
    python scan_now.py --no-intraday  # skip the heavy 6H..4D crypto frames

The same swallow pattern as the live bots is used (imported from scan_all).
"""

import argparse
import datetime
import os

import scan_all as s
import scan_crypto as cr

M = 1_000_000
B = 1_000_000_000

MIN_STOCK_CAP = 250 * M   # smallest cap tier the bots scan (small-cap floor)
CRYPTO_WM_TOP = 1000      # weekly/monthly crypto universe (ranks 1-1000)
CRYPTO_INTRADAY_TOP = 300 # intraday crypto universe (ranks 1-300)


def stock_universe():
    """Every US stock at or above the small-cap floor (all tiers the bots cover)."""
    rows = s.get_us_stocks_with_caps()
    return [sym for sym, cap in rows if cap >= MIN_STOCK_CAP]


def scan_stocks():
    """All US stocks on weekly + monthly. Returns (kind, t, df, tf_label, dir)."""
    universe = stock_universe()
    print(f"  Stocks: {len(universe)} tickers (all tiers >= ${MIN_STOCK_CAP/B:.2g}B)")
    groups = [("STOCK", universe, "Stocks")]
    out = []
    for tf in s.TIMEFRAMES:   # weekly (keep forming bar) + monthly (drop it)
        for k, t, df, direction in s.run_timeframe(tf, groups):
            out.append((k, t, df, tf["label"], direction))
    return out


def scan_crypto_frames(crypto, frames):
    """Scan a crypto list across `frames`, checking each coin's most recent CLOSED
    candle (no freshness gate, unlike the live bot). Returns match tuples."""
    needed_bases = {tf["base"] for tf in frames}
    bases = {}
    for b in needed_bases:
        print(f"\n  Downloading {b} candles for {len(crypto)} coins...")
        bases[b] = cr.download_base(crypto, b, cr.BASE_PERIODS[b])
        print(f"    got data for {len(bases[b])}/{len(crypto)} coins")

    out = []
    for tf in frames:
        base_data = bases[tf["base"]]
        n = 0
        for t, df in base_data.items():
            try:
                d = cr.resample_ohlc(df, tf["rkw"]) if tf["rkw"] else df
                d = d.iloc[:-1]          # drop the still-forming candle
                if len(d) < 2 or s.is_flat(d):
                    continue
                # A candle is either green or red, so at most one of these fires.
                if s.check_pattern(d):
                    out.append(("CRYPTO", t, d, tf["label"], "long")); n += 1
                elif s.check_pattern_bearish(d):
                    out.append(("CRYPTO", t, d, tf["label"], "short")); n += 1
            except Exception:
                continue
        print(f"  [{tf['label']}] {n} match(es) from {len(base_data)} coins")
    return out


def scan_crypto(do_intraday=True):
    """Crypto coverage: top 1000 on weekly/monthly, top 300 on intraday frames."""
    out = []
    top_wm = s.get_top_crypto(CRYPTO_WM_TOP)
    print(f"  Crypto weekly/monthly: {len(top_wm)} coins (top {CRYPTO_WM_TOP})")
    out += scan_crypto_frames(top_wm, cr.WM_FRAMES)
    if do_intraday:
        top_intraday = s.get_top_crypto(CRYPTO_INTRADAY_TOP)
        print(f"\n  Crypto intraday: {len(top_intraday)} coins (top {CRYPTO_INTRADAY_TOP})")
        out += scan_crypto_frames(top_intraday, cr.INTRADAY_FRAMES)
    return out


def draw(matches):
    """Save a chart PNG for every match and print where each landed."""
    for kind, t, df, tf_label, direction in matches:
        name = s.COMMODITIES.get(t, t)
        print(f"  [{tf_label}] {name} ({kind}) - {s.pattern_name(direction)}")
        try:
            path = s.save_chart(t, kind, df, tf_label, direction=direction)
            print(f"    chart -> {path}")
        except Exception as e:
            print(f"    (could not draw chart: {e})")


def main():
    parser = argparse.ArgumentParser(description="Manual pattern scan -> chart PNGs")
    parser.add_argument("--only", choices=["stocks", "crypto"],
                        help="limit to one asset class (default: both)")
    parser.add_argument("--no-intraday", action="store_true",
                        help="skip the heavy 6H..4D crypto frames")
    args = parser.parse_args()

    # Save every chart from this run into one fresh, timestamped folder so each
    # manual run is self-contained. save_chart() writes under s.CHART_DIR.
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = os.path.join(s.CHART_DIR, f"scan_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    s.CHART_DIR = run_dir

    print("Scanning -- this can take several minutes for the full universe.\n")
    matches = []
    if args.only in (None, "stocks"):
        print("=== STOCKS ===")
        matches += scan_stocks()
    if args.only in (None, "crypto"):
        print("\n=== CRYPTO ===")
        matches += scan_crypto(do_intraday=not args.no_intraday)

    print("\n" + "=" * 40)
    if not matches:
        print("No matches this scan -- no charts to draw.")
        print("=" * 40)
        return

    print(f"MATCHES FOUND: {len(matches)}")
    draw(matches)
    print("=" * 40)
    print(f"\nCharts saved in: {run_dir}")

    # Pop the folder open in File Explorer (Windows only).
    if os.name == "nt":
        try:
            os.startfile(run_dir)
        except Exception as e:
            print(f"  (could not auto-open folder: {e})")


if __name__ == "__main__":
    main()
