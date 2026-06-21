"""
Manual scanner -- run it by hand to get the latest pattern matches as chart PNGs.

ONE file, the SAME coverage as all your scheduled bots. With no flags it runs
every bot; with --bot it runs just one. The seven scanner bots are:

    stock_mega        US stocks > $200B                 (weekly + monthly)
    stock_large       US stocks $10B - $200B            (weekly + monthly)
    stock_mid         US stocks $2B - $10B              (weekly + monthly)
    stock_small       US stocks $250M - $2B             (weekly + monthly)
    crypto_top1000    crypto rank 300 - 1000            (weekly + monthly)
    crypto_wm         crypto rank 1 - 300               (weekly + monthly)
    crypto_intraday   crypto rank 1 - 300               (6H,8H,12H,1D,2D,3D,4D)

Unlike the live bots it does NOT log to signals.csv and does NOT send Telegram/
email, and it ignores the "candle just closed" freshness gate -- it simply checks
each ticker's most recent CLOSED candle. Every match is saved as a candlestick
PNG into a fresh, timestamped folder under the charts/ directory, which opens in
File Explorer when the run finishes.

    python scan_now.py                      # every bot (slow)
    python scan_now.py --bot stock_mid      # just one bot
    python scan_now.py --no-intraday        # every bot except crypto_intraday

The same swallow pattern as the live bots is used (imported from scan_all).
"""

import argparse
import datetime
import os

import scan_all as s
import scan_crypto as cr
import scan_segment as seg


def scan_crypto_frames(crypto, frames):
    """Scan a crypto list across `frames`, checking each coin's most recent CLOSED
    candle (no freshness gate, unlike the live bot). Returns match tuples
    (kind, ticker, df, tf_label, direction)."""
    needed_bases = {tf["base"] for tf in frames}
    bases = {}
    for b in needed_bases:
        print(f"    downloading {b} candles for {len(crypto)} coins...")
        bases[b] = cr.download_base(crypto, b, cr.BASE_PERIODS[b])
        print(f"      got data for {len(bases[b])}/{len(crypto)} coins")

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
        print(f"    [{tf['label']}] {n} match(es) from {len(base_data)} coins")
    return out


def scan_stock_tier(key):
    """One stock cap-tier bot (stock_mega/large/mid/small) on weekly + monthly."""
    seg_def = seg.SEGMENTS[key]
    low, high = seg_def["cap"]
    universe = seg.build_stock_universe(low, high)
    print(f"    {seg_def['label']}: {len(universe)} stocks")
    groups = [("STOCK", universe, seg_def["label"])]
    out = []
    for tf in s.TIMEFRAMES:   # weekly (keep forming bar) + monthly (drop it)
        for k, t, df, direction in s.run_timeframe(tf, groups):
            out.append((k, t, df, tf["label"], direction))
    return out


def scan_crypto_top1000():
    """The crypto_top1000 bot: ranks 300-1000 on weekly + monthly."""
    low, high = seg.SEGMENTS["crypto_top1000"]["rank"]
    universe = seg.build_crypto_universe(low, high)
    print(f"    Crypto rank {low}-{high}: {len(universe)} coins")
    return scan_crypto_frames(universe, cr.WM_FRAMES)


def scan_crypto_wm():
    """The crypto_wm bot: top 300 on weekly + monthly."""
    universe = s.get_top_crypto(cr.CRYPTO_TOP_N)
    print(f"    Crypto top {cr.CRYPTO_TOP_N}: {len(universe)} coins")
    return scan_crypto_frames(universe, cr.WM_FRAMES)


def scan_crypto_intraday():
    """The crypto_intraday bot: top 300 on the 6H..4D frames."""
    universe = s.get_top_crypto(cr.CRYPTO_TOP_N)
    print(f"    Crypto top {cr.CRYPTO_TOP_N}: {len(universe)} coins")
    return scan_crypto_frames(universe, cr.INTRADAY_FRAMES)


# Each bot name -> the function that scans exactly what that scheduled bot scans.
BOTS = {
    "stock_mega":      lambda: scan_stock_tier("stock_mega"),
    "stock_large":     lambda: scan_stock_tier("stock_large"),
    "stock_mid":       lambda: scan_stock_tier("stock_mid"),
    "stock_small":     lambda: scan_stock_tier("stock_small"),
    "crypto_top1000":  scan_crypto_top1000,
    "crypto_wm":       scan_crypto_wm,
    "crypto_intraday": scan_crypto_intraday,
}


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
    parser = argparse.ArgumentParser(
        description="Manual pattern scan -> chart PNGs (one bot or all of them)")
    parser.add_argument("--bot", choices=list(BOTS),
                        help="run just one bot (default: all of them)")
    parser.add_argument("--no-intraday", action="store_true",
                        help="when running all bots, skip the heavy crypto_intraday bot")
    args = parser.parse_args()

    # Save every chart from this run into one fresh, timestamped folder so each
    # manual run is self-contained. save_chart() writes under s.CHART_DIR.
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = os.path.join(s.CHART_DIR, f"scan_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    s.CHART_DIR = run_dir

    if args.bot:
        todo = [args.bot]
    else:
        todo = list(BOTS)
        if args.no_intraday:
            todo.remove("crypto_intraday")

    print(f"Scanning bot(s): {', '.join(todo)}")
    print("(this can take several minutes for the full universe)\n")

    matches = []
    for name in todo:
        print(f"=== {name} ===")
        matches += BOTS[name]()
        print()

    print("=" * 40)
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
