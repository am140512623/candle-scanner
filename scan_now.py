"""
Manual scanner -- run it whenever you want a fresh set of pattern charts.

    python scan_now.py                 # weekly timeframe, all asset groups
    python scan_now.py --both          # weekly AND monthly
    python scan_now.py --monthly       # monthly only
    python scan_now.py --only crypto   # just crypto (or: stocks / commodities)

It scans the same universe as scan_all.py using the SAME swallow pattern, then
saves a candlestick PNG for every match into a fresh, timestamped folder under
the charts/ directory. It does NOT log to signals.csv and does NOT send
Telegram/email -- it just makes the pictures and opens the folder when done.
"""

import argparse
import datetime
import os

import scan_all as s


def build_groups(only):
    """The asset groups to scan: all of them, or just the one named by --only."""
    groups = []
    if only in (None, "stocks"):
        stocks = sorted(set(s.get_sp500()) | set(s.get_nasdaq100()))
        print(f"  Stocks: {len(stocks)} unique (S&P 500 + Nasdaq-100)")
        groups.append(("STOCK", stocks, "Stocks"))
    if only in (None, "crypto"):
        crypto = s.get_top_crypto(s.CRYPTO_TOP_N)
        print(f"  Crypto: {len(crypto)} (top {s.CRYPTO_TOP_N}, stablecoins dropped)")
        groups.append(("CRYPTO", crypto, "Crypto"))
    if only in (None, "commodities"):
        commodities = list(s.COMMODITIES.keys())
        print(f"  Commodities (ETFs): {len(commodities)}")
        groups.append(("COMMODITY", commodities, "Commodities"))
    return groups


def chosen_timeframes(args):
    """Which timeframe(s) to run, taken straight from scan_all.TIMEFRAMES."""
    weekly = next(tf for tf in s.TIMEFRAMES if not tf["monthly_only"])
    monthly = next(tf for tf in s.TIMEFRAMES if tf["monthly_only"])
    if args.both:
        return [weekly, monthly]
    if args.monthly:
        return [monthly]
    return [weekly]


def main():
    parser = argparse.ArgumentParser(description="Manual pattern scan -> chart PNGs")
    parser.add_argument("--both", action="store_true", help="scan weekly AND monthly")
    parser.add_argument("--monthly", action="store_true",
                        help="scan monthly instead of weekly")
    parser.add_argument("--only", choices=["stocks", "crypto", "commodities"],
                        help="limit to one asset group (default: all)")
    args = parser.parse_args()

    # Save every chart from this run into one fresh, timestamped folder so each
    # manual run is self-contained. save_chart() writes under s.CHART_DIR, so we
    # just point that at the run folder.
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = os.path.join(s.CHART_DIR, f"scan_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    s.CHART_DIR = run_dir

    print("Building ticker universe...")
    groups = build_groups(args.only)

    matches = []  # (kind, ticker, df, timeframe_label, direction)
    for tf in chosen_timeframes(args):
        print(f"\n--- {tf['label']} ({tf['interval']}) ---")
        for kind, tickers, label in groups:
            for t, df, direction in s.scan(tickers, label, tf["interval"],
                                           tf["period"], tf["closed_only"]):
                matches.append((kind, t, df, tf["label"], direction))

    print("\n" + "=" * 40)
    if not matches:
        print("No matches this scan -- no charts to draw.")
        print("=" * 40)
        return

    print(f"MATCHES FOUND: {len(matches)}")
    for kind, t, df, tf_label, direction in matches:
        name = s.COMMODITIES.get(t, t)
        print(f"  [{tf_label}] {name} ({kind}) - {s.pattern_name(direction)}")
        try:
            path = s.save_chart(t, kind, df, tf_label, direction=direction)
            print(f"    chart -> {path}")
        except Exception as e:
            print(f"    (could not draw chart: {e})")
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
