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

    python scan_now.py            # show a numbered menu, type a number to pick
    python scan_now.py 3          # run bot #3 straight away (no menu)
    python scan_now.py 0          # run ALL bots
    python scan_now.py stock_mid  # pick by name too, if you prefer
    python scan_now.py --no-intraday   # all bots except the heavy crypto_intraday

Pick crypto_intraday (7) and it ALSO asks which frame (6H..4D); you can answer
on the command line too:
    python scan_now.py 7 1        # crypto_intraday, frame #1 (6H)
    python scan_now.py 7 12H      # crypto_intraday, the 12H frame
    python scan_now.py 7 0        # crypto_intraday, all frames

Menu numbering (type):
    0 = ALL    1 = stock_mega   2 = stock_large   3 = stock_mid
    4 = stock_small   5 = crypto_top1000   6 = crypto_wm   7 = crypto_intraday
Frame numbering (crypto_intraday only):
    0 = ALL    1 = 6H   2 = 8H   3 = 12H   4 = 1D   5 = 2D   6 = 3D   7 = 4D

Charts land in charts/<type>[_<frame>]_<timestamp>/<FRAME>_<date>/ -- the folder
references the type and the frame, like the old pre-bot version. The same swallow
pattern as the live bots is used (imported from scan_all). No CSV, no Telegram.
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
        bases[b] = s.dedupe_crypto(bases[b])   # drop wrapped/bridged clones

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


def scan_crypto_intraday(frames=None):
    """The crypto_intraday bot: top 300 on the 6H..4D frames (all, or a subset)."""
    universe = s.get_top_crypto(cr.CRYPTO_TOP_N)
    print(f"    Crypto top {cr.CRYPTO_TOP_N}: {len(universe)} coins")
    return scan_crypto_frames(universe, frames or cr.INTRADAY_FRAMES)


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

# The order the bots show up in the numbered menu (1..7). Number 0 = ALL.
ORDER = ["stock_mega", "stock_large", "stock_mid", "stock_small",
         "crypto_top1000", "crypto_wm", "crypto_intraday"]
LABELS = {
    "stock_mega":      "Mega-Cap stocks (>$200B)        weekly+monthly",
    "stock_large":     "Large-Cap stocks ($10B-$200B)   weekly+monthly",
    "stock_mid":       "Mid-Cap stocks ($2B-$10B)       weekly+monthly",
    "stock_small":     "Small-Cap stocks ($250M-$2B)    weekly+monthly",
    "crypto_top1000":  "Crypto rank 300-1000            weekly+monthly",
    "crypto_wm":       "Crypto top 300                  weekly+monthly",
    "crypto_intraday": "Crypto top 300                  6H,8H,12H,1D,2D,3D,4D",
}


def print_menu():
    """Show the numbered list of bots to pick from."""
    print("Which bot do you want to scan?\n")
    print("   0) ALL bots")
    for i, key in enumerate(ORDER, 1):
        print(f"   {i}) {key:<16}{LABELS[key]}")
    print()


def resolve_choice(raw):
    """Turn a menu answer (a number, 'all', or a bot name) into a list of bot
    keys to run, or None if it isn't recognised."""
    raw = (raw or "").strip().lower()
    if raw in ("0", "all"):
        return list(ORDER)
    if raw.isdigit() and 1 <= int(raw) <= len(ORDER):
        return [ORDER[int(raw) - 1]]
    if raw in BOTS:
        return [raw]
    return None


# crypto_intraday frames, in menu order (1..7). Number 0 = ALL of them.
INTRADAY_LABELS = [tf["label"] for tf in cr.INTRADAY_FRAMES]   # 6H,8H,12H,1D,2D,3D,4D


def choose_intraday_frames(raw=None):
    """Pick which crypto_intraday timeframe(s) to scan. Shows a menu if `raw` is
    None; accepts a number (1..7), 'all'/0, or a label like '6H'. Returns the
    list of frame dicts to scan."""
    if raw is None:
        print("\nWhich timeframe for crypto_intraday?\n")
        print("   0) ALL frames")
        for i, lbl in enumerate(INTRADAY_LABELS, 1):
            print(f"   {i}) {lbl}")
        raw = input("\nEnter number: ")
    raw = (raw or "").strip().lower()
    if raw in ("0", "all", ""):
        return cr.INTRADAY_FRAMES
    if raw.isdigit() and 1 <= int(raw) <= len(INTRADAY_LABELS):
        want = {INTRADAY_LABELS[int(raw) - 1]}
    else:
        want = {x.strip().upper() for x in raw.split(",") if x.strip()}
    chosen = [tf for tf in cr.INTRADAY_FRAMES if tf["label"] in want]
    return chosen or cr.INTRADAY_FRAMES


def run_folder_name(todo, intraday_frames):
    """Name the output folder after the type (and frame, for a single intraday
    frame) so it's easy to tell runs apart -- e.g. crypto_intraday_6H, stock_mid."""
    if todo == list(ORDER):
        return "ALL"
    if len(todo) == 1:
        name = todo[0]
        if name == "crypto_intraday" and len(intraday_frames) == 1:
            name += "_" + intraday_frames[0]["label"]
        return name
    return "_".join(todo)


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
    parser.add_argument("choice", nargs="?",
                        help="bot number 1-7, a bot name, or 0/all. "
                             "Omit to get the numbered menu.")
    parser.add_argument("frame", nargs="?",
                        help="for crypto_intraday only: frame number 1-7, a label "
                             "like 6H, or 0/all. Omit to get the frame menu.")
    parser.add_argument("--no-intraday", action="store_true",
                        help="when running all bots, skip the heavy crypto_intraday bot")
    args = parser.parse_args()

    # No choice on the command line -> show the menu and ask for a number.
    raw = args.choice
    if raw is None:
        print_menu()
        raw = input("Enter number: ")

    todo = resolve_choice(raw)
    if todo is None:
        raise SystemExit(f"Unknown choice {raw!r}. Pick 0-{len(ORDER)} or a bot name.")
    if args.no_intraday and len(todo) > 1 and "crypto_intraday" in todo:
        todo.remove("crypto_intraday")

    # If we're running ONLY crypto_intraday, also ask which timeframe(s).
    intraday_frames = cr.INTRADAY_FRAMES
    if todo == ["crypto_intraday"]:
        intraday_frames = choose_intraday_frames(args.frame)

    # Save every chart from this run into one fresh folder, named after the type
    # (and frame) + a timestamp. save_chart() writes its own TF_date subfolders
    # under s.CHART_DIR, so the final path references both the type and the frame.
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = os.path.join(s.CHART_DIR, f"{run_folder_name(todo, intraday_frames)}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    s.CHART_DIR = run_dir

    print(f"\nScanning bot(s): {', '.join(todo)}")
    if todo == ["crypto_intraday"]:
        print(f"Frames: {', '.join(tf['label'] for tf in intraday_frames)}")
    print("(this can take several minutes for the full universe)\n")

    matches = []
    for name in todo:
        print(f"=== {name} ===")
        if name == "crypto_intraday":
            matches += scan_crypto_intraday(intraday_frames)
        else:
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
