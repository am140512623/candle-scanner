"""
Bot #2 -- multi-timeframe crypto scanner.

Scans the top 300 crypto for the same Liquidity Grab pattern on EVERY clean
timeframe that is bigger than 4 hours and smaller than 1 week:

    6h, 8h, 12h, 1-day, 2-day, 3-day, 4-day

Yahoo only serves 1h and 1d candles natively, so the 6h/8h/12h frames are built
from hourly data and the 2d/3d/4d frames from daily data (crypto trades 24/7, so
these merge cleanly). Alerts go to a SEPARATE Telegram bot.
"""

import datetime
import os

import pandas as pd

import scan_all as s

# --- Separate bot for these crypto alerts ---
s.BOT_TOKEN = os.environ.get("CRYPTO_BOT_TOKEN") or "YOUR_CRYPTO_BOT_TOKEN"
s.CHAT_IDS = [
    "7788611624",   # A
    "6173185769",   # m
]
if os.environ.get("CRYPTO_CHAT_ID"):
    s.CHAT_IDS = [c.strip() for c in os.environ["CRYPTO_CHAT_ID"].split(",") if c.strip()]

CRYPTO_TOP_N = 300

# Each timeframe: which base candles to pull, and how to merge them.
# rule=None means use the base candles as-is.
TIMEFRAMES = [
    {"label": "6H",  "base": "1h", "rule": "6h"},
    {"label": "8H",  "base": "1h", "rule": "8h"},
    {"label": "12H", "base": "1h", "rule": "12h"},
    {"label": "1D",  "base": "1d", "rule": None},
    {"label": "2D",  "base": "1d", "rule": "2D"},
    {"label": "3D",  "base": "1d", "rule": "3D"},
    {"label": "4D",  "base": "1d", "rule": "4D"},
]

# How much history to pull for each base candle size.
BASE_PERIODS = {"1h": "60d", "1d": "1y"}

# Origin used when merging daily candles into 2D/3D/4D. A fixed epoch makes the
# candle boundaries deterministic, so we can tell -- by the calendar alone --
# exactly which day each one closes on (see closes_today).
RESAMPLE_ORIGIN = "epoch"
_EPOCH = datetime.date(1970, 1, 1)
# Multi-day frames close every Nth day; the rest close every day.
_MULTIDAY = {"2D": 2, "3D": 3, "4D": 4}

# How long each frame's candle lasts. Used to turn a candle's start timestamp
# into the wall-clock moment it actually closed (start + duration).
FRAME_DURATION = {
    "6H":  datetime.timedelta(hours=6),
    "8H":  datetime.timedelta(hours=8),
    "12H": datetime.timedelta(hours=12),
    "1D":  datetime.timedelta(days=1),
    "2D":  datetime.timedelta(days=2),
    "3D":  datetime.timedelta(days=3),
    "4D":  datetime.timedelta(days=4),
}

# Freshness gate. These bots are stateless, so a candle could be alerted twice if
# a trigger double-fires, fires late, or is replayed. We only alert a candle in
# the brief window right after it closes; anything that closed more than
# CRYPTO_FRESHNESS_MIN minutes ago is treated as already-handled and stays
# silent. (The download+scan of 300 coins takes a couple minutes, so the window
# must comfortably exceed normal run latency -- 30 min is a safe default.)
FRESHNESS_WINDOW = datetime.timedelta(
    minutes=int(os.environ.get("CRYPTO_FRESHNESS_MIN", "30")))


def selected_frames():
    """Which timeframes to scan this run. CRYPTO_FRAMES (e.g. '6H,12H') lets a
    workflow scan only the frame(s) that just closed; unset = scan them all."""
    raw = os.environ.get("CRYPTO_FRAMES")
    if not raw:
        return TIMEFRAMES
    want = {x.strip().upper() for x in raw.split(",") if x.strip()}
    return [tf for tf in TIMEFRAMES if tf["label"] in want]


def closes_today(label):
    """True if a candle of this timeframe closes at 00:00 UTC today.

    6H/8H/12H/1D close every day, so they're always 'today'. 2D/3D/4D only close
    every 2nd/3rd/4th day -- counted from the same fixed epoch the resampler uses
    -- which is how we avoid re-alerting the same multi-day candle on its off-days.
    """
    if label not in _MULTIDAY:
        return True
    today = datetime.datetime.now(datetime.timezone.utc).date()
    days = (today - _EPOCH).days
    return days % _MULTIDAY[label] == 0


def candle_closed_at(label, candle_start):
    """UTC datetime when a `label` candle whose bar starts at candle_start closed.

    Yahoo/pandas label each bar by its left edge (its start), so the close is just
    start + the frame's duration. Daily bars come back tz-naive (UTC midnight) and
    hourly bars tz-aware -- normalise both to UTC.
    """
    ts = pd.Timestamp(candle_start)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.to_pydatetime() + FRAME_DURATION[label]


def is_fresh(label, candle_start, now=None):
    """True only if this candle closed within the freshness window (i.e. just now).

    Blocks re-alerts: a candle that closed more than FRESHNESS_WINDOW ago is
    stale, so a late / bunched / replayed run won't fire on it a second time.
    """
    now = now or datetime.datetime.now(datetime.timezone.utc)
    age = now - candle_closed_at(label, candle_start)
    return datetime.timedelta(0) <= age <= FRESHNESS_WINDOW


def download_base(tickers, interval, period):
    """Download OHLC for all tickers at one interval; return {ticker: DataFrame}."""
    out = {}
    for batch in s.chunked(tickers, s.BATCH_SIZE):
        data = s.yf.download(batch, period=period, interval=interval,
                             group_by="ticker", auto_adjust=True,
                             threads=True, progress=False)
        for t in batch:
            try:
                df = data[t] if len(batch) > 1 else data
                df = df.dropna()
                if not df.empty:
                    out[t] = df
            except (KeyError, IndexError):
                continue
    return out


def resample_ohlc(df, rule):
    """Merge candles into a bigger timeframe (e.g. 1h -> 6h)."""
    agg = df.resample(rule, origin=RESAMPLE_ORIGIN).agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"})
    return agg.dropna()


def run(crypto):
    """Scan a given list of crypto tickers across the selected timeframes.
    Returns the match count. Reused by the segmented rank bots (scan_segment.py).

    Which timeframes run is controlled by CRYPTO_FRAMES so a workflow can scan
    ONLY the frame that just closed -- giving near-on-close alerts with no repeats.
    """
    frames = selected_frames()
    # 2D/3D only "close" some days; skip them entirely on their off-days.
    frames = [tf for tf in frames if closes_today(tf["label"])]
    if not frames:
        print("No timeframe closes right now -- nothing to scan.")
        return 0
    print("Scanning timeframes: " + ", ".join(tf["label"] for tf in frames))

    # Pull each base candle size once, then reuse for every timeframe.
    needed_bases = {tf["base"] for tf in frames}
    bases = {}
    for b in needed_bases:
        print(f"\nDownloading {b} candles...")
        bases[b] = download_base(crypto, b, BASE_PERIODS[b])
        print(f"  got data for {len(bases[b])}/{len(crypto)} coins")

    total = 0
    for tf in frames:
        base_data = bases[tf["base"]]
        matches, stale = [], 0
        for t, df in base_data.items():
            try:
                d = resample_ohlc(df, tf["rule"]) if tf["rule"] else df
                d = d.iloc[:-1]          # drop the still-forming candle
                if len(d) < 2:
                    continue
                # Freshness gate: only the candle that JUST closed is eligible.
                # Skips stale candles so a late/duplicate run can't re-alert them.
                if not is_fresh(tf["label"], d.index[-1]):
                    stale += 1
                    continue
                if s.check_pattern(d):
                    matches.append((t, d))
            except Exception:
                continue
        print(f"\n[{tf['label']}] {len(matches)} match(es) from {len(base_data)} coins"
              + (f" ({stale} skipped: candle not fresh)" if stale else ""))
        for t, d in matches:
            total += 1
            yahoo, tv = s.chart_links("CRYPTO", t)
            msg = (f"[{tf['label']}] MATCH: {t} (CRYPTO) formed your Liquidity Grab pattern!\n"
                   f"Yahoo: {yahoo}\nTradingView: {tv}")
            print("  " + msg.splitlines()[0])
            try:
                chart_path = s.save_chart(t, "CRYPTO", d, tf["label"])
                s.send_telegram_photo(chart_path, msg)
            except Exception as e:
                print(f"    (could not draw chart: {e})")
                s.send_telegram_alert(msg)

    print("\n" + "=" * 40)
    print(f"TOTAL MATCHES across all timeframes: {total}")
    print("=" * 40)
    return total


def main():
    # Token now comes from the CRYPTO_BOT_TOKEN secret (no hardcoded fallback).
    # Log readiness -- never the token itself -- so a run shows at a glance
    # whether alerts will actually send.
    print("Telegram: " + ("configured" if s._telegram_ready()
                           else "NOT configured -- alerts will be skipped"))
    print("Building crypto universe...")
    crypto = s.get_top_crypto(CRYPTO_TOP_N)
    print(f"  Crypto: {len(crypto)} from top {CRYPTO_TOP_N} (stablecoins dropped)")
    run(crypto)


if __name__ == "__main__":
    main()
