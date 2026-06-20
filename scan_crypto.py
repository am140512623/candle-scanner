"""
Bot #2 -- intraday multi-timeframe crypto scanner.

Scans the top 300 crypto for the same Liquidity Grab pattern on EVERY clean
timeframe that is bigger than 4 hours and smaller than 1 week:

    6h, 8h, 12h, 1-day, 2-day, 3-day, 4-day

Yahoo only serves 1h and 1d candles natively, so the 6h/8h/12h frames are built
from hourly data and the 2d/3d/4d frames from daily data (crypto trades 24/7, so
these merge cleanly). Alerts go to a SEPARATE Telegram bot (CRYPTO_BOT_TOKEN).

The weekly (1W) and monthly (1M) frames are a DIFFERENT bot entirely -- see
scan_crypto_wm.py, which reuses this module's engine (resample / freshness /
WM_FRAMES) but sends to its own bot. The two never mix.
"""

import datetime
import os

import pandas as pd

import scan_all as s

# --- Separate bot for these (intraday) crypto alerts ---
# This script's main() is the INTRADAY bot: 6H..4D frames only -> CRYPTO_BOT_TOKEN.
# The weekly/monthly frames (1W, 1M) are a SEPARATE bot handled by scan_crypto_wm.py,
# which reuses this module's engine with its own token. The two never mix.
s.BOT_TOKEN = os.environ.get("CRYPTO_BOT_TOKEN") or "YOUR_CRYPTO_BOT_TOKEN"
s.CHAT_IDS = [
    "7788611624",   # A
    "6173185769",   # m
]
if os.environ.get("CRYPTO_CHAT_ID"):
    s.CHAT_IDS = [c.strip() for c in os.environ["CRYPTO_CHAT_ID"].split(",") if c.strip()]

CRYPTO_TOP_N = 300

# Origin used when merging daily candles into 2D/3D/4D. A fixed epoch makes the
# candle boundaries deterministic, so we can tell -- by the calendar alone --
# exactly which day each one closes on (see closes_today).
RESAMPLE_ORIGIN = "epoch"
_EPOCH = datetime.date(1970, 1, 1)

# Each timeframe: which base candles to pull, and how to merge them. "rkw" is the
# keyword args passed straight to df.resample(); rkw=None means use the base
# candles as-is. The Nd frames merge from a fixed epoch (deterministic day
# boundaries); the weekly frame uses Monday-anchored, left-labelled weeks so each
# bar is stamped at its own Monday start; the monthly frame uses month-start (MS)
# bars stamped at the 1st -- both keeping the "bar labelled at its start"
# convention the freshness logic below relies on.
#
# Two DISJOINT groups, each driven by its own bot/script (they never mix):
#   INTRADAY_FRAMES -> scan_crypto.py     (this file)        -> CRYPTO_BOT_TOKEN
#   WM_FRAMES       -> scan_crypto_wm.py  (weekly + monthly) -> CRYPTO_WM_BOT_TOKEN
INTRADAY_FRAMES = [
    {"label": "6H",  "base": "1h", "rkw": {"rule": "6h",  "origin": RESAMPLE_ORIGIN}},
    {"label": "8H",  "base": "1h", "rkw": {"rule": "8h",  "origin": RESAMPLE_ORIGIN}},
    {"label": "12H", "base": "1h", "rkw": {"rule": "12h", "origin": RESAMPLE_ORIGIN}},
    {"label": "1D",  "base": "1d", "rkw": None},
    {"label": "2D",  "base": "1d", "rkw": {"rule": "2D", "origin": RESAMPLE_ORIGIN}},
    {"label": "3D",  "base": "1d", "rkw": {"rule": "3D", "origin": RESAMPLE_ORIGIN}},
    {"label": "4D",  "base": "1d", "rkw": {"rule": "4D", "origin": RESAMPLE_ORIGIN}},
]
WM_FRAMES = [
    {"label": "1W",  "base": "1d", "rkw": {"rule": "W-MON", "label": "left", "closed": "left"}},
    {"label": "1M",  "base": "1d", "rkw": {"rule": "MS",    "label": "left", "closed": "left"}},
]
# Full catalog -- used only by the per-label helpers (closes_today, durations).
TIMEFRAMES = INTRADAY_FRAMES + WM_FRAMES

# How much history to pull for each base candle size. The daily pull needs to be
# long enough to leave plenty of monthly bars after resampling (2y -> ~24 months).
BASE_PERIODS = {"1h": "60d", "1d": "2y"}

# Multi-day frames close every Nth day; 6H/8H/12H/1D close every day. Weekly and
# monthly are calendar-based and handled specially in closes_today.
_MULTIDAY = {"2D": 2, "3D": 3, "4D": 4}

# How long each frame's candle lasts. Used to turn a candle's start timestamp
# into the wall-clock moment it actually closed (start + duration). 1M is NOT
# here -- months vary in length, so its close is derived from the calendar.
FRAME_DURATION = {
    "6H":  datetime.timedelta(hours=6),
    "8H":  datetime.timedelta(hours=8),
    "12H": datetime.timedelta(hours=12),
    "1D":  datetime.timedelta(days=1),
    "2D":  datetime.timedelta(days=2),
    "3D":  datetime.timedelta(days=3),
    "4D":  datetime.timedelta(days=4),
    "1W":  datetime.timedelta(days=7),
}

# Freshness gate. These bots are stateless, so a candle could be alerted twice if
# a trigger double-fires, fires late, or is replayed. We only alert a candle in
# the brief window right after it closes; anything that closed more than
# CRYPTO_FRESHNESS_MIN minutes ago is treated as already-handled and stays
# silent. (The download+scan of 300 coins takes a couple minutes, so the window
# must comfortably exceed normal run latency -- 30 min is a safe default.)
FRESHNESS_WINDOW = datetime.timedelta(
    minutes=int(os.environ.get("CRYPTO_FRESHNESS_MIN", "30")))


def selected_frames(catalog):
    """Which timeframes to scan this run, chosen from `catalog` (this bot's own
    frame group). CRYPTO_FRAMES (e.g. '6H,12H') lets a workflow scan only the
    frame(s) that just closed; unset = scan all of the bot's frames."""
    raw = os.environ.get("CRYPTO_FRAMES")
    if not raw:
        return list(catalog)
    want = {x.strip().upper() for x in raw.split(",") if x.strip()}
    return [tf for tf in catalog if tf["label"] in want]


def closes_today(label):
    """True if a candle of this timeframe closes at 00:00 UTC today.

    6H/8H/12H/1D close every day, so they're always 'today'. 2D/3D/4D only close
    every 2nd/3rd/4th day -- counted from the same fixed epoch the resampler uses
    -- which is how we avoid re-alerting the same multi-day candle on its off-days.
    1W closes only on Mondays (Mon-Sun weeks) and 1M only on the 1st of the month,
    so on every other day they self-skip and the just-closed candle isn't re-scanned.
    """
    today = datetime.datetime.now(datetime.timezone.utc).date()
    if label == "1W":
        return today.weekday() == 0      # Monday
    if label == "1M":
        return today.day == 1
    if label not in _MULTIDAY:
        return True
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
    if label == "1M":
        # Months vary in length, so derive the close from the calendar, not a
        # fixed duration: a month bar labelled at its 1st closes on the next 1st.
        return (ts + pd.offsets.MonthBegin(1)).to_pydatetime()
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


def resample_ohlc(df, rkw):
    """Merge candles into a bigger timeframe (e.g. 1h -> 6h). rkw is passed
    straight to df.resample() (rule + anchoring), so each frame controls its own
    bin boundaries and labelling."""
    agg = df.resample(**rkw).agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"})
    return agg.dropna()


def run(crypto, catalog=INTRADAY_FRAMES, bot="crypto_intraday"):
    """Scan a list of crypto tickers across the selected timeframes, sending each
    match via whatever bot token is currently set on scan_all (s.BOT_TOKEN).
    Returns the match count.

    `catalog` is this bot's frame group (INTRADAY_FRAMES or WM_FRAMES). Which of
    those actually run is further controlled by CRYPTO_FRAMES, so a workflow can
    scan ONLY the frame that just closed -- near-on-close alerts with no repeats.
    """
    frames = selected_frames(catalog)
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
                d = resample_ohlc(df, tf["rkw"]) if tf["rkw"] else df
                d = d.iloc[:-1]          # drop the still-forming candle
                if len(d) < 2:
                    continue
                # Freshness gate: only the candle that JUST closed is eligible.
                # Skips stale candles so a late/duplicate run can't re-alert them.
                if not is_fresh(tf["label"], d.index[-1]):
                    stale += 1
                    continue
                if s.is_flat(d):
                    continue          # pegged stablecoin -- noise, skip it
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
            s.log_signal("CRYPTO", t, tf["label"], d, bot=bot)
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
    # Token comes from the CRYPTO_BOT_TOKEN secret (no hardcoded fallback).
    # Log readiness -- never the token itself -- so a run shows at a glance
    # whether alerts will actually send.
    print("Telegram: " + ("configured" if s._telegram_ready()
                           else "NOT configured -- alerts will be skipped"))
    print("Building crypto universe...")
    crypto = s.get_top_crypto(CRYPTO_TOP_N)
    print(f"  Crypto: {len(crypto)} from top {CRYPTO_TOP_N} (stablecoins dropped)")
    run(crypto, INTRADAY_FRAMES)


if __name__ == "__main__":
    main()
