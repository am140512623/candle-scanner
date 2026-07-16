"""
Bot #4 -- US index-futures scanner (US500 / US100 / US30).

Scans the three big US index futures for the same Liquidity Grab pattern on EVERY
clean timeframe from 4 hours up to 1 year:

    4H, 6H, 8H, 12H, 1D, 2D, 3D, 4D, 1W, 1M, 3M, 6M, 1Y

We use the CME futures (not the cash indices or ETFs) because they trade ~23h a
day, so the intraday 4H..12H candles align cleanly -- the cash indices only trade
~6.5h, which would leave ragged intraday bars. The mapping is:

    SPY -> US500  (S&P 500 ETF, real shares)
    QQQ -> US100  (Nasdaq-100 ETF, real shares)
    DIA -> US30   (Dow 30 ETF, real shares)

Yahoo serves 1h and 1d candles natively. The 4H/6H/8H/12H frames are built from
hourly data; everything 1D and above is built from daily data (which goes back to
~2000, so even the yearly frame has 20+ candles of history). Alerts go to this
bot's OWN Telegram bot (US_INDEX_BOT_TOKEN); nothing here mixes with the crypto
or stock bots.

This module reuses scan_all's engine (pattern check, charts, Telegram, logging)
and scan_crypto's generic helpers (download_base, resample_ohlc). The freshness
gate is the same idea as the crypto bots: a candle is only alerted in the short
window right after it closes, so a late or repeated run never double-alerts.
"""

import datetime
import os

import pandas as pd

import scan_all as s
import scan_crypto as sc
import reclaim_pattern            # grab -> reverse candle -> reclaim (LONG)
import bb_chart                   # shared per-signal chart styling

# --- This bot's own Telegram credentials ---
# NOTE: set AFTER every import above. Some scan_* modules assign scan_all.BOT_TOKEN
# at import time, so this assignment must come last to win. reclaim_pattern is
# deliberately free of that side effect.
s.BOT_TOKEN = os.environ.get("US_INDEX_BOT_TOKEN") or "YOUR_US_INDEX_BOT_TOKEN"
# Same recipients as the other bots by default; each must tap Start on THIS bot
# once. Override with the US_INDEX_CHAT_ID secret (comma-separated) if needed.
s.CHAT_IDS = [
    "7788611624",   # A
    "6173185769",   # m
]
if os.environ.get("US_INDEX_CHAT_ID"):
    s.CHAT_IDS = [c.strip() for c in os.environ["US_INDEX_CHAT_ID"].split(",") if c.strip()]

# The universe: three index ETFs you BUY as real shares (NOT futures/CFDs), with
# the friendly US### names shown in alerts and the TradingView symbol for the link.
INDICES = {
    "SPY": {"name": "US500 (S&P 500 — SPY)",    "tv": "SPY"},
    "QQQ": {"name": "US100 (Nasdaq 100 — QQQ)", "tv": "QQQ"},
    "DIA": {"name": "US30 (Dow 30 — DIA)",      "tv": "DIA"},
}
# Teach save_chart the friendly labels (it looks names up in COMMODITIES).
s.COMMODITIES.update({sym: meta["name"] for sym, meta in INDICES.items()})

RESAMPLE_ORIGIN = "epoch"
_EPOCH = datetime.date(1970, 1, 1)

# Every timeframe from 4H to 1Y. "base" is which candle size to pull; "rkw" is the
# kwargs handed straight to df.resample() (rule + anchoring), or None to use the
# base candles as-is. Daily-built frames are anchored at calendar starts so each
# bar is stamped at its own start (the freshness logic relies on this):
#   2D/3D/4D -> fixed epoch        1W -> Monday-anchored weeks
#   1M -> month start              3M -> quarter start (Jan/Apr/Jul/Oct)
#   6M -> half-year (Jan/Jul)      1Y -> year start (Jan 1)
FRAMES_CATALOG = [
    {"label": "4H",  "base": "1h", "rkw": {"rule": "4h",  "origin": RESAMPLE_ORIGIN}},
    {"label": "6H",  "base": "1h", "rkw": {"rule": "6h",  "origin": RESAMPLE_ORIGIN}},
    {"label": "8H",  "base": "1h", "rkw": {"rule": "8h",  "origin": RESAMPLE_ORIGIN}},
    {"label": "12H", "base": "1h", "rkw": {"rule": "12h", "origin": RESAMPLE_ORIGIN}},
    {"label": "1D",  "base": "1d", "rkw": None},
    {"label": "2D",  "base": "1d", "rkw": {"rule": "2D", "origin": RESAMPLE_ORIGIN}},
    {"label": "3D",  "base": "1d", "rkw": {"rule": "3D", "origin": RESAMPLE_ORIGIN}},
    {"label": "4D",  "base": "1d", "rkw": {"rule": "4D", "origin": RESAMPLE_ORIGIN}},
    {"label": "1W",  "base": "1d", "rkw": {"rule": "W-MON", "label": "left", "closed": "left"}},
    {"label": "1M",  "base": "1d", "rkw": {"rule": "MS",  "label": "left", "closed": "left"}},
    {"label": "3M",  "base": "1d", "rkw": {"rule": "QS",  "label": "left", "closed": "left"}},
    {"label": "6M",  "base": "1d", "rkw": {"rule": "2QS", "label": "left", "closed": "left"}},
    {"label": "1Y",  "base": "1d", "rkw": {"rule": "YS",  "label": "left", "closed": "left"}},
]

# How much history to pull for each base candle size. 1h is capped by Yahoo to a
# couple of months (plenty for the 4H..12H frames); the daily pull goes to "max"
# so the 1Y frame still has 20+ yearly candles of history.
BASE_PERIODS = {"1h": "60d", "1d": "max"}

# Multi-day frames close every Nth day (counted from the epoch). The intraday and
# 1D frames close "every day"; weekly/monthly/quarterly/half-year/yearly are
# calendar-based and handled directly in closes_today.
_MULTIDAY = {"2D": 2, "3D": 3, "4D": 4}

# How long each fixed-length frame lasts -- used to turn a candle's start stamp
# into the wall-clock moment it closed. The variable-length calendar frames
# (1M/3M/6M/1Y) are NOT here; their close is derived from the calendar instead.
FRAME_DURATION = {
    "4H":  datetime.timedelta(hours=4),
    "6H":  datetime.timedelta(hours=6),
    "8H":  datetime.timedelta(hours=8),
    "12H": datetime.timedelta(hours=12),
    "1D":  datetime.timedelta(days=1),
    "2D":  datetime.timedelta(days=2),
    "3D":  datetime.timedelta(days=3),
    "4D":  datetime.timedelta(days=4),
    "1W":  datetime.timedelta(days=7),
}

# Freshness gate. This bot is stateless, so a candle could be alerted twice if a
# trigger fires late, bunches, or is replayed. We only alert a candle in the brief
# window right after it closes; anything older is treated as already-handled and
# stays silent. The window is a bit wider than the crypto bots' (90 min default)
# so a simple hourly trigger reliably catches each just-closed intraday candle.
FRESHNESS_WINDOW = datetime.timedelta(
    minutes=int(os.environ.get("INDEX_FRESHNESS_MIN", "90")))


def selected_frames(catalog):
    """Which timeframes to scan this run. INDEX_FRAMES (e.g. '4H,1D') lets a
    trigger scan only the frame(s) that just closed; unset = scan all of them."""
    raw = os.environ.get("INDEX_FRAMES")
    if not raw:
        return list(catalog)
    want = {x.strip().upper() for x in raw.split(",") if x.strip()}
    return [tf for tf in catalog if tf["label"] in want]


def closes_today(label):
    """True if a candle of this timeframe closes at 00:00 UTC today.

    4H/6H/8H/12H/1D close every day (the freshness gate then picks the one that
    just closed). 2D/3D/4D close every 2nd/3rd/4th day from a fixed epoch. The
    calendar frames close on their own boundaries: 1W on Mondays, 1M on the 1st,
    3M on the 1st of Jan/Apr/Jul/Oct, 6M on the 1st of Jan/Jul, 1Y on Jan 1st.
    On every other day they self-skip, so a just-closed candle isn't re-scanned.
    """
    today = datetime.datetime.now(datetime.timezone.utc).date()
    if label == "1W":
        return today.weekday() == 0          # Monday
    if label == "1M":
        return today.day == 1
    if label == "3M":
        return today.day == 1 and today.month in (1, 4, 7, 10)
    if label == "6M":
        return today.day == 1 and today.month in (1, 7)
    if label == "1Y":
        return today.day == 1 and today.month == 1
    if label not in _MULTIDAY:
        return True
    days = (today - _EPOCH).days
    return days % _MULTIDAY[label] == 0


def candle_closed_at(label, candle_start):
    """UTC datetime when a `label` candle whose bar starts at candle_start closed.

    Bars are labelled at their left edge (their start), so the close is start +
    the frame's length. Fixed-length frames use FRAME_DURATION; the calendar
    frames (1M/3M/6M/1Y) vary in length, so their close is derived from the
    calendar (next month/quarter/half-year/year start). Daily-built bars come back
    tz-naive (UTC midnight) and hourly bars tz-aware -- normalise both to UTC.
    """
    ts = pd.Timestamp(candle_start)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    if label == "1M":
        return (ts + pd.offsets.MonthBegin(1)).to_pydatetime()
    if label == "3M":
        return (ts + pd.DateOffset(months=3)).to_pydatetime()
    if label == "6M":
        return (ts + pd.DateOffset(months=6)).to_pydatetime()
    if label == "1Y":
        return (ts + pd.DateOffset(years=1)).to_pydatetime()
    return ts.to_pydatetime() + FRAME_DURATION[label]


def is_fresh(label, candle_start, now=None):
    """True only if this candle closed within the freshness window (i.e. just now).
    Blocks re-alerts: a candle that closed longer ago than FRESHNESS_WINDOW is
    stale, so a late / bunched / replayed run won't fire on it a second time."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    age = now - candle_closed_at(label, candle_start)
    return datetime.timedelta(0) <= age <= FRESHNESS_WINDOW


def chart_links(ticker):
    """Yahoo link (exact match to our data) + TradingView link (continuous
    front-month contract, e.g. ES1!) for one index future."""
    yahoo = f"https://finance.yahoo.com/quote/{ticker}"
    tv = f"https://www.tradingview.com/symbols/{INDICES[ticker]['tv']}/"
    return yahoo, tv


# ---------------------------------------------------------------------------
# EXTRA PATTERN -- grab -> reverse candle -> reclaim (LONG), 4H..1D only.
# Runs on the SAME universe (US500 / US100 / US30) as the pattern above, on its
# own RECLAIM_FRAMES subset, and logs under its own IDXRC_ id-prefix so its rows
# never collide with this bot's main signals.
# ---------------------------------------------------------------------------
RECLAIM_FRAMES = {"4H", "6H", "8H", "12H", "1D"}


def save_reclaim_chart(ticker, df, tf_label, info, bars=80):
    """Candlestick PNG marking the grab, the reverse candle and the signal candle,
    with the reclaimed level drawn across. No Bollinger bands -- this pattern does
    not use them."""
    label = INDICES[ticker]["name"]
    candle = df.index[-1].date()
    out_dir = os.path.join(s.CHART_DIR, f"IDXRC_{tf_label}_{candle}")
    os.makedirs(out_dir, exist_ok=True)

    plot_df = df.tail(min(bars, len(df))).copy()
    roles = {
        df.index[info["grab_idx"]]: "GRAB",
        df.index[info["opp_idx"]]: "REVERSE",
        df.index[-1]: "SIGNAL",
    }
    out = os.path.join(out_dir, f"IDXRC_{tf_label}_{s._safe_name(label)}_{candle}.png")
    return bb_chart.render(
        plot_df, [], f"{label} — Grab → Reverse Candle → Reclaim ({tf_label})",
        out, "IDXRC_", roles, hline=info["opp_open"])


def _alert_reclaim(ticker, tf_label, d, info, bot):
    """Log + send one reclaim match (chart if possible, else text)."""
    name = INDICES[ticker]["name"]
    yahoo, tv = chart_links(ticker)
    msg = (f"[{tf_label}] MATCH: {name} (INDEX) — "
           f"grab → reverse candle → reclaim (LONG)\n"
           f"[reclaimed the reverse candle's open {info['opp_open']:.2f} "
           f"after {info['waited']} candle(s)]\n"
           f"Yahoo: {yahoo}\nTradingView: {tv}")
    # Telegram is UTF-8, but a Windows cp1252 console raises on "→"/"—" -- and an
    # exception here would abort the whole scan. Only the local echo is degraded.
    print("  " + msg.splitlines()[0].encode("ascii", "replace").decode())
    chart_path = None
    try:
        chart_path = save_reclaim_chart(ticker, d, tf_label, info)
    except Exception as e:
        print(f"    (could not draw chart: {e})")
    s.log_signal("INDEX", ticker, tf_label, d, bot=bot, direction="long",
                 id_prefix="IDXRC_", chart=s.chart_rel_path(chart_path))
    if chart_path:
        try:
            s.send_telegram_photo(chart_path, msg)
        except Exception as e:
            print(f"    (could not send photo: {e})")
            s.send_telegram_alert(msg)
    else:
        s.send_telegram_alert(msg)


def run(tickers=None, catalog=FRAMES_CATALOG, bot="us_index"):
    """Scan the index futures across the selected timeframes, sending each match
    via this bot's Telegram token. Returns the match count.

    Which frames actually run is controlled by INDEX_FRAMES (so a trigger can scan
    ONLY the frame that just closed) and by closes_today (calendar frames self-skip
    on their off-days). The freshness gate ensures each candle alerts at most once.
    """
    tickers = list(INDICES) if tickers is None else tickers
    frames = selected_frames(catalog)
    frames = [tf for tf in frames if closes_today(tf["label"])]
    if not frames:
        print("No timeframe closes right now -- nothing to scan.")
        return 0
    print("Scanning timeframes: " + ", ".join(tf["label"] for tf in frames))

    # Pull each base candle size once, then reuse for every timeframe that needs it.
    needed_bases = {tf["base"] for tf in frames}
    bases = {}
    for b in needed_bases:
        print(f"\nDownloading {b} candles...")
        bases[b] = sc.download_base(tickers, b, BASE_PERIODS[b])
        print(f"  got data for {len(bases[b])}/{len(tickers)} symbols")

    total = 0
    for tf in frames:
        base_data = bases[tf["base"]]
        matches, reclaims, stale = [], [], 0
        for t, df in base_data.items():
            try:
                d = sc.resample_ohlc(df, tf["rkw"]) if tf["rkw"] else df
                d = d.iloc[:-1]          # drop the still-forming candle
                if len(d) < 2:
                    continue
                # Freshness gate: only the candle that JUST closed is eligible.
                if not is_fresh(tf["label"], d.index[-1]):
                    stale += 1
                    continue
                # A candle is either green or red, so at most one of these fires.
                if s.check_pattern(d):
                    matches.append((t, d, "long"))
                elif s.check_pattern_bearish(d):
                    matches.append((t, d, "short"))
                # Independent extra pattern, 4H..1D only. Its signal candle is a
                # LATER bar than the grab, so it can fire on the same bar as the
                # main pattern or on its own; they never interfere.
                if tf["label"] in RECLAIM_FRAMES:
                    r_matched, r_info = reclaim_pattern.reclaim_signal(d)
                    if r_matched:
                        reclaims.append((t, d, r_info))
            except Exception:
                continue
        print(f"\n[{tf['label']}] {len(matches)} match(es), "
              f"{len(reclaims)} reclaim(s) from {len(base_data)} symbols"
              + (f" ({stale} skipped: candle not fresh)" if stale else ""))
        for t, d, info in reclaims:
            total += 1
            _alert_reclaim(t, tf["label"], d, info, bot + "_reclaim")
        for t, d, direction in matches:
            total += 1
            name = INDICES[t]["name"]
            yahoo, tv = chart_links(t)
            msg = (f"[{tf['label']}] MATCH: {name} (INDEX) formed your {s.pattern_name(direction)} pattern!\n"
                   f"Yahoo: {yahoo}\nTradingView: {tv}")
            print("  " + msg.splitlines()[0])
            chart_path = None
            try:
                chart_path = s.save_chart(t, "INDEX", d, tf["label"], direction=direction)
            except Exception as e:
                print(f"    (could not draw chart: {e})")
            s.log_signal("INDEX", t, tf["label"], d, bot=bot, direction=direction,
                         chart=s.chart_rel_path(chart_path))
            if chart_path:
                try:
                    s.send_telegram_photo(chart_path, msg)
                except Exception as e:
                    print(f"    (could not send photo: {e})")
                    s.send_telegram_alert(msg)
            else:
                s.send_telegram_alert(msg)

    print("\n" + "=" * 40)
    print(f"TOTAL MATCHES across all timeframes: {total}")
    print("=" * 40)
    return total


def main():
    print("Telegram (US index bot): " + ("configured" if s._telegram_ready()
                                          else "NOT configured -- alerts will be skipped"))
    print("Universe: " + ", ".join(f"{m['name']} [{sym}]" for sym, m in INDICES.items()))
    run()


if __name__ == "__main__":
    main()
