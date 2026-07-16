"""
Bot #7 -- Bollinger-Band Liquidity-Grab scanner (LONG only).

The SAME liquidity-grab ("swallow") engine as the stock/crypto/RSI bots, with ONE
extra gate on the trigger candle -- a Bollinger-Band breakout on the firing bar:

    STAGE 1 -- liquidity grab (the EXACT same `_swallow` engine): a green trigger
        candle whose body covers the prior green candle body(s) and whose low
        sweeps below their low(s). Fires on the 2-candle or 3-candle layout.

    STAGE 2 -- Bollinger breakout on that same trigger candle:
        - it OPENS below the UPPER band, and
        - it CLOSES above the UPPER band
      i.e. the grab candle punches up THROUGH the top band in a single bar.

ENTRY = a grab whose trigger candle closes above the upper Bollinger band while
having opened below it, on the just-closed candle. This is the Python twin of
bb_liquidity_grab.pine.

This bot has its OWN Telegram bot (BB_GRAB_BOT_TOKEN); nothing here mixes with the
other scanners. It reuses scan_all's engine (grab check, charts, Telegram, logging,
universe builders) and scan_crypto's download/resample helpers -- exactly the way
scan_rsi_div.py does, so none of the existing bots are touched.

Frames run from 4H up to 1Y (4H, 6H, 8H, 12H, 1D, 2D, 3D, 4D, 1W, 1M, 3M, 6M,
1Y) -- the 1H frame is intentionally skipped. The universe is sharded with
BB_SEGMENT so each workflow run scans one slice (commodities / indices / crypto
[top 100] / stock_mega / stock_large / stock_mid / stock_small), and BB_FRAMES
narrows the scan to the frame(s) that just closed. A freshness gate makes sure
each candle is alerted at most once.

Run one slice locally (PowerShell):
    $env:BB_SEGMENT="commodities"; python scan_bb_grab.py
"""

import datetime
import os

import mplfinance as mpf
import numpy as np
import pandas as pd

import scan_all as s
import scan_crypto as sc

# --- This bot's OWN Telegram credentials (no hardcoded fallback token) ---
s.BOT_TOKEN = os.environ.get("BB_GRAB_BOT_TOKEN") or "YOUR_BB_GRAB_BOT_TOKEN"
s.CHAT_IDS = [
    "7788611624",   # A
    "6173185769",   # m
]
if os.environ.get("BB_GRAB_CHAT_ID"):
    s.CHAT_IDS = [c.strip() for c in os.environ["BB_GRAB_CHAT_ID"].split(",") if c.strip()]

# ---------------------------------------------------------------------------
# PATTERN PARAMETERS  (mirror the .pine defaults / standard Bollinger settings)
# ---------------------------------------------------------------------------
BB_LEN  = int(os.environ.get("BB_LEN", "20"))     # SMA / stdev length for the bands
BB_MULT = float(os.environ.get("BB_MULT", "2.0")) # stdev multiplier for the outer bands
# Enough bars to warm up the band (needs BB_LEN closes) plus the swallow's lookback.
MIN_BARS = BB_LEN + 5


def _bollinger(close, length=BB_LEN, mult=BB_MULT):
    """Classic Bollinger Bands matching TradingView's `ta.sma` + `ta.stdev`.
    Returns (basis, upper, lower). `ta.stdev` is the POPULATION std (ddof=0)."""
    basis = close.rolling(length).mean()
    dev = close.rolling(length).std(ddof=0)
    upper = basis + mult * dev
    lower = basis - mult * dev
    return basis, upper, lower


def bb_grab(df):
    """(matched, info) for the LONG Bollinger-breakout liquidity grab, evaluated on
    the LAST candle of `df` (the runner has already trimmed to the just-closed one).

    matched -> the last candle is a bullish `_swallow` grab AND it opened below the
               upper band and closed above it (a single-bar breakout through the top).
    info    -> None if no match, else the band values on the trigger candle for the
               chart: {"basis", "upper", "lower"}.
    """
    if df is None or len(df) < MIN_BARS:
        return (False, None)

    # Same grab engine as every other bot -- must fire on the last candle.
    if not s._swallow(df, bullish=True):
        return (False, None)

    basis, upper, lower = _bollinger(df["Close"])
    o = df["Open"].iloc[-1]
    c = df["Close"].iloc[-1]
    up = upper.iloc[-1]
    if pd.isna(o) or pd.isna(c) or pd.isna(up):
        return (False, None)

    # The grab candle must punch UP THROUGH the top band in this one bar:
    # open below the upper band, close above it.
    if o < up and c > up:
        info = {
            "basis": float(basis.iloc[-1]),
            "upper": float(up),
            "lower": float(lower.iloc[-1]),
        }
        return (True, info)
    return (False, None)


def bb_grab_reclaim(df):
    """(matched, info) for the EXTRA signal: a Bollinger-breakout grab, then the
    candle right after it is OPPOSITE (red), then the FIRST candle to close above
    that red candle's OPEN -- which must be the just-closed candle. See bb_reclaim.

    Imported lazily: bb_reclaim imports this module for MIN_BARS/_bollinger, so a
    top-level import here would be circular."""
    import bb_reclaim
    return bb_reclaim.reclaim_signal(df, bb_reclaim.breakout_gate)


# ---------------------------------------------------------------------------
# TIMEFRAMES -- 4H up to 1Y (built from 1h and 1d Yahoo candles). This bot
# intentionally skips the 1H frame: it scans 4H -> 1W and 1W -> 1Y only.
# ---------------------------------------------------------------------------
RESAMPLE_ORIGIN = "epoch"
_EPOCH = datetime.date(1970, 1, 1)

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

# 1h base is capped by Yahoo (~2 months of hourly). Daily -> max so the bands have
# plenty of history even on the yearly frame.
BASE_PERIODS = {"1h": "60d", "1d": "max"}

_MULTIDAY = {"2D": 2, "3D": 3, "4D": 4}

FRAME_DURATION = {
    "1H":  datetime.timedelta(hours=1),
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

FRESHNESS_WINDOW = datetime.timedelta(
    minutes=int(os.environ.get("BB_FRESHNESS_MIN", "90")))


def selected_frames(catalog):
    """Which frames to scan this run. BB_FRAMES (e.g. '1H,1D') narrows to the
    frame(s) that just closed; unset = all of them."""
    raw = os.environ.get("BB_FRAMES")
    if not raw:
        return list(catalog)
    want = {x.strip().upper() for x in raw.split(",") if x.strip()}
    return [tf for tf in catalog if tf["label"] in want]


def closes_today(label):
    """True if a candle of this frame closes at 00:00 UTC today. Intraday and 1D
    close every day; 2D/3D/4D every Nth day from the epoch; the calendar frames on
    their own boundaries (1W Mondays, 1M the 1st, 3M Jan/Apr/Jul/Oct, 6M Jan/Jul,
    1Y Jan 1). On off-days a frame self-skips so a closed candle isn't re-scanned."""
    today = datetime.datetime.now(datetime.timezone.utc).date()
    if label == "1W":
        return today.weekday() == 0
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
    return (today - _EPOCH).days % _MULTIDAY[label] == 0


def candle_closed_at(label, candle_start):
    """UTC datetime a `label` candle (labelled at its start) closed."""
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
    """True only if this candle closed within the freshness window (blocks re-alerts)."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    age = now - candle_closed_at(label, candle_start)
    return datetime.timedelta(0) <= age <= FRESHNESS_WINDOW


# ---------------------------------------------------------------------------
# UNIVERSE -- sharded by BB_SEGMENT so each run scans one manageable slice
# ---------------------------------------------------------------------------
B = 1_000_000_000
M = 1_000_000

# Commodity + index ETFs you buy as real shares (kind = COMM / INDEX for links).
COMMODITY_TICKERS = list(s.COMMODITIES.keys())
INDEX_TICKERS = ["SPY", "QQQ", "DIA"]
s.COMMODITIES.setdefault("SPY", "US500 (S&P 500 — SPY)")
s.COMMODITIES.setdefault("QQQ", "US100 (Nasdaq 100 — QQQ)")
s.COMMODITIES.setdefault("DIA", "US30 (Dow 30 — DIA)")

# Stock cap tiers, same bands as scan_segment.py.
_STOCK_CAP = {
    "stock_mega":  (200 * B, None),
    "stock_large": (10 * B, 200 * B),
    "stock_mid":   (2 * B, 10 * B),
    "stock_small": (250 * M, 2 * B),
}
CRYPTO_TOP_N = int(os.environ.get("BB_CRYPTO_TOP_N", "100"))


def build_universe(segment):
    """Return (kind, [tickers], is_crypto) for one segment slice."""
    if segment == "commodities":
        return "COMM", COMMODITY_TICKERS, False
    if segment == "indices":
        return "INDEX", INDEX_TICKERS, False
    if segment == "crypto":
        return "CRYPTO", s.get_top_crypto(CRYPTO_TOP_N), True
    if segment in _STOCK_CAP:
        low, high = _STOCK_CAP[segment]
        lo = low or 0
        hi = high if high is not None else float("inf")
        rows = s.get_us_stocks_with_caps()
        return "STOCK", [sym for sym, cap in rows if lo <= cap < hi], False
    raise SystemExit(
        "Set BB_SEGMENT to one of: commodities, indices, crypto, "
        "stock_mega, stock_large, stock_mid, stock_small  (got %r)" % segment)


# ---------------------------------------------------------------------------
# CHART -- candles + the three Bollinger lines, with the grab candle(s) highlighted.
# ---------------------------------------------------------------------------
def save_bb_chart(ticker, kind, df, tf_label, info, bars=80):
    """Candlestick PNG overlaid with the Bollinger Bands (basis + upper/lower) and a
    highlight over the grab candle(s) that punched up through the top band. The aqua
    badge + colour identify this as signal ① -- see bb_chart."""
    import bb_chart

    label = s.COMMODITIES.get(ticker, ticker)
    candle = df.index[-1].date()
    out_dir = os.path.join(s.CHART_DIR, f"BBGRAB_{tf_label}_{candle}")
    os.makedirs(out_dir, exist_ok=True)

    bars = min(bars, len(df))
    plot_df = df.tail(bars).copy()

    # Here the grab candle IS the signal candle; the one before it is the candle
    # it swallowed.
    roles = {df.index[-2]: "GRAB", df.index[-1]: "SIGNAL"}

    out = os.path.join(out_dir, f"BBGRAB_{tf_label}_{s._safe_name(label)}_{candle}.png")
    return bb_chart.render(
        plot_df, bb_chart.band_addplots(plot_df, _bollinger),
        f"{label} — Bollinger Breakout → Liquidity Grab ({tf_label})",
        out, "BBGRAB_", roles)


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------
def _alert(kind, ticker, tf_label, d, info, bot):
    """Log + send one match (chart if possible, else text)."""
    yahoo, tv = s.chart_links(kind, ticker)
    name = s.COMMODITIES.get(ticker, ticker)
    msg = (f"[{tf_label}] MATCH: {name} ({kind}) — "
           f"Bollinger breakout → liquidity grab (LONG) "
           f"[closed above upper band]\n"
           f"Yahoo: {yahoo}\nTradingView: {tv}")
    print("  " + msg.splitlines()[0])
    chart_path = None
    try:
        chart_path = save_bb_chart(ticker, kind, d, tf_label, info)
    except Exception as e:
        print(f"    (could not draw chart: {e})")
    # BBGRAB_ namespace so these rows never collide with the plain grab bot's, even
    # on the same ticker/timeframe/candle.
    s.log_signal(kind, ticker, tf_label, d, bot=bot, direction="long",
                 id_prefix="BBGRAB_", chart=s.chart_rel_path(chart_path))
    if chart_path:
        try:
            s.send_telegram_photo(chart_path, msg)
        except Exception as e:
            print(f"    (could not send photo: {e})")
            s.send_telegram_alert(msg)
    else:
        s.send_telegram_alert(msg)


def run(segment, catalog=FRAMES_CATALOG, bot="bb_grab"):
    """Scan one universe segment across the selected frames. Returns match count."""
    kind, tickers, is_crypto = build_universe(segment)
    print(f"=== Segment: {segment} ({kind}) — {len(tickers)} tickers ===")

    frames = [tf for tf in selected_frames(catalog) if closes_today(tf["label"])]
    if not frames:
        print("No timeframe closes right now -- nothing to scan.")
        return 0
    print("Scanning timeframes: " + ", ".join(tf["label"] for tf in frames))

    needed_bases = {tf["base"] for tf in frames}
    bases = {}
    for b in needed_bases:
        print(f"\nDownloading {b} candles...")
        bases[b] = sc.download_base(tickers, b, BASE_PERIODS[b])
        print(f"  got data for {len(bases[b])}/{len(tickers)} symbols")
        if is_crypto:
            bases[b] = s.dedupe_crypto(bases[b])   # drop wrapped/bridged clones

    import bb_reclaim

    total = 0
    for tf in frames:
        base_data = bases[tf["base"]]
        matches, reclaims, stale = [], [], 0
        for t, df in base_data.items():
            try:
                d = sc.resample_ohlc(df, tf["rkw"]) if tf["rkw"] else df
                d = d.iloc[:-1]                      # drop the still-forming candle
                if len(d) < MIN_BARS:
                    continue
                if not is_fresh(tf["label"], d.index[-1]):
                    stale += 1
                    continue
                if is_crypto and s.is_flat(d):
                    continue                         # pegged stablecoin -- noise
                matched, info = bb_grab(d)
                if matched:
                    matches.append((t, d, info))
                # Independent extra signal -- a breakout grab that has since been
                # reclaimed. Both can fire on the same symbol/frame; they are
                # different candles, logged under different id-prefixes.
                r_matched, r_info = bb_grab_reclaim(d)
                if r_matched:
                    reclaims.append((t, d, r_info))
            except Exception:
                continue
        print(f"\n[{tf['label']}] {len(matches)} match(es), "
              f"{len(reclaims)} reclaim(s) from {len(base_data)} symbols"
              + (f" ({stale} skipped: candle not fresh)" if stale else ""))
        for t, d, info in matches:
            total += 1
            _alert(kind, t, tf["label"], d, info, bot)
        for t, d, info in reclaims:
            total += 1
            bb_reclaim.alert(kind, t, tf["label"], d, info, bot + "_reclaim",
                             "BBGRABRC_", "Breakout Grab → Opposite Candle → Reclaim",
                             "grab closed above the upper band")

    print("\n" + "=" * 40)
    print(f"{segment}: {total} total match(es) across all frames")
    print("=" * 40)
    return total


def main():
    print("Telegram (Bollinger-grab bot): " + ("configured" if s._telegram_ready()
          else "NOT configured -- alerts will be skipped"))
    segment = (os.environ.get("BB_SEGMENT") or "commodities").strip()
    run(segment)


if __name__ == "__main__":
    main()
