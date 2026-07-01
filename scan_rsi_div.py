"""
Bot #6 -- RSI Oversold-Divergence -> Liquidity-Grab scanner (LONG only).

A two-stage bullish setup, scanned across the WHOLE universe (commodity ETFs,
US index ETFs, top crypto, and every US stock cap tier) on EVERY frame from 1H
up to 1Y:

    1H, 4H, 6H, 8H, 12H, 1D, 2D, 3D, 4D, 1W, 1M, 3M, 6M, 1Y

STAGE 1 -- RSI oversold divergence (the whole thing under 30):
    Price prints a LOWER low between two RSI(14) pivot lows while RSI prints a
    HIGHER low, and BOTH RSI pivot lows sit under 30. If BOTH are under 20 the
    setup is flagged STRONG and alerts with a different label/emoji.

STAGE 2 -- liquidity grab (the EXACT same `_swallow` engine as the stock/crypto
    bots): a green trigger candle whose body covers the prior green candle
    body(s) and whose low sweeps below their low(s). Fires on the 2-candle or
    3-candle layout.

ENTRY = a grab that lands within a window AFTER a valid oversold divergence, on
the just-closed candle. This is the Python twin of rsi_divergence_liquidity_grab.pine.

This bot has its OWN Telegram bot (RSI_DIV_BOT_TOKEN); nothing here mixes with
the other scanners. It reuses scan_all's engine (grab check, charts, Telegram,
logging, universe builders) and scan_crypto's download/resample helpers.

The universe is sharded with RSI_SEGMENT so each workflow run scans one slice
(commodities / indices / crypto / stock_mega / stock_large / stock_mid /
stock_small), and RSI_FRAMES narrows the scan to the frame(s) that just closed.
A freshness gate makes sure each candle is alerted at most once.

Run one slice locally (PowerShell):
    $env:RSI_SEGMENT="commodities"; python scan_rsi_div.py
"""

import datetime
import os

import mplfinance as mpf
import numpy as np
import pandas as pd

import scan_all as s
import scan_crypto as sc

# --- This bot's OWN Telegram credentials (no hardcoded fallback token) ---
s.BOT_TOKEN = os.environ.get("RSI_DIV_BOT_TOKEN") or "YOUR_RSI_DIV_BOT_TOKEN"
s.CHAT_IDS = [
    "7788611624",   # A
    "6173185769",   # m
]
if os.environ.get("RSI_DIV_CHAT_ID"):
    s.CHAT_IDS = [c.strip() for c in os.environ["RSI_DIV_CHAT_ID"].split(",") if c.strip()]

# ---------------------------------------------------------------------------
# PATTERN PARAMETERS  (mirror the .pine defaults)
# ---------------------------------------------------------------------------
RSI_LEN      = int(os.environ.get("RSI_LEN", "14"))
OS_LEVEL     = float(os.environ.get("RSI_OS_LEVEL", "30"))     # whole divergence under this
STRONG_LEVEL = float(os.environ.get("RSI_STRONG_LEVEL", "20")) # both legs under this = STRONG
PIV_LB       = int(os.environ.get("RSI_PIV_LB", "3"))          # RSI pivot bars to the left
PIV_RB       = int(os.environ.get("RSI_PIV_RB", "3"))          # RSI pivot bars to the right
MIN_GAP      = int(os.environ.get("RSI_MIN_GAP", "3"))         # min bars between the two RSI lows
MAX_GAP      = int(os.environ.get("RSI_MAX_GAP", "60"))        # max bars between the two RSI lows
GRAB_WINDOW  = int(os.environ.get("RSI_GRAB_WINDOW", "30"))    # max bars from divergence to the grab
# Enough bars to warm up RSI and print two separated, confirmed pivot lows. The two
# lows only need MIN_GAP between them, so this stays small -- real frames have far more.
MIN_BARS     = RSI_LEN + MIN_GAP + PIV_LB + PIV_RB + 10


def _wilder_rsi(close, length=RSI_LEN):
    """Wilder's RSI, matching TradingView's ta.rsi (RMA smoothing)."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    roll_down = down.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = roll_up / roll_down
    return 100.0 - 100.0 / (1.0 + rs)


def _pivot_lows(values, lb, rb):
    """Integer positions of strict pivot lows: a bar strictly lower than every one
    of its `lb` left neighbours and `rb` right neighbours (TradingView pivotlow)."""
    n = len(values)
    out = []
    for i in range(lb, n - rb):
        v = values[i]
        if np.isnan(v):
            continue
        left = values[i - lb:i]
        right = values[i + 1:i + rb + 1]
        if np.isnan(left).any() or np.isnan(right).any():
            continue
        if (left > v).all() and (right > v).all():
            out.append(i)
    return out


def divergence_grab(df):
    """(matched, strong, info) for the LONG oversold-divergence -> grab setup,
    evaluated on the LAST candle of `df` (the runner has already trimmed to the
    just-closed candle).

    matched -> a valid oversold divergence completed within GRAB_WINDOW bars before
               the last candle, AND that last candle is a bullish `_swallow` grab.
    strong  -> both RSI pivot lows of that divergence are under STRONG_LEVEL (<20).
    info    -> None if no match, else the two divergence pivots for the chart:
               {"d1": (pos, price_low, rsi), "d2": (pos, price_low, rsi)}
               (d1 = older/left low, d2 = newer/right low).

    Bottoms are read from the wicks (df['Low']), never the bodies -- same as the
    scanner's grab, which sweeps df['Low'].
    """
    if df is None or len(df) < MIN_BARS:
        return (False, False, None)

    rsi = _wilder_rsi(df["Close"])
    piv = _pivot_lows(rsi.values, PIV_LB, PIV_RB)
    if len(piv) < 2:
        return (False, False, None)

    lows = df["Low"].values
    rvals = rsi.values
    last = len(df) - 1

    # Only the LAST TWO bottoms count. d2 (the right low) must be the MOST RECENT
    # pivot low: if another RSI bottom printed after it, this isn't a fresh
    # last-two-bottoms setup, so we don't signal. No fallback to older pairs.
    strong = None
    info = None
    cur, prev = piv[-1], piv[-2]                     # newest two pivot lows
    gap = cur - prev
    if (last - cur) <= GRAB_WINDOW and MIN_GAP <= gap <= MAX_GAP:
        price_ll = lows[cur] < lows[prev]           # price lower low (wick)
        rsi_hl = rvals[cur] > rvals[prev]           # RSI higher low
        both_os = rvals[cur] < OS_LEVEL and rvals[prev] < OS_LEVEL
        if price_ll and rsi_hl and both_os:
            # No THIRD bottom between the second low and the grab: nothing may dip
            # below d2's low until the grab candle itself. If a lower low prints in
            # between, that's a third bottom -> not accepted.
            between = lows[cur + 1:last]            # bars strictly between d2 and the grab
            if between.size == 0 or float(between.min()) >= lows[cur]:
                strong = bool(rvals[cur] < STRONG_LEVEL and rvals[prev] < STRONG_LEVEL)
                info = {"d1": (int(prev), float(lows[prev]), float(rvals[prev])),
                        "d2": (int(cur),  float(lows[cur]),  float(rvals[cur]))}

    if strong is None:
        return (False, False, None)
    # Same grab engine as the stock/crypto bots -- must fire on the last candle.
    if not s._swallow(df, bullish=True):
        return (False, False, None)
    return (True, strong, info)


# ---------------------------------------------------------------------------
# TIMEFRAMES -- 1H up to 1Y (built from 1h and 1d Yahoo candles)
# ---------------------------------------------------------------------------
RESAMPLE_ORIGIN = "epoch"
_EPOCH = datetime.date(1970, 1, 1)

FRAMES_CATALOG = [
    {"label": "1H",  "base": "1h", "rkw": None},
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

# 1h base is capped by Yahoo (~2 months of hourly). Daily -> max so the divergence
# has plenty of history even on the yearly frame.
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
    minutes=int(os.environ.get("RSI_FRESHNESS_MIN", "90")))


def selected_frames(catalog):
    """Which frames to scan this run. RSI_FRAMES (e.g. '1H,1D') narrows to the
    frame(s) that just closed; unset = all of them."""
    raw = os.environ.get("RSI_FRAMES")
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
# UNIVERSE -- sharded by RSI_SEGMENT so each run scans one manageable slice
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
CRYPTO_TOP_N = int(os.environ.get("RSI_CRYPTO_TOP_N", "300"))


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
        "Set RSI_SEGMENT to one of: commodities, indices, crypto, "
        "stock_mega, stock_large, stock_mid, stock_small  (got %r)" % segment)


# ---------------------------------------------------------------------------
# CHART -- candles + RSI panel with the divergence drawn on both. STRONG (<20)
# gets a distinct theme + magenta so it stands out at a glance.
# ---------------------------------------------------------------------------
def save_div_chart(ticker, kind, df, tf_label, strong, info, bars=None):
    """Candlestick PNG with an RSI(14) sub-panel. Draws the divergence line across
    the two price lows AND across the two RSI lows (with pivot markers), the 30/20
    oversold lines, and highlights the grab candle(s). Normal = blue theme; STRONG
    (both legs <20) = a darker theme with magenta so it's instantly recognisable."""
    label = s.COMMODITIES.get(ticker, ticker)
    candle = df.index[-1].date()
    out_dir = os.path.join(s.CHART_DIR, f"RSIDIV_{tf_label}_{candle}")
    os.makedirs(out_dir, exist_ok=True)

    last = len(df) - 1
    d1pos, p1, r1 = info["d1"]
    d2pos, p2, r2 = info["d2"]
    if bars is None:                                  # show from the first low + margin
        bars = max(40, (last - d1pos) + 12)
    bars = min(bars, len(df))
    plot_df = df.tail(bars).copy()
    offset = len(df) - len(plot_df)
    i1, i2 = d1pos - offset, d2pos - offset           # pivot positions inside plot_df
    idx = plot_df.index

    # Compute RSI over the FULL history, then slice to the visible window, so the
    # RSI line spans the whole chart like the candles (no blank 14-bar warm-up gap
    # at the left edge).
    rsi = _wilder_rsi(df["Close"]).reindex(idx)
    have_pts = 0 <= i1 < len(idx) and 0 <= i2 < len(idx) and i2 > i1

    # Both divergence lines are TRENDLINES that touch the two lows on their panel:
    # the price line runs across the two candle tails (wick lows), and the RSI line
    # runs across the two RSI pivot lows -- touching the bottom of the RSI curve, the
    # same way the price line touches the candle tails. No offset, so neither floats.
    rsi_line = pd.Series(np.nan, index=idx)           # line across the two RSI lows
    if have_pts:
        for k in range(i1, i2 + 1):
            rsi_line.iloc[k] = r1 + (r2 - r1) * (k - i1) / (i2 - i1)

    # colour scheme: STRONG pops in magenta on a dark theme; normal is calm blue.
    theme = "nightclouds" if strong else "charles"
    accent = "magenta" if strong else "royalblue"
    # secondary_y=False keeps EVERY RSI-panel series on ONE shared y-axis. Without
    # it mplfinance auto-splits them onto two axes with different scales, so the
    # divergence line and 30/20 levels float at the wrong height vs the RSI curve.
    aps = [
        mpf.make_addplot(rsi, panel=1, color="teal", width=1.1, ylabel="RSI (14)",
                         secondary_y=False),
        mpf.make_addplot(pd.Series(OS_LEVEL, index=idx), panel=1, color="gray",
                         width=0.7, linestyle="--", secondary_y=False),
        mpf.make_addplot(pd.Series(STRONG_LEVEL, index=idx), panel=1, color="orange",
                         width=0.7, linestyle="--", secondary_y=False),
    ]
    if have_pts:                                       # thin divergence line so it doesn't cover the RSI curve
        aps.append(mpf.make_addplot(rsi_line, panel=1, color=accent, width=0.9,
                                    secondary_y=False))

    tag = "🔥 STRONG " if strong else ""
    out = os.path.join(out_dir, f"RSIDIV_{tf_label}_{s._safe_name(label)}_{candle}.png")
    kwargs = dict(
        type="candle", style=theme, addplot=aps, panel_ratios=(3, 1),
        title=f"\n{label} — {tag}RSI Oversold Divergence → Grab ({tf_label})",
        ylabel="Price",
        vlines=dict(vlines=[df.index[-2], df.index[-1]], colors=accent,
                    alpha=0.30, linewidths=9),
        savefig=dict(fname=out, dpi=120, bbox_inches="tight"),
    )
    if have_pts:                                       # price line across the two tails (wick lows)
        kwargs["alines"] = dict(alines=[[(idx[i1], p1), (idx[i2], p2)]],
                                colors=[accent], linewidths=[2.2])
    mpf.plot(plot_df, **kwargs)
    return out


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------
def _alert(kind, ticker, tf_label, d, strong, info, bot):
    """Log + send one match (chart if possible, else text). STRONG (<20) gets a
    distinct 🔥 label + magenta chart so it's obvious at a glance."""
    yahoo, tv = s.chart_links(kind, ticker)
    name = s.COMMODITIES.get(ticker, ticker)
    tag = "🔥 STRONG " if strong else ""
    extra = " [both RSI legs <20]" if strong else " [both RSI legs <30]"
    msg = (f"[{tf_label}] {tag}MATCH: {name} ({kind}) — "
           f"RSI oversold divergence → liquidity grab (LONG){extra}\n"
           f"Yahoo: {yahoo}\nTradingView: {tv}")
    print("  " + msg.splitlines()[0])
    # RSIDIV_ namespace so these rows never collide with the plain grab bot's, even
    # on the same ticker/timeframe/candle.
    s.log_signal(kind, ticker, tf_label, d, bot=bot, direction="long", id_prefix="RSIDIV_")
    try:
        chart_path = save_div_chart(ticker, kind, d, tf_label, strong, info)
        s.send_telegram_photo(chart_path, msg)
    except Exception as e:
        print(f"    (could not draw chart: {e})")
        s.send_telegram_alert(msg)


def run(segment, catalog=FRAMES_CATALOG, bot="rsi_div"):
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

    total = 0
    for tf in frames:
        base_data = bases[tf["base"]]
        matches, stale = [], 0
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
                matched, strong, info = divergence_grab(d)
                if matched:
                    matches.append((t, d, strong, info))
            except Exception:
                continue
        print(f"\n[{tf['label']}] {len(matches)} match(es) from {len(base_data)} symbols"
              + (f" ({stale} skipped: candle not fresh)" if stale else ""))
        for t, d, strong, info in matches:
            total += 1
            _alert(kind, t, tf["label"], d, strong, info, bot)

    print("\n" + "=" * 40)
    print(f"{segment}: {total} total match(es) across all frames")
    print("=" * 40)
    return total


def main():
    print("Telegram (RSI-divergence bot): " + ("configured" if s._telegram_ready()
          else "NOT configured -- alerts will be skipped"))
    segment = (os.environ.get("RSI_SEGMENT") or "commodities").strip()
    run(segment)


if __name__ == "__main__":
    main()
