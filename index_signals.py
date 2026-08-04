"""
Sharp-Drop -> Higher-Low -> Peak-Break signals for the US indices, sent to
Telegram. SEPARATE from the other scanners -- it imports your existing Telegram
config from scan_all and changes nothing there.

Tickers scanned:
    US30   -> DIA  (Dow Jones)
    US100  -> QQQ  (Nasdaq 100)
    US500  -> SPY  (S&P 500)
    GOLD   -> GLD  (SPDR Gold Shares -- spot gold bullion)
    SILVER -> SLV  (iShares Silver Trust -- physical silver)

Every alert carries a chart PNG showing the five candles that make the pattern, so
the numbers in the caption can be checked against the picture.

Timeframes 30m and up: 30m, 1h, 2h, 4h, 1D, 1W -- except the metals, which are
scanned on 4h and above only (see ASSET_FRAMES).

    python index_signals.py            # scan once, alert on the latest CLOSED candle
    python index_signals.py --print    # also print matches to the console

Schedule it (Task Scheduler / cron / your cloud runner) as often as you like.
Each (asset, timeframe, candle) is alerted only once -- state kept in
.index_signals_seen.txt next to this file.

A candle is judged as soon as it stops trading (see _drop_unclosed), so a daily
signal goes out the same evening rather than waiting for the next session's row
to show up in the data.
"""

import argparse
import os

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import bb_chart                   # shared per-signal chart styling

# --- Telegram: the @us3indexbot bot (its OWN token, separate from scan_all) ---
# Get the token from @BotFather for @us3indexbot, and the chat id you want alerts
# in. Best practice: set them as environment variables so the token isn't in code.
BOT_TOKEN = os.environ.get("US_INDEX_BOT_TOKEN") or "YOUR_US_INDEX_BOT_TOKEN"
CHAT_IDS  = ["7788611624", "6173185769"]   # same recipients as scan_indices
if os.environ.get("US_INDEX_CHAT_ID"):
    CHAT_IDS = [c.strip() for c in os.environ["US_INDEX_CHAT_ID"].split(",") if c.strip()]


def send_telegram_alert(message):
    if "YOUR_" in BOT_TOKEN or not CHAT_IDS:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    for chat_id in CHAT_IDS:
        try:
            requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=15)
        except Exception:
            pass


def _safe_print(text):
    """Print without ever raising. A console that isn't UTF-8 (Windows cp1252) chokes
    on the 🟢 in the alert, and that exception used to land in the scan loop's
    handler -- losing the whole signal, chart and Telegram message included."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


def send_telegram_photo(image_path, caption):
    """Send the chart with the alert text as its caption. Returns False if any
    recipient failed, so the caller can fall back to a plain text alert."""
    if "YOUR_" in BOT_TOKEN or not CHAT_IDS:
        return True                     # nothing configured: not a failure
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    ok = True
    for chat_id in CHAT_IDS:
        try:
            with open(image_path, "rb") as f:
                r = requests.post(url, data={"chat_id": chat_id, "caption": caption},
                                  files={"photo": f}, timeout=30)
            if not r.ok:
                print(f"  Telegram photo error to {chat_id}: {r.status_code} {r.text}")
                ok = False
        except Exception as e:
            print(f"  Error sending photo to {chat_id}: {e}")
            ok = False
    return ok


ASSETS = {
    "DIA": "US30 (Dow Jones — DIA)",
    "QQQ": "US100 (Nasdaq 100 — QQQ)",
    "SPY": "US500 (S&P 500 — SPY)",
    "GLD": "GOLD (Spot Gold — GLD)",
    "SLV": "SILVER (Spot Silver — SLV)",
}

# Which frames each asset is scanned on. Absent from here = every frame below.
# The metals are wanted on 4h and ABOVE only (same floor as the scan_indices bot),
# so the sub-4h frames stay index-only.
ASSET_FRAMES = {
    "GLD": {"4h", "1D", "1W"},
    "SLV": {"4h", "1D", "1W"},
}

# 30m and every frame above it. 2h/4h are resampled from 1h candles.
FRAMES = [
    {"label": "30m", "interval": "30m", "period": "60d",  "resample": None},
    {"label": "1h",  "interval": "60m", "period": "360d", "resample": None},
    {"label": "2h",  "interval": "60m", "period": "360d", "resample": "2h"},
    {"label": "4h",  "interval": "60m", "period": "360d", "resample": "4h"},
    {"label": "1D",  "interval": "1d",  "period": "2y",   "resample": None},
    {"label": "1W",  "interval": "1wk", "period": "max",  "resample": None},
]

# Pattern thresholds (indices move less than single stocks, so a smaller drop).
MIN_DROP_PCT  = 0.02    # the sharp drop must lose at least 2%
MIN_DROP_BARS = 4
MAX_DROP_BARS = 30
MAX_NON_RED   = 0       # interior drop candles all red (first/last may differ)
PIVOT_K       = 2
SEARCH_BARS   = 120
MAX_RECOVERY  = 40

SEEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         ".index_signals_seen.txt")


def _swing_highs(high, k):
    out = []
    for i in range(k, len(high) - k):
        w = high[i - k:i + k + 1]
        if high[i] == w.max() and high[i] > w[0] and high[i] > w[-1]:
            out.append(i)
    return out


def detect(df):
    """Return a dict of levels if the LAST candle completes the pattern, else None.
    Rules: sharp all-red drop -> peak (below the drop's top) -> higher low (also
    below the drop's top) -> a GREEN candle closing above that peak = entry."""
    o = df["Open"].to_numpy(float);  c = df["Close"].to_numpy(float)
    h = df["High"].to_numpy(float);  l = df["Low"].to_numpy(float)
    n = len(c)
    if n < 15:
        return None
    last = n - 1

    # PEAK = most recent swing high the last GREEN candle freshly closes above.
    peak_bar = None
    for ip in reversed(_swing_highs(h, PIVOT_K)):
        if ip >= last or ip < last - SEARCH_BARS:
            if ip < last - SEARCH_BARS:
                break
            continue
        fresh = ip + 1 >= last or np.max(c[ip + 1:last]) <= h[ip]
        if c[last] > h[ip] and c[last] > o[last] and c[last - 1] <= h[ip] and fresh:
            peak_bar = ip
            break
    if peak_bar is None or peak_bar >= last - 1:
        return None
    peak = h[peak_bar]

    # HIGHER LOW = lowest low between the peak and the bar before the entry.
    seg = l[peak_bar + 1:last]
    low2_bar = peak_bar + 1 + int(seg.argmin())
    low2 = l[low2_bar]

    # BOTTOM = deepest low before the peak (the real spike low).
    start = max(0, peak_bar - MAX_RECOVERY)
    seg2 = l[start:peak_bar]
    if len(seg2) == 0:
        return None
    low1_bar = start + int(seg2.argmin())
    low1 = l[low1_bar]

    # DROP = the contiguous red run that ends at the bottom (ignore earlier moves).
    run_start, greens, b = low1_bar, 0, low1_bar - 1
    while b >= 0:
        if c[b] < o[b]:
            run_start, b = b, b - 1
        elif greens < MAX_NON_RED:
            greens, run_start, b = greens + 1, b, b - 1
        else:
            break
    ds_bar = max(0, run_start - 1)
    ds_top = h[ds_bar]

    # validations
    if low2 <= low1:                      # higher low
        return None
    if not (peak < ds_top and low2 < ds_top):   # shadow: stay under the drop's top
        return None
    drop_bars = low1_bar - ds_bar
    if not (MIN_DROP_BARS <= drop_bars <= MAX_DROP_BARS):
        return None
    drop_pct = (ds_top - low1) / ds_top if ds_top > 0 else 0.0
    if drop_pct < MIN_DROP_PCT:
        return None
    non_red = sum(1 for i in range(ds_bar + 1, low1_bar) if c[i] >= o[i])
    if non_red > MAX_NON_RED:
        return None

    # The *_bar positions are what the chart marks up, so the picture and the
    # numbers in the alert can never drift apart.
    return {"entry": float(peak), "low1": float(low1), "low2": float(low2),
            "drop_pct": float(drop_pct), "bar_time": df.index[last],
            "ds_bar": ds_bar, "low1_bar": low1_bar,
            "peak_bar": peak_bar, "low2_bar": low2_bar}


# Chart PNGs land next to this script, in the same charts/ folder the other bots
# use -- the workflow archives that folder to the private `charts` branch.
CHART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")


def _safe_name(text):
    return "".join(c if c.isalnum() else "_" for c in text)


def save_chart(name, df, tf_label, m, min_bars=60, lead=12):
    """Save the signal's chart: the five pattern candles labelled, and the entry
    level drawn across. The window always reaches back past the drop's top (plus
    `lead` bars of run-up) so the whole setup is visible, never just the break."""
    candle = df.index[-1].date()
    out_dir = os.path.join(CHART_DIR, f"IDXSD_{tf_label}_{candle}")
    os.makedirs(out_dir, exist_ok=True)

    need = len(df) - m["ds_bar"] + lead
    plot_df = df.tail(min(max(min_bars, need), len(df))).copy()
    roles = {
        df.index[m["ds_bar"]]:   "DROP TOP",
        df.index[m["low1_bar"]]: "BOTTOM",
        df.index[m["peak_bar"]]: "PEAK",
        df.index[m["low2_bar"]]: "HIGHER LOW",
        df.index[-1]:            "SIGNAL",
    }
    out = os.path.join(out_dir,
                       f"IDXSD_{tf_label}_{_safe_name(name)}_{candle}.png")
    return bb_chart.render(
        plot_df, [], f"{name} — Sharp Drop → Higher Low → Peak Break ({tf_label})",
        out, "IDXSD_", roles, hline=m["entry"])


def _resample(df, rule):
    return df.resample(rule).agg({"Open": "first", "High": "max",
                                  "Low": "min", "Close": "last"}).dropna()


# ---------------------------------------------------------------------------
# IS THE NEWEST CANDLE FINISHED?
# ---------------------------------------------------------------------------
# yfinance hands back the still-forming candle as the last row, so this bot used
# to drop that row unconditionally. That also threw away a candle that HAD just
# finished: after Friday's close the last daily row IS Friday, so a signal there
# only surfaced once Monday's row appeared -- a full trading day late, with price
# already away from the entry. So work out when each candle actually stops
# trading and keep it once that moment has passed.
MARKET_TZ   = "America/New_York"
SESSION_END = (16, 0)       # US cash close -- every ticker here is a US ETF
SETTLE_MIN  = 15            # let Yahoo finish writing the closing print

INTRADAY_SPAN = {"30m": "30min", "1h": "1h", "2h": "2h", "4h": "4h"}


def _bar_end(ts, label):
    """The moment the candle starting at `ts` stops trading (market time)."""
    if label in ("1D", "1W"):
        # Daily/weekly rows are tz-naive dates, and a weekly row is labelled with
        # its MONDAY -- so that week finishes on the Friday. In a holiday-shortened
        # week this waits until the Friday close: late by a day, never early.
        day = pd.Timestamp(ts).normalize()
        if label == "1W":
            day += pd.Timedelta(days=4)
        end = day + pd.Timedelta(hours=SESSION_END[0], minutes=SESSION_END[1])
        return end.tz_localize(MARKET_TZ)

    end = ts + pd.Timedelta(INTRADAY_SPAN[label])
    # The last candle of a session is cut short by the close: a 1h candle opening
    # at 15:30 is done at 16:00, not 16:30.
    close = ts.normalize() + pd.Timedelta(hours=SESSION_END[0], minutes=SESSION_END[1])
    return min(end, close)


def _drop_unclosed(df, label, now=None):
    """Trim the newest row only if it is still forming."""
    if df.empty:
        return df
    ts = df.index[-1]
    if label not in ("1D", "1W") and ts.tzinfo is None:
        ts = ts.tz_localize(MARKET_TZ)      # defensive: intraday should be aware
    now = now if now is not None else pd.Timestamp.now(tz=MARKET_TZ)
    if now < _bar_end(ts, label) + pd.Timedelta(minutes=SETTLE_MIN):
        return df.iloc[:-1]
    return df


def _load_seen():
    if not os.path.exists(SEEN_FILE):
        return set()
    with open(SEEN_FILE) as f:
        return set(line.strip() for line in f if line.strip())


def _mark_seen(key):
    with open(SEEN_FILE, "a") as f:
        f.write(key + "\n")


def scan(do_print=False):
    seen = _load_seen()
    for ticker, name in ASSETS.items():
        allowed = ASSET_FRAMES.get(ticker)
        for fr in FRAMES:
            if allowed is not None and fr["label"] not in allowed:
                continue
            try:
                raw = yf.download(ticker, interval=fr["interval"], period=fr["period"],
                                  auto_adjust=False, progress=False)
                if raw is None or raw.empty:
                    continue
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                df = _resample(raw, fr["resample"]) if fr["resample"] else raw
                df = _drop_unclosed(df, fr["label"])   # latest CLOSED candle only
                if len(df) < 15:
                    continue
                m = detect(df)
                if not m:
                    continue
                key = f"{ticker}|{fr['label']}|{m['bar_time']}"
                if key in seen:
                    continue
                # "Drop size" alone reads as "the market is down this much now".
                # It isn't -- it's the earlier fall that set the pattern up, and
                # the signal itself is a break UPWARDS. Say so.
                msg = (
                    "🟢 BUY SIGNAL — Sharp Drop → Higher Low → Peak Break\n"
                    f"{name}\n"
                    f"Timeframe: {fr['label']}\n"
                    f"Entry (peak broken): {m['entry']:.2f}\n"
                    f"Setup drop, already recovered: −{m['drop_pct']*100:.1f}% "
                    f"({m['low1']:.2f} was the bottom)\n"
                    f"Higher low since: {m['low2']:.2f}\n"
                    f"Candle closed: {m['bar_time']}"
                )
                if do_print:
                    _safe_print(msg + "\n")
                chart_path = None
                try:
                    chart_path = save_chart(name, df, fr["label"], m)
                except Exception as e:
                    print(f"  {ticker} {fr['label']}: could not draw chart: {e}")
                # Text fallback if the picture didn't go out. A recipient who did
                # get the photo may see the text twice -- better than a lost signal.
                if not chart_path or not send_telegram_photo(chart_path, msg):
                    send_telegram_alert(msg)
                _mark_seen(key)
                seen.add(key)
            except Exception as e:
                if do_print:
                    print(f"  {ticker} {fr['label']}: {e}")


def main():
    ap = argparse.ArgumentParser(description="US index pattern signals -> Telegram")
    ap.add_argument("--print", action="store_true", dest="show",
                    help="also print matches to the console")
    args = ap.parse_args()
    scan(do_print=args.show)


if __name__ == "__main__":
    main()
