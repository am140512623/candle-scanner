"""
Sharp-Drop -> Higher-Low -> Peak-Break signals for the US indices, sent to
Telegram. SEPARATE from the other scanners -- it imports your existing Telegram
config from scan_all and changes nothing there.

Real, buyable ETF SHARES only (NOT CFDs / futures / contracts):
    US30   -> DIA  (Dow Jones)
    US100  -> QQQ  (Nasdaq 100)
    US500  -> SPY  (S&P 500)

Timeframes 30m and up: 30m, 1h, 2h, 4h, 1D, 1W.

    python index_signals.py            # scan once, alert on the latest CLOSED candle
    python index_signals.py --print    # also print matches to the console

Schedule it (Task Scheduler / cron / your cloud runner) as often as you like.
Each (asset, timeframe, candle) is alerted only once -- state kept in
.index_signals_seen.txt next to this file.
"""

import argparse
import os

import numpy as np
import pandas as pd
import requests
import yfinance as yf

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

# Real assets you can BUY as shares (no CFDs, no leverage, no bidding).
ASSETS = {
    "DIA": "US30 (Dow Jones — DIA)",
    "QQQ": "US100 (Nasdaq 100 — QQQ)",
    "SPY": "US500 (S&P 500 — SPY)",
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

    return {"entry": float(peak), "low1": float(low1), "low2": float(low2),
            "drop_pct": float(drop_pct), "bar_time": df.index[last]}


def _resample(df, rule):
    return df.resample(rule).agg({"Open": "first", "High": "max",
                                  "Low": "min", "Close": "last"}).dropna()


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
        for fr in FRAMES:
            try:
                raw = yf.download(ticker, interval=fr["interval"], period=fr["period"],
                                  auto_adjust=False, progress=False)
                if raw is None or raw.empty:
                    continue
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                df = _resample(raw, fr["resample"]) if fr["resample"] else raw
                df = df.iloc[:-1]              # judge the latest CLOSED candle only
                if len(df) < 15:
                    continue
                m = detect(df)
                if not m:
                    continue
                key = f"{ticker}|{fr['label']}|{m['bar_time']}"
                if key in seen:
                    continue
                msg = (
                    "🟢 BUY SIGNAL — Sharp Drop → Higher Low → Peak Break\n"
                    f"{name}\n"
                    f"Timeframe: {fr['label']}\n"
                    f"Entry (peak broken): {m['entry']:.2f}\n"
                    f"Drop size: {m['drop_pct']*100:.1f}%\n"
                    f"Candle closed: {m['bar_time']}\n"
                    "Real asset — buyable ETF share, NOT a CFD."
                )
                if do_print:
                    print(msg + "\n")
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
