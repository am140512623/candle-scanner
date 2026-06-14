"""
Bot #2 -- multi-timeframe crypto scanner.

Scans the top 200 crypto for the same Liquidity Grab pattern on EVERY timeframe
that is bigger than 4 hours and smaller than 1 week:

    6h, 8h, 12h, 1-day, 2-day, 3-day

Yahoo only serves 1h and 1d candles natively, so the 6h/8h/12h frames are built
from hourly data and the 2d/3d frames from daily data (crypto trades 24/7, so
these merge cleanly). Alerts go to a SEPARATE Telegram bot.
"""

import os

import pandas as pd

import scan_all as s

# --- Separate bot for these crypto alerts ---
s.BOT_TOKEN = os.environ.get("CRYPTO_BOT_TOKEN") or "8980241214:AAGCgROHDxMAjakuuU5c7mrzEESsy9PxRD4"
s.CHAT_IDS = [
    "7788611624",   # A
    "6173185769",   # m
]
if os.environ.get("CRYPTO_CHAT_ID"):
    s.CHAT_IDS = [c.strip() for c in os.environ["CRYPTO_CHAT_ID"].split(",") if c.strip()]

CRYPTO_TOP_N = 200

# Each timeframe: which base candles to pull, and how to merge them.
# rule=None means use the base candles as-is.
TIMEFRAMES = [
    {"label": "6H",  "base": "1h", "rule": "6h"},
    {"label": "8H",  "base": "1h", "rule": "8h"},
    {"label": "12H", "base": "1h", "rule": "12h"},
    {"label": "1D",  "base": "1d", "rule": None},
    {"label": "2D",  "base": "1d", "rule": "2D"},
    {"label": "3D",  "base": "1d", "rule": "3D"},
]

# How much history to pull for each base candle size.
BASE_PERIODS = {"1h": "60d", "1d": "1y"}


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
    agg = df.resample(rule).agg({"Open": "first", "High": "max",
                                 "Low": "min", "Close": "last"})
    return agg.dropna()


def main():
    print("Building crypto universe...")
    crypto = s.get_top_crypto(CRYPTO_TOP_N)
    print(f"  Crypto: {len(crypto)} from top {CRYPTO_TOP_N} (stablecoins dropped)")

    # Pull each base candle size once, then reuse for every timeframe.
    needed_bases = {tf["base"] for tf in TIMEFRAMES}
    bases = {}
    for b in needed_bases:
        print(f"\nDownloading {b} candles...")
        bases[b] = download_base(crypto, b, BASE_PERIODS[b])
        print(f"  got data for {len(bases[b])}/{len(crypto)} coins")

    total = 0
    for tf in TIMEFRAMES:
        base_data = bases[tf["base"]]
        matches = []
        for t, df in base_data.items():
            try:
                d = resample_ohlc(df, tf["rule"]) if tf["rule"] else df
                d = d.iloc[:-1]          # drop the still-forming candle
                if len(d) < 2:
                    continue
                if s.check_pattern(d):
                    matches.append((t, d))
            except Exception:
                continue
        print(f"\n[{tf['label']}] {len(matches)} match(es) from {len(base_data)} coins")
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


if __name__ == "__main__":
    main()
