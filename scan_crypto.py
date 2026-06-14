"""
Bot #2 -- DAILY crypto-only scanner.

Scans the top 200 crypto on the DAILY timeframe (each candle = 1 day, "bigger
than 4h, less than a week") for the same Liquidity Grab pattern, and sends to a
SEPARATE Telegram bot so these alerts stay apart from the weekly/monthly ones.

Reuses all the logic from scan_all.py -- only the bot, the asset set, and the
timeframe are different.
"""

import os
import datetime

import scan_all as s

# --- Separate bot for crypto-daily alerts ---
# Fill CRYPTO_BOT_TOKEN with the token from your NEW bot (see chat).
s.BOT_TOKEN = os.environ.get("CRYPTO_BOT_TOKEN") or "8980241214:AAGCgROHDxMAjakuuU5c7mrzEESsy9PxRD4"
# Recipients for this bot (each must tap Start on the NEW bot once).
s.CHAT_IDS = [
    "7788611624",   # A
    "6173185769",   # m
]
if os.environ.get("CRYPTO_CHAT_ID"):
    s.CHAT_IDS = [c.strip() for c in os.environ["CRYPTO_CHAT_ID"].split(",") if c.strip()]

CRYPTO_TOP_N = 200
INTERVAL = "1d"     # daily candle
PERIOD = "6mo"      # plenty of daily history for the pattern + chart


def main():
    print("Building crypto universe...")
    crypto = s.get_top_crypto(CRYPTO_TOP_N)
    print(f"  Crypto: {len(crypto)} from top {CRYPTO_TOP_N} (stablecoins dropped)")

    print(f"\nScanning {INTERVAL} (daily) candles...\n")
    matches = s.scan(crypto, "Crypto", INTERVAL, PERIOD, closed_only=True)

    print("\n" + "=" * 40)
    if matches:
        print(f"MATCHES FOUND: {len(matches)}")
        for t, df in matches:
            yahoo, tv = s.chart_links("CRYPTO", t)
            msg = (f"[DAILY] MATCH: {t} (CRYPTO) formed your Liquidity Grab pattern!\n"
                   f"Yahoo: {yahoo}\nTradingView: {tv}")
            print("  " + msg)
            try:
                chart_path = s.save_chart(t, "CRYPTO", df, "DAILY")
                print(f"    chart -> {chart_path}")
                s.send_telegram_photo(chart_path, msg)
            except Exception as e:
                print(f"    (could not draw chart: {e})")
                s.send_telegram_alert(msg)
    else:
        print("No matches this scan.")
    print("=" * 40)


if __name__ == "__main__":
    main()
