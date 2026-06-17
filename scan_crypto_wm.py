"""
Bot #3 -- weekly + monthly crypto scanner (@topcrypto300wmbot).

Scans the SAME top 300 crypto as the intraday bot (scan_crypto.py), but ONLY on
the two high timeframes:

    1W  -- Monday-anchored weeks (Mon-Sun), closing Monday 00:00 UTC
    1M  -- calendar months, closing on the 1st at 00:00 UTC

It reuses scan_crypto's whole engine (download, resample, freshness gate, charts)
and just points it at the WM_FRAMES group and its own Telegram bot. Nothing here
overlaps with the intraday bot -- separate frames, separate token.

Token comes from the CRYPTO_WM_BOT_TOKEN secret (no hardcoded fallback).
"""

import os

import scan_all as s
import scan_crypto as sc

# --- This bot's own Telegram credentials ---
s.BOT_TOKEN = os.environ.get("CRYPTO_WM_BOT_TOKEN") or "YOUR_CRYPTO_WM_BOT_TOKEN"
# Same recipients as the other crypto bots; each must tap Start on THIS bot once.
s.CHAT_IDS = [
    "7788611624",   # A
    "6173185769",   # m
]
if os.environ.get("CRYPTO_CHAT_ID"):
    s.CHAT_IDS = [c.strip() for c in os.environ["CRYPTO_CHAT_ID"].split(",") if c.strip()]


def main():
    print("Telegram (weekly/monthly bot): " + ("configured" if s._telegram_ready()
                                                else "NOT configured -- alerts will be skipped"))
    print("Building crypto universe...")
    crypto = s.get_top_crypto(sc.CRYPTO_TOP_N)
    print(f"  Crypto: {len(crypto)} from top {sc.CRYPTO_TOP_N} (stablecoins dropped)")
    # Scan only the weekly/monthly frames. CRYPTO_FRAMES (e.g. '1W') can still
    # narrow it further; empty = both. 1W self-skips except Mondays, 1M except 1st.
    sc.run(crypto, sc.WM_FRAMES)


if __name__ == "__main__":
    main()
