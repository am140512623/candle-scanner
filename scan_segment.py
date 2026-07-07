"""
Segmented Liquidity-Grab scanners -- ONE engine, many bots.

Each bot runs THIS script and just picks a different slice of the market with the
SEGMENT environment variable. All of these scan the WEEKLY (and MONTHLY) candle:

    crypto_top300_weekly  crypto rank 1 - 300 on the weekly frame -> its own bot
    crypto_top1000   crypto market-cap rank 300 - 1000   -> your crypto bot
    stock_mega       US stocks  > $200B                  -> your stock bot
    stock_large      US stocks  $10B - $200B             -> your stock bot
    stock_mid        US stocks  $2B - $10B               -> your stock bot
    stock_small      US stocks  $250M - $2B              -> your stock bot

(Crypto rank 1-300 is handled separately by bot #2, scan_crypto.py -- which now
covers 6h..4d intraday PLUS weekly and monthly for those top 300 coins.)

Each segment has its OWN Telegram bot. The workflow passes that bot's token in as
SEG_TOKEN (from a GitHub secret); alerts go to the recipients in DEFAULT_CHATS
(override per run with SEG_CHAT). Every recipient must tap Start on a bot once
before Telegram will let it message them.

Run one locally (PowerShell):
    $env:SEGMENT="stock_mid"; python scan_segment.py
"""

import os

import scan_all as s

B = 1_000_000_000   # billion
M = 1_000_000       # million

# Who receives the alerts (Telegram numeric user IDs). Same people as the other
# bots; each must press Start on every bot once. Not secret, so kept in code.
DEFAULT_CHATS = ["7788611624", "6173185769"]

# Each segment = which slice of the market + a human label.
#   crypto -> "rank": (low, high) market-cap rank, inclusive low / exclusive high
#   stock  -> "cap":  (low, high) USD market cap,  inclusive low / exclusive high (None = no bound)
SEGMENTS = {
    "crypto_top300_weekly": {"asset": "crypto", "rank": (1, 300),   "label": "Crypto Top 300 (Weekly)"},
    "crypto_top1000": {"asset": "crypto", "rank": (300, 1000),      "label": "Crypto rank 300-1000"},
    "stock_mega":     {"asset": "stock",  "cap": (200 * B, None),   "label": "Mega-Cap Stocks (>$200B)"},
    "stock_large":    {"asset": "stock",  "cap": (10 * B, 200 * B), "label": "Large-Cap Stocks ($10B-$200B)"},
    "stock_mid":      {"asset": "stock",  "cap": (2 * B, 10 * B),   "label": "Mid-Cap Stocks ($2B-$10B)"},
    "stock_small":    {"asset": "stock",  "cap": (250 * M, 2 * B),  "label": "Small-Cap Stocks ($250M-$2B)"},
}


def configure_telegram():
    """Point scan_all's Telegram sender at THIS segment's bot.

    SEG_TOKEN (the segment's own bot token, injected from a GitHub secret) is
    required in the cloud; if it's missing locally we leave scan_all's default
    token in place so a manual test still runs.
    """
    if os.environ.get("SEG_TOKEN"):
        s.BOT_TOKEN = os.environ["SEG_TOKEN"]
    chat = os.environ.get("SEG_CHAT")
    s.CHAT_IDS = [x.strip() for x in chat.split(",") if x.strip()] if chat else list(DEFAULT_CHATS)


def timeframes_for(asset):
    """Weekly + monthly timeframes, with the right 'just-closed candle' handling.

    Yahoo labels each weekly bar by its Monday. Crypto trades the weekend, so its
    week only rolls over Monday 00:00 UTC -- we scan Monday and DROP the forming
    new week (closed_only=True). Stocks finish trading Friday with no weekend bar,
    so we scan Saturday and KEEP that bar (closed_only=False); it already holds the
    full Mon-Fri week. Monthly always drops the forming current month.
    """
    weekly_closed = (asset == "crypto")
    return [
        {"interval": "1wk", "period": "2y",  "label": "WEEKLY",  "monthly_only": False, "closed_only": weekly_closed},
        {"interval": "1mo", "period": "max", "label": "MONTHLY", "monthly_only": True,  "closed_only": True},
    ]


def build_crypto_universe(rank_low, rank_high):
    """Crypto tickers whose market-cap rank is in [rank_low, rank_high)."""
    ranked = s.get_top_crypto(rank_high)        # cleaned, in rank order
    return ranked[rank_low - 1:rank_high]


def build_stock_universe(cap_low, cap_high):
    """US stock tickers whose market cap is in [cap_low, cap_high)."""
    low = cap_low or 0
    high = cap_high if cap_high is not None else float("inf")
    rows = s.get_us_stocks_with_caps()
    return [sym for sym, cap in rows if low <= cap < high]


def scan_segment(universe, label, kind, asset, bot):
    """Scan one slice on weekly + monthly, sending each match to Telegram.

    SCAN_MODE (set by the workflow's cron) picks the timeframe:
        weekly  -> only the weekly candle
        monthly -> only the monthly candle
        unset   -> both (handy for a manual run)
    """
    mode = (os.environ.get("SCAN_MODE") or "").lower()
    groups = [(kind, universe, label)]
    total = 0
    for tf in timeframes_for(asset):
        is_monthly = tf["monthly_only"]
        if mode == "weekly" and is_monthly:
            continue
        if mode == "monthly" and not is_monthly:
            continue
        for k, t, df, direction in s.run_timeframe(tf, groups):
            total += 1
            yahoo, tv = s.chart_links(k, t)
            msg = (f"[{tf['label']}] MATCH: {t} ({k}) formed your {s.pattern_name(direction)} pattern!\n"
                   f"Yahoo: {yahoo}\nTradingView: {tv}")
            print("  " + msg.splitlines()[0])
            s.log_signal(k, t, tf["label"], df, bot=bot, direction=direction)
            try:
                chart_path = s.save_chart(t, k, df, tf["label"], direction=direction)
                s.send_telegram_photo(chart_path, msg)
            except Exception as e:
                print(f"    (could not draw chart: {e})")
                s.send_telegram_alert(msg)
    return total


def main():
    key = (os.environ.get("SEGMENT") or "").strip()
    if key not in SEGMENTS:
        raise SystemExit(f"Set SEGMENT to one of: {', '.join(SEGMENTS)}  (got {key!r})")
    seg = SEGMENTS[key]
    configure_telegram()
    kind = "CRYPTO" if seg["asset"] == "crypto" else "STOCK"
    print(f"=== Segment: {seg['label']} ({key}) ===")

    if seg["asset"] == "crypto":
        low, high = seg["rank"]
        universe = build_crypto_universe(low, high)
        print(f"  {len(universe)} coins in rank {low}-{high}")
    else:
        low, high = seg["cap"]
        universe = build_stock_universe(low, high)
        print(f"  {len(universe)} stocks in this cap tier")

    total = scan_segment(universe, seg["label"], kind, seg["asset"], bot=key)
    print("\n" + "=" * 40)
    print(f"{seg['label']}: {total} total match(es)")
    print("=" * 40)


if __name__ == "__main__":
    main()
