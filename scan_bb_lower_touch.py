"""
Bot #8 -- Bollinger-Band LOWER-TOUCH Liquidity-Grab scanner (LONG only).

The SAME liquidity-grab ("swallow") engine as scan_bb_grab, but with a DIFFERENT
band gate on the trigger candle:

    STAGE 1 -- liquidity grab (the EXACT same `_swallow` engine): a green trigger
        candle whose body covers the prior green candle body(s) and whose low
        sweeps below their low(s). Fires on the 2-candle or 3-candle layout.

    STAGE 2 -- LOWER-BAND TOUCH on that same trigger candle: the candle's range
        TOUCHES the LOWER Bollinger band, i.e. the bottom line passes THROUGH the
        candle:  low <= lower  AND  high >= lower. (The upper band and basis are
        NOT used as a gate here -- only the bottom line matters.)

ENTRY = a grab whose trigger candle touches the lower Bollinger band, on the
just-closed candle. This is the Python twin of bb_lower_touch_grab.pine.

By DEFAULT this sends into the SAME Telegram bot as scan_bb_grab (BB_GRAB_BOT_TOKEN)
-- "the Bollinger band bot" -- so its alerts land in the same chat. Set
BB_LOWER_BOT_TOKEN (and optionally BB_LOWER_CHAT_ID) to route it to its own bot
instead. Logged signals use the BBLOWER_ id-prefix so they never collide with the
breakout bot's BBGRAB_ rows, even on the same ticker/timeframe/candle.

It reuses scan_bb_grab's scaffolding wholesale (frame catalog, universe builders,
freshness gate, Bollinger helper) so none of the existing bots are touched.

Run one slice locally (PowerShell):
    $env:BB_SEGMENT="commodities"; python scan_bb_lower_touch.py
"""

import os

import mplfinance as mpf
import pandas as pd

import scan_all as s
import scan_crypto as sc
import scan_bb_grab as bbg   # reuse frames / universe / freshness / _bollinger

# --- This bot's Telegram creds. Default = the existing Bollinger bot's token, so
#     lower-touch alerts land in the SAME chat as the breakout bot. A dedicated
#     BB_LOWER_BOT_TOKEN overrides it if you ever want to split them out. ---
s.BOT_TOKEN = (os.environ.get("BB_LOWER_BOT_TOKEN")
               or os.environ.get("BB_GRAB_BOT_TOKEN")
               or "YOUR_BB_GRAB_BOT_TOKEN")
s.CHAT_IDS = [
    "7788611624",   # A
    "6173185769",   # m
]
_chat = os.environ.get("BB_LOWER_CHAT_ID") or os.environ.get("BB_GRAB_CHAT_ID")
if _chat:
    s.CHAT_IDS = [c.strip() for c in _chat.split(",") if c.strip()]


# ---------------------------------------------------------------------------
# DETECTOR -- swallow grab + the trigger candle touching the LOWER band.
# ---------------------------------------------------------------------------
def bb_lower_touch(df):
    """(matched, info) for the LONG lower-band-touch liquidity grab, evaluated on
    the LAST candle of `df` (the runner has already trimmed to the just-closed one).

    matched -> the last candle is a bullish `_swallow` grab AND the lower Bollinger
               band passes through it (low <= lower <= high).
    info    -> None if no match, else the band values on the trigger candle for the
               chart: {"basis", "upper", "lower"}.
    """
    if df is None or len(df) < bbg.MIN_BARS:
        return (False, None)

    # Same grab engine as every other bot -- must fire on the last candle.
    if not s._swallow(df, bullish=True):
        return (False, None)

    basis, upper, lower = bbg._bollinger(df["Close"])
    h = df["High"].iloc[-1]
    l = df["Low"].iloc[-1]
    lo = lower.iloc[-1]
    if pd.isna(h) or pd.isna(l) or pd.isna(lo):
        return (False, None)

    # The grab candle must TOUCH the lower band: bottom line inside its range.
    if l <= lo and h >= lo:
        info = {
            "basis": float(basis.iloc[-1]),
            "upper": float(upper.iloc[-1]),
            "lower": float(lo),
        }
        return (True, info)
    return (False, None)


# ---------------------------------------------------------------------------
# CHART -- candles + the three Bollinger lines, grab candle(s) highlighted.
# ---------------------------------------------------------------------------
def save_lower_chart(ticker, kind, df, tf_label, info, bars=80):
    """Candlestick PNG overlaid with the Bollinger Bands and a highlight over the
    grab candle(s) that touched the lower band."""
    label = s.COMMODITIES.get(ticker, ticker)
    candle = df.index[-1].date()
    out_dir = os.path.join(s.CHART_DIR, f"BBLOWER_{tf_label}_{candle}")
    os.makedirs(out_dir, exist_ok=True)

    bars = min(bars, len(df))
    plot_df = df.tail(bars).copy()
    basis, upper, lower = bbg._bollinger(plot_df["Close"])

    aps = [
        mpf.make_addplot(upper, color="crimson", width=1.1),
        mpf.make_addplot(basis, color="darkorange", width=1.0, linestyle="--"),
        mpf.make_addplot(lower, color="royalblue", width=1.1),
    ]

    out = os.path.join(out_dir, f"BBLOWER_{tf_label}_{s._safe_name(label)}_{candle}.png")
    mpf.plot(
        plot_df,
        type="candle",
        style="charles",
        addplot=aps,
        title=f"\n{label} — Lower-Band Touch → Liquidity Grab ({tf_label})",
        ylabel="Price",
        vlines=dict(vlines=[df.index[-2], df.index[-1]], colors="teal",
                    alpha=0.30, linewidths=9),
        savefig=dict(fname=out, dpi=120, bbox_inches="tight"),
    )
    return out


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------
def _alert(kind, ticker, tf_label, d, info, bot):
    """Log + send one match (chart if possible, else text)."""
    yahoo, tv = s.chart_links(kind, ticker)
    name = s.COMMODITIES.get(ticker, ticker)
    msg = (f"[{tf_label}] MATCH: {name} ({kind}) — "
           f"lower-band touch → liquidity grab (LONG) "
           f"[touched the lower Bollinger band]\n"
           f"Yahoo: {yahoo}\nTradingView: {tv}")
    print("  " + msg.splitlines()[0])
    chart_path = None
    try:
        chart_path = save_lower_chart(ticker, kind, d, tf_label, info)
    except Exception as e:
        print(f"    (could not draw chart: {e})")
    # BBLOWER_ namespace so these rows never collide with the breakout bot's BBGRAB_.
    s.log_signal(kind, ticker, tf_label, d, bot=bot, direction="long",
                 id_prefix="BBLOWER_", chart=s.chart_rel_path(chart_path))
    if chart_path:
        try:
            s.send_telegram_photo(chart_path, msg)
        except Exception as e:
            print(f"    (could not send photo: {e})")
            s.send_telegram_alert(msg)
    else:
        s.send_telegram_alert(msg)


def run(segment, catalog=bbg.FRAMES_CATALOG, bot="bb_lower_touch"):
    """Scan one universe segment across the selected frames. Returns match count.
    Mirrors scan_bb_grab.run but swaps in the lower-touch detector + alert."""
    kind, tickers, is_crypto = bbg.build_universe(segment)
    print(f"=== Segment: {segment} ({kind}) — {len(tickers)} tickers ===")

    frames = [tf for tf in bbg.selected_frames(catalog) if bbg.closes_today(tf["label"])]
    if not frames:
        print("No timeframe closes right now -- nothing to scan.")
        return 0
    print("Scanning timeframes: " + ", ".join(tf["label"] for tf in frames))

    needed_bases = {tf["base"] for tf in frames}
    bases = {}
    for b in needed_bases:
        print(f"\nDownloading {b} candles...")
        bases[b] = sc.download_base(tickers, b, bbg.BASE_PERIODS[b])
        print(f"  got data for {len(bases[b])}/{len(tickers)} symbols")
        if is_crypto:
            bases[b] = s.dedupe_crypto(bases[b])

    total = 0
    for tf in frames:
        base_data = bases[tf["base"]]
        matches, stale = [], 0
        for t, df in base_data.items():
            try:
                d = sc.resample_ohlc(df, tf["rkw"]) if tf["rkw"] else df
                d = d.iloc[:-1]                      # drop the still-forming candle
                if len(d) < bbg.MIN_BARS:
                    continue
                if not bbg.is_fresh(tf["label"], d.index[-1]):
                    stale += 1
                    continue
                if is_crypto and s.is_flat(d):
                    continue                         # pegged stablecoin -- noise
                matched, info = bb_lower_touch(d)
                if matched:
                    matches.append((t, d, info))
            except Exception:
                continue
        print(f"\n[{tf['label']}] {len(matches)} match(es) from {len(base_data)} symbols"
              + (f" ({stale} skipped: candle not fresh)" if stale else ""))
        for t, d, info in matches:
            total += 1
            _alert(kind, t, tf["label"], d, info, bot)

    print("\n" + "=" * 40)
    print(f"{segment}: {total} total match(es) across all frames")
    print("=" * 40)
    return total


def main():
    print("Telegram (Bollinger lower-touch bot): " + ("configured" if s._telegram_ready()
          else "NOT configured -- alerts will be skipped"))
    segment = (os.environ.get("BB_SEGMENT") or "commodities").strip()
    run(segment)


if __name__ == "__main__":
    main()
