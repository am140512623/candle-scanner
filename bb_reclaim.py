"""
Shared stage: GRAB -> OPPOSITE CANDLE -> RECLAIM  (LONG only).

An EXTRA signal layered on top of BOTH Bollinger bots. It does not replace or
touch their existing signal -- each bot keeps firing its own grab alert exactly
as before, and additionally fires this one when the layout below completes.

The layout (uptrend case):

    1) GRAB -- the usual liquidity grab ("swallow"), on candle g, passing that
       bot's OWN Bollinger gate on that same candle:
         - scan_bb_grab      -> g opened below the upper band and closed above it
         - scan_bb_lower_touch -> the lower band passes through g's range
       So the grab half is unchanged; only what happens AFTER it is new.

    2) OPPOSITE CANDLE -- the candle IMMEDIATELY after the grab (g+1) is the
       other colour: a RED candle (close < open). Its OPEN becomes the level.

    3) RECLAIM -- the FIRST candle after that red one to CLOSE above the red
       candle's OPEN is the signal candle (the one the arrow points at). Only
       the first one counts: once price has closed back above the level, the
       setup is spent and later closes above it are not re-signalled.

    The reclaim must land within MAX_WAIT candles of the red candle, otherwise
    the setup goes stale and is dropped.

Unlike the plain grab bots, the signal candle here is NOT the grab candle -- it
comes some bars later, so the detector looks back from the just-closed candle
for a completed layout ending on it.
"""

import os

import pandas as pd

import scan_all as s
import scan_bb_grab as bbg   # reuse MIN_BARS / _bollinger / chart scaffolding
import bb_chart              # per-signal colours, badge, candle role labels


# How many candles after the opposite (red) candle the reclaim may still arrive.
MAX_WAIT = int(os.environ.get("BB_RECLAIM_MAX_WAIT", "10"))


# ---------------------------------------------------------------------------
# BAND GATES -- each bot's existing Bollinger rule, evaluated on ANY candle `i`
# instead of only the last one (the grab now sits back in the series).
# ---------------------------------------------------------------------------
def breakout_gate(df, i, basis, upper, lower):
    """scan_bb_grab's rule: candle i opens below the upper band, closes above it."""
    o, c, up = df["Open"].iloc[i], df["Close"].iloc[i], upper.iloc[i]
    if pd.isna(o) or pd.isna(c) or pd.isna(up):
        return None
    if o < up and c > up:
        return {"basis": float(basis.iloc[i]), "upper": float(up),
                "lower": float(lower.iloc[i])}
    return None


def lower_touch_gate(df, i, basis, upper, lower):
    """scan_bb_lower_touch's rule: the lower band passes through candle i's range."""
    h, l, lo = df["High"].iloc[i], df["Low"].iloc[i], lower.iloc[i]
    if pd.isna(h) or pd.isna(l) or pd.isna(lo):
        return None
    if l <= lo and h >= lo:
        return {"basis": float(basis.iloc[i]), "upper": float(upper.iloc[i]),
                "lower": float(lo)}
    return None


# ---------------------------------------------------------------------------
# DETECTOR
# ---------------------------------------------------------------------------
def reclaim_signal(df, band_gate, max_wait=MAX_WAIT):
    """(matched, info) for the grab -> opposite candle -> reclaim layout, where the
    RECLAIM lands on the LAST candle of `df` (the runner has already trimmed to the
    just-closed one).

    `band_gate` is breakout_gate or lower_touch_gate -- the calling bot's own
    Bollinger rule, applied to the grab candle.

    info -> None if no match, else the band values on the grab candle plus
            {"grab_idx", "opp_idx", "opp_open", "waited"} for the chart/message.
    """
    if df is None or len(df) < bbg.MIN_BARS + 2:
        return (False, None)

    opens, closes = df["Open"], df["Close"]
    last = len(df) - 1
    c_last = closes.iloc[-1]
    if pd.isna(c_last):
        return (False, None)

    basis, upper, lower = bbg._bollinger(closes)

    # Walk candidate positions for the opposite candle, newest first: the freshest
    # completed structure wins. r is the red candle, r-1 the grab it followed.
    earliest = max(last - max_wait, 1)
    for r in range(last - 1, earliest - 1, -1):
        o_r, c_r = opens.iloc[r], closes.iloc[r]
        if pd.isna(o_r) or pd.isna(c_r):
            continue
        if not c_r < o_r:
            continue                      # candle after the grab isn't opposite
        if not c_last > o_r:
            continue                      # the last candle didn't reclaim the level
        if bool((closes.iloc[r + 1:last] > o_r).any()):
            continue                      # something already closed above it -- spent

        g = r - 1
        if g < bbg.MIN_BARS - 1:
            continue                      # not enough history to warm the band up
        # Same grab engine as every other bot, evaluated as of candle g.
        if not s._swallow(df.iloc[:g + 1], bullish=True):
            continue
        gate_info = band_gate(df, g, basis, upper, lower)
        if gate_info is None:
            continue                      # grab didn't pass this bot's band rule

        info = dict(gate_info)
        info.update({"grab_idx": g, "opp_idx": r, "opp_open": float(o_r),
                     "waited": last - r})
        return (True, info)
    return (False, None)


# ---------------------------------------------------------------------------
# CHART -- candles + bands, with the grab / opposite / signal candles marked and
# the reclaimed level drawn across.
# ---------------------------------------------------------------------------
def save_reclaim_chart(ticker, df, tf_label, info, prefix, title, bars=80):
    """Candlestick PNG for one reclaim match. `prefix` namespaces the output dir the
    same way each bot's own charts do (BBGRABRC_ / BBLOWERRC_) and picks the colour
    scheme + badge that identify this signal -- see bb_chart."""
    label = s.COMMODITIES.get(ticker, ticker)
    candle = df.index[-1].date()
    out_dir = os.path.join(s.CHART_DIR, f"{prefix}{tf_label}_{candle}")
    os.makedirs(out_dir, exist_ok=True)

    bars = min(bars, len(df))
    plot_df = df.tail(bars).copy()

    # The three candles that make the pattern. Any that fall outside the plotted
    # window are simply not marked.
    roles = {
        df.index[info["grab_idx"]]: "GRAB",
        df.index[info["opp_idx"]]: "OPPOSITE",
        df.index[-1]: "SIGNAL",
    }

    out = os.path.join(out_dir, f"{prefix}{tf_label}_{s._safe_name(label)}_{candle}.png")
    return bb_chart.render(
        plot_df, bb_chart.band_addplots(plot_df, bbg._bollinger),
        f"{label} — {title} ({tf_label})", out, prefix, roles,
        hline=info["opp_open"])


# ---------------------------------------------------------------------------
# ALERT
# ---------------------------------------------------------------------------
def alert(kind, ticker, tf_label, d, info, bot, prefix, title, gate_note):
    """Log + send one reclaim match (chart if possible, else text)."""
    yahoo, tv = s.chart_links(kind, ticker)
    name = s.COMMODITIES.get(ticker, ticker)
    msg = (f"[{tf_label}] MATCH: {name} ({kind}) — {title} (LONG)\n"
           f"[{gate_note}; reclaimed the opposite candle's open "
           f"{info['opp_open']:.4f} after {info['waited']} candle(s)]\n"
           f"Yahoo: {yahoo}\nTradingView: {tv}")
    print("  " + msg.splitlines()[0])
    chart_path = None
    try:
        chart_path = save_reclaim_chart(ticker, d, tf_label, info, prefix, title)
    except Exception as e:
        print(f"    (could not draw chart: {e})")
    # Own id-prefix so these rows never collide with the bot's plain grab rows.
    s.log_signal(kind, ticker, tf_label, d, bot=bot, direction="long",
                 id_prefix=prefix, chart=s.chart_rel_path(chart_path))
    if chart_path:
        try:
            s.send_telegram_photo(chart_path, msg)
        except Exception as e:
            print(f"    (could not send photo: {e})")
            s.send_telegram_alert(msg)
    else:
        s.send_telegram_alert(msg)
