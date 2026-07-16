"""
GRAB -> REVERSE CANDLE -> RECLAIM  (LONG only) -- the standalone pattern.

This has nothing to do with the Bollinger bands. It lived on the Bollinger bots
for a while, but the backtest showed the band gate threw away ~92% of the setups
and the UPPER band gate actively made it worse (+0.174R -> +0.131R), so the
pattern now runs ungated on the US index bot where it belongs.

The layout (uptrend case):

    1) GRAB -- the usual liquidity grab ("swallow"): a green candle whose body
       covers the prior green candle body(s) and whose low sweeps below their
       low(s), then closes green. Same `_swallow` engine as every other bot.

    2) REVERSE CANDLE -- the candle IMMEDIATELY after the grab is the other
       colour: a RED candle (close < open). Its OPEN becomes the level.

    3) RECLAIM -- the FIRST candle to CLOSE above that red candle's OPEN is the
       signal candle. Only the first one counts: once price has closed back above
       the level the setup is spent, and later closes above it are not
       re-signalled. Must land within MAX_WAIT candles or the setup goes stale.

Unlike the plain grab bots, the signal candle is NOT the grab candle -- it comes
some bars later, so the detector looks back from the just-closed candle for a
completed layout ending on it.

Deliberately imports only scan_all (for `_swallow`). It must NOT import any of the
scan_* bot modules: those mutate scan_all.BOT_TOKEN at import time, which would
hijack the Telegram token of whichever bot imported this.

Backtested (backtest_reclaim.py, arm "reclaim"): +0.174R/trade vs +0.083R for
taking the grab alone, over 29,687 daily S&P 500 trades -- see that file for the
survivorship-bias caveat that bounds those numbers.
"""

import os

import pandas as pd

import scan_all as s

# How many candles after the reverse candle the reclaim may still arrive.
MAX_WAIT = int(os.environ.get("RECLAIM_MAX_WAIT", "10"))

# Grab needs up to 3 candles, then the reverse candle, then the signal.
MIN_BARS = 6


def reclaim_signal(df, max_wait=MAX_WAIT):
    """(matched, info) for the grab -> reverse candle -> reclaim layout, where the
    RECLAIM lands on the LAST candle of `df` (the caller has already trimmed to the
    just-closed candle).

    info -> None if no match, else {"grab_idx", "opp_idx", "opp_open", "waited"}.
    """
    if df is None or len(df) < MIN_BARS:
        return (False, None)

    opens, closes = df["Open"], df["Close"]
    last = len(df) - 1
    c_last = closes.iloc[-1]
    if pd.isna(c_last):
        return (False, None)

    # Walk candidate positions for the reverse candle, newest first: the freshest
    # completed structure wins. r is the reverse candle, r-1 the grab it followed.
    earliest = max(last - max_wait, 1)
    for r in range(last - 1, earliest - 1, -1):
        o_r, c_r = opens.iloc[r], closes.iloc[r]
        if pd.isna(o_r) or pd.isna(c_r):
            continue
        if not c_r < o_r:
            continue                      # candle after the grab isn't the reverse
        if not c_last > o_r:
            continue                      # the last candle didn't reclaim the level
        if bool((closes.iloc[r + 1:last] > o_r).any()):
            continue                      # something already closed above it -- spent

        g = r - 1
        if g < 1:
            continue                      # need a candle for the grab to swallow
        if not s._swallow(df.iloc[:g + 1], bullish=True):
            continue

        return (True, {"grab_idx": g, "opp_idx": r, "opp_open": float(o_r),
                       "waited": last - r})
    return (False, None)
