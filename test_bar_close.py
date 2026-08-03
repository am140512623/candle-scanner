"""
The US index bot must alert on a candle as soon as that candle STOPS TRADING --
not a bar later.

yfinance returns the still-forming candle as the last row, and the bot used to
drop that row unconditionally. On the daily frame that also discarded a candle
that had already finished: after Friday's close the last row IS Friday, so the
signal waited for Monday's row to appear -- a full trading day, with price gone
from the entry. index_signals._drop_unclosed replaces the blind drop, and this
pins its behaviour on both sides: nothing acted on early, nothing held back.

Run:  python test_bar_close.py
"""

import pandas as pd

import index_signals as ix

NY = ix.MARKET_TZ


def ny(text):
    return pd.Timestamp(text, tz=NY)


def frame(stamps):
    """A minimal OHLC frame indexed by the given timestamps."""
    idx = pd.DatetimeIndex(stamps)
    return pd.DataFrame({"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5},
                        index=idx)


def check(label, stamps, now, keep_last, why):
    df = frame(stamps)
    got = ix._drop_unclosed(df, label, now=ny(now))
    kept = len(got) == len(df)
    assert kept == keep_last, (
        f"{label} @ {now}: expected last bar "
        f"{'KEPT' if keep_last else 'DROPPED'} ({why}), got "
        f"{'KEPT' if kept else 'DROPPED'}")
    # Dropping must only ever remove the newest row.
    assert len(got) == len(df) - (0 if keep_last else 1)
    return 1


def main():
    n = 0

    # --- DAILY: the bug this fix exists for --------------------------------
    week = ["2026-07-29", "2026-07-30", "2026-07-31"]
    n += check("1D", week, "2026-07-31 12:00", False, "Friday still trading")
    n += check("1D", week, "2026-07-31 16:05", False, "inside the settle window")
    n += check("1D", week, "2026-07-31 16:20", True,  "Friday has closed")
    n += check("1D", week, "2026-08-02 09:00", True,  "the weekend after")
    # Monday's partial row must not be judged; Friday's stays the last candle.
    mon = week + ["2026-08-03"]
    n += check("1D", mon, "2026-08-03 12:00", False, "Monday still trading")

    # --- WEEKLY: rows are labelled with the MONDAY, so the week ends Friday --
    wk = ["2026-07-20", "2026-07-27"]
    n += check("1W", wk, "2026-07-29 16:20", False, "midweek")
    n += check("1W", wk, "2026-07-31 15:00", False, "Friday still trading")
    n += check("1W", wk, "2026-07-31 16:20", True,  "the week has closed")

    # --- INTRADAY: labelled with the bar START, tz-aware -------------------
    half = ["2026-07-31 11:30", "2026-07-31 12:00"]
    n += check("30m", [ny(t) for t in half], "2026-07-31 12:20", False,
               "12:00 bar runs to 12:30")
    n += check("30m", [ny(t) for t in half], "2026-07-31 12:46", True,
               "12:30 close plus settle")

    # The final candle of a session is cut short by the 16:00 close: a 1h bar
    # opening 15:30 is done at 16:00, not 16:30.
    hourly = [ny("2026-07-31 14:30"), ny("2026-07-31 15:30")]
    n += check("1h", hourly, "2026-07-31 16:16", True, "clipped to the close")
    n += check("1h", hourly, "2026-07-31 15:50", False, "session still open")

    # 4h buckets land on 08:00 / 12:00; the 12:00 one ends exactly at the close.
    four = [ny("2026-07-31 08:00"), ny("2026-07-31 12:00")]
    n += check("4h", four, "2026-07-31 16:20", True,  "12:00 bucket ends 16:00")
    n += check("4h", four, "2026-07-31 14:00", False, "12:00 bucket still filling")
    n += check("4h", [ny("2026-07-31 04:00"), ny("2026-07-31 08:00")],
               "2026-07-31 12:20", True, "08:00 bucket ends 12:00")

    # --- the bar is never judged before its close, on any frame ------------
    for label, ts, end in (("1D", "2026-07-31", "2026-07-31 16:00"),
                           ("1W", "2026-07-27", "2026-07-31 16:00"),
                           ("30m", ny("2026-07-31 12:00"), "2026-07-31 12:30"),
                           ("1h", ny("2026-07-31 15:30"), "2026-07-31 16:00"),
                           ("4h", ny("2026-07-31 08:00"), "2026-07-31 12:00")):
        got = ix._bar_end(pd.Timestamp(ts), label)
        assert got == ny(end), f"{label} {ts}: bar ends {got}, expected {ny(end)}"
        n += 1

    print(f"OK: {n} checks -- closed candles are acted on the same session, "
          f"forming candles are still ignored.")


if __name__ == "__main__":
    main()
