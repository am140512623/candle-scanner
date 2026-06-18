"""
Score past Liquidity-Grab signals -- did the price rise or fall after each alert?

The scanners (scan_all / scan_crypto / scan_segment) append every alert to
signals.csv, recording the entry price and the candle it formed in. This script
reads that log, pulls each ticker's price history, and measures the % change at
fixed horizons after the alert:

    +1d, +3d, +7d, +30d, +90d, +180d

The short horizons judge the fast intraday bots; the long ones (30d-180d) judge
the weekly and monthly bots. A horizon column stays blank until that much time
has actually passed, then fills in and never changes (historical closes are final).

It also applies a STOP LOSS: each signal carries the lowest low of its two
pattern candles (stop_level). If price ever trades down to that level afterward,
the signal is marked stopped_out on that date with the loss locked in -- the
trade is treated as closed there, regardless of what price does later.

It writes results.csv (one scored row per signal) and prints a win-rate summary
per timeframe.

Run it on a daily schedule. It rebuilds results.csv from scratch each time, so
it is safe to run as often as you like -- no state of its own to corrupt.

Optional: set SCORE_BOT_TOKEN (+ SCORE_CHAT_ID, comma-separated) to also receive
the win-rate summary as a Telegram message.
"""

import csv
import datetime
import os

import pandas as pd
import yfinance as yf

import scan_all as s

RESULTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")

# (column suffix, days after the candle close) -- covers the fast intraday bots
# and the slower weekly/monthly ones in one sweep.
HORIZONS = [("1d", 1), ("3d", 3), ("7d", 7), ("30d", 30), ("90d", 90), ("180d", 180)]

RESULT_FIELDS = (
    ["signal_id", "alert_date", "candle_date", "kind", "ticker",
     "timeframe", "entry_close", "stop_level"]
    + [f"chg_{name}" for name, _ in HORIZONS]
    + ["last_chg", "stop_date", "stop_chg", "result", "status"]
)


def load_signals():
    """Read signals.csv into a list of dict rows (empty list if it doesn't exist)."""
    try:
        with open(s.SIGNALS_CSV, newline="") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []


def history_for(ticker, start_date):
    """Daily closes for `ticker` from start_date onward (auto-adjusted), or None."""
    try:
        hist = yf.download(ticker, start=start_date.isoformat(), interval="1d",
                           auto_adjust=True, progress=False)
        if hist is None or hist.empty:
            return None
        # yfinance returns a column MultiIndex for a single ticker too; flatten it.
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        return hist
    except Exception as e:
        print(f"  ({ticker}: history fetch failed: {e})")
        return None


def close_on_or_after(hist, target_date):
    """First daily close on or after target_date, or None if no bar exists yet
    (i.e. that horizon hasn't happened -- or fully settled -- yet)."""
    sub = hist.loc[[d.date() >= target_date for d in hist.index]]
    if sub.empty:
        return None
    return float(sub["Close"].iloc[0])


def stop_breach_date(hist, candle_date, stop_level):
    """Earliest date AFTER the signal candle whose low traded down to stop_level
    (the trade's invalidation price), or None if it never has."""
    if hist is None or not stop_level:
        return None
    after = hist.loc[[d.date() > candle_date for d in hist.index]]
    hit = after.loc[after["Low"] <= stop_level]
    if hit.empty:
        return None
    return hit.index[0].date()


def score_one(row, hist):
    """Score one signal: horizon % changes, stop-loss check, and a final verdict."""
    candle_date = datetime.date.fromisoformat(row["candle_date"])
    entry = float(row["entry_close"])
    stop_level = float(row["stop_level"]) if row.get("stop_level") else 0.0
    out = dict(row)
    last_chg, filled = "", 0
    for name, days in HORIZONS:
        col = f"chg_{name}"
        out[col] = ""
        if hist is None or entry <= 0:
            continue
        price = close_on_or_after(hist, candle_date + datetime.timedelta(days=days))
        if price is None:
            continue  # not enough time has passed yet
        pct = (price - entry) / entry * 100
        out[col] = f"{pct:+.2f}"
        last_chg = out[col]
        filled += 1
    out["last_chg"] = last_chg

    # Stop loss: did price ever retreat to the pattern's low? If so the trade is
    # closed there with the locked-in loss, whatever the horizons say afterward.
    stop_dt = stop_breach_date(hist, candle_date, stop_level)
    if stop_dt and entry > 0:
        out["stop_date"] = stop_dt.isoformat()
        out["stop_chg"] = f"{(stop_level - entry) / entry * 100:+.2f}"
        out["result"], out["status"] = "stopped_out", "stopped_out"
        return out
    out["stop_date"], out["stop_chg"] = "", ""

    if not last_chg:
        out["result"], out["status"] = "", "pending"
    else:
        val = float(last_chg)
        out["result"] = "rose" if val > 0 else "fell" if val < 0 else "flat"
        # "closed" once the longest horizon is in; otherwise still maturing.
        out["status"] = "closed" if filled == len(HORIZONS) else "open"
    return out


def summarize(results):
    """Win-rate per timeframe. A signal counts as a win if it rose; a stop-out
    counts as a loss. Pending (too-new) signals are excluded from the rate."""
    by_tf = {}
    for r in results:
        if r["result"] in ("rose", "fell", "flat", "stopped_out"):
            tf = r["timeframe"]
            wins, stops, total = by_tf.get(tf, (0, 0, 0))
            by_tf[tf] = (
                wins + (1 if r["result"] == "rose" else 0),
                stops + (1 if r["result"] == "stopped_out" else 0),
                total + 1,
            )
    lines = ["Liquidity Grab signal performance (price up after alert):"]
    for tf in sorted(by_tf):
        wins, stops, total = by_tf[tf]
        rate = 100 * wins / total if total else 0
        extra = f", {stops} stopped out" if stops else ""
        lines.append(f"  {tf:>4}: {wins}/{total} rose  ({rate:.0f}%){extra}")
    pending = sum(1 for r in results if r["status"] == "pending")
    if pending:
        lines.append(f"  ({pending} signal(s) too new to score yet)")
    return "\n".join(lines)


def maybe_telegram(summary):
    """Send the summary to Telegram only if SCORE_BOT_TOKEN is configured."""
    token = os.environ.get("SCORE_BOT_TOKEN")
    if not token:
        return
    s.BOT_TOKEN = token
    chat = os.environ.get("SCORE_CHAT_ID")
    if chat:
        s.CHAT_IDS = [c.strip() for c in chat.split(",") if c.strip()]
    s.send_telegram_alert(summary)


def main():
    signals = load_signals()
    if not signals:
        print("No signals logged yet (signals.csv is empty/missing) -- nothing to score.")
        return

    # Fetch each ticker's history once, from its earliest signal date.
    earliest = {}
    for r in signals:
        cd = datetime.date.fromisoformat(r["candle_date"])
        t = r["ticker"]
        earliest[t] = min(earliest.get(t, cd), cd)
    print(f"Scoring {len(signals)} signal(s) across {len(earliest)} ticker(s)...")
    hist_cache = {t: history_for(t, start) for t, start in earliest.items()}

    results = [score_one(r, hist_cache.get(r["ticker"])) for r in signals]

    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in RESULT_FIELDS})
    print(f"Wrote {RESULTS_CSV}")

    summary = summarize(results)
    print("\n" + summary)
    maybe_telegram(summary)


if __name__ == "__main__":
    main()
