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

The VERDICT for each signal is a take-profit vs stop-loss race:

    - Take-profit targets sit every +10% above entry (+10%, +20%, +30%, ...).
    - The stop is the lowest low of the two pattern candles (stop_level).
    - Walking forward day by day, whichever is hit first decides the outcome:
        took_profit  -- reached the first +10% target before the stop  (win)
        stopped_out  -- hit the stop first                             (loss)
        open         -- neither hit yet
        pending      -- not enough price history after the candle yet
    - best_target_pct records the highest +10% rung price reached, so a runner
      that hit +120% before stopping is captured.

It writes results.csv (per-signal detail) and summary.csv (per-bot totals +
win-rate), and prints a win-rate summary per timeframe.

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
SUMMARY_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "summary.csv")

SUMMARY_FIELDS = ["bot", "total_signals", "scored", "took_profit",
                  "stopped_out", "open", "win_rate_pct"]

# Take-profit ladder: targets sit every TP_STEP percent above entry (+10%, +20%,
# +30%, ...). There's no fixed ceiling -- we just record the highest rung price
# actually reached, so a runner that hits +120% is captured.
TP_STEP = 10        # percent between take-profit rungs

# (column suffix, days after the candle close) -- a separate, purely informational
# lens: where price sat at each age, regardless of the TP/stop verdict.
HORIZONS = [("1d", 1), ("3d", 3), ("7d", 7), ("30d", 30), ("90d", 90), ("180d", 180)]

RESULT_FIELDS = (
    ["signal_id", "bot", "alert_date", "candle_date", "kind", "ticker",
     "timeframe", "entry_close", "stop_level"]
    + [f"chg_{name}" for name, _ in HORIZONS]
    + ["last_chg", "best_target_pct", "tp_date", "stop_date", "stop_chg",
       "result", "status"]
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


def evaluate_trade(hist, candle_date, entry, stop_level):
    """Walk the daily bars after the signal and decide the outcome:

        take_profit  -- price reached the first +TP_STEP% target before the stop
        stopped_out  -- price hit the stop (the pattern low) first
        open         -- has run for a while but hit neither yet
        pending      -- not enough price history after the candle yet

    Also records best_target_pct (highest +10% rung price reached while the trade
    was live) and the dates. Same-day ties (a bar whose high hits a target AND
    whose low hits the stop) are scored conservatively as stopped_out, since daily
    bars can't tell us which came first.
    """
    blank = {"result": "pending", "status": "pending", "tp_date": "",
             "stop_date": "", "stop_chg": "", "best_target_pct": ""}
    if hist is None or entry <= 0:
        return blank
    after = hist.loc[[d.date() > candle_date for d in hist.index]]
    if after.empty:
        return blank

    # First day the stop is breached (low trades down to the pattern low).
    stop_dt = None
    if stop_level:
        breached = after.loc[after["Low"] <= stop_level]
        if not breached.empty:
            stop_dt = breached.index[0].date()

    # Window the trade was live: every bar strictly before the stop day (so a
    # same-day target doesn't count -- the conservative tie-break above).
    live = after.loc[[d.date() < stop_dt for d in after.index]] if stop_dt else after

    best_pct, tp_date = 0, ""
    if not live.empty:
        gains = (live["High"] - entry) / entry * 100
        reached = gains[gains >= TP_STEP]
        if not reached.empty:
            best_pct = int(reached.max() // TP_STEP) * TP_STEP   # floor to a rung
            tp_date = reached.index[0].date().isoformat()        # first +10% hit

    if tp_date:
        result = "took_profit"
    elif stop_dt:
        result = "stopped_out"
    else:
        result = "open"
    return {
        "result": result,
        "status": result,
        "tp_date": tp_date,
        "stop_date": stop_dt.isoformat() if stop_dt else "",
        "stop_chg": (f"{(stop_level - entry) / entry * 100:+.2f}"
                     if stop_dt else ""),
        "best_target_pct": best_pct,
    }


def score_one(row, hist):
    """Score one signal: the TP/stop verdict plus informational horizon changes."""
    candle_date = datetime.date.fromisoformat(row["candle_date"])
    entry = float(row["entry_close"])
    stop_level = float(row["stop_level"]) if row.get("stop_level") else 0.0
    out = dict(row)

    # Informational horizons: where price sat at each age (independent of TP/stop).
    last_chg = ""
    for name, days in HORIZONS:
        col = f"chg_{name}"
        out[col] = ""
        if hist is None or entry <= 0:
            continue
        price = close_on_or_after(hist, candle_date + datetime.timedelta(days=days))
        if price is None:
            continue  # not enough time has passed yet
        out[col] = f"{(price - entry) / entry * 100:+.2f}"
        last_chg = out[col]
    out["last_chg"] = last_chg

    # The verdict: which came first, a take-profit target or the stop?
    out.update(evaluate_trade(hist, candle_date, entry, stop_level))
    return out


def summarize(results):
    """Win-rate per timeframe. A win = took profit (hit +10% before the stop);
    a loss = stopped out. Unresolved (open / too-new) signals are excluded."""
    by_tf = {}
    for r in results:
        if r["result"] in ("took_profit", "stopped_out"):
            tf = r["timeframe"]
            wins, total = by_tf.get(tf, (0, 0))
            by_tf[tf] = (wins + (1 if r["result"] == "took_profit" else 0), total + 1)
    lines = ["Liquidity Grab signal performance (take-profit vs stop):"]
    for tf in sorted(by_tf):
        wins, total = by_tf[tf]
        rate = 100 * wins / total if total else 0
        lines.append(f"  {tf:>4}: {wins}/{total} hit target  ({rate:.0f}%)")
    unresolved = sum(1 for r in results if r["result"] in ("open", "pending"))
    if unresolved:
        lines.append(f"  ({unresolved} signal(s) still open / too new)")
    return "\n".join(lines)


def build_summary(results):
    """One totals row per bot: how many signals it fired and how they turned out.
    `scored` = took_profit + stopped_out; `open` = still unresolved / too new;
    win_rate = took_profit / scored."""
    by_bot = {}
    for r in results:
        bot = r.get("bot") or "unknown"
        agg = by_bot.setdefault(
            bot, {"total_signals": 0, "took_profit": 0,
                  "stopped_out": 0, "open": 0})
        agg["total_signals"] += 1
        res = r["result"]
        if res == "took_profit":
            agg["took_profit"] += 1
        elif res == "stopped_out":
            agg["stopped_out"] += 1
        else:                       # open or pending
            agg["open"] += 1

    rows = []
    for bot in sorted(by_bot):
        a = by_bot[bot]
        scored = a["took_profit"] + a["stopped_out"]
        rate = 100 * a["took_profit"] / scored if scored else 0
        rows.append({
            "bot": bot,
            "total_signals": a["total_signals"],
            "scored": scored,
            "took_profit": a["took_profit"],
            "stopped_out": a["stopped_out"],
            "open": a["open"],
            "win_rate_pct": f"{rate:.0f}",
        })
    return rows


def write_summary(rows):
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Wrote {SUMMARY_CSV}")


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

    write_summary(build_summary(results))

    summary = summarize(results)
    print("\n" + summary)
    maybe_telegram(summary)


if __name__ == "__main__":
    main()
