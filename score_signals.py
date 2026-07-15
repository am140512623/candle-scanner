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

The VERDICT for each signal uses a TRAILING take-profit vs a stop-loss:

    - Take-profit targets sit every 10% in the trade's favour (up for a long,
      down for a short).
    - The stop is the lowest low of the two pattern candles for a long, or the
      highest high for a short (stop_level).
    - Before any profit, the stop protects the trade. Once price reaches the
      first +10% rung it rides up, then exits when price pulls back 10% below its
      highest rung -- banking (peak rung - 10%) as realized_pct. E.g. a peak of
      +60% that pulls back to +50% exits with +50% locked in.
    - Outcomes: took_profit (win), stopped_out (loss), open (still running),
      pending (no price history after the candle yet). best_target_pct records
      the highest rung reached, so a +120% runner is captured.

It writes results.csv (per-signal detail) and summary.csv (per-bot totals,
win-rate, and average win size), and prints a win-rate summary per timeframe.

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
                  "stopped_out", "open", "win_rate_pct", "avg_win_pct"]

# Take-profit ladder + trailing exit. Targets sit every TP_STEP percent above
# entry (+10%, +20%, ...). Once price reaches the first rung the trade rides up,
# and exits when price pulls back TRAIL_PCT from its highest rung -- banking
# (peak rung - TRAIL_PCT). Example: peaks at +60%, drops to +50% -> exit, +50%
# locked in. There's no ceiling on how high the peak can climb.
TP_STEP = 10        # percent between take-profit rungs
TRAIL_PCT = 10      # pull-back from the peak rung that triggers the exit

# (column suffix, days after the candle close) -- a separate, purely informational
# lens: where price sat at each age, regardless of the TP/stop verdict.
HORIZONS = [("1d", 1), ("3d", 3), ("7d", 7), ("30d", 30), ("90d", 90), ("180d", 180)]

RESULT_FIELDS = (
    ["signal_id", "bot", "alert_date", "candle_date", "kind", "direction", "ticker",
     "timeframe", "entry_close", "stop_level", "chart"]
    + [f"chg_{name}" for name, _ in HORIZONS]
    + ["last_chg", "best_target_pct", "realized_pct", "tp_date", "exit_date",
       "stop_date", "stop_chg", "result", "status"]
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


def evaluate_trade(hist, candle_date, entry, stop_level, direction="long"):
    """Walk the daily bars after the signal and decide the outcome with a TRAILING
    take-profit. The logic is mirrored for shorts:

      LONG (bullish grab):
        - Before any profit, the pattern-LOW stop protects the trade.
        - Once price rises +TP_STEP% it rides up; exits when it pulls back
          TRAIL_PCT below its highest rung, banking (peak - TRAIL_PCT).
      SHORT (bearish breakdown):
        - Before any profit, the pattern-HIGH stop protects the trade (a rally
          back above the swept high invalidates it).
        - Once price falls -TP_STEP% it rides down; exits when it bounces back
          TRAIL_PCT above its lowest rung, banking the same (peak - TRAIL_PCT).

    realized_pct / best_target_pct are always the trade's P/L magnitude (positive
    for a win) regardless of direction, so summaries treat both sides alike.

    Outcomes:
        took_profit  -- exited via the trailing target  (win, see realized_pct)
        stopped_out  -- hit the stop before ever reaching TP_STEP%  (loss)
        open         -- in profit and still running (or running, neither hit)
        pending      -- not enough price history after the candle yet

    Daily-bar convention: the trailing exit is only checked on bars AFTER the one
    that set a new peak (we can't know the intra-day order of a bar's own high and
    low), so a single spiky bar never stops itself out at break-even.
    """
    blank = {"result": "pending", "status": "pending", "best_target_pct": "",
             "realized_pct": "", "tp_date": "", "exit_date": "",
             "stop_date": "", "stop_chg": ""}
    if hist is None or entry <= 0:
        return blank
    after = hist.loc[[d.date() > candle_date for d in hist.index]]
    if after.empty:
        return blank

    short = direction == "short"
    peak_rung = 0          # highest TP_STEP% rung reached so far (in profit direction)
    tp_date = ""           # first day price reached TP_STEP% in our favour
    activated = False      # has the trade reached the first rung yet?

    for ts, bar in after.iterrows():
        day = ts.date()
        high = float(bar["High"])
        low = float(bar["Low"])
        # Profit moves down for a short, up for a long. The favourable extreme is
        # the bar's low (short) or high (long); the stop sits the other way.
        if short:
            profit_pct = (entry - low) / entry * 100
            stop_hit = stop_level and high >= stop_level
        else:
            profit_pct = (high - entry) / entry * 100
            stop_hit = stop_level and low <= stop_level

        if activated:
            # Trailing exit, measured against the peak as it stood BEFORE this bar.
            trail_pct = peak_rung - TRAIL_PCT
            trail_price = entry * (1 - trail_pct / 100) if short else entry * (1 + trail_pct / 100)
            pulled_back = high >= trail_price if short else low <= trail_price
            if pulled_back:
                return {"result": "took_profit", "status": "took_profit",
                        "best_target_pct": peak_rung, "realized_pct": trail_pct,
                        "tp_date": tp_date, "exit_date": day.isoformat(),
                        "stop_date": "", "stop_chg": ""}
        elif stop_hit:
            # Not in profit yet and the stop gave way -> a clean loss. stop_chg is
            # the P/L at the stop (negative): below entry for a long, above for a short.
            stop_chg = (entry - stop_level) / entry * 100 if short else (stop_level - entry) / entry * 100
            return {"result": "stopped_out", "status": "stopped_out",
                    "best_target_pct": 0, "realized_pct": "",
                    "tp_date": "", "exit_date": "",
                    "stop_date": day.isoformat(),
                    "stop_chg": f"{stop_chg:+.2f}"}

        # Update the peak from this bar's favourable move (after the exit checks).
        if profit_pct >= TP_STEP:
            rung = int(profit_pct // TP_STEP) * TP_STEP
            if rung > peak_rung:
                peak_rung = rung
            if not activated:
                activated = True
                tp_date = day.isoformat()

    # Ran out of history without an exit: still open.
    return {"result": "open", "status": "open",
            "best_target_pct": peak_rung if activated else 0,
            "realized_pct": "", "tp_date": tp_date, "exit_date": "",
            "stop_date": "", "stop_chg": ""}


def score_one(row, hist):
    """Score one signal: the TP/stop verdict plus informational horizon changes."""
    candle_date = datetime.date.fromisoformat(row["candle_date"])
    entry = float(row["entry_close"])
    stop_level = float(row["stop_level"]) if row.get("stop_level") else 0.0
    direction = row.get("direction") or "long"   # pre-column rows are all long
    out = dict(row)
    out["direction"] = direction

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
    out.update(evaluate_trade(hist, candle_date, entry, stop_level, direction))
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
                  "stopped_out": 0, "open": 0, "win_pct_sum": 0.0})
        agg["total_signals"] += 1
        res = r["result"]
        if res == "took_profit":
            agg["took_profit"] += 1
            try:
                agg["win_pct_sum"] += float(r.get("realized_pct") or 0)
            except ValueError:
                pass
        elif res == "stopped_out":
            agg["stopped_out"] += 1
        else:                       # open or pending
            agg["open"] += 1

    rows = []
    for bot in sorted(by_bot):
        a = by_bot[bot]
        scored = a["took_profit"] + a["stopped_out"]
        rate = 100 * a["took_profit"] / scored if scored else 0
        avg_win = a["win_pct_sum"] / a["took_profit"] if a["took_profit"] else 0
        rows.append({
            "bot": bot,
            "total_signals": a["total_signals"],
            "scored": scored,
            "took_profit": a["took_profit"],
            "stopped_out": a["stopped_out"],
            "open": a["open"],
            "win_rate_pct": f"{rate:.0f}",
            "avg_win_pct": f"{avg_win:.0f}",
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
