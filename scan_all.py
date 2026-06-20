"""
Weekly Liquidity-Grab scanner across the S&P 500, Nasdaq-100, and top-200 crypto.

Run it manually whenever you want (e.g. once a week after the weekly candle closes):
    python scan_all.py

It fetches the ticker universe automatically, downloads candles in batches,
runs your pattern check on each, and prints every match. Fill in BOT_TOKEN /
CHAT_ID below if you also want Telegram alerts.
"""

import csv
import datetime
import io
import logging
import os
import smtplib
import time
from email.message import EmailMessage

import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save files
import mplfinance as mpf
import pandas as pd
import requests
import yfinance as yf

# Silence yfinance's "possibly delisted" noise for coins Yahoo doesn't carry.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Folder where chart PNGs get saved (next to this script).
CHART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
# Read from environment (used by the cloud / GitHub Actions secrets) if present,
# otherwise fall back to these local values.
BOT_TOKEN = os.environ.get("BOT_TOKEN") or "YOUR_BOT_TOKEN"

# One or more Telegram recipients. Add more by appending their chat IDs.
# (Each person must tap Start on the bot once before they can receive messages.)
CHAT_IDS = [
    "7788611624",   # A
    "6173185769",   # m
]
# Allow the cloud secret CHAT_ID to override (comma-separated list supported).
if os.environ.get("CHAT_ID"):
    CHAT_IDS = [c.strip() for c in os.environ["CHAT_ID"].split(",") if c.strip()]

# --- Email results (optional) ---
# To turn on: set EMAIL_ENABLED = True and fill in the 3 lines below.
# For Gmail, EMAIL_APP_PASSWORD must be a 16-char Google "App Password"
# (NOT your normal password) -- see the setup note in chat.
EMAIL_ENABLED = False
EMAIL_FROM = "am140512623@gmail.com"          # the Gmail that sends
EMAIL_APP_PASSWORD = "your_16_char_app_password"  # Google App Password, no spaces
EMAIL_TO = "am140512623@gmail.com"            # where to receive the report
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465

# Timeframes to scan. Weekly runs every time; monthly runs only once a new
# month has started (so it scans the just-CLOSED monthly candle, not the
# half-formed one). closed_only=True drops the still-forming last candle.
TIMEFRAMES = [
    {"interval": "1wk", "period": "2y",  "label": "WEEKLY",  "monthly_only": False, "closed_only": False},
    {"interval": "1mo", "period": "max", "label": "MONTHLY", "monthly_only": True,  "closed_only": True},
]

BATCH_SIZE = 50     # tickers per Yahoo download request
CRYPTO_TOP_N = 200  # how many top coins to pull from CoinGecko

# Remembers which month the monthly scan last ran, so it only runs once per month.
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".monthly_state.txt")

# Commodities via ownable ETFs (NOT futures/CFDs). You buy these as normal
# shares on a stock broker. Label shows the asset name + the ticker you buy.
COMMODITIES = {
    "GLD": "Gold (GLD)",
    "IAU": "Gold (IAU)",
    "SLV": "Silver (SLV)",
    "PPLT": "Platinum (PPLT)",
    "PALL": "Palladium (PALL)",
    "CPER": "Copper (CPER)",
    "USO": "Oil (USO)",
    "BNO": "Brent Oil (BNO)",
    "UNG": "Natural Gas (UNG)",
    "CORN": "Corn (CORN)",
    "WEAT": "Wheat (WEAT)",
    "SOYB": "Soybeans (SOYB)",
    "DBC": "Broad Commodities (DBC)",
    "PDBC": "Broad Commodities (PDBC)",
    "DBA": "Agriculture (DBA)",
    "GSG": "Broad Commodities (GSG)",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (weekly-scanner)"}


# ---------------------------------------------------------------------------
# YOUR PATTERN
# ---------------------------------------------------------------------------
# A pegged stablecoin barely moves: over a recent window its entire high-to-low
# range is a tiny fraction of its price. We drop those -- a "Liquidity Grab" on a
# coin glued to ~$1 is just decimal noise. Judged by VOLATILITY, not price level,
# so a genuinely volatile coin that happens to trade near $1 is still kept. The
# window is measured in CANDLES, so on each timeframe it spans a sensible amount
# of real time (20 x 6H ~ 5 days; 20 weekly ~ 5 months) -- and a real asset is
# never that flat over those spans, so stocks are unaffected too.
FLAT_LOOKBACK = 20      # candles to measure flatness over
FLAT_THRESHOLD = 0.02   # skip if the full range is under 2% of price


def is_flat(df, lookback=FLAT_LOOKBACK, threshold=FLAT_THRESHOLD):
    """True if `df` barely moves over its last `lookback` candles (a peg/stablecoin)."""
    try:
        recent = df.tail(lookback)
        if len(recent) < 2:
            return False
        price = float(recent["Close"].iloc[-1])
        if price <= 0:
            return False
        rng = float(recent["High"].max()) - float(recent["Low"].min())
        return (rng / price) < threshold
    except Exception:
        return False


# Shared engine for both patterns. There's NO shape requirement on the lead-in
# candle(s) -- only the trigger (the last, newest candle) has to qualify: it must
# be a decisive candle of the right colour whose wick sweeps PAST the reference
# candle(s) and whose close lands across their close(s).
#
#   bullish -> green, strong close near its high, low sweeps BELOW the reference
#              low(s), close ABOVE the reference close(s)
#   bearish -> red,   strong close near its low,  high sweeps ABOVE the reference
#              high(s), close BELOW the reference close(s)
#
# It fires on EITHER layout (whichever holds):
#   2-candle: the trigger clears just the single candle right before it -- sweeps
#             past its extreme and closes across its close.
#   3-candle: the trigger clears the wider two-candle consolidation block --
#             sweeps past BOTH extremes and closes past BOTH closes.
def _liquidity_grab(df, bullish):
    if df is None or len(df) < 2:
        return False

    o, c, h, l = (df["Open"].iloc[-1], df["Close"].iloc[-1],
                  df["High"].iloc[-1], df["Low"].iloc[-1])
    if pd.isna([o, c]).any():
        return False

    # The trigger candle: right colour + a decisive close (body beats the tail).
    if bullish:
        if not (c > o and (c - o) > (h - c)):
            return False
    else:
        if not (c < o and (o - c) > (c - l)):
            return False

    def clears(idxs):
        """True if the trigger sweeps past, and closes across, every ref candle."""
        highs = [df["High"].iloc[i] for i in idxs]
        lows = [df["Low"].iloc[i] for i in idxs]
        closes = [df["Close"].iloc[i] for i in idxs]
        if pd.isna(closes).any():
            return False
        if bullish:
            return l < min(lows) and c > max(closes)
        return h > max(highs) and c < min(closes)

    # Fire on EITHER form: the trigger clears just the candle before it (2-candle),
    # or it clears the wider two-candle block (3-candle).
    if clears([-2]):
        return True
    if len(df) >= 3 and clears([-3, -2]):
        return True
    return False


def check_pattern(df):
    return _liquidity_grab(df, bullish=True)


# ---------------------------------------------------------------------------
# THE BEARISH MIRROR (short setups)
# ---------------------------------------------------------------------------
def check_pattern_bearish(df):
    return _liquidity_grab(df, bullish=False)


def pattern_name(direction):
    """Human label for the alert/chart. Long = the original bullish grab, short =
    the bearish mirror."""
    return "Bearish Liquidity Grab (SHORT)" if direction == "short" else "Liquidity Grab"


# ---------------------------------------------------------------------------
# TICKER UNIVERSE
# ---------------------------------------------------------------------------
def _read_wiki_tables(url):
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))


def get_sp500():
    tables = _read_wiki_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    syms = tables[0]["Symbol"].astype(str).tolist()
    return [s.replace(".", "-").strip() for s in syms]


def get_nasdaq100():
    tables = _read_wiki_tables("https://en.wikipedia.org/wiki/Nasdaq-100")
    for tbl in tables:
        for col in ("Ticker", "Symbol"):
            if col in tbl.columns:
                syms = tbl[col].astype(str).tolist()
                return [s.replace(".", "-").strip() for s in syms]
    return []


# Stablecoins are useless for this pattern, so they're dropped from the universe.
STABLECOINS = {"USDT", "USDC", "DAI", "BUSD", "TUSD", "USDE", "FDUSD", "USDS", "PYUSD"}
# Browser-like UA: Yahoo's screener endpoint rejects the plain scanner UA.
_BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def get_top_crypto(n=200):
    """Top n crypto tickers by market cap, stablecoins dropped, in rank order.

    Pulls the list straight from Yahoo's own crypto screener, so every ticker is
    guaranteed to have Yahoo candle data (no CoinGecko->Yahoo symbol mismatches,
    which used to skip ~half of the lower-ranked coins). Falls back to CoinGecko
    if Yahoo's screener is ever unavailable.
    """
    try:
        quotes = _yahoo_crypto_quotes(n)
        seen, tickers = set(), []
        for q in quotes[:n]:
            sym = (q.get("symbol") or "").upper()        # already 'BTC-USD'
            if not sym or sym in seen or sym.replace("-USD", "") in STABLECOINS:
                continue
            seen.add(sym)
            tickers.append(sym)
        if tickers:
            return tickers
        print("  Yahoo crypto screener returned nothing; falling back to CoinGecko")
    except Exception as e:
        print(f"  Yahoo crypto screener failed ({e}); falling back to CoinGecko")
    return _coingecko_top_crypto(n)


def _yahoo_crypto_quotes(n):
    """Fetch up to n crypto quotes (symbol + market cap) from Yahoo, cap-sorted."""
    session = requests.Session()
    session.headers.update({"User-Agent": _BROWSER_UA})
    try:
        session.get("https://finance.yahoo.com", timeout=15)  # prime cookies
    except requests.RequestException:
        pass
    out, start = [], 0
    while len(out) < n:
        r = session.get(
            "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved",
            params={"scrIds": "all_cryptocurrencies_us", "count": 250, "start": start},
            timeout=30,
        )
        r.raise_for_status()
        result = r.json()["finance"]["result"][0]
        quotes = result.get("quotes", [])
        if not quotes:
            break
        out.extend(quotes)
        start += 250
        if start >= result.get("total", 0):
            break
        time.sleep(0.3)
    return out


def _coingecko_top_crypto(n):
    """Fallback: top n crypto from CoinGecko, mapped to Yahoo SYMBOL-USD tickers."""
    out = []
    per_page = 250
    page = 1
    while len(out) < n:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={"vs_currency": "usd", "order": "market_cap_desc",
                    "per_page": min(per_page, n - len(out)), "page": page},
            headers=HEADERS, timeout=30,
        )
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        out.extend(rows)
        page += 1
        time.sleep(1)  # be polite to the free API
    seen, tickers = set(), []
    for c in out[:n]:
        sym = c["symbol"].upper()
        if sym in STABLECOINS or sym in seen:
            continue
        seen.add(sym)
        tickers.append(f"{sym}-USD")
    return tickers


def get_us_stocks_with_caps():
    """Every US-listed stock with a market cap, from the free Nasdaq screener.

    One HTTP call, no API key. Returns a list of (yahoo_symbol, market_cap_usd)
    sorted biggest-cap first. Used by the segmented cap-tier bots to slice the
    market into Mega / Large / Mid / Small buckets.
    """
    headers = {"User-Agent": "Mozilla/5.0 (scanner)", "Accept": "application/json"}
    resp = requests.get(
        "https://api.nasdaq.com/api/screener/stocks",
        params={"tableonly": "true", "limit": "10000", "download": "true"},
        headers=headers, timeout=30,
    )
    resp.raise_for_status()
    rows = (resp.json().get("data") or {}).get("rows") or []
    out = {}
    for r in rows:
        sym = (r.get("symbol") or "").strip().upper()
        if not sym or "^" in sym:
            continue
        try:
            cap = float((r.get("marketCap") or "").strip())
        except ValueError:
            continue
        if cap <= 0:
            continue
        # Nasdaq uses '/' or '.' for share classes; Yahoo uses '-' (BRK/B -> BRK-B).
        yahoo = sym.replace("/", "-").replace(".", "-")
        out[yahoo] = cap   # dedupe; last write wins
    return sorted(out.items(), key=lambda kv: kv[1], reverse=True)


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# ---------------------------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------------------------
def chart_links(kind, ticker):
    """Tappable chart links: Yahoo (exact match to our data) + TradingView (nicer view)."""
    yahoo = f"https://finance.yahoo.com/quote/{ticker}"   # ticker IS the Yahoo symbol
    if kind == "CRYPTO":
        tv_sym = ticker.replace("-USD", "USD").replace("-", "")  # BTC-USD -> BTCUSD
    else:
        tv_sym = ticker
    tradingview = f"https://www.tradingview.com/symbols/{tv_sym}/"
    return yahoo, tradingview


def _telegram_ready():
    return "YOUR_" not in BOT_TOKEN and bool(CHAT_IDS)


def send_telegram_alert(message):
    if not _telegram_ready():
        return  # not configured yet
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    for chat_id in CHAT_IDS:
        try:
            r = requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=15)
            if not r.ok:
                print(f"  Telegram error to {chat_id}: {r.status_code} {r.text}")
        except Exception as e:
            print(f"  Error sending message to {chat_id}: {e}")


def send_email_report(subject, body, attachments):
    if not EMAIL_ENABLED:
        return
    if "your_16_char" in EMAIL_APP_PASSWORD:
        print("  Email skipped: app password not set.")
        return
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg.set_content(body)
    for path in attachments:
        try:
            with open(path, "rb") as f:
                msg.add_attachment(f.read(), maintype="image", subtype="png",
                                   filename=os.path.basename(path))
        except Exception as e:
            print(f"  Could not attach {path}: {e}")
    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30) as smtp:
            smtp.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
            smtp.send_message(msg)
        print(f"  Email sent to {EMAIL_TO}")
    except Exception as e:
        print(f"  Email error: {e}")


def send_telegram_photo(image_path, caption):
    if not _telegram_ready():
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    for chat_id in CHAT_IDS:
        try:
            with open(image_path, "rb") as f:
                r = requests.post(url, data={"chat_id": chat_id, "caption": caption},
                                  files={"photo": f}, timeout=30)
            if not r.ok:
                print(f"  Telegram photo error to {chat_id}: {r.status_code} {r.text}")
        except Exception as e:
            print(f"  Error sending photo to {chat_id}: {e}")


# ---------------------------------------------------------------------------
# SIGNAL LOG (so we can later track what the price did after each alert)
# ---------------------------------------------------------------------------
# Every alert appends one row here; score_signals.py reads it back, looks up the
# price afterwards, and works out whether each signal rose or fell. The bots are
# stateless, so this file is committed back to the repo by the workflow -- that
# is what gives the signals a memory across runs.
SIGNALS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signals.csv")
SIGNAL_FIELDS = [
    "signal_id", "bot", "alert_date", "candle_date", "kind", "direction", "ticker",
    "timeframe", "entry_close", "stop_level",
]


def _existing_signal_ids():
    """IDs already in signals.csv, so the same candle is never logged twice."""
    ids = set()
    try:
        with open(SIGNALS_CSV, newline="") as f:
            for row in csv.DictReader(f):
                ids.add(row.get("signal_id", ""))
    except FileNotFoundError:
        pass
    return ids


def _infer_bot(kind, timeframe):
    """Best guess at which bot produced an old row that predates the `bot` column,
    from its kind + timeframe. Stock cap tiers are indistinguishable, so unknown."""
    tf = (timeframe or "").upper()
    if kind == "CRYPTO":
        if tf in {"6H", "8H", "12H", "1D", "2D", "3D", "4D"}:
            return "crypto_intraday"
        if tf in {"1W", "1M"}:
            return "crypto_wm"
    return "unknown"


def _migrate_signals_schema():
    """Keep signals.csv on the current column set. If an older file is missing
    newer columns (e.g. it predates `bot`), rewrite it with the full header --
    back-filling `bot` where it can be inferred -- so appended rows stay aligned
    instead of silently shifting into the wrong columns."""
    if not os.path.exists(SIGNALS_CSV):
        return
    with open(SIGNALS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == SIGNAL_FIELDS:
            return  # already current -- nothing to do
        rows = list(reader)
    for r in rows:
        if not r.get("bot"):
            r["bot"] = _infer_bot(r.get("kind", ""), r.get("timeframe", ""))
        if not r.get("direction"):
            # Every signal logged before this column existed was the bullish grab.
            r["direction"] = "long"
    with open(SIGNALS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SIGNAL_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in SIGNAL_FIELDS})
    print("  (migrated signals.csv to the current column layout)")


def log_signal(kind, ticker, timeframe, df, bot="scan_all", direction="long"):
    """Record one match in signals.csv: its entry (close) price and the candle it
    formed in, keyed by a unique signal_id. `bot` names which of the scanners
    found it (e.g. stock_mega, crypto_intraday) so per-bot totals can be tallied.
    `direction` is "long" (bullish grab) or "short" (bearish breakdown).
    Re-logging the same candle is a no-op, so re-runs and overlapping schedules
    can't create duplicates."""
    try:
        candle_date = pd.Timestamp(df.index[-1]).date().isoformat()
        entry_close = float(df["Close"].iloc[-1])
        if direction == "short":
            # Short stop = the highest high of the two pattern candles -- the swept
            # high. If price climbs back above it the breakdown has failed.
            stop_level = float(max(df["High"].iloc[-2], df["High"].iloc[-1]))
        else:
            # Long stop = the lowest low of the two pattern candles. The pattern
            # requires candle 2 to swallow candle 1's low, so this is normally c2's
            # low -- the price the trade is invalidated at if the market retreats.
            stop_level = float(min(df["Low"].iloc[-2], df["Low"].iloc[-1]))
    except Exception as e:
        print(f"    (could not log signal for {ticker}: {e})")
        return
    # Long IDs keep their original shape so historical rows never re-log; shorts
    # get a SHORT_ namespace so a green and red signal on the same candle coexist.
    prefix = "SHORT_" if direction == "short" else ""
    signal_id = f"{prefix}{kind}_{ticker}_{timeframe}_{candle_date}"
    _migrate_signals_schema()   # align an older file before we append to it
    if signal_id in _existing_signal_ids():
        return
    new_file = not os.path.exists(SIGNALS_CSV)
    with open(SIGNALS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SIGNAL_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({
            "signal_id": signal_id,
            "bot": bot,
            "alert_date": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
            "candle_date": candle_date,
            "kind": kind,
            "direction": direction,
            "ticker": ticker,
            "timeframe": timeframe,
            "entry_close": f"{entry_close:.10g}",
            "stop_level": f"{stop_level:.10g}",
        })
    print(f"    logged signal {signal_id} @ {entry_close:.6g}")


# ---------------------------------------------------------------------------
# CHARTS
# ---------------------------------------------------------------------------
def _safe_name(text):
    return "".join(c if c.isalnum() else "_" for c in text)


def save_chart(ticker, kind, df, timeframe="WEEKLY", bars=40, direction="long"):
    """Save a candlestick PNG with the two pattern candles highlighted."""
    label = COMMODITIES.get(ticker, ticker)          # show "Gold (GLD)" instead of a symbol
    candle = df.index[-1].date()                     # the candle this match is in
    # One folder per timeframe per candle date, e.g. charts\WEEKLY_2026-06-08
    out_dir = os.path.join(CHART_DIR, f"{timeframe}_{candle}")
    os.makedirs(out_dir, exist_ok=True)
    plot_df = df.tail(bars)
    pattern_dates = [df.index[-2], df.index[-1]]  # the two candles that triggered
    # Keep long filenames byte-identical to before; tag shorts so the two don't collide.
    tag = "SHORT_" if direction == "short" else ""
    out = os.path.join(out_dir, f"{timeframe}_{tag}{kind}_{_safe_name(label)}_{candle}.png")
    mpf.plot(
        plot_df,
        type="candle",
        style="charles",
        title=f"\n{label} ({kind}) - {timeframe} {pattern_name(direction)}",
        ylabel="Price",
        vlines=dict(vlines=pattern_dates, colors="royalblue", alpha=0.25, linewidths=10),
        savefig=dict(fname=out, dpi=120, bbox_inches="tight"),
    )
    return out


# ---------------------------------------------------------------------------
# SCAN
# ---------------------------------------------------------------------------
def scan(tickers, label, interval, period, closed_only=False):
    matches, scanned = [], 0
    for batch in chunked(tickers, BATCH_SIZE):
        data = yf.download(batch, period=period, interval=interval,
                           group_by="ticker", auto_adjust=True,
                           threads=True, progress=False)
        for t in batch:
            try:
                df = data[t] if len(batch) > 1 else data
                df = df.dropna()
                if closed_only and len(df) > 0:
                    df = df.iloc[:-1]   # drop the still-forming candle
                if df.empty:
                    continue
                scanned += 1
                if is_flat(df):
                    continue          # pegged stablecoin -- noise, skip it
                # A candle is either green or red, so at most one of these fires.
                if check_pattern(df):
                    matches.append((t, df.copy(), "long"))
                elif check_pattern_bearish(df):
                    matches.append((t, df.copy(), "short"))
            except (KeyError, IndexError):
                continue
    print(f"[{label}] scanned {scanned}/{len(tickers)} (rest had no Yahoo data)")
    return matches


def _last_monthly_run():
    try:
        with open(STATE_FILE) as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def _mark_monthly_run(month_key):
    with open(STATE_FILE, "w") as f:
        f.write(month_key)


def run_timeframe(tf, groups):
    """Scan every asset group on one timeframe; return list of
    (kind, ticker, df, direction)."""
    print(f"\n--- {tf['label']} ({tf['interval']}) ---")
    found = []
    for kind, tickers, label in groups:
        results = scan(tickers, label, tf["interval"], tf["period"], tf["closed_only"])
        found += [(kind, t, df, direction) for t, df, direction in results]
    return found


def main():
    print("Building ticker universe...")
    sp500 = get_sp500()
    ndx = get_nasdaq100()
    stocks = sorted(set(sp500) | set(ndx))
    print(f"  Stocks: {len(sp500)} S&P 500 + {len(ndx)} Nasdaq-100 = {len(stocks)} unique")

    crypto = get_top_crypto(CRYPTO_TOP_N)
    print(f"  Crypto: {len(crypto)} from top {CRYPTO_TOP_N} (stablecoins dropped)")

    commodities = list(COMMODITIES.keys())
    print(f"  Commodities (ETFs): {len(commodities)}")

    groups = [
        ("STOCK", stocks, "Stocks"),
        ("CRYPTO", crypto, "Crypto"),
        ("COMMODITY", commodities, "Commodities"),
    ]

    # SCAN_MODE lets the cloud run weekly and monthly on separate schedules.
    #   "weekly"  -> only the weekly timeframe
    #   "monthly" -> only the monthly timeframe
    #   unset     -> local mode: weekly every time, monthly once per new month
    mode = (os.environ.get("SCAN_MODE") or "").lower()
    this_month = datetime.date.today().strftime("%Y-%m")
    monthly_due_local = _last_monthly_run() != this_month

    all_matches = []  # (kind, ticker, df, timeframe_label, direction)
    for tf in TIMEFRAMES:
        is_monthly = tf["monthly_only"]
        if mode == "weekly":
            run = not is_monthly
        elif mode == "monthly":
            run = is_monthly
        else:  # local auto mode
            run = (not is_monthly) or monthly_due_local
        if not run:
            if is_monthly:
                print(f"\n--- {tf['label']} skipped (already scanned for {this_month}) ---")
            continue
        found = run_timeframe(tf, groups)
        all_matches += [(kind, t, df, tf["label"], direction) for kind, t, df, direction in found]
        if is_monthly and mode != "weekly":
            _mark_monthly_run(this_month)  # remember we did the monthly pass

    print("\n" + "=" * 40)
    if all_matches:
        print(f"MATCHES FOUND: {len(all_matches)}")
        chart_folders = set()
        chart_paths = []
        email_lines = []
        for kind, t, df, tf_label, direction in all_matches:
            yahoo, tradingview = chart_links(kind, t)
            msg = (f"[{tf_label}] MATCH: {COMMODITIES.get(t, t)} ({kind}) "
                   f"formed your {pattern_name(direction)} pattern!\n"
                   f"Yahoo: {yahoo}\nTradingView: {tradingview}")
            print("  " + msg)
            email_lines.append(msg)
            log_signal(kind, t, tf_label, df, direction=direction)
            try:
                chart_path = save_chart(t, kind, df, tf_label, direction=direction)
                print(f"    chart -> {chart_path}")
                chart_folders.add(os.path.dirname(chart_path))
                chart_paths.append(chart_path)
                send_telegram_photo(chart_path, msg)
            except Exception as e:
                print(f"    (could not draw chart: {e})")
                send_telegram_alert(msg)
        # Email one summary with all charts attached.
        today = datetime.date.today()
        send_email_report(
            subject=f"Liquidity Grab scan {today}: {len(all_matches)} match(es)",
            body="Matches found:\n\n" + "\n".join(email_lines) + "\n\n(charts attached)",
            attachments=chart_paths,
        )
        # Pop the chart folder(s) open in File Explorer (local Windows only;
        # skipped on the cloud, which has no desktop).
        if os.name == "nt" and not os.environ.get("CI"):
            for folder in chart_folders:
                print(f"\nOpening charts folder: {folder}")
                try:
                    os.startfile(folder)
                except Exception as e:
                    print(f"  (could not auto-open folder: {e})")
    else:
        print("No matches this scan.")
        send_email_report(
            subject=f"Liquidity Grab scan {datetime.date.today()}: no matches",
            body="No matches this scan.",
            attachments=[],
        )
    print("=" * 40)


if __name__ == "__main__":
    main()
