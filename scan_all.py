"""
Weekly Liquidity-Grab scanner across the S&P 500, Nasdaq-100, and top-200 crypto.

Run it manually whenever you want (e.g. once a week after the weekly candle closes):
    python scan_all.py

It fetches the ticker universe automatically, downloads candles in batches,
runs your pattern check on each, and prints every match. Fill in BOT_TOKEN /
CHAT_ID below if you also want Telegram alerts.
"""

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
BOT_TOKEN = os.environ.get("BOT_TOKEN") or "8853901166:AAFXsym_oE-8rzDN6pEKBS5hT8lh9_hnAWM"
CHAT_ID = os.environ.get("CHAT_ID") or "7788611624"

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
# YOUR PATTERN (unchanged)
# ---------------------------------------------------------------------------
def check_pattern(df):
    if df is None or len(df) < 2:
        return False

    c1_open, c1_close, c1_high, c1_low = df["Open"].iloc[-2], df["Close"].iloc[-2], df["High"].iloc[-2], df["Low"].iloc[-2]
    c2_open, c2_close, c2_high, c2_low = df["Open"].iloc[-1], df["Close"].iloc[-1], df["High"].iloc[-1], df["Low"].iloc[-1]

    if pd.isna([c1_open, c1_close, c2_open, c2_close]).any():
        return False

    c1_green = c1_close > c1_open
    c2_green = c2_close > c2_open

    body_one = c1_close - c1_open
    upper_tail_one = c1_high - c1_close
    long_upper_tail = upper_tail_one > 0
    small_body_one = body_one < (c1_high - c1_low) * 0.7

    body_two = c2_close - c2_open
    upper_tail_two = c2_high - c2_close
    strong_close = body_two > upper_tail_two
    engulfs_body = c2_close > c1_close and c2_open <= c1_close

    bottom_swallow = c2_low < c1_low

    return c1_green and c2_green and long_upper_tail and small_body_one and strong_close and engulfs_body and bottom_swallow


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


def get_top_crypto(n=200):
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
    # Yahoo uses SYMBOL-USD (e.g. BTC-USD). Stablecoins are useless for this pattern.
    stable = {"USDT", "USDC", "DAI", "BUSD", "TUSD", "USDE", "FDUSD", "USDS", "PYUSD"}
    seen, tickers = set(), []
    for c in out[:n]:
        sym = c["symbol"].upper()
        if sym in stable or sym in seen:
            continue
        seen.add(sym)
        tickers.append(f"{sym}-USD")
    return tickers


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
    return "YOUR_" not in BOT_TOKEN and "YOUR_" not in CHAT_ID


def send_telegram_alert(message):
    if not _telegram_ready():
        return  # not configured yet
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": CHAT_ID, "text": message}, timeout=15)
        if not r.ok:
            print(f"  Telegram error {r.status_code}: {r.text}")
    except Exception as e:
        print(f"  Error sending message: {e}")


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
    try:
        with open(image_path, "rb") as f:
            r = requests.post(url, data={"chat_id": CHAT_ID, "caption": caption},
                              files={"photo": f}, timeout=30)
        if not r.ok:
            print(f"  Telegram photo error {r.status_code}: {r.text}")
    except Exception as e:
        print(f"  Error sending photo: {e}")


# ---------------------------------------------------------------------------
# CHARTS
# ---------------------------------------------------------------------------
def _safe_name(text):
    return "".join(c if c.isalnum() else "_" for c in text)


def save_chart(ticker, kind, df, timeframe="WEEKLY", bars=40):
    """Save a candlestick PNG with the two pattern candles highlighted."""
    label = COMMODITIES.get(ticker, ticker)          # show "Gold (GLD)" instead of a symbol
    candle = df.index[-1].date()                     # the candle this match is in
    # One folder per timeframe per candle date, e.g. charts\WEEKLY_2026-06-08
    out_dir = os.path.join(CHART_DIR, f"{timeframe}_{candle}")
    os.makedirs(out_dir, exist_ok=True)
    plot_df = df.tail(bars)
    pattern_dates = [df.index[-2], df.index[-1]]  # the two candles that triggered
    out = os.path.join(out_dir, f"{timeframe}_{kind}_{_safe_name(label)}_{candle}.png")
    mpf.plot(
        plot_df,
        type="candle",
        style="charles",
        title=f"\n{label} ({kind}) - {timeframe} Liquidity Grab",
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
                if check_pattern(df):
                    matches.append((t, df.copy()))
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
    """Scan every asset group on one timeframe; return list of (kind, ticker, df)."""
    print(f"\n--- {tf['label']} ({tf['interval']}) ---")
    found = []
    for kind, tickers, label in groups:
        results = scan(tickers, label, tf["interval"], tf["period"], tf["closed_only"])
        found += [(kind, t, df) for t, df in results]
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

    all_matches = []  # (kind, ticker, df, timeframe_label)
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
        all_matches += [(kind, t, df, tf["label"]) for kind, t, df in found]
        if is_monthly and mode != "weekly":
            _mark_monthly_run(this_month)  # remember we did the monthly pass

    print("\n" + "=" * 40)
    if all_matches:
        print(f"MATCHES FOUND: {len(all_matches)}")
        chart_folders = set()
        chart_paths = []
        email_lines = []
        for kind, t, df, tf_label in all_matches:
            yahoo, tradingview = chart_links(kind, t)
            msg = (f"[{tf_label}] MATCH: {COMMODITIES.get(t, t)} ({kind}) "
                   f"formed your Liquidity Grab pattern!\n"
                   f"Yahoo: {yahoo}\nTradingView: {tradingview}")
            print("  " + msg)
            email_lines.append(msg)
            try:
                chart_path = save_chart(t, kind, df, tf_label)
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
