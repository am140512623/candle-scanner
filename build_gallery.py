"""
Build an interactive visual-review page from results.csv.

This is the YEAR-END review tool. The scanners archive each signal's chart and
record its path in the `chart` column; the score job carries that column into
results.csv alongside the win/loss verdict. This script turns that into a single
self-contained gallery.html where you:

    * see every signal as a PICTURE next to its data (ticker, timeframe, date,
      bot, result, realized %),
    * flip a Keep / Delete toggle on any signal whose chart fails your visual check,
    * watch the summary at the top (win rate, avg win, net %) recompute LIVE as you
      exclude the ones that don't pass -- nothing is deleted from any CSV,
    * filter by bot / timeframe / result, and
    * download the kept set as a CSV if you want a record of the filtered result.

The images are referenced by their repo-relative path (the `chart` column), so open
gallery.html from a checkout that also has the charts/ folder (i.e. the private data
repo) and the pictures render -- offline, no login.

Run:  python build_gallery.py [results.csv] [gallery.html]
Env:  GALLERY_TITLE  (page heading, optional)
"""

import csv
import json
import os
import sys

RESULTS_CSV = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
OUT_HTML = sys.argv[2] if len(sys.argv) > 2 else "gallery.html"
TITLE = os.environ.get("GALLERY_TITLE", "Signal review — keep or delete")


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build(rows):
    # Only carry the fields the page needs; keep the payload small.
    keep_fields = ["signal_id", "bot", "ticker", "timeframe", "candle_date",
                   "direction", "entry_close", "stop_level", "chart",
                   "realized_pct", "best_target_pct", "stop_chg", "result", "status"]
    data = [{k: r.get(k, "") for k in keep_fields} for r in rows]
    payload = json.dumps(data)
    return HTML_TEMPLATE.replace("__TITLE__", TITLE).replace("__DATA__", payload)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.4 system-ui, sans-serif; background: #0f1115; color: #e6e6e6; }
  header { position: sticky; top: 0; z-index: 5; background: #171a21; border-bottom: 1px solid #2a2f3a;
           padding: 12px 16px; box-shadow: 0 2px 8px rgba(0,0,0,.3); }
  h1 { margin: 0 0 8px; font-size: 16px; }
  .stats { display: flex; flex-wrap: wrap; gap: 18px; align-items: baseline; }
  .stat b { font-size: 20px; }
  .stat span { color: #9aa4b2; font-size: 12px; margin-left: 4px; }
  .win { color: #4ade80; } .loss { color: #f87171; }
  .controls { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; align-items: center; }
  select, button { background: #222834; color: #e6e6e6; border: 1px solid #38404e; border-radius: 6px;
                   padding: 5px 9px; font: inherit; }
  button { cursor: pointer; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 14px; padding: 16px; }
  .card { background: #171a21; border: 1px solid #2a2f3a; border-radius: 10px; overflow: hidden; display: flex; flex-direction: column; }
  .card.excluded { opacity: .38; filter: grayscale(.7); }
  .card img { width: 100%; display: block; background: #000; aspect-ratio: 3/2; object-fit: contain; }
  .card .nopic { display: flex; align-items: center; justify-content: center; height: 160px; color: #6b7280; background: #12151b; }
  .meta { padding: 9px 11px; display: flex; flex-direction: column; gap: 4px; }
  .row1 { display: flex; justify-content: space-between; align-items: baseline; }
  .tk { font-weight: 700; font-size: 15px; }
  .tag { font-size: 11px; color: #9aa4b2; }
  .verdict { font-size: 12px; font-weight: 600; }
  .foot { display: flex; justify-content: space-between; align-items: center; padding: 8px 11px; border-top: 1px solid #262b35; }
  label.kd { display: flex; gap: 6px; align-items: center; cursor: pointer; user-select: none; }
  .sid { font-size: 10px; color: #5b6472; word-break: break-all; }
</style></head><body>
<header>
  <h1>__TITLE__</h1>
  <div class="stats">
    <div class="stat"><b id="s-kept">0</b><span>kept</span></div>
    <div class="stat"><b id="s-win" class="win">0</b><span>wins</span></div>
    <div class="stat"><b id="s-loss" class="loss">0</b><span>losses</span></div>
    <div class="stat"><b id="s-wr">0%</b><span>win rate</span></div>
    <div class="stat"><b id="s-avg">0%</b><span>avg win</span></div>
    <div class="stat"><b id="s-net">0%</b><span>net (win% + stop%)</span></div>
    <div class="stat"><b id="s-excl">0</b><span>excluded</span></div>
  </div>
  <div class="controls">
    <select id="f-bot"><option value="">All bots</option></select>
    <select id="f-tf"><option value="">All timeframes</option></select>
    <select id="f-res">
      <option value="">All results</option>
      <option value="took_profit">Wins</option>
      <option value="stopped_out">Losses</option>
      <option value="open">Open/other</option>
    </select>
    <button id="only-scored">Only scored (win/loss)</button>
    <button id="dl">Download kept as CSV</button>
    <span class="tag" id="shown"></span>
  </div>
</header>
<div class="grid" id="grid"></div>
<script>
const DATA = __DATA__;
const excluded = new Set();      // signal_ids the reviewer has deleted
let onlyScored = false;

const isWin = r => r.result === "took_profit";
const isLoss = r => r.result === "stopped_out";
const num = v => { const n = parseFloat(v); return isNaN(n) ? 0 : n; };

function visible() {
  const b = document.getElementById("f-bot").value;
  const t = document.getElementById("f-tf").value;
  const rr = document.getElementById("f-res").value;
  return DATA.filter(r =>
    (!b || r.bot === b) && (!t || r.timeframe === t) &&
    (!rr || r.result === rr) &&
    (!onlyScored || isWin(r) || isLoss(r)));
}

function recompute() {
  // Stats are over the KEPT + scored signals across the WHOLE set (not just the
  // current filter view), so filtering never changes the headline result.
  const kept = DATA.filter(r => !excluded.has(r.signal_id));
  const wins = kept.filter(isWin), losses = kept.filter(isLoss);
  const wr = (wins.length + losses.length) ? 100 * wins.length / (wins.length + losses.length) : 0;
  const avg = wins.length ? wins.reduce((a, r) => a + num(r.realized_pct), 0) / wins.length : 0;
  const net = wins.reduce((a, r) => a + num(r.realized_pct), 0) + losses.reduce((a, r) => a + num(r.stop_chg), 0);
  document.getElementById("s-kept").textContent = kept.length;
  document.getElementById("s-win").textContent = wins.length;
  document.getElementById("s-loss").textContent = losses.length;
  document.getElementById("s-wr").textContent = wr.toFixed(1) + "%";
  document.getElementById("s-avg").textContent = avg.toFixed(1) + "%";
  const netEl = document.getElementById("s-net");
  netEl.textContent = (net >= 0 ? "+" : "") + net.toFixed(0) + "%";
  netEl.className = net >= 0 ? "win" : "loss";
  document.getElementById("s-excl").textContent = excluded.size;
}

function verdictHtml(r) {
  if (isWin(r)) return `<span class="verdict win">WIN ${r.realized_pct||""}%</span>`;
  if (isLoss(r)) return `<span class="verdict loss">LOSS ${r.stop_chg||""}%</span>`;
  return `<span class="verdict" style="color:#9aa4b2">${r.result||r.status||"—"}</span>`;
}

function render() {
  const grid = document.getElementById("grid");
  const rows = visible();
  grid.innerHTML = "";
  for (const r of rows) {
    const card = document.createElement("div");
    card.className = "card" + (excluded.has(r.signal_id) ? " excluded" : "");
    const pic = r.chart
      ? `<img loading="lazy" src="${r.chart}" alt="${r.ticker}" onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'nopic',textContent:'chart not found: '+this.getAttribute('src')}))">`
      : `<div class="nopic">no chart archived</div>`;
    card.innerHTML = pic + `
      <div class="meta">
        <div class="row1"><span class="tk">${r.ticker}</span>${verdictHtml(r)}</div>
        <div class="tag">${r.bot} · ${r.timeframe} · ${r.candle_date} · entry ${r.entry_close}</div>
        <div class="sid">${r.signal_id}</div>
      </div>
      <div class="foot">
        <label class="kd"><input type="checkbox" ${excluded.has(r.signal_id) ? "" : "checked"}
          data-id="${r.signal_id}"> Keep</label>
      </div>`;
    card.querySelector("input").addEventListener("change", e => {
      const id = e.target.dataset.id;
      if (e.target.checked) excluded.delete(id); else excluded.add(id);
      card.classList.toggle("excluded", !e.target.checked);
      recompute();
    });
    grid.appendChild(card);
  }
  document.getElementById("shown").textContent = rows.length + " shown";
}

function fillSelect(id, key) {
  const el = document.getElementById(id);
  [...new Set(DATA.map(r => r[key]).filter(Boolean))].sort()
    .forEach(v => el.add(new Option(v, v)));
}

function downloadKept() {
  const kept = DATA.filter(r => !excluded.has(r.signal_id));
  if (!kept.length) return;
  const cols = Object.keys(kept[0]);
  const esc = v => `"${String(v).replace(/"/g, '""')}"`;
  const csv = [cols.join(",")].concat(kept.map(r => cols.map(c => esc(r[c])).join(","))).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.download = "kept_signals.csv";
  a.click();
}

["f-bot", "f-tf", "f-res"].forEach(id => document.getElementById(id).addEventListener("change", render));
document.getElementById("only-scored").addEventListener("click", e => {
  onlyScored = !onlyScored;
  e.target.style.background = onlyScored ? "#2b5" : "";
  render();
});
document.getElementById("dl").addEventListener("click", downloadKept);
fillSelect("f-bot", "bot");
fillSelect("f-tf", "timeframe");
recompute();
render();
</script></body></html>"""


def main():
    if not os.path.exists(RESULTS_CSV):
        sys.exit(f"No {RESULTS_CSV} found. Run score_signals.py first (or pass a path).")
    rows = load_rows(RESULTS_CSV)
    html = build(rows)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    scored = sum(1 for r in rows if r.get("result") in ("took_profit", "stopped_out"))
    withpic = sum(1 for r in rows if r.get("chart"))
    print(f"Wrote {OUT_HTML}: {len(rows)} signals ({scored} scored, {withpic} with a chart).")
    print("Open it from a folder that also has the charts/ tree so the pictures render.")


if __name__ == "__main__":
    main()
