"""
Shared chart styling for the Bollinger bots, so the pictures tell you WHICH signal
fired without having to read the filename.

Both Bollinger signals used to be drawn with the same teal highlight -- the charts
were indistinguishable. Now two independent things are encoded:

    SHAPE  = which band the grab touched
        ▲  upper band  (the breakout / "touch up")
        ▼  lower band  (the "touch down")

    COLOUR = which signal it is
        white  plain upper-band breakout grab      (scan_bb_grab)
        green  plain lower-band touch grab         (scan_bb_lower_touch)
        gold   grab → reverse candle → reclaim     (scan_indices, no band involved)
        blue   sharp drop → higher low → peak break (index_signals, no band)

IMPORTANT: all of these are LONG (BUY) setups. On the Bollinger charts the arrow
says which BAND was touched -- it is NOT the trade direction, and a ▼ is still a
buy. The chart repeats this in a footnote so a down arrow is never misread as a
sell. The index pattern has no band, so its ▲ just means long.
"""

import matplotlib
matplotlib.use("Agg")           # headless: these run in CI, never on a desktop

import matplotlib.pyplot as plt
import mplfinance as mpf

WHITE = "#ffffff"
GREEN = "#2eaa48"
GOLD  = "#d4a017"
GRAY  = "#9aa0a6"               # stands in for white where white would vanish
RED   = "#c0392b"
BLUE  = "#2f6fd0"

# One entry per id-prefix used by the bots.
#   marker/face/edge/text -> the signal marker under the signal candle + its badge
#   shade                 -> the vertical highlight (white is invisible on the
#                            chart's white background, so ① shades gray instead)
# The band arrow is easy to misread as a sell, so the Bollinger charts spell it
# out. The index pattern has no band, so it gets its own note instead.
_BAND_NOTE = ("LONG (BUY) setup. The arrow shows which BAND was touched, "
              "NOT the trade direction — a ▼ is still a buy.")
_IDX_NOTE  = ("LONG (BUY) setup. Entry = close of the SIGNAL candle, the first to "
              "close back above the REVERSE candle's open.")
# The drop is the SETUP, not the current move -- reading "drop size" as "the market
# is down that much right now" is the easy mistake, so the picture says otherwise.
_SD_NOTE   = ("LONG (BUY) setup. The DROP TOP → BOTTOM fall is the SETUP that already "
              "happened; entry is the SIGNAL candle closing above the PEAK.")

STYLES = {
    "BBGRAB_": {
        "tag": "① UPPER-BAND TOUCH ▲ → GRAB",
        "marker": "^", "face": WHITE, "edge": "black", "text": "black",
        "shade": GRAY, "grab_shade": GRAY, "note": _BAND_NOTE,
        "level_label": "LEVEL",
    },
    "BBLOWER_": {
        "tag": "② LOWER-BAND TOUCH ▼ → GRAB",
        "marker": "v", "face": GREEN, "edge": "black", "text": "white",
        "shade": GREEN, "grab_shade": GREEN, "note": _BAND_NOTE,
        "level_label": "LEVEL",
    },
    # The standalone pattern on the US index bot. No band is involved, so the
    # shape just means LONG here rather than encoding which band was touched.
    "IDXRC_": {
        "tag": "GRAB ▲ → REVERSE CANDLE → RECLAIM  (LONG)",
        "marker": "^", "face": GOLD, "edge": "black", "text": "black",
        "shade": GOLD, "grab_shade": GRAY, "note": _IDX_NOTE,
        "level_label": "RECLAIM LEVEL (the reverse candle's open)",
    },
    # The US index bot's own pattern (index_signals). Five candles matter here
    # rather than three, so it colours each role separately instead of reusing the
    # single shade -- see `role_colors`.
    "IDXSD_": {
        "tag": "SHARP DROP → HIGHER LOW → PEAK BREAK ▲  (LONG)",
        "marker": "^", "face": BLUE, "edge": "black", "text": "white",
        "shade": BLUE, "grab_shade": GRAY, "note": _SD_NOTE,
        "level_label": "ENTRY — the PEAK this candle closed above",
        "role_colors": {"DROP TOP": GRAY, "BOTTOM": RED, "PEAK": GOLD,
                        "HIGHER LOW": GREEN, "SIGNAL": BLUE},
        "role_text": {"BOTTOM": "white", "SIGNAL": "white"},
    },
}


def band_addplots(plot_df, bollinger):
    """The three Bollinger lines, drawn the same way on every chart."""
    basis, upper, lower = bollinger(plot_df["Close"])
    return [
        mpf.make_addplot(upper, color="crimson", width=1.1),
        mpf.make_addplot(basis, color="darkorange", width=1.0, linestyle="--"),
        mpf.make_addplot(lower, color="royalblue", width=1.1),
    ]


# Where each role's label sits, in points below the candle's low. Each role gets
# its OWN row: the pattern candles are often adjacent, and a label box is far wider
# than one candle, so same-row boxes would collide.
ROW = {"GRAB": -16, "REVERSE": -34, "SIGNAL": -58,
       # index_signals' five roles: neighbours alternate rows so their boxes,
       # which are wider than a candle, never overlap.
       "DROP TOP": -16, "BOTTOM": -34, "PEAK": -16, "HIGHER LOW": -34}


def _role_color(role, style):
    override = style.get("role_colors", {}).get(role)
    if override:
        return override
    if role == "REVERSE":
        return RED
    if role == "GRAB":
        return style["grab_shade"]
    return style["shade"]


def _role_text_color(role, style):
    """Label text colour. Styles that need it override per role; everything else
    keeps the original rule (only the red REVERSE box needs white)."""
    return style.get("role_text", {}).get(role) or ("white" if role == "REVERSE"
                                                    else "black")


def _fit_labels_below(fig, ax, plot_df, shown):
    """Grow the y-axis downwards if a role label would hang off the bottom.

    The labels are offset in POINTS below each candle's low, so a role sitting on
    the chart's own lowest low (BOTTOM, always) lands on the date axis. Work out
    how far each box reaches in data units and lower the limit only if needed --
    charts whose labels already fit are left exactly as they were."""
    if not shown:
        return
    lo, hi = ax.get_ylim()
    ax_in = fig.get_size_inches()[1] * ax.get_position().height
    if ax_in <= 0:
        return
    per_pt = (hi - lo) / (ax_in * 72.0)         # data units per typographic point
    need = min(plot_df["Low"].loc[ts] - (abs(ROW.get(role, -16)) + 12) * per_pt
               for ts, role in shown.items())
    if need < lo:
        ax.set_ylim(need, hi)


def render(plot_df, aps, title, out, prefix, roles, hline=None):
    """Draw and save one signal chart.

    `roles` maps a timestamp -> that candle's part in the pattern ("GRAB",
    "OPPOSITE", "SIGNAL"). Timestamps outside the plotted window are skipped.
    `hline` is an optional price level to run across (the reclaimed level).
    """
    style = STYLES[prefix]
    shown = {ts: r for ts, r in roles.items() if ts in plot_df.index}

    kw = dict(
        type="candle",
        style="charles",
        title=f"\n{title}",
        ylabel="Price",
        returnfig=True,
    )
    if aps:                     # the index pattern draws no bands -> no addplots
        kw["addplot"] = aps
    if shown:
        kw["vlines"] = dict(vlines=list(shown),
                            colors=[_role_color(r, style) for r in shown.values()],
                            alpha=0.28, linewidths=9)
    if hline is not None:
        # SOLID, not dashed: the basis is already an orange DASHED line, and a gold
        # dashed level next to it was impossible to tell apart.
        kw["hlines"] = dict(hlines=[hline], colors=[style["shade"]],
                            linestyle="-", linewidths=1.8)

    fig, axes = mpf.plot(plot_df, **kw)
    ax = axes[0]

    # The badge: which of the signals this picture is.
    ax.text(0.012, 0.975, style["tag"], transform=ax.transAxes,
            va="top", ha="left", fontsize=9.5, fontweight="bold",
            color=style["text"], zorder=11,
            bbox=dict(facecolor=style["face"], edgecolor=style["edge"],
                      linewidth=1.0, boxstyle="round,pad=0.45"))

    if hline is not None:
        ax.annotate(style["level_label"],
                    xy=(0.015, hline), xycoords=("axes fraction", "data"),
                    va="bottom", ha="left", fontsize=7, fontweight="bold",
                    color="black", zorder=11,
                    bbox=dict(facecolor=style["shade"], edgecolor="black",
                              linewidth=0.6, boxstyle="round,pad=0.25"))

    # The big arrow under the signal candle: shape = band, colour = signal.
    span = plot_df["High"].max() - plot_df["Low"].min()
    for ts, role in shown.items():
        x = plot_df.index.get_loc(ts)
        if role == "SIGNAL":
            ax.plot([x], [plot_df["Low"].loc[ts] - span * 0.045],
                    marker=style["marker"], markersize=15,
                    markerfacecolor=style["face"], markeredgecolor=style["edge"],
                    markeredgewidth=1.2, linestyle="None", clip_on=False, zorder=11)
        # Name each candle's part in the pattern, under the candle itself.
        ax.annotate(role, xy=(x, plot_df["Low"].loc[ts]),
                    xytext=(0, ROW.get(role, -16)),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=7, fontweight="bold", zorder=11,
                    color=_role_text_color(role, style),
                    bbox=dict(facecolor=_role_color(role, style),
                              edgecolor="black", linewidth=0.6,
                              boxstyle="round,pad=0.25"))

    _fit_labels_below(fig, ax, plot_df, shown)

    ax.text(0.5, -0.17, style["note"], transform=ax.transAxes, ha="center",
            va="top", fontsize=7.5, color="#555555", style="italic")

    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out
