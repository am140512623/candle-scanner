"""
Shared chart styling for the Bollinger bots, so the pictures tell you WHICH signal
fired without having to read the filename.

Every signal used to be drawn with the same teal highlight -- the charts were
indistinguishable. Now two independent things are encoded, so any combination is
readable at a glance:

    SHAPE  = which band the grab touched
        ▲  upper band  (the breakout / "touch up")
        ▼  lower band  (the "touch down")

    COLOUR = which signal it is
        white  plain upper-band breakout grab      (scan_bb_grab)
        green  plain lower-band touch grab         (scan_bb_lower_touch)
        gold   grab → opposite candle → reclaim    (bb_reclaim, on either bot)

So a gold ▲ is "breakout grab, then reclaimed", a gold ▼ is "lower-touch grab, then
reclaimed", and the two plain grabs keep white ▲ / green ▼.

IMPORTANT: all of these are LONG (BUY) setups. The arrow says which BAND was
touched -- it is NOT the trade direction. A ▼ is still a buy. The chart repeats this
in a footnote so a down arrow is never misread as a sell.
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

# One entry per id-prefix used by the bots.
#   marker/face/edge/text -> the signal marker under the signal candle + its badge
#   shade                 -> the vertical highlight (white is invisible on the
#                            chart's white background, so ① shades gray instead)
STYLES = {
    "BBGRAB_": {
        "tag": "① UPPER-BAND TOUCH ▲ → GRAB",
        "marker": "^", "face": WHITE, "edge": "black", "text": "black",
        "shade": GRAY, "grab_shade": GRAY,
    },
    "BBLOWER_": {
        "tag": "② LOWER-BAND TOUCH ▼ → GRAB",
        "marker": "v", "face": GREEN, "edge": "black", "text": "white",
        "shade": GREEN, "grab_shade": GREEN,
    },
    "BBGRABRC_": {
        "tag": "③ UPPER-BAND TOUCH ▲ → GRAB → OPPOSITE → RECLAIM",
        "marker": "^", "face": GOLD, "edge": "black", "text": "black",
        "shade": GOLD, "grab_shade": GRAY,     # grab keeps its parent's colour
    },
    "BBLOWERRC_": {
        "tag": "③ LOWER-BAND TOUCH ▼ → GRAB → OPPOSITE → RECLAIM",
        "marker": "v", "face": GOLD, "edge": "black", "text": "black",
        "shade": GOLD, "grab_shade": GREEN,    # grab keeps its parent's colour
    },
}

FOOTNOTE = "All signals are LONG (BUY). The arrow shows which BAND was touched, not the trade direction."


def band_addplots(plot_df, bollinger):
    """The three Bollinger lines, drawn the same way on every chart."""
    basis, upper, lower = bollinger(plot_df["Close"])
    return [
        mpf.make_addplot(upper, color="crimson", width=1.1),
        mpf.make_addplot(basis, color="darkorange", width=1.0, linestyle="--"),
        mpf.make_addplot(lower, color="royalblue", width=1.1),
    ]


def _role_color(role, style):
    if role == "OPPOSITE":
        return RED
    if role == "GRAB":
        return style["grab_shade"]
    return style["shade"]


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
        addplot=aps,
        title=f"\n{title}",
        ylabel="Price",
        returnfig=True,
    )
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
        ax.annotate("RECLAIM LEVEL (opposite candle's open)",
                    xy=(0.015, hline), xycoords=("axes fraction", "data"),
                    va="bottom", ha="left", fontsize=7, fontweight="bold",
                    color="black", zorder=11,
                    bbox=dict(facecolor=style["shade"], edgecolor="black",
                              linewidth=0.6, boxstyle="round,pad=0.25"))

    # The big arrow under the signal candle: shape = band, colour = signal.
    span = plot_df["High"].max() - plot_df["Low"].min()
    # Each role gets its OWN row. The pattern candles are often adjacent, and a
    # label box is far wider than one candle, so same-row boxes would collide.
    ROW = {"GRAB": -16, "OPPOSITE": -34, "SIGNAL": -58}
    for ts, role in shown.items():
        x = plot_df.index.get_loc(ts)
        if role == "SIGNAL":
            ax.plot([x], [plot_df["Low"].loc[ts] - span * 0.045],
                    marker=style["marker"], markersize=15,
                    markerfacecolor=style["face"], markeredgecolor=style["edge"],
                    markeredgewidth=1.2, linestyle="None", clip_on=False, zorder=11)
        # Name each candle's part in the pattern, under the candle itself.
        ax.annotate(role, xy=(x, plot_df["Low"].loc[ts]),
                    xytext=(0, ROW[role]),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=7, fontweight="bold", zorder=11,
                    color="white" if role == "OPPOSITE" else "black",
                    bbox=dict(facecolor=_role_color(role, style),
                              edgecolor="black", linewidth=0.6,
                              boxstyle="round,pad=0.25"))

    ax.text(0.5, -0.115, FOOTNOTE, transform=ax.transAxes, ha="center", va="top",
            fontsize=7.5, color="#555555", style="italic")

    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out
