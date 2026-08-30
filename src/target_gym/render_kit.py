"""Shared control-room rendering toolkit.

Every non-pygame environment draws its frame through this module, so the whole
suite reads as one instrument suite rather than eleven unrelated plots. The
visual language is taken from the nuclear-reactor renderer, which was the
standard the rest were measured against:

* a dark console ground with a cool blue-grey ink,
* a **schematic** of the plant on the left, animated by live state,
* an **instrument stack** on the right -- horizontal gauge bars with limit
  ticks and target markers,
* a **strip chart** across the bottom carrying the controlled variable against
  its setpoint,
* a monospaced header with the run clock and a status pill.

The point of the schematic is not decoration. Each one is drawn from state that
the *controller* often cannot see -- riser voidage, thermal mass, the kiln's
axial profile -- so a frame shows both what the agent knows and what it is
actually up against.

Typical use::

    from target_gym import render_kit as rk

    gauges = [rk.Gauge("POWER", f"{p:.2f}", frac=p, color=rk.CYAN)]
    fig = rk.frame(
        title="MY PLANT", step=step, params=params,
        schematic=lambda ax: draw_my_plant(ax, state, params),
        gauges=gauges,
        strips=[rk.Strip(t, [rk.Series(values, "x", rk.CYAN)], ylabel="x")],
    )
    return rk.finish(fig), history
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# ---------------------------------------------------------------------------
# Palette -- shared with target_gym.reactor.rendering
# ---------------------------------------------------------------------------

BG = "#080c14"  # console ground
PANEL = "#0e1824"  # inset panel fill
WELL = "#0a1018"  # gauge trough
FRAME = "#1e3248"  # hairlines and borders
TEXT = "#c4d8ec"  # primary ink
DIM = "#4a6678"  # labels, axis furniture

CYAN = "#00bcd4"
BLUE = "#42a5f5"
GREEN = "#66bb6a"
RED = "#ef5350"
AMBER = "#ffca28"
PURPLE = "#ab47bc"
ORANGE = "#ff8a50"
TEAL = "#26a69a"
PINK = "#ec407a"

MONO = "monospace"

# Re-exported so environment renderers need only import this module.
patches = mpatches

# Status pill levels -> (colour, label default)
NOMINAL, WATCH, ALARM = "nominal", "watch", "alarm"
_STATUS_COLOR = {NOMINAL: GREEN, WATCH: AMBER, ALARM: RED}


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------


def clamp01(x) -> float:
    return float(np.clip(x, 0.0, 1.0))


def lerp_hex(c0: str, c1: str, t: float) -> str:
    """Blend two hex colours."""
    a = np.array(mcolors.to_rgb(c0))
    b = np.array(mcolors.to_rgb(c1))
    return mcolors.to_hex(a + (b - a) * clamp01(t))


def incandescence(T, T_lo: float, T_hi: float) -> tuple:
    """Hot-body colour ramp: cool grey -> dull red -> orange -> yellow-white.

    Used wherever a surface is hot enough to glow -- fuel pins, furnace crown,
    kiln charge -- so temperature reads without consulting a number.
    """
    f = clamp01((float(T) - T_lo) / (T_hi - T_lo))
    if f < 0.25:
        g = 0.35 + f * 1.0
        return (g, g * 0.9, g * 0.85)
    if f < 0.50:
        t = (f - 0.25) / 0.25
        return (0.6 + 0.4 * t, 0.25 + 0.25 * t, 0.10)
    if f < 0.75:
        t = (f - 0.50) / 0.25
        return (1.0, 0.50 + 0.30 * t, 0.10)
    t = (f - 0.75) / 0.25
    return (1.0, 0.80 + 0.20 * t, 0.10 + 0.85 * t)


def duty_hex(frac, c_low=BLUE, c_high=ORANGE):
    """Saturated low-to-high ramp for process fluids and surfaces.

    Preferred over :func:`thermal_hex` inside vessels: matplotlib's coolwarm
    passes through near-white at its midpoint, which renders a mid-range
    temperature as muddy grey and reads as "no signal" rather than "middle".
    """
    return lerp_hex(c_low, c_high, clamp01(frac))


def thermal_hex(T, T_lo: float, T_hi: float) -> str:
    """Cold-to-hot ramp for fluids: blue -> white -> red."""
    return mcolors.rgb2hex(
        plt.cm.coolwarm(clamp01((float(T) - T_lo) / (T_hi - T_lo)))[:3]
    )


# ---------------------------------------------------------------------------
# Schematic primitives
# ---------------------------------------------------------------------------


def schematic_axes(ax, xlim=(0.0, 1.0), ylim=(0.0, 1.0)):
    """Prepare an axis for freehand schematic drawing."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    ax.set_facecolor(BG)
    return ax


def vessel(ax, x, y, w, h, *, fc=PANEL, ec=FRAME, lw=2.2, pad=0.02, **kw):
    """A rounded equipment outline."""
    p = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=mpatches.BoxStyle("Round", pad=pad),
        fc=fc,
        ec=ec,
        lw=lw,
        **kw,
    )
    ax.add_patch(p)
    return p


def fill_level(ax, x, y, w, h, frac, *, color=BLUE, alpha=0.75, ec="none"):
    """A vertical fill inside a vessel -- tank level, state of charge, holdup."""
    fh = h * clamp01(frac)
    if fh <= 1e-4:
        return None
    p = mpatches.Rectangle((x, y), w, fh, fc=color, alpha=alpha, ec=ec, zorder=2)
    ax.add_patch(p)
    return p


def disc(ax, x, y, r, *, fc=WELL, ec=DIM, lw=1.6, alpha=1.0, zorder=5):
    """A circular element -- pump, valve, rotor hub, sensor."""
    c = mpatches.Circle((x, y), r, fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(c)
    return c


def pipe(ax, pts, *, color=DIM, lw=3.0, alpha=0.9, zorder=1):
    """A run of pipe or duct through a list of (x, y) points."""
    pts = np.asarray(pts, dtype=float)
    ax.plot(
        pts[:, 0],
        pts[:, 1],
        color=color,
        lw=lw,
        alpha=alpha,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=zorder,
    )


def flow_arrow(
    ax, x0, y0, x1, y1, *, color=CYAN, lw=1.6, alpha=0.9, zorder=4, mutation=11
):
    """A directed flow marker."""
    ax.add_patch(
        mpatches.FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=mutation,
            color=color,
            lw=lw,
            alpha=alpha,
            zorder=zorder,
        )
    )


def label(
    ax,
    x,
    y,
    text,
    *,
    color=DIM,
    size=7.5,
    ha="center",
    va="center",
    weight="normal",
    zorder=6,
    **kw,
):
    """Monospaced annotation, the schematic's only typography."""
    return ax.text(
        x,
        y,
        text,
        color=color,
        fontsize=size,
        ha=ha,
        va=va,
        family=MONO,
        fontweight=weight,
        zorder=zorder,
        **kw,
    )


def caption(ax, text, *, color=DIM, size=7.5):
    """A single caption under a schematic."""
    ax.text(
        0.5,
        -0.045,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        color=color,
        fontsize=size,
        family=MONO,
    )


def glow(ax, x, y, r, *, color=CYAN, strength=1.0, layers=7, zorder=1):
    """Soft radial bloom -- reaction intensity, flame, incandescence."""
    s = clamp01(strength)
    if s <= 0.01:
        return
    for i in range(layers, 0, -1):
        ax.add_patch(
            mpatches.Circle(
                (x, y),
                r * i / layers,
                fc=color,
                ec="none",
                alpha=0.055 * s,
                zorder=zorder,
            )
        )


def gradient_bar(ax, x, y, w, h, values, cmap_fn, *, ec=FRAME, lw=1.0, zorder=2):
    """A horizontal strip coloured cell-by-cell.

    For distributed plants: the kiln's axial profile, a column's tray
    compositions, a regenerator stack.
    """
    n = len(values)
    if n == 0:
        return
    cw = w / n
    for i, v in enumerate(values):
        ax.add_patch(
            mpatches.Rectangle(
                (x + i * cw, y),
                cw,
                h,
                fc=cmap_fn(v),
                ec="none",
                zorder=zorder,
            )
        )
    ax.add_patch(
        mpatches.Rectangle((x, y), w, h, fc="none", ec=ec, lw=lw, zorder=zorder + 1)
    )


# ---------------------------------------------------------------------------
# Instrument gauges
# ---------------------------------------------------------------------------


@dataclass
class Gauge:
    """One horizontal instrument bar.

    ``frac`` is 0..1, or -1..1 when ``bipolar`` (zero at the bar centre, which
    is how a signed quantity like rod reactivity or battery power should read).
    ``limit_frac`` draws a red trip line; ``target_frac`` a white setpoint
    marker.
    """

    label: str
    value: str
    frac: float
    color: str = CYAN
    limit_frac: Optional[float] = None
    target_frac: Optional[float] = None
    bipolar: bool = False
    neg_color: Optional[str] = None
    hidden: bool = False  # mark quantities the controller cannot measure


def hbar(ax, y, g: Gauge, *, bar_h=0.062, x0=0.30, bw=0.46):
    """Draw one gauge bar. Mirrors the reactor's instrument styling."""
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x0, y),
            bw,
            bar_h,
            boxstyle=mpatches.BoxStyle("Round", pad=0.004),
            fc=WELL,
            ec=FRAME,
            lw=1,
        )
    )
    pad_x, pad_y = 0.005, 0.007
    inner_h = bar_h - 2 * pad_y

    if g.bipolar:
        mid = x0 + bw / 2
        c = float(np.clip(g.frac, -1, 1))
        fw = (bw / 2) * abs(c)
        if fw > 0.002:
            xs = mid if c >= 0 else mid - fw
            ax.add_patch(
                mpatches.Rectangle(
                    (xs, y + pad_y),
                    fw,
                    inner_h,
                    fc=g.color if c >= 0 else (g.neg_color or RED),
                    alpha=0.82,
                    ec="none",
                )
            )
        ax.plot(
            [mid, mid], [y - 0.003, y + bar_h + 0.003], color=DIM, lw=0.8, alpha=0.6
        )
    else:
        fw = bw * clamp01(g.frac)
        if fw > 0.002:
            ax.add_patch(
                mpatches.Rectangle(
                    (x0 + pad_x, y + pad_y),
                    max(fw - 2 * pad_x, 0.0),
                    inner_h,
                    fc=g.color,
                    alpha=0.82,
                    ec="none",
                )
            )

    if g.limit_frac is not None:
        lx = x0 + bw * clamp01(g.limit_frac)
        ax.plot(
            [lx, lx],
            [y - 0.006, y + bar_h + 0.006],
            color=RED,
            lw=1.4,
            ls="--",
            alpha=0.75,
        )
    if g.target_frac is not None:
        tx = x0 + bw * clamp01(g.target_frac)
        s = 0.013
        ax.fill(
            [tx - s, tx + s, tx],
            [y + bar_h + 0.010, y + bar_h + 0.010, y + bar_h - 0.003],
            color="white",
            alpha=0.9,
            zorder=5,
        )

    ax.text(
        x0 - 0.02,
        y + bar_h / 2,
        g.label,
        ha="right",
        va="center",
        fontsize=8,
        color=DIM,
        family=MONO,
        fontweight="bold",
    )
    ax.text(
        x0 + bw + 0.02,
        y + bar_h / 2,
        g.value,
        ha="left",
        va="center",
        fontsize=8.5,
        color=TEXT,
        family=MONO,
    )
    if g.hidden:
        # Unmeasurable quantities get a marker, so a frame distinguishes what
        # the plant instruments read from what only the simulator knows.
        ax.plot(
            [x0 + bw + 0.175],
            [y + bar_h / 2],
            marker="o",
            ms=3.2,
            color=PURPLE,
            alpha=0.85,
        )


def gauge_panel(
    ax,
    gauges: Sequence[Gauge],
    *,
    title="INSTRUMENTS",
    status=None,
    status_text=None,
    footnote=None,
):
    """Draw the instrument stack, top-aligned, with an optional status pill."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(BG)

    ax.text(
        0.5,
        0.975,
        title,
        ha="center",
        va="top",
        fontsize=9.5,
        color=DIM,
        family=MONO,
        fontweight="bold",
    )

    # Centre the stack vertically so panels with few gauges do not read as
    # empty, and cap the top so the first bar's target marker clears the title.
    # Tighten the pitch for tall stacks so the status pill and the hidden-state
    # note below still have room.
    gap = 0.098 if len(gauges) <= 6 else 0.085
    top = min(0.845, 0.5 + (len(gauges) - 1) * gap / 2)
    for i, g in enumerate(gauges):
        hbar(ax, top - i * gap, g)

    y_after = top - len(gauges) * gap

    if status is not None:
        c = _STATUS_COLOR.get(status, DIM)
        txt = status_text or status.upper()
        ax.plot([0.455], [y_after - 0.022], marker="o", ms=6, color=c)
        ax.text(
            0.485,
            y_after - 0.022,
            txt,
            ha="left",
            va="center",
            fontsize=9,
            color=c,
            family=MONO,
            fontweight="bold",
        )

    if footnote:
        ax.text(
            0.5,
            y_after - 0.110,
            footnote,
            ha="center",
            va="top",
            fontsize=7,
            color=DIM,
            family=MONO,
            alpha=0.85,
        )

    # Legend for hidden quantities, drawn once if any gauge is marked.
    if any(g.hidden for g in gauges):
        ax.text(
            0.5,
            y_after - 0.072,
            "● HIDDEN from the controller",
            ha="center",
            va="top",
            fontsize=6.5,
            color=PURPLE,
            family=MONO,
            alpha=0.8,
        )


# ---------------------------------------------------------------------------
# Strip charts
# ---------------------------------------------------------------------------


@dataclass
class Series:
    """One trace on a strip chart."""

    values: Sequence[float]
    label: str
    color: str = CYAN
    ls: str = "-"
    lw: float = 1.5
    alpha: float = 1.0
    fill_to: Optional[Sequence[float]] = None


@dataclass
class Strip:
    """One strip-chart panel."""

    t: Sequence[float]
    series: Sequence[Series]
    ylabel: str = ""
    xlabel: str = ""
    ylim: Optional[tuple] = None
    bands: Sequence[tuple] = field(default_factory=tuple)  # (lo, hi, colour)
    lines: Sequence[tuple] = field(default_factory=tuple)  # (y, colour, label)


def draw_strip(ax, s: Strip, *, legend=True):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(FRAME)
    ax.tick_params(colors=DIM, labelsize=7)
    ax.grid(color=FRAME, alpha=0.5, lw=0.5)

    if len(s.t) < 2:
        return

    for lo, hi, c in s.bands:
        ax.axhspan(lo, hi, color=c, alpha=0.10, zorder=0)
    for y, c, lbl in s.lines:
        ax.axhline(y, color=c, lw=1.0, ls="--", alpha=0.65, zorder=1)
        if lbl:
            ax.text(
                s.t[-1],
                y,
                f" {lbl}",
                color=c,
                fontsize=6,
                family=MONO,
                va="center",
                ha="left",
                alpha=0.85,
            )

    for ser in s.series:
        ax.plot(
            s.t,
            ser.values,
            color=ser.color,
            lw=ser.lw,
            ls=ser.ls,
            alpha=ser.alpha,
            label=ser.label,
            zorder=3,
        )
        if ser.fill_to is not None:
            ax.fill_between(
                s.t, ser.values, ser.fill_to, color=ser.color, alpha=0.07, zorder=2
            )

    if s.ylabel:
        ax.set_ylabel(s.ylabel, fontsize=7, color=DIM, family=MONO)
    if s.xlabel:
        ax.set_xlabel(s.xlabel, fontsize=7, color=DIM, family=MONO)
    if s.ylim:
        ax.set_ylim(*s.ylim)
    if legend and any(ser.label for ser in s.series):
        ax.legend(
            loc="upper right",
            fontsize=6,
            framealpha=0.4,
            facecolor=PANEL,
            edgecolor=FRAME,
            labelcolor=TEXT,
            ncol=2,
        )


# ---------------------------------------------------------------------------
# Frame assembly
# ---------------------------------------------------------------------------


def format_clock(seconds: float) -> str:
    hrs, rem = divmod(float(seconds), 3600)
    mins, secs = divmod(rem, 60)
    return f"T+{int(hrs):02d}:{int(mins):02d}:{int(secs):02d}"


def frame(
    *,
    title: str,
    step: int,
    elapsed_s: float,
    schematic: Callable,
    gauges: Sequence[Gauge],
    strips: Sequence[Strip],
    subtitle: Optional[str] = None,
    schematic_title: Optional[str] = None,
    gauge_title: str = "INSTRUMENTS",
    status: Optional[str] = None,
    status_text: Optional[str] = None,
    footnote: Optional[str] = None,
    figsize=(14.0, 7.5),
    width_ratios=(1.0, 1.1),
    dpi=100,
):
    """Assemble one control-room frame and return the matplotlib figure.

    Layout matches the reactor: schematic upper-left, instrument stack
    upper-right, strip chart(s) across the bottom.
    """
    fig = plt.figure(figsize=figsize, facecolor=BG, dpi=dpi)

    fig.text(
        0.50,
        0.975,
        title,
        ha="center",
        va="top",
        fontsize=14,
        color=TEXT,
        fontweight="bold",
        family=MONO,
    )
    fig.text(
        0.03,
        0.975,
        f"STEP {int(step)}",
        ha="left",
        va="top",
        fontsize=10,
        color=DIM,
        family=MONO,
    )
    fig.text(
        0.97,
        0.975,
        format_clock(elapsed_s),
        ha="right",
        va="top",
        fontsize=10,
        color=DIM,
        family=MONO,
    )
    if subtitle:
        fig.text(
            0.50,
            0.937,
            subtitle,
            ha="center",
            va="top",
            fontsize=8,
            color=DIM,
            family=MONO,
        )

    n_strips = max(len(strips), 1)
    gs = fig.add_gridspec(
        1 + n_strips,
        2,
        width_ratios=list(width_ratios),
        height_ratios=[2.8] + [1.0 / n_strips * 1.0] * n_strips,
        hspace=0.30 if n_strips > 1 else 0.22,
        wspace=0.08,
        left=0.03,
        right=0.97,
        top=0.915 if subtitle else 0.93,
        bottom=0.065,
    )

    ax_s = fig.add_subplot(gs[0, 0], facecolor=BG)
    schematic_axes(ax_s)
    if schematic_title:
        ax_s.text(
            0.5,
            1.045,
            schematic_title,
            transform=ax_s.transAxes,
            ha="center",
            va="top",
            fontsize=9.5,
            color=DIM,
            family=MONO,
            fontweight="bold",
        )
    schematic(ax_s)

    ax_g = fig.add_subplot(gs[0, 1], facecolor=BG)
    gauge_panel(
        ax_g,
        gauges,
        title=gauge_title,
        status=status,
        status_text=status_text,
        footnote=footnote,
    )

    for i, s in enumerate(strips):
        ax = fig.add_subplot(gs[1 + i, :])
        draw_strip(ax, s)

    return fig


def finish(fig) -> np.ndarray:
    """Rasterise a figure to an RGB array and close it."""
    canvas = FigureCanvas(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[
        ..., :3
    ]
    plt.close(fig)
    return image


def make_render_hook(render_fn, history_keys, *, stride=10):
    """Build the ``_render(cls, screen, state, params, frames, clock)`` adapter.

    Every environment's hook is the same shape, so it is generated rather than
    copied: reset the history when a new episode starts, call *render_fn* every
    *stride* steps, and append the frame.
    """

    def _render(cls, screen, state, params, frames, clock, stride=stride):
        if state is None:
            state = getattr(cls, "state", None)
            if state is None:
                raise ValueError("No state provided")
        if not hasattr(cls, "history") or state.time == 1:
            cls.history = {k: [] for k in history_keys}
        if state.time % stride == 0 or state.time == 1:
            frame_img, cls.history = render_fn(state, params, state.time, cls.history)
            frames.append(frame_img)
            cls.frames = frames
        return frames, screen, clock

    return _render
