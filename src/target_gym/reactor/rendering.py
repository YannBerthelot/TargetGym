"""Control-room rendering for the nuclear reactor, a pressurised water reactor.

The task (PHYSICS.md): follow the turbine's load while holding Tavg on its
program, Tref, and the axial offset inside its band, with rods that move at one
speed and boron that reaches the core late.

Left -- the pressure vessel: fuel tinted by temperature, rods entering from the
top with the rod insertion limit beside them, a Cherenkov glow for power, a
xenon fog, and the axial power split between the core's halves beside it.
Boron is printed under it with the boronmeter's late reading.

Right -- the instruments: power against load, Tavg against Tref with its dead
band and DNB limit, the hot leg against its trip, the axial offset against its
band and the day's allowance outside it, the rods against their insertion
limit, boron, and the step's cost.

Bottom -- Tavg against Tref above the floor, and the axial offset in its band.
"""

import matplotlib.colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from target_gym import render_kit as rk
from target_gym.reactor.env import (
    compute_reward,
    rod_insertion_limit,
    tavg_dnb_limit,
)

_BG = rk.BG
_FRAME = rk.FRAME
_TEXT = rk.TEXT
_DIM = rk.DIM
_CYAN = rk.CYAN
_RED = rk.RED

HISTORY_KEYS = ("t", "T_avg", "T_ref", "ao", "n", "load", "reward")


# -- Colour helpers ------------------------------------------------------------
def _fuel_rgb(T, T_lo=600.0, T_hi=2800.0):
    """Fuel incandescence: cool gray -> dull red -> orange -> yellow-white."""
    f = float(np.clip((T - T_lo) / (T_hi - T_lo), 0, 1))
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


def _coolant_hex(T, T_lo=550.0, T_hi=700.0):
    """Coolant colour for arrows: blue (cold) -> red (hot)."""
    rgba = plt.cm.coolwarm(float(np.clip((T - T_lo) / (T_hi - T_lo), 0, 1)))
    return matplotlib.colors.rgb2hex(rgba[:3])


# -- Reactor vessel drawing ----------------------------------------------------
def _offset(state, params):
    """The axial offset, (top - bottom) / total: ``delta_I`` over power."""
    return float(state.delta_I) / max(float(state.n), 1e-6)


def _draw_offset(ax, state, params, x, y0, h):
    """The power in each half of the core, and the offset against its band."""
    n = float(state.n)
    dI = float(state.delta_I)
    top, bottom = 0.5 * (n + dI), 0.5 * (n - dI)
    w = 0.05
    for share, yy in ((top, y0 + h / 2), (bottom, y0)):
        frac = rk.clamp01(share / (0.5 * params.n_max))
        ax.add_patch(
            mpatches.Rectangle(
                (x, yy), w, h / 2, fc=rk.WELL, ec=_FRAME, lw=1.0, zorder=3
            )
        )
        ax.add_patch(
            mpatches.Rectangle(
                (x, yy), w * frac, h / 2, fc=_CYAN, alpha=0.7, ec="none", zorder=4
            )
        )
    ao = _offset(state, params)
    outside = abs(ao - params.ao_ref) > params.ao_band
    rk.label(
        ax,
        x + w / 2,
        y0 + h + 0.035,
        f"AO\n{ao:+.3f}",
        color=_RED if outside else _TEXT,
        size=6.5,
    )
    rk.label(ax, x + w + 0.012, y0 + 0.75 * h, "top", color=_DIM, size=6, ha="left")
    rk.label(ax, x + w + 0.012, y0 + 0.25 * h, "bottom", color=_DIM, size=6, ha="left")


def _draw_vessel(ax, state, params):
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.14, 1.12)
    ax.axis("off")

    # Vessel outline (pressure vessel body)
    vx, vy, vw, vh = 0.10, 0.0, 0.80, 1.0
    vessel = mpatches.FancyBboxPatch(
        (vx, vy),
        vw,
        vh,
        boxstyle=mpatches.BoxStyle("Round", pad=0.045),
        fc="#111e2c",
        ec=_FRAME,
        lw=2.5,
    )
    ax.add_patch(vessel)

    # Dome top
    dome = mpatches.Wedge(
        (vx + vw / 2, vy + vh - 0.01),
        vw * 0.43,
        0,
        180,
        fc="#111e2c",
        ec=_FRAME,
        lw=2.5,
    )
    ax.add_patch(dome)

    # -- Core region --
    cx, cy, cw, ch = 0.20, 0.10, 0.60, 0.72
    ax.add_patch(mpatches.Rectangle((cx, cy), cw, ch, fc="#060e18", ec="none"))

    # -- Coolant channels (tinted fill between rods) --
    T_cool = float(state.T_coolant)
    cool_rgba = plt.cm.coolwarm(float(np.clip((T_cool - 550) / 150, 0, 1)))
    ax.add_patch(
        mpatches.Rectangle(
            (cx, cy),
            cw,
            ch,
            fc=cool_rgba,
            alpha=0.18,
            ec="none",
            zorder=1,
        )
    )

    # -- Cherenkov glow --
    n = float(state.n)
    glow_i = float(np.clip(n / params.n_max, 0, 1))
    for k in range(8):
        r = 0.05 + k * 0.04
        a = glow_i * 0.20 * (1 - k / 8)
        ax.add_patch(
            mpatches.Ellipse(
                (0.50, cy + ch * 0.45),
                r * 2.2,
                r * 3.5,
                fc=(0.20, 0.50, 1.0),
                alpha=a,
                ec="none",
                zorder=1,
            )
        )

    # -- Fuel assemblies --
    n_rods = 7
    rod_xs = np.linspace(cx + 0.045, cx + cw - 0.045, n_rods)
    fc = _fuel_rgb(float(state.T_fuel))
    rod_w = 0.030
    for x in rod_xs:
        ax.add_patch(
            mpatches.Rectangle(
                (x - rod_w / 2, cy + 0.02),
                rod_w,
                ch - 0.04,
                fc=fc,
                ec="#1a1a1a",
                lw=0.5,
                zorder=3,
            )
        )

    # -- Control rods --
    # Rods enter from the top: ``rod_in`` is the fraction of the stroke in.
    rod_in = float(state.rod_in)
    tip_y = cy + (1.0 - rod_in) * ch
    shaft_top = vy + vh + 0.12

    ctrl_xs = rod_xs[1::2]  # interleaved with fuel assemblies
    cw_rod = 0.020
    for x in ctrl_xs:
        # Shaft
        ax.add_patch(
            mpatches.Rectangle(
                (x - cw_rod / 2, tip_y),
                cw_rod,
                shaft_top - tip_y,
                fc="#282828",
                ec="#444",
                lw=0.6,
                zorder=4,
            )
        )
        # Absorber tip (darker, wider)
        ax.add_patch(
            mpatches.Rectangle(
                (x - cw_rod * 0.8, tip_y),
                cw_rod * 1.6,
                0.025,
                fc="#0a0a0a",
                ec="#555",
                lw=0.6,
                zorder=5,
            )
        )

    # -- Xenon-135 fog --
    xe = float(state.Xe_hat)
    xe_a = float(np.clip((xe - 0.5) * 0.35, 0, 0.45))
    if xe_a > 0.005:
        for dy in (0.0, 0.15, -0.15, 0.30, -0.08):
            ax.add_patch(
                mpatches.Ellipse(
                    (0.50, cy + ch * 0.50 + dy),
                    cw * 0.80,
                    ch * 0.22,
                    fc="#7b1fa2",
                    alpha=xe_a * 0.45,
                    ec="none",
                    zorder=2,
                )
            )

    # -- Coolant flow arrows --
    cool_hex = _coolant_hex(T_cool)
    arr_in = dict(arrowstyle="->", color=_CYAN, lw=1.8)
    arr_out = dict(arrowstyle="->", color=cool_hex, lw=1.8)
    ax.annotate(
        "",
        xy=(0.28, cy + 0.04),
        xytext=(0.28, cy - 0.06),
        arrowprops=arr_in,
    )
    ax.annotate(
        "",
        xy=(0.72, cy + ch + 0.06),
        xytext=(0.72, cy + ch - 0.04),
        arrowprops=arr_out,
    )
    ax.text(
        0.28,
        cy - 0.08,
        "IN",
        ha="center",
        va="top",
        fontsize=6,
        color=_DIM,
        family="monospace",
    )
    ax.text(
        0.72,
        cy + ch + 0.08,
        "OUT",
        ha="center",
        va="bottom",
        fontsize=6,
        color=_DIM,
        family="monospace",
    )

    # -- Labels --
    boron = float(state.boron)
    meter = float(state.boron_meter[-1])
    burn = getattr(state, "burn_rate", None)  # reactor v9: the fuel's burnup
    drift = (
        f"  ·  burnup {float(burn) * 86400.0:.1f} ppm/day ●"
        if burn is not None and float(burn) > 0.0
        else ""
    )
    rk.label(
        ax,
        0.50,
        -0.10,
        f"BORON {boron:.0f} ppm ●  ·  meter {meter:.0f} (late)" f"{drift}",
        color=_CYAN,
        size=7,
    )
    # The rod insertion limit at this power, on the core's left edge.
    ril = float(rod_insertion_limit(state.n, params))
    ril_y = cy + (1.0 - ril) * ch
    ax.plot(
        [cx - 0.03, cx + 0.02], [ril_y, ril_y], color=_RED, lw=1.4, ls="--", zorder=6
    )
    rk.label(ax, cx - 0.035, ril_y, "RIL", color=_RED, size=6.5, ha="right")
    rk.label(
        ax,
        cx - 0.035,
        tip_y,
        f"rods\n{rod_in * 100:.0f}% in",
        color=_TEXT,
        size=6.5,
        ha="right",
        va="top",
    )

    _draw_offset(ax, state, params, cx + cw + 0.035, cy, ch)


def _status(state, params):
    """``(status, text)``: trips and timed limits first, then the bands."""
    T_avg = 0.5 * (float(state.T_coolant) + float(state.T_cold))
    ao = _offset(state, params)
    if float(state.below_floor) > 0.0:
        return rk.ALARM, (
            f"TAVG UNDER FLOOR {float(state.below_floor) / 60:.0f}"
            f"/{params.floor_allowed / 60:.0f} min"
        )
    if float(state.above_dnb) > 0.0:
        return rk.ALARM, "TAVG OVER DNB LIMIT"
    if float(state.ril_outside) > 0.0:
        return rk.WATCH, "RODS PAST INSERTION LIMIT"
    if abs(ao - params.ao_ref) > params.ao_band:
        return rk.WATCH, "AXIAL OFFSET OUT OF BAND"
    if abs(T_avg - float(state.T_ref)) > params.e_tol:
        return rk.WATCH, "TAVG OFF PROGRAM"
    return rk.NOMINAL, None


def render_reactor(state, params, step, history):
    """Render one control-room frame and append data to *history*."""
    p = params
    T_avg = 0.5 * (float(state.T_coolant) + float(state.T_cold))
    T_ref = float(state.T_ref)
    ao = _offset(state, p)
    history["t"].append(step * p.delta_t / 3600.0)
    history["T_avg"].append(T_avg)
    history["T_ref"].append(T_ref)
    history["ao"].append(ao)
    history["n"].append(float(state.n))
    history["load"].append(float(state.target_n))
    history["reward"].append(float(compute_reward(state, p)))

    dnb = float(tavg_dnb_limit(p))
    t_lo, t_hi = p.T_avg_min - 4.0, dnb + 4.0  # Tavg's gauge span
    tf = lambda T: (float(T) - t_lo) / (t_hi - t_lo)  # noqa: E731
    h_lo = p.T_avg_min
    hf = lambda T: (float(T) - h_lo) / (p.T_coolant_max + 5.0 - h_lo)  # noqa: E731
    ao_span = 2.0 * p.ao_band
    ril = float(rod_insertion_limit(state.n, p))
    meter = float(state.boron_meter[-1])

    gauges = [
        rk.Gauge(
            "POWER",
            f"{float(state.n) * 100:.1f} %",
            float(state.n) / p.n_max,
            _CYAN,
            limit_frac=1.0,
            target_frac=float(state.target_n) / p.n_max,
        ),
        rk.Gauge(
            "T AVG",
            f"{T_avg - 273.15:.1f} C",
            tf(T_avg),
            rk.ORANGE,
            limit_frac=tf(dnb),
            target_frac=tf(T_ref),
            band=(tf(T_ref - p.e_tol), tf(T_ref + p.e_tol)),
        ),
        rk.Gauge(
            "T HOT",
            f"{float(state.T_coolant) - 273.15:.1f} C",
            hf(state.T_coolant),
            rk.RED,
            limit_frac=hf(p.T_coolant_max),
        ),
        rk.Gauge(
            "AXIAL OFFSET",
            f"{ao:+.3f}",
            (ao - p.ao_ref) / ao_span,
            rk.BLUE,
            bipolar=True,
            neg_color=rk.BLUE,
            band=(-p.ao_band / ao_span, p.ao_band / ao_span),
        ),
        rk.Gauge(
            "AO TIME OUT",
            f"{float(state.ao_outside) / 60:.0f} / {p.ao_allowed / 60:.0f} min",
            float(state.ao_outside) / p.ao_allowed,
            rk.AMBER,
            limit_frac=1.0,
        ),
        rk.Gauge(
            "RODS IN",
            f"{float(state.rod_in) * 100:.0f} %",
            float(state.rod_in),
            rk.DIM,
            limit_frac=ril,
        ),
        rk.Gauge(
            "BORON",
            f"{meter:.0f} ppm",
            meter / (2.0 * p.boron_ref),
            _CYAN,
            target_frac=0.5,
        ),
        rk.cost_gauge(history["reward"], p, "$"),
    ]

    strips = [
        rk.Strip(
            history["t"],
            [
                rk.Series(
                    history["T_avg"], "Tavg", rk.ORANGE, fill_to=history["T_ref"]
                ),
                rk.Series(
                    history["T_ref"], "Tref", "white", ls="--", lw=1.0, alpha=0.7
                ),
            ],
            ylabel="Tavg  [K]",
            lines=[(p.T_avg_min, _RED, "floor")],
        ),
        rk.Strip(
            history["t"],
            [rk.Series(history["ao"], "axial offset", rk.BLUE)],
            ylabel="AO",
            xlabel="hours",
            bands=[(p.ao_ref - p.ao_band, p.ao_ref + p.ao_band, rk.GREEN)],
            lines=[(p.ao_ref, _DIM, "")],
        ),
    ]
    status, status_text = _status(state, p)
    fig = rk.frame(
        title="NUCLEAR  REACTOR  —  PRESSURISED  WATER",
        step=step,
        elapsed_s=step * p.delta_t,
        schematic=lambda ax: _draw_vessel(ax, state, p),
        schematic_title="NUCLEAR  CORE",
        gauges=gauges,
        strips=strips,
        status=status,
        status_text=status_text,
        subtitle="Tavg on its program  ·  axial offset in its band  ·  boron late",
    )
    return rk.finish(fig), history


# -- save_video hook -----------------------------------------------------------
def _render(cls, screen, state, params, frames, clock, stride: int | None = None):
    """Render hook used by ``save_video`` -- subsamples frames by *stride*.

    The stride is derived from the episode via ``render_kit.frame_stride``
    rather than fixed: a fixed stride chosen for one episode length leaves a
    clip of another length with a handful of frames.
    """
    if stride is None:
        stride = rk.frame_stride(params)
    if state is None:
        if cls.state is None:
            raise ValueError("No state provided")
        state = cls.state

    if not hasattr(cls, "history") or state.time <= 1 or not frames:
        cls.history = {k: [] for k in HISTORY_KEYS}

    step = state.time
    if step % stride == 0 or step <= 1 or not frames:
        # The renderer wants the physics clock: it labels elapsed time as
        # ``step * delta_t`` seconds, and ``delta_t`` is the sub-step.
        frame, cls.history = render_reactor(
            state, params, int(state.physics_time), cls.history
        )
        frames.append(frame)
        cls.frames = frames

    return frames, screen, clock
