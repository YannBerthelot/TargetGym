"""Control-room rendering for the two-stage pH neutralisation.

Each tank is tinted the way a universal indicator would show it, so the
equivalence point reads as a colour change rather than a number. Stage 1 takes
the influent, the buffer, the main base valve and the basin's pump-back; stage
2 the trim valve, and its probe is the discharge the permit judges. Effluent
below the sewer floor is diverted into the basin. The buffer flow and the
influent's acid strength are marked hidden: the controller sees neither.
"""

import numpy as np

from target_gym import render_kit as rk
from target_gym.pc_gym.ph_neutralization.env import acid_scale, compute_reward

HISTORY_KEYS = ("t", "pH", "pH_s1", "target", "reward")

# Universal-indicator stops from strongly acidic to strongly alkaline.
_INDICATOR = [
    (2.0, "#d32f2f"),
    (4.0, "#f57c00"),
    (6.0, "#fdd835"),
    (7.0, "#43a047"),
    (8.5, "#00897b"),
    (10.0, "#1e88e5"),
    (12.0, "#6a1b9a"),
]


def indicator_hex(pH):
    """Colour the solution the way universal indicator would."""
    pH = float(np.clip(pH, _INDICATOR[0][0], _INDICATOR[-1][0]))
    for (p0, c0), (p1, c1) in zip(_INDICATOR, _INDICATOR[1:]):
        if pH <= p1:
            return rk.lerp_hex(c0, c1, (pH - p0) / (p1 - p0))
    return _INDICATOR[-1][1]


def _tank(ax, x, y, w, h, pH, name):
    col = indicator_hex(pH)
    rk.vessel(ax, x, y, w, h, fc="#0b1420", ec=rk.FRAME, lw=2.0)
    rk.fill_level(ax, x, y, w, h, 0.82, color=col, alpha=0.55)
    cx = x + w / 2
    ax.plot([cx, cx], [y + h * 0.30, y + h * 0.48], color=rk.DIM, lw=1.8, zorder=5)
    for dx in (-0.045, 0.045):
        ax.plot(
            [cx, cx + dx],
            [y + h * 0.30, y + h * 0.30 - 0.02],
            color=rk.TEXT,
            lw=2.0,
            alpha=0.7,
            zorder=5,
        )
    rk.label(
        ax,
        cx,
        y + h * 0.62,
        f"pH {pH:.2f}",
        color=rk.TEXT,
        size=12,
        weight="bold",
        zorder=8,
    )
    rk.label(ax, cx, y - 0.035, name, color=rk.DIM, size=7, weight="bold")
    return col


def _feed(ax, x, y_in, name, val, vmax, c, hidden=False, note=None):
    """A reagent line straight down into a tank, labelled above it."""
    f = rk.clamp01(val / max(vmax, 1e-9))
    top = 0.88 if note is None else 0.855
    rk.pipe(
        ax,
        [(x, top), (x, y_in + 0.03)],
        color=c,
        lw=1.0 + 3.0 * f,
        alpha=0.45 + 0.4 * f,
    )
    rk.flow_arrow(ax, x, y_in + 0.05, x, y_in - 0.02, color=c, lw=1.0 + 1.5 * f)
    rk.label(
        ax, x, 0.965, name, color=c, size=6.5, weight="normal" if hidden else "bold"
    )
    rk.label(ax, x, 0.93, f"{val:.2f} mL/s", color=rk.TEXT, size=6.5)
    if note is not None:
        rk.label(ax, x, 0.895, note, color=rk.PURPLE, size=6)


def _draw_plant(ax, state, params):
    p = params
    pH, pH1 = float(state.pH), float(state.pH_s1)
    q1, q2, q3, qt = (
        float(state.q1),
        float(state.q2),
        float(state.q3),
        float(state.q3_trim),
    )
    diverting = pH < p.sewer_floor

    # Stage 1 and its feeds.
    x1, y1, w1, h1 = 0.04, 0.36, 0.34, 0.34
    top = y1 + h1
    acid = f"acid ×{float(acid_scale(state, p)):.2f} ●"
    _feed(ax, 0.08, top, "INFLUENT", q1, 1.5 * p.q1, rk.RED, note=acid)
    _feed(ax, 0.21, top, "BUFFER ●", q2, p.q2_max, rk.PURPLE, hidden=True)
    _feed(ax, 0.34, top, "BASE q3", q3, p.q3_max, rk.BLUE)
    _tank(ax, x1, y1, w1, h1, pH1, "STAGE 1")

    # Stage 2 and the trim valve.
    x2, y2, w2, h2 = 0.52, 0.36, 0.30, 0.34
    _feed(ax, x2 + w2 * 0.75, y2 + h2, "TRIM", qt, p.q3_trim_max, rk.CYAN)
    rk.pipe(
        ax,
        [(x1 + w1 + 0.02, y1 + 0.06), (x2 - 0.02, y2 + 0.06)],
        color=indicator_hex(pH1),
        lw=3,
    )
    rk.flow_arrow(
        ax, x2 - 0.06, y2 + 0.06, x2 - 0.015, y2 + 0.06, color=indicator_hex(pH1)
    )
    col = _tank(ax, x2, y2, w2, h2, pH, "STAGE 2  (discharge)")

    # The discharge: to the sewer inside the floor, else to the basin.
    ox = x2 + w2 + 0.02
    rk.pipe(ax, [(ox, y2 + 0.06), (0.93, y2 + 0.06)], color=rk.FRAME, lw=4)
    sewer_c = rk.DIM if diverting else col
    rk.pipe(ax, [(0.93, y2 + 0.06), (0.93, 0.68)], color=sewer_c, lw=3)
    rk.flow_arrow(ax, 0.93, 0.62, 0.93, 0.70, color=sewer_c)
    rk.label(ax, 0.93, 0.73, "SEWER", color=sewer_c, size=6.5, weight="bold")
    div_c = rk.RED if diverting else rk.DIM
    rk.pipe(ax, [(0.93, y2 + 0.06), (0.93, 0.25)], color=div_c, lw=3)
    rk.label(ax, 0.975, 0.33, f"< pH\n{p.sewer_floor:.1f}", color=div_c, size=6)

    # The diversion basin, pumped back to stage 1.
    bx, by, bw, bh = 0.58, 0.05, 0.38, 0.15
    rk.vessel(ax, bx, by, bw, bh, fc="#0b1420", ec=div_c, lw=1.6)
    full = float(state.basin) / p.basin_volume
    rk.fill_level(ax, bx, by, bw, bh, full, color=rk.ORANGE, alpha=0.55)
    rk.label(
        ax,
        bx + bw / 2,
        by + bh / 2,
        f"BASIN  {full * 100:.0f} %  of {p.basin_volume:.0f} L",
        color=rk.TEXT,
        size=7,
        weight="bold",
        zorder=8,
    )
    pump = not diverting and float(state.basin) > 0.0
    pc = rk.ORANGE if pump else rk.FRAME
    rk.pipe(
        ax,
        [(bx, by + bh / 2), (x1 + w1 * 0.5, by + bh / 2), (x1 + w1 * 0.5, y1 - 0.06)],
        color=pc,
        lw=2,
    )
    rk.flow_arrow(ax, x1 + w1 * 0.5, y1 - 0.10, x1 + w1 * 0.5, y1 - 0.045, color=pc)
    rk.label(ax, 0.30, by + bh / 2 + 0.03, "pump-back", color=pc, size=6)

    rk.caption(
        ax,
        f"permit pH {p.permit_low:g}-{p.permit_high:g}  ·  "
        f"{p.excursion_limit / 60:.0f} min excursion is a violation",
    )


def render_ph(state, params, step, history):
    p = params
    history["t"].append(step * p.delta_t / 60.0)
    history["pH"].append(float(state.pH))
    history["pH_s1"].append(float(state.pH_s1))
    history["target"].append(float(state.target_pH))
    history["reward"].append(float(compute_reward(state, p)))

    pH = float(state.pH)
    exc = float(state.excursion)
    if pH < p.sewer_floor:
        status, status_text = rk.ALARM, "DIVERTING"
    elif exc > 0.0:
        status, status_text = rk.WATCH, f"OUTSIDE PERMIT {exc / 60:.0f} min"
    else:
        status, status_text = rk.NOMINAL, "IN PERMIT"
    f = lambda v: float(v) / 14.0  # noqa: E731, the pH scale

    gauges = [
        rk.Gauge(
            "pH OUT",
            f"{pH:.2f}",
            f(pH),
            indicator_hex(pH),
            limit_frac=f(p.permit_high),
            target_frac=f(state.target_pH),
        ),
        rk.Gauge(
            "BASE q3",
            f"{float(state.q3):.2f} mL/s",
            (float(state.q3) - p.q3_min) / (p.q3_max - p.q3_min),
            rk.BLUE,
        ),
        rk.Gauge(
            "TRIM",
            f"{float(state.q3_trim):.2f} mL/s",
            float(state.q3_trim) / p.q3_trim_max,
            rk.CYAN,
        ),
        rk.Gauge(
            "INFLUENT",
            f"{float(state.q1):.1f} mL/s",
            float(state.q1) / (1.5 * p.q1),
            rk.RED,
            target_frac=1.0 / 1.5,
        ),
        rk.Gauge(
            "BUFFER q2",
            f"{float(state.q2):.2f} mL/s",
            float(state.q2) / p.q2_max,
            rk.PURPLE,
            hidden=True,
        ),
        rk.Gauge(
            "EXCURSION",
            f"{exc / 60:.1f} / {p.excursion_limit / 60:.0f} min",
            exc / p.excursion_limit,
            rk.AMBER,
            limit_frac=1.0,
        ),
        rk.Gauge(
            "BASIN",
            f"{100 * float(state.basin) / p.basin_volume:.1f} %",
            float(state.basin) / p.basin_volume,
            rk.ORANGE,
            limit_frac=1.0,
        ),
        rk.cost_gauge(history["reward"], p, "L"),
    ]

    strip = rk.Strip(
        history["t"],
        [
            rk.Series(history["pH"], "discharge pH", rk.CYAN),
            rk.Series(history["pH_s1"], "stage 1 pH", rk.DIM, lw=1.0),
            rk.Series(
                history["target"], "set point", "white", ls="--", lw=1.0, alpha=0.6
            ),
        ],
        ylabel="pH",
        xlabel="minutes",
        bands=[(p.permit_low, p.permit_high, rk.GREEN)],
        lines=[(p.sewer_floor, rk.RED, "sewer floor")],
    )
    fig = rk.frame(
        title="pH  NEUTRALISATION",
        step=step,
        elapsed_s=step * p.delta_t,
        schematic=lambda ax: _draw_plant(ax, state, p),
        schematic_title="TWO-STAGE PLANT",
        gauges=gauges,
        strips=[strip],
        status=status,
        status_text=status_text,
        subtitle="the titration curve swings the gain  ·  buffering is unmeasured",
    )
    return rk.finish(fig), history


_render = rk.make_render_hook(render_ph, HISTORY_KEYS)
