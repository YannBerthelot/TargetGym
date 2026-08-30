"""Control-room rendering for the pH neutralisation reactor.

The tank is tinted the way a universal indicator would show it, so the
equivalence point reads as a colour change rather than a number. The buffer
stream is drawn dashed and marked hidden: it is what flattens the titration
curve and the controller never sees it.
"""

import numpy as np

from target_gym import render_kit as rk
from target_gym.pc_gym.ph_neutralization.env import compute_reward

HISTORY_KEYS = ("t", "pH", "target", "q3", "q2", "reward")

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


def _draw_reactor(ax, state, params):
    pH = float(state.pH)
    q3 = float(state.q3)
    q2 = float(state.q2)
    col = indicator_hex(pH)

    vx, vy, vw, vh = 0.26, 0.14, 0.46, 0.54
    rk.vessel(ax, vx, vy, vw, vh, fc="#0b1420", ec=rk.FRAME, lw=2.2)
    rk.fill_level(ax, vx, vy, vw, vh, 0.80, color=col, alpha=0.55)

    # Impeller.
    cx = vx + vw / 2
    ax.plot([cx, cx], [vy + vh * 0.34, vy + vh + 0.06], color=rk.DIM, lw=2, zorder=5)
    for dx in (-0.07, 0.07):
        ax.plot(
            [cx, cx + dx],
            [vy + vh * 0.34, vy + vh * 0.34 - 0.026],
            color=rk.TEXT,
            lw=2.2,
            alpha=0.7,
            zorder=5,
        )

    # Three feeds. Acid is fixed, base is the manipulated variable, buffer is
    # the unmeasured disturbance.
    feeds = [
        (0.10, "ACID q1", params.q1, params.q1, rk.RED, False),
        (0.50, "BUFFER q2", q2, params.q2_max, rk.PURPLE, True),
        (0.90, "BASE q3", q3, params.q3_max, rk.BLUE, False),
    ]
    for fx, name, val, vmax, c, hidden in feeds:
        f = rk.clamp01(val / max(vmax, 1e-9))
        ls_pipe = c if not hidden else rk.PURPLE
        rk.pipe(
            ax,
            [
                (fx, 0.93),
                (fx, 0.80),
                (cx + (fx - 0.5) * 0.30, 0.80),
                (cx + (fx - 0.5) * 0.30, vy + vh + 0.02),
            ],
            color=ls_pipe,
            lw=1.0 + 3.2 * f,
            alpha=0.45 + 0.4 * f,
        )
        rk.flow_arrow(
            ax,
            cx + (fx - 0.5) * 0.30,
            vy + vh + 0.05,
            cx + (fx - 0.5) * 0.30,
            vy + vh - 0.02,
            color=ls_pipe,
            lw=1.0 + 1.6 * f,
        )
        rk.label(
            ax,
            fx,
            0.965,
            name,
            color=ls_pipe,
            size=7,
            weight="bold" if not hidden else "normal",
        )
        rk.label(ax, fx, 0.90, f"{val:.2f} mL/s", color=rk.TEXT, size=7)
        if hidden:
            rk.label(ax, fx, 0.865, "● hidden", color=rk.PURPLE, size=6)

    # Effluent and probe.
    rk.pipe(ax, [(cx, vy), (cx, 0.06), (0.92, 0.06)], color=rk.FRAME, lw=4)
    rk.flow_arrow(ax, 0.82, 0.06, 0.94, 0.06, color=col, lw=1.6)
    rk.disc(ax, vx + vw - 0.055, vy + vh * 0.55, 0.022, fc=rk.WELL, ec=col, lw=1.8)
    rk.label(ax, vx + vw - 0.055, vy + vh * 0.55 - 0.048, "pH", color=rk.DIM, size=6)

    # Sits inside the liquid, clear of the three feed pipes above the tank.
    rk.label(
        ax,
        cx,
        vy + vh - 0.075,
        f"pH {pH:.2f}",
        color=rk.TEXT,
        size=16,
        weight="bold",
        zorder=8,
    )
    rk.caption(ax, "buffer flattens the titration curve by an order of magnitude")


def render_ph(state, params, step, history):
    history["t"].append(step * params.delta_t / 60.0)
    history["pH"].append(float(state.pH))
    history["target"].append(float(state.target_pH))
    history["q3"].append(float(state.q3))
    history["q2"].append(float(state.q2))
    history["reward"].append(float(compute_reward(state, params)))

    err = abs(float(state.pH) - float(state.target_pH))
    status = rk.NOMINAL if err < 0.2 else (rk.WATCH if err < 0.8 else rk.ALARM)
    span = params.pH_max - params.pH_min
    f = lambda v: (float(v) - params.pH_min) / span

    gauges = [
        rk.Gauge(
            "pH",
            f"{float(state.pH):.3f}",
            f(state.pH),
            indicator_hex(state.pH),
            target_frac=f(state.target_pH),
        ),
        rk.Gauge("TARGET", f"{float(state.target_pH):.3f}", f(state.target_pH), rk.DIM),
        rk.Gauge(
            "BASE q3",
            f"{float(state.q3):.2f} mL/s",
            (float(state.q3) - params.q3_min) / (params.q3_max - params.q3_min),
            rk.BLUE,
            limit_frac=1.0,
        ),
        rk.Gauge(
            "BUFFER q2",
            f"{float(state.q2):.3f} mL/s",
            float(state.q2) / params.q2_max,
            rk.PURPLE,
            hidden=True,
        ),
        rk.Gauge(
            "REWARD",
            f"{history['reward'][-1]:+.3f}",
            rk.clamp01(history["reward"][-1]),
            rk.GREEN,
        ),
    ]

    strip = rk.Strip(
        history["t"],
        [
            rk.Series(history["pH"], "pH", rk.CYAN, fill_to=history["target"]),
            rk.Series(history["target"], "target", "white", ls="--", lw=1.0, alpha=0.6),
        ],
        ylabel="pH",
        xlabel="minutes",
        bands=[(6.5, 8.0, rk.GREEN)],
    )
    fig = rk.frame(
        title="pH  NEUTRALISATION",
        step=step,
        elapsed_s=step * params.delta_t,
        schematic=lambda ax: _draw_reactor(ax, state, params),
        schematic_title="REACTOR",
        gauges=gauges,
        strips=[strip],
        status=status,
        subtitle="45x gain variation across the range  ·  buffering is unmeasured",
    )
    return rk.finish(fig), history


_render = rk.make_render_hook(render_ph, HISTORY_KEYS, stride=4)
