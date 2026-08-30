"""Control-room rendering for the quadruple-tank process.

The schematic is the plant's defining feature: each pump splits its flow, most
of it going to the *diagonal* upper tank, which then drains into the lower tank
on the other side. That cross-coupling is why independent loops struggle, and
the picture shows it as plumbing rather than as a gain matrix.
"""

from target_gym import render_kit as rk
from target_gym.pc_gym.four_tank.env import compute_reward

HISTORY_KEYS = ("t", "h1", "h2", "h3", "h4", "t1", "t2", "v1", "v2", "reward")


def _tank(ax, x, y, w, h, level_frac, color, name, value, target_frac=None):
    rk.vessel(ax, x, y, w, h, fc="#0b1420", ec=rk.FRAME, lw=1.8, pad=0.012)
    rk.fill_level(ax, x, y, w, h, level_frac, color=color, alpha=0.6)
    if target_frac is not None:
        ax.plot(
            [x - 0.015, x + w + 0.015],
            [y + h * target_frac] * 2,
            color="white",
            lw=1.2,
            ls="--",
            alpha=0.85,
            zorder=6,
        )
    rk.label(ax, x + w / 2, y + h - 0.028, name, color=rk.DIM, size=7.5, weight="bold")
    rk.label(ax, x + w / 2, y - 0.038, value, color=rk.TEXT, size=8)


def _draw_tanks(ax, state, params):
    hmax = params.h_max
    f = lambda v: rk.clamp01(float(v) / hmax)

    tw, th = 0.20, 0.26
    ux, uy = 0.16, 0.58  # upper row
    lx, ly = 0.16, 0.16  # lower row
    rx = 0.60

    _tank(
        ax, ux, uy, tw, th, f(state.h3), rk.TEAL, "TANK 3", f"{float(state.h3):.2f} m"
    )
    _tank(
        ax, rx, uy, tw, th, f(state.h4), rk.TEAL, "TANK 4", f"{float(state.h4):.2f} m"
    )
    _tank(
        ax,
        lx,
        ly,
        tw,
        th,
        f(state.h1),
        rk.CYAN,
        "TANK 1",
        f"{float(state.h1):.2f} m",
        target_frac=f(state.target_h1),
    )
    _tank(
        ax,
        rx,
        ly,
        tw,
        th,
        f(state.h2),
        rk.CYAN,
        "TANK 2",
        f"{float(state.h2):.2f} m",
        target_frac=f(state.target_h2),
    )

    # Upper tanks drain into the lower tank directly below.
    for cx in (ux + tw / 2, rx + tw / 2):
        rk.pipe(ax, [(cx, uy), (cx, ly + th)], color=rk.FRAME, lw=3)
        rk.flow_arrow(ax, cx, uy - 0.02, cx, ly + th + 0.01, color=rk.TEAL, lw=1.2)

    # Pumps. Each splits: gamma to the lower tank on its own side, 1-gamma to
    # the diagonal upper tank. The diagonal path is the cross-coupling.
    v1 = float(state.v1) / max(params.v_max, 1e-9)
    v2 = float(state.v2) / max(params.v_max, 1e-9)
    for px, name, v, g, own_x, diag_x, col in (
        (0.07, "PUMP 1", v1, params.gamma1, lx + tw / 2, rx + tw / 2, rk.GREEN),
        (0.93, "PUMP 2", v2, params.gamma2, rx + tw / 2, ux + tw / 2, rk.AMBER),
    ):
        rk.disc(ax, px, 0.42, 0.035, ec=col)
        rk.label(ax, px, 0.42, f"{v * 100:.0f}", color=col, size=6.5, zorder=7)
        rk.label(ax, px, 0.365, name, color=rk.DIM, size=6.5)
        lw_own = 0.8 + 3.0 * v * g
        lw_diag = 0.8 + 3.0 * v * (1 - g)
        rk.pipe(
            ax,
            [(px, 0.455), (px, ly + th + 0.10), (own_x, ly + th + 0.10)],
            color=col,
            lw=lw_own,
            alpha=0.65,
        )
        rk.pipe(
            ax,
            [(px, 0.455), (px, uy + th + 0.10), (diag_x, uy + th + 0.10)],
            color=col,
            lw=lw_diag,
            alpha=0.5,
        )
        rk.flow_arrow(
            ax,
            diag_x,
            uy + th + 0.10,
            diag_x,
            uy + th + 0.015,
            color=col,
            lw=1.0,
            alpha=0.7,
        )

    rk.caption(ax, "each pump splits: most flow to the DIAGONAL upper tank")


def render_four_tank(state, params, step, history):
    history["t"].append(step * params.delta_t)
    for k, v in (
        ("h1", state.h1),
        ("h2", state.h2),
        ("h3", state.h3),
        ("h4", state.h4),
        ("t1", state.target_h1),
        ("t2", state.target_h2),
        ("v1", state.v1),
        ("v2", state.v2),
    ):
        history[k].append(float(v))
    history["reward"].append(float(compute_reward(state, params)))

    e = max(
        abs(float(state.h1) - float(state.target_h1)),
        abs(float(state.h2) - float(state.target_h2)),
    )
    status = rk.NOMINAL if e < 0.03 else (rk.WATCH if e < 0.15 else rk.ALARM)
    f = lambda v: rk.clamp01(float(v) / params.h_max)

    gauges = [
        rk.Gauge(
            "LEVEL h1",
            f"{float(state.h1):.3f} m",
            f(state.h1),
            rk.CYAN,
            target_frac=f(state.target_h1),
        ),
        rk.Gauge(
            "LEVEL h2",
            f"{float(state.h2):.3f} m",
            f(state.h2),
            rk.CYAN,
            target_frac=f(state.target_h2),
        ),
        rk.Gauge("LEVEL h3", f"{float(state.h3):.3f} m", f(state.h3), rk.TEAL),
        rk.Gauge("LEVEL h4", f"{float(state.h4):.3f} m", f(state.h4), rk.TEAL),
        rk.Gauge(
            "PUMP v1",
            f"{float(state.v1):.2f} V",
            float(state.v1) / params.v_max,
            rk.GREEN,
            limit_frac=1.0,
        ),
        rk.Gauge(
            "PUMP v2",
            f"{float(state.v2):.2f} V",
            float(state.v2) / params.v_max,
            rk.AMBER,
            limit_frac=1.0,
        ),
        rk.Gauge(
            "REWARD",
            f"{history['reward'][-1]:.3f}",
            rk.clamp01(history["reward"][-1]),
            rk.GREEN,
        ),
    ]

    strip = rk.Strip(
        history["t"],
        [
            rk.Series(history["h1"], "h1", rk.CYAN, fill_to=history["t1"]),
            rk.Series(history["t1"], "target h1", "white", ls="--", lw=1.0, alpha=0.55),
            rk.Series(history["h2"], "h2", rk.BLUE, fill_to=history["t2"]),
            rk.Series(history["t2"], "target h2", rk.DIM, ls="--", lw=1.0, alpha=0.75),
        ],
        ylabel="level  [m]",
        xlabel="seconds",
    )
    fig = rk.frame(
        title="QUADRUPLE  TANK  PROCESS",
        step=step,
        elapsed_s=step * params.delta_t,
        schematic=lambda ax: _draw_tanks(ax, state, params),
        schematic_title="PLANT",
        gauges=gauges,
        strips=[strip],
        status=status,
        subtitle=f"cross-coupled  ·  gamma1 {params.gamma1:.2f}  gamma2 {params.gamma2:.2f}",
    )
    return rk.finish(fig), history


_render = rk.make_render_hook(render_four_tank, HISTORY_KEYS, stride=10)
