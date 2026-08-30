"""Control-room rendering for the first-order system.

The plant is a single lag, so the schematic is a single tank: the input valve
sets inflow, the tank level is the state, and the drain is the time constant.
It is the simplest plant in the suite and the frame says so.
"""

from target_gym import render_kit as rk
from target_gym.pc_gym.first_order.env import compute_reward

HISTORY_KEYS = ("t", "x", "target", "u", "reward")


def _draw_plant(ax, state, params):
    x = float(state.x)
    tgt = float(state.target_x)
    u = float(state.u)
    span = params.x_max - params.x_min
    frac = rk.clamp01((x - params.x_min) / span)
    tfrac = rk.clamp01((tgt - params.x_min) / span)

    tx, ty, tw, th = 0.30, 0.14, 0.40, 0.62
    rk.vessel(ax, tx, ty, tw, th, fc="#0b1420")
    rk.fill_level(ax, tx, ty, tw, th, frac, color=rk.CYAN, alpha=0.55)

    # Setpoint band across the tank -- the level the controller is aiming at.
    ax.plot(
        [tx - 0.03, tx + tw + 0.03],
        [ty + th * tfrac] * 2,
        color="white",
        lw=1.3,
        ls="--",
        alpha=0.8,
        zorder=5,
    )
    rk.label(
        ax, tx + tw + 0.05, ty + th * tfrac, "target", color=rk.TEXT, size=7, ha="left"
    )

    # Inflow: arrow width follows the commanded input, direction follows sign.
    u_frac = abs(u) / max(params.u_max, 1e-9)
    up = u >= 0
    col = rk.GREEN if up else rk.AMBER
    rk.pipe(
        ax,
        [(0.09, 0.88), (0.09, ty + th + 0.06), (tx + tw / 2, ty + th + 0.06)],
        color=rk.FRAME,
        lw=4,
    )
    rk.flow_arrow(
        ax,
        tx + tw / 2,
        ty + th + 0.06,
        tx + tw / 2,
        ty + th - 0.02,
        color=col,
        lw=1.2 + 2.6 * u_frac,
        mutation=10 + 12 * u_frac,
    )
    rk.label(ax, 0.09, 0.93, f"u = {u:+.2f}", color=col, size=8)

    # Drain: the lag itself.
    rk.pipe(
        ax, [(tx + tw / 2, ty), (tx + tw / 2, 0.07), (0.90, 0.07)], color=rk.FRAME, lw=4
    )
    rk.flow_arrow(ax, 0.80, 0.07, 0.92, 0.07, color=rk.DIM, lw=1.4)
    rk.label(ax, 0.91, 0.13, f"tau = {params.tau:.2f} s", color=rk.DIM, size=7.5)

    rk.label(
        ax,
        tx + tw / 2,
        ty + th + 0.14,
        f"x = {x:+.3f}",
        color=rk.TEXT,
        size=11,
        weight="bold",
    )
    rk.caption(ax, f"first-order lag   K = {params.K:.2f}")


def render_first_order(state, params, step, history):
    history["t"].append(step * params.delta_t)
    history["x"].append(float(state.x))
    history["target"].append(float(state.target_x))
    history["u"].append(float(state.u))
    history["reward"].append(float(compute_reward(state, params)))

    err = abs(float(state.x) - float(state.target_x))
    status = rk.NOMINAL if err < 0.05 else (rk.WATCH if err < 0.3 else rk.ALARM)

    span = params.x_max - params.x_min
    gauges = [
        rk.Gauge(
            "OUTPUT x",
            f"{float(state.x):+.3f}",
            (float(state.x) - params.x_min) / span,
            rk.CYAN,
            target_frac=(float(state.target_x) - params.x_min) / span,
        ),
        rk.Gauge(
            "TARGET",
            f"{float(state.target_x):+.3f}",
            (float(state.target_x) - params.x_min) / span,
            rk.DIM,
        ),
        rk.Gauge(
            "INPUT u",
            f"{float(state.u):+.3f}",
            float(state.u) / params.u_max,
            rk.GREEN,
            bipolar=True,
            neg_color=rk.AMBER,
        ),
        rk.Gauge(
            "ERROR",
            f"{float(state.target_x) - float(state.x):+.3f}",
            (float(state.target_x) - float(state.x)) / span,
            rk.PURPLE,
            bipolar=True,
            neg_color=rk.PURPLE,
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
            rk.Series(history["x"], "x", rk.CYAN, fill_to=history["target"]),
            rk.Series(history["target"], "target", "white", ls="--", lw=1.0, alpha=0.6),
        ],
        ylabel="x",
        xlabel="seconds",
    )
    fig = rk.frame(
        title="FIRST  ORDER  SYSTEM",
        step=step,
        elapsed_s=step * params.delta_t,
        schematic=lambda ax: _draw_plant(ax, state, params),
        schematic_title="PLANT",
        gauges=gauges,
        strips=[strip],
        status=status,
        subtitle="single lag  ·  the suite's sanity check",
    )
    return rk.finish(fig), history


_render = rk.make_render_hook(render_first_order, HISTORY_KEYS, stride=10)
