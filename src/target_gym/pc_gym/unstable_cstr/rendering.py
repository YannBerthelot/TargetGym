"""Control-room rendering for the LDPE autoclave.

The schematic is the autoclave: contents tinted by temperature between the cold
feed and the decomposition trip, the initiator injection drawn by its rate.
Held at an unstable point, the reactor either heats until it trips or cools
until polymerisation stops; the strip shows which way it is going. The feed
is drawn as it enters now, its temperature wandering and stepping with the
scenario's event; the controller reads neither.
"""

from target_gym import render_kit as rk
from target_gym.pc_gym.unstable_cstr.env import compute_reward, events


def _feed(state, params):
    """``(T_f, initiator efficiency, flow scale)`` entering now."""
    _, T_f, f, scale = events(state, params, float(state.time) * params.delta_t)
    return float(T_f), float(f) / params.f_eff, float(scale)


HISTORY_KEYS = ("t", "target", "T", "I_f", "reward")


def _draw_reactor(ax, state, params):
    T = float(state.T)
    frac = rk.clamp01((T - params.T_feed) / (params.T_trip - params.T_feed))
    body = rk.duty_hex(frac, "#1a3a5c", rk.ORANGE)
    vx, vy, vw, vh = 0.36, 0.10, 0.28, 0.78
    rk.vessel(ax, vx, vy, vw, vh, fc="#0b1420", ec=rk.FRAME, lw=2.2)
    rk.fill_level(ax, vx, vy, vw, vh, 0.95, color=body, alpha=0.5)
    cx = vx + vw / 2
    ax.plot([cx, cx], [vy + 0.05, vy + vh + 0.05], color=rk.DIM, lw=2, zorder=5)
    T_f, eff, scale = _feed(state, params)
    moved = abs(T_f - params.T_feed) > params.feed_T_wander or scale != 1.0
    rk.label(
        ax,
        0.06,
        0.94,
        f"ETHYLENE  {T_f:.1f} K  ×{scale:.2f} flow  ●",
        color=rk.ORANGE if moved else rk.PURPLE,
        size=7,
        ha="left",
    )
    if eff < 1.0:
        rk.label(
            ax,
            0.06,
            0.90,
            f"initiator efficiency {eff:.0%}  ●",
            color=rk.PURPLE,
            size=7,
            ha="left",
        )
    at = float(state.event_at)
    now = float(state.time) * params.delta_t
    if at < 1e8 and now < at:
        rk.label(
            ax,
            0.94,
            0.94,
            f"event in {at - now:.0f} s",
            color=rk.AMBER,
            size=7,
            ha="right",
        )
    rate = rk.clamp01(
        (float(state.I_f) - params.I_f_min) / (params.I_f_max - params.I_f_min)
    )
    rk.glow(ax, vx - 0.04, vy + vh * 0.5, 0.08, color=rk.AMBER, strength=rate * 2.0)
    rk.label(
        ax, cx, vy + vh + 0.10, f"T = {T:.1f} K", color=rk.TEXT, size=11, weight="bold"
    )
    rk.caption(ax, "adiabatic autoclave  ·  initiator holds the unstable point")


def render_autoclave(state, params, step, history):
    history["t"].append(step * params.delta_t / 60.0)
    history["target"].append(float(state.target_T))
    history["T"].append(float(state.T))
    history["I_f"].append(float(state.I_f))
    history["reward"].append(float(compute_reward(state, params)))

    err = abs(float(state.T) - float(state.target_T))
    status = rk.NOMINAL if err < 2.0 else (rk.WATCH if err < 10.0 else rk.ALARM)
    span = params.T_trip - params.T_feed
    gauges = [
        rk.Gauge(
            "T REACTOR",
            f"{float(state.T):.1f} K",
            (float(state.T) - params.T_feed) / span,
            rk.ORANGE,
            target_frac=(float(state.target_T) - params.T_feed) / span,
            limit_frac=1.0,
        ),
        rk.Gauge(
            "INITIATOR",
            f"{float(state.I_f) * 1e4:.2f} e-4",
            rk.clamp01(float(state.I_f) / params.I_f_max),
            rk.AMBER,
        ),
        rk.Gauge(
            "FEED T",
            f"{_feed(state, params)[0]:.1f} K",
            # Zero at the nominal feed; full scale past the wander and the step.
            (_feed(state, params)[0] - params.T_feed)
            / (1.5 * max(params.feed_T_wander, abs(params.feed_T_step), 1e-9)),
            rk.ORANGE,
            bipolar=True,
            neg_color=rk.BLUE,
            hidden=True,
        ),
        rk.cost_gauge(history["reward"], params),
    ]
    strip = rk.Strip(
        history["t"],
        [
            rk.Series(history["T"], "T", rk.ORANGE, fill_to=history["target"]),
            rk.Series(
                history["target"], "setpoint", "white", ls="--", lw=1.0, alpha=0.6
            ),
        ],
        ylabel="T  [K]",
        xlabel="minutes",
    )
    fig = rk.frame(
        title="LDPE  AUTOCLAVE",
        step=step,
        elapsed_s=step * params.delta_t,
        schematic=lambda ax: _draw_reactor(ax, state, params),
        schematic_title="REACTOR",
        gauges=gauges,
        strips=[strip],
        status=status,
        subtitle="free-radical polymerisation  ·  initiator feed is the input",
    )
    return rk.finish(fig), history


_render = rk.make_render_hook(render_autoclave, HISTORY_KEYS)
