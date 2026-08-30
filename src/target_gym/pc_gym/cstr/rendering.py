"""Control-room rendering for the jacketed CSTR.

The schematic is the reactor itself: contents tinted by bulk temperature, a
cooling jacket tinted by coolant temperature, and a reaction bloom whose
intensity follows conversion. The exothermic loop the controller is fighting --
hotter contents burn reactant faster, which releases more heat -- is the thing
the picture is meant to make legible.
"""

from target_gym import render_kit as rk
from target_gym.pc_gym.cstr.env import compute_reward

HISTORY_KEYS = ("t", "C_a", "target", "T", "T_c", "reward")


def _draw_reactor(ax, state, params):
    C_a = float(state.C_a)
    T = float(state.T)
    T_c = float(state.T_c)

    vx, vy, vw, vh = 0.26, 0.16, 0.46, 0.62
    jacket_pad = 0.055

    # Cooling jacket -- the manipulated variable, drawn as the outer shell.
    # Coolant: deep blue when cold (strong cooling), pale once it has warmed
    # toward the reactor. Red would read as danger on a cooling jacket.
    jc_frac = rk.clamp01((T_c - params.T_c_min) / (params.T_c_max - params.T_c_min))
    jc = rk.duty_hex(jc_frac, "#1565c0", "#8aa8bd")
    rk.vessel(
        ax,
        vx - jacket_pad,
        vy - jacket_pad * 0.6,
        vw + 2 * jacket_pad,
        vh + jacket_pad * 1.2,
        fc=jc,
        ec=rk.FRAME,
        lw=1.8,
        alpha=0.30,
    )
    rk.label(
        ax,
        vx - jacket_pad - 0.02,
        vy + vh * 0.5,
        "JACKET",
        color=rk.DIM,
        size=7,
        ha="right",
        rotation=90,
    )

    # Reactor contents, tinted by bulk temperature.
    body = rk.duty_hex(
        (T - params.T_min) / (params.T_max - params.T_min), "#1a3a5c", rk.ORANGE
    )
    rk.vessel(ax, vx, vy, vw, vh, fc="#0b1420", ec=rk.FRAME, lw=2.2)
    rk.fill_level(ax, vx, vy, vw, vh, 0.84, color=body, alpha=0.45)

    # Reaction bloom: how much of the feed has been consumed.
    conv = rk.clamp01((params.Caf - C_a) / max(params.Caf, 1e-9))
    rk.glow(ax, vx + vw / 2, vy + vh * 0.42, 0.16, color=rk.AMBER, strength=conv * 2.2)

    # Impeller.
    cx, cy = vx + vw / 2, vy + vh * 0.40
    ax.plot([cx, cx], [cy, vy + vh + 0.05], color=rk.DIM, lw=2, zorder=5)
    for dx in (-0.075, 0.075):
        ax.plot(
            [cx, cx + dx], [cy, cy - 0.028], color=rk.TEXT, lw=2.4, alpha=0.75, zorder=5
        )

    # Feed and product.
    rk.pipe(
        ax,
        [(0.06, 0.90), (cx - 0.14, 0.90), (cx - 0.14, vy + vh - 0.02)],
        color=rk.FRAME,
        lw=4,
    )
    rk.flow_arrow(
        ax, cx - 0.14, vy + vh + 0.03, cx - 0.14, vy + vh - 0.03, color=rk.CYAN, lw=1.5
    )
    rk.label(
        ax,
        0.06,
        0.945,
        f"FEED  Caf {params.Caf:.2f}  Ti {params.Ti:.0f}K",
        color=rk.DIM,
        size=7,
        ha="left",
    )

    rk.pipe(ax, [(cx, vy), (cx, 0.07), (0.90, 0.07)], color=rk.FRAME, lw=4)
    rk.flow_arrow(ax, 0.80, 0.07, 0.93, 0.07, color=body, lw=1.6)
    rk.label(
        ax, 0.93, 0.135, f"PRODUCT  Ca {C_a:.3f}", color=rk.TEXT, size=7.5, ha="right"
    )

    # Coolant loop.
    rk.flow_arrow(ax, 0.10, 0.30, 0.10, 0.58, color=jc, lw=1.8)
    rk.label(ax, 0.10, 0.25, f"Tc {T_c:.1f}K", color=jc, size=7.5)

    rk.label(
        ax, cx, vy + vh + 0.14, f"T = {T:.1f} K", color=rk.TEXT, size=11, weight="bold"
    )
    rk.caption(ax, "exothermic A -> B   ·   jacket removes the heat")


def render_cstr(state, params, step, history):
    history["t"].append(step * params.delta_t)
    history["C_a"].append(float(state.C_a))
    history["target"].append(float(state.target_CA))
    history["T"].append(float(state.T))
    history["T_c"].append(float(state.T_c))
    history["reward"].append(float(compute_reward(state, params)))

    err = abs(float(state.C_a) - float(state.target_CA))
    status = rk.NOMINAL if err < 0.01 else (rk.WATCH if err < 0.04 else rk.ALARM)

    ca_span = params.C_a_max - params.C_a_min
    t_span = params.T_max - params.T_min
    tc_span = params.T_c_max - params.T_c_min
    conv = rk.clamp01((params.Caf - float(state.C_a)) / params.Caf)

    gauges = [
        rk.Gauge(
            "Ca",
            f"{float(state.C_a):.4f}",
            (float(state.C_a) - params.C_a_min) / ca_span,
            rk.CYAN,
            target_frac=(float(state.target_CA) - params.C_a_min) / ca_span,
        ),
        rk.Gauge(
            "TARGET Ca",
            f"{float(state.target_CA):.4f}",
            (float(state.target_CA) - params.C_a_min) / ca_span,
            rk.DIM,
        ),
        rk.Gauge(
            "T REACTOR",
            f"{float(state.T):.1f} K",
            (float(state.T) - params.T_min) / t_span,
            rk.ORANGE,
            limit_frac=1.0,
        ),
        rk.Gauge(
            "T COOLANT",
            f"{float(state.T_c):.1f} K",
            (float(state.T_c) - params.T_c_min) / tc_span,
            rk.BLUE,
        ),
        rk.Gauge("CONVERSION", f"{conv * 100:.1f} %", conv, rk.AMBER),
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
            rk.Series(history["C_a"], "Ca", rk.CYAN, fill_to=history["target"]),
            rk.Series(history["target"], "target", "white", ls="--", lw=1.0, alpha=0.6),
        ],
        ylabel="Ca  [mol/L]",
        xlabel="minutes",
    )
    fig = rk.frame(
        title="CSTR  —  JACKETED  REACTOR",
        step=step,
        elapsed_s=step * params.delta_t * 60.0,
        schematic=lambda ax: _draw_reactor(ax, state, params),
        schematic_title="REACTOR",
        gauges=gauges,
        strips=[strip],
        status=status,
        subtitle="exothermic first-order reaction  ·  coolant temperature is the input",
    )
    return rk.finish(fig), history


_render = rk.make_render_hook(render_cstr, HISTORY_KEYS, stride=10)
