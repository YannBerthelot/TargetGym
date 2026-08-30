"""Control-room rendering for the regenerative float-glass furnace.

The schematic is a section through the furnace with both regenerator chambers
drawn either side. Firing alternates between them every 25 minutes, so the
picture shows which side is burning and which is being reheated -- the checker
stacks carry the furnace's memory and none of them is measured.
"""

import numpy as np

from target_gym import render_kit as rk
from target_gym.glass_furnace.env import (
    N_REGEN_NODES,
    compute_reward,
    reversal_phase,
)

HISTORY_KEYS = (
    "t",
    "T_crown",
    "target",
    "T_melt",
    "T_work",
    "fuel",
    "m_batch",
    "T_air",
    "reward",
)

_T_LO, _T_HI = 400.0, 1750.0


def _draw_furnace(ax, state, params):
    T_crown = float(state.T_crown)
    T_melt = float(state.T_melt)
    T_work = float(state.T_work)
    T_gas = float(state.T_gas)
    fuel = float(state.fuel_flow)
    fires_a = bool(np.asarray(reversal_phase(state.time, params)) > 0.5)

    # Furnace shell.
    fx, fy, fw, fh = 0.28, 0.30, 0.44, 0.34
    crown_c = rk.incandescence(T_crown, 1200.0, 1750.0)
    rk.vessel(ax, fx, fy, fw, fh, fc="#0b1420", ec=rk.FRAME, lw=2.2)

    # Crown: the controlled surface, glowing with its own temperature.
    ax.add_patch(
        rk.patches.Rectangle(
            (fx, fy + fh * 0.66),
            fw,
            fh * 0.34,
            fc=crown_c,
            ec="none",
            alpha=0.55,
            zorder=3,
        )
    )
    rk.label(
        ax,
        fx + fw / 2,
        fy + fh * 0.83,
        f"CROWN  {T_crown:.0f} C",
        color=rk.TEXT,
        size=9,
        weight="bold",
        zorder=6,
    )

    # Combustion space and flame.
    rk.glow(
        ax,
        fx + fw / 2,
        fy + fh * 0.58,
        0.14,
        color=rk.AMBER,
        strength=rk.clamp01(
            (fuel - params.fuel_min) / (params.fuel_max - params.fuel_min)
        ),
        zorder=2,
    )
    rk.label(
        ax,
        fx + fw / 2,
        fy + fh * 0.58,
        f"gas {T_gas:.0f} C",
        color=rk.AMBER,
        size=7,
        zorder=6,
    )

    # Melt and working end.
    melt_c = rk.incandescence(T_melt, 900.0, 1750.0)
    ax.add_patch(
        rk.patches.Rectangle(
            (fx, fy + 0.012), fw, fh * 0.36, fc=melt_c, ec="none", alpha=0.70, zorder=3
        )
    )
    rk.label(
        ax,
        fx + fw * 0.32,
        fy + fh * 0.19,
        f"melt {T_melt:.0f}",
        color="#20140a",
        size=7,
        weight="bold",
        zorder=6,
    )
    rk.label(
        ax,
        fx + fw * 0.78,
        fy + fh * 0.19,
        f"work {T_work:.0f}",
        color="#20140a",
        size=7,
        weight="bold",
        zorder=6,
    )

    # Batch blanket floating on the melt at the doghouse end.
    bf = rk.clamp01(float(state.m_batch) / params.m_batch_full)
    if bf > 0.005:
        ax.add_patch(
            rk.patches.Rectangle(
                (fx + 0.008, fy + fh * 0.37),
                fw * 0.46 * bf,
                fh * 0.07,
                fc="#9aa7b3",
                ec="none",
                alpha=0.85,
                zorder=5,
            )
        )
        rk.label(
            ax,
            fx + 0.02,
            fy + fh * 0.47,
            f"batch {bf * 100:.0f}%",
            color="#9aa7b3",
            size=6.5,
            ha="left",
            zorder=6,
        )

    # Regenerator chambers. Hot end at the top of each stack.
    for side, (label_txt, temps, firing) in enumerate(
        (
            ("A", np.asarray(state.T_rA), fires_a),
            ("B", np.asarray(state.T_rB), not fires_a),
        )
    ):
        rxc = 0.13 if side == 0 else 0.87
        ry0, rh, rw = 0.16, 0.44, 0.13
        rk.vessel(
            ax,
            rxc - rw / 2,
            ry0,
            rw,
            rh,
            fc="#0b1420",
            ec=rk.AMBER if firing else rk.FRAME,
            lw=2.4 if firing else 1.4,
        )
        nh = rh / N_REGEN_NODES
        for i, T in enumerate(temps):  # index 0 is the hot end
            ax.add_patch(
                rk.patches.Rectangle(
                    (rxc - rw / 2 + 0.006, ry0 + rh - (i + 1) * nh + 0.004),
                    rw - 0.012,
                    nh - 0.008,
                    fc=rk.incandescence(float(T), _T_LO, _T_HI),
                    ec="none",
                    alpha=0.8,
                    zorder=3,
                )
            )
        rk.label(
            ax,
            rxc,
            ry0 + rh + 0.045,
            f"REGEN {label_txt}",
            color=rk.AMBER if firing else rk.DIM,
            size=7.5,
            weight="bold",
        )
        rk.label(
            ax,
            rxc,
            ry0 - 0.035,
            "FIRING" if firing else "EXHAUST",
            color=rk.AMBER if firing else rk.BLUE,
            size=7,
        )
        # Duct to the furnace.
        xin = fx if side == 0 else fx + fw
        rk.pipe(
            ax,
            [(rxc, ry0 + rh - 0.02), (rxc, fy + fh * 0.60), (xin, fy + fh * 0.60)],
            color=rk.AMBER if firing else rk.BLUE,
            lw=2.6,
            alpha=0.8,
        )
        if firing:
            rk.flow_arrow(
                ax,
                xin - 0.05 if side == 0 else xin + 0.05,
                fy + fh * 0.60,
                xin,
                fy + fh * 0.60,
                color=rk.AMBER,
                lw=1.6,
            )

    rk.label(
        ax,
        0.5,
        0.94,
        f"FUEL {fuel:.3f} kg/s   ·   preheat {float(state.T_air_preheat):.0f} C",
        color=rk.TEXT,
        size=8,
    )
    rk.label(ax, 0.5, 0.895, "● checker stacks are HIDDEN", color=rk.PURPLE, size=6.5)
    rk.caption(ax, f"regenerators reverse every {params.reversal_period / 60:.0f} min")


def render_glass_furnace(state, params, step, history):
    history["t"].append(step * params.delta_t / 3600.0)
    history["T_crown"].append(float(state.T_crown))
    history["target"].append(float(state.target_T_crown))
    history["T_melt"].append(float(state.T_melt))
    history["T_work"].append(float(state.T_work))
    history["fuel"].append(float(state.fuel_flow))
    history["m_batch"].append(float(state.m_batch))
    history["T_air"].append(float(state.T_air_preheat))
    history["reward"].append(float(compute_reward(state, params)))

    err = abs(float(state.T_crown) - float(state.target_T_crown))
    status = rk.NOMINAL if err < 8 else (rk.WATCH if err < 25 else rk.ALARM)
    span = params.T_crown_max - params.T_crown_min
    f = lambda T: (float(T) - params.T_crown_min) / span

    gauges = [
        rk.Gauge(
            "T CROWN",
            f"{float(state.T_crown):.1f} C",
            f(state.T_crown),
            rk.ORANGE,
            target_frac=f(state.target_T_crown),
            limit_frac=1.0,
        ),
        rk.Gauge(
            "TARGET",
            f"{float(state.target_T_crown):.1f} C",
            f(state.target_T_crown),
            rk.DIM,
        ),
        rk.Gauge(
            "T MELT",
            f"{float(state.T_melt):.1f} C",
            f(state.T_melt),
            rk.RED,
            hidden=True,
        ),
        rk.Gauge(
            "T WORK",
            f"{float(state.T_work):.1f} C",
            f(state.T_work),
            rk.AMBER,
            hidden=True,
        ),
        rk.Gauge(
            "AIR PREHEAT",
            f"{float(state.T_air_preheat):.0f} C",
            (float(state.T_air_preheat) - _T_LO) / (_T_HI - _T_LO),
            rk.CYAN,
            hidden=True,
        ),
        rk.Gauge(
            "BATCH",
            f"{float(state.m_batch) / 1000:.1f} t",
            float(state.m_batch) / params.m_batch_full,
            rk.BLUE,
            hidden=True,
        ),
        rk.Gauge(
            "FUEL",
            f"{float(state.fuel_flow):.3f} kg/s",
            (float(state.fuel_flow) - params.fuel_min)
            / (params.fuel_max - params.fuel_min),
            rk.GREEN,
            limit_frac=1.0,
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
            rk.Series(
                history["T_crown"], "crown", rk.ORANGE, fill_to=history["target"]
            ),
            rk.Series(history["target"], "target", "white", ls="--", lw=1.0, alpha=0.6),
            rk.Series(history["T_melt"], "melt (hidden)", rk.RED, lw=1.1, alpha=0.7),
        ],
        ylabel="temperature  [C]",
        xlabel="hours",
    )
    fig = rk.frame(
        title="GLASS  FURNACE  —  REGENERATIVE",
        step=step,
        elapsed_s=step * params.delta_t,
        schematic=lambda ax: _draw_furnace(ax, state, params),
        schematic_title="FURNACE  SECTION",
        gauges=gauges,
        strips=[strip],
        status=status,
        subtitle="float glass  ·  6 of 9 states hidden  ·  multi-hour transients",
    )
    return rk.finish(fig), history


_render = rk.make_render_hook(render_glass_furnace, HISTORY_KEYS, stride=10)
