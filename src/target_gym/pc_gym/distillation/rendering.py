"""Control-room rendering for the binary distillation column.

The column is drawn as its composition profile: every one of the 41 stages is a
band tinted from bottoms-rich to distillate-rich. Only the two ends are
measured -- the interior profile is the column's memory and is hidden -- so the
schematic shows exactly what the controller is flying blind through. The
product analysers are chromatographs that report a sample one cycle late, so
the strips draw the true product beside the late reading the controller gets.
The feed rate is metered; its composition is not.
"""

import numpy as np

from target_gym import render_kit as rk
from target_gym.pc_gym.distillation.env import N_STAGES, compute_reward

HISTORY_KEYS = (
    "t",
    "yD",
    "xB",
    "yD_read",
    "xB_read",
    "tyD",
    "txB",
    "F",
    "reward",
)


def _readings(state, params):
    """``(yD, xB)`` as the analysers report them: late and with scatter."""
    from target_gym.registry import REGISTRY

    obs = REGISTRY["distillation"].make_env().get_obs(state, params)
    return float(obs[0]), float(obs[1])


def _comp_hex(x):
    """Bottoms-rich (heavy) to distillate-rich (light)."""
    return rk.duty_hex(x, "#1a3550", rk.CYAN)


def _draw_column(ax, state, params):
    x = np.asarray(state.x)
    yD, xB = float(x[-1]), float(x[0])
    L, V = float(state.L), float(state.V)

    cx, cw = 0.44, 0.20
    y0, y1 = 0.16, 0.80
    n = len(x)
    band = (y1 - y0) / n

    rk.vessel(
        ax,
        cx - 0.012,
        y0 - 0.012,
        cw + 0.024,
        (y1 - y0) + 0.024,
        fc="#0b1420",
        ec=rk.FRAME,
        lw=2.0,
        pad=0.008,
    )
    for i, xi in enumerate(x):
        ax.add_patch(
            rk.patches.Rectangle(
                (cx, y0 + i * band),
                cw,
                band * 0.92,
                fc=_comp_hex(float(xi)),
                ec="none",
                alpha=0.85,
                zorder=3,
            )
        )

    # Feed enters mid-column.
    fy = y0 + (y1 - y0) * 0.5
    F = float(state.F)
    ff = rk.clamp01(F / ((1.0 + params.feed_step) * params.F))
    rk.pipe(ax, [(0.08, fy), (cx, fy)], color=rk.AMBER, lw=1.2 + 2.4 * ff)
    rk.flow_arrow(ax, cx - 0.05, fy, cx + 0.005, fy, color=rk.AMBER, lw=1.4)
    rk.label(
        ax,
        0.08,
        fy + 0.045,
        f"FEED  F {F:.2f} kmol/min",
        color=rk.AMBER,
        size=7,
        ha="left",
    )
    rk.label(
        ax,
        0.08,
        fy - 0.045,
        f"zF {float(state.zF):.3f}  ● hidden",
        color=rk.PURPLE,
        size=6.5,
        ha="left",
    )

    # Condenser and reflux.
    rk.vessel(
        ax,
        cx + cw + 0.10,
        y1 - 0.06,
        0.16,
        0.09,
        fc=rk.PANEL,
        ec=rk.FRAME,
        lw=1.5,
        pad=0.008,
    )
    rk.label(ax, cx + cw + 0.18, y1 - 0.015, "CONDENSER", color=rk.DIM, size=6.5)
    rk.pipe(
        ax,
        [
            (cx + cw / 2, y1 + 0.02),
            (cx + cw / 2, y1 + 0.07),
            (cx + cw + 0.18, y1 + 0.07),
            (cx + cw + 0.18, y1 + 0.03),
        ],
        color=rk.FRAME,
        lw=2.4,
    )
    lf = rk.clamp01((L - params.L_min) / (params.L_max - params.L_min))
    rk.pipe(
        ax,
        [
            (cx + cw + 0.18, y1 - 0.06),
            (cx + cw + 0.18, y1 - 0.115),
            (cx + cw / 2 + 0.02, y1 - 0.115),
        ],
        color=rk.BLUE,
        lw=1.2 + 3.0 * lf,
        alpha=0.85,
    )
    rk.flow_arrow(
        ax,
        cx + cw / 2 + 0.06,
        y1 - 0.115,
        cx + cw / 2 + 0.01,
        y1 - 0.115,
        color=rk.BLUE,
        lw=1.4,
    )
    rk.label(
        ax,
        cx + cw + 0.19,
        y1 - 0.155,
        f"REFLUX  L {L:.3f}",
        color=rk.BLUE,
        size=7,
        ha="center",
    )

    # Distillate.
    rk.flow_arrow(
        ax,
        cx + cw + 0.26,
        y1 - 0.015,
        cx + cw + 0.34,
        y1 - 0.015,
        color=_comp_hex(yD),
        lw=1.8,
    )
    rk.label(
        ax,
        cx + cw + 0.30,
        y1 + 0.035,
        f"yD {yD:.4f}",
        color=_comp_hex(yD),
        size=8.5,
        weight="bold",
    )

    # Reboiler and boil-up.
    rk.vessel(
        ax,
        cx - cw - 0.10,
        y0 - 0.03,
        0.16,
        0.09,
        fc=rk.PANEL,
        ec=rk.FRAME,
        lw=1.5,
        pad=0.008,
    )
    rk.label(ax, cx - cw - 0.02, y0 - 0.062, "REBOILER", color=rk.DIM, size=6.5)
    vf = rk.clamp01((V - params.V_min) / (params.V_max - params.V_min))
    rk.glow(ax, cx - cw - 0.02, y0 + 0.015, 0.07, color=rk.RED, strength=vf)
    rk.pipe(
        ax,
        [
            (cx - cw - 0.02, y0 + 0.06),
            (cx - cw - 0.02, y0 + 0.10),
            (cx + 0.02, y0 + 0.10),
        ],
        color=rk.RED,
        lw=1.2 + 3.0 * vf,
        alpha=0.85,
    )
    rk.flow_arrow(
        ax, cx + 0.02, y0 + 0.10, cx + cw / 2, y0 + 0.10, color=rk.RED, lw=1.4
    )
    rk.label(
        ax, cx - cw - 0.02, y0 + 0.135, f"BOIL-UP  V {V:.3f}", color=rk.RED, size=7
    )

    # Bottoms.
    rk.flow_arrow(
        ax, cx + cw / 2, y0 - 0.02, cx + cw / 2, y0 - 0.075, color=_comp_hex(xB), lw=1.8
    )
    rk.label(
        ax,
        cx + cw / 2,
        y0 - 0.105,
        f"xB {xB:.4f}",
        color=_comp_hex(xB),
        size=8.5,
        weight="bold",
    )

    rk.label(
        ax,
        cx + cw + 0.045,
        y0 + (y1 - y0) * 0.5,
        f"{N_STAGES} stages\n● interior HIDDEN",
        color=rk.PURPLE,
        size=6.5,
        ha="left",
    )
    rk.caption(ax, "ill-conditioned: reflux and boil-up move both products together")


def render_distillation(state, params, step, history):
    x = np.asarray(state.x)
    history["t"].append(step * params.delta_t)
    history["yD"].append(float(x[-1]))
    history["xB"].append(float(x[0]))
    yD_read, xB_read = _readings(state, params)
    history["yD_read"].append(yD_read)
    history["xB_read"].append(xB_read)
    history["tyD"].append(float(state.target_yD))
    history["txB"].append(float(state.target_xB))
    history["F"].append(float(state.F))
    history["reward"].append(float(compute_reward(state, params)))

    # Off spec past either zone's edge (PHYSICS.md, Limits).
    e = max(
        float(state.target_yD) - float(x[-1]),
        float(x[0]) - float(state.target_xB),
    )
    status = rk.NOMINAL if e <= 0.0 else (rk.WATCH if e < 0.01 else rk.ALARM)

    gauges = [
        rk.Gauge(
            "yD  READ",
            f"{yD_read:.5f}",
            yD_read,
            rk.CYAN,
            target_frac=float(state.target_yD),
        ),
        rk.Gauge(
            "xB  READ",
            f"{xB_read:.5f}",
            xB_read * 20.0,
            rk.TEAL,
            target_frac=float(state.target_xB) * 20.0,
        ),
        rk.Gauge(
            "REFLUX L",
            f"{float(state.L):.3f}",
            (float(state.L) - params.L_min) / (params.L_max - params.L_min),
            rk.BLUE,
            limit_frac=1.0,
        ),
        rk.Gauge(
            "BOIL-UP V",
            f"{float(state.V):.3f}",
            (float(state.V) - params.V_min) / (params.V_max - params.V_min),
            rk.RED,
            limit_frac=1.0,
        ),
        rk.Gauge(
            "FEED F",
            f"{float(state.F):.3f} kmol/min",
            float(state.F) / ((1.0 + params.feed_step) * params.F),
            rk.AMBER,
            target_frac=1.0 / (1.0 + params.feed_step),
        ),
        rk.Gauge(
            "FEED zF",
            f"{float(state.zF):.4f}",
            (float(state.zF) - params.zF_nominal + 2 * params.zF_step)
            / (4 * params.zF_step),
            rk.PURPLE,
            hidden=True,
        ),
        rk.cost_gauge(history["reward"], params, "kmol"),
    ]

    strips = [
        rk.Strip(
            history["t"],
            [
                rk.Series(history["yD"], "yD true", rk.CYAN, fill_to=history["tyD"]),
                rk.Series(
                    history["yD_read"], "analyser (late)", rk.TEXT, ls=":", lw=1.2
                ),
                rk.Series(history["tyD"], "spec", "white", ls="--", lw=1.0, alpha=0.6),
            ],
            ylabel="yD  top",
        ),
        rk.Strip(
            history["t"],
            [
                rk.Series(history["xB"], "xB true", rk.TEAL, fill_to=history["txB"]),
                rk.Series(
                    history["xB_read"], "analyser (late)", rk.TEXT, ls=":", lw=1.2
                ),
                rk.Series(history["txB"], "spec", "white", ls="--", lw=1.0, alpha=0.6),
            ],
            ylabel="xB  bottom",
            xlabel="minutes",
        ),
    ]
    fig = rk.frame(
        title="BINARY  DISTILLATION  —  COLUMN  A",
        step=step,
        elapsed_s=step * params.delta_t * 60.0,
        schematic=lambda ax: _draw_column(ax, state, params),
        schematic_title="COLUMN",
        gauges=gauges,
        strips=strips,
        status=status,
        subtitle="dual composition control  ·  alpha 1.5  ·  condition number ~140",
    )
    return rk.finish(fig), history


_render = rk.make_render_hook(render_distillation, HISTORY_KEYS)
