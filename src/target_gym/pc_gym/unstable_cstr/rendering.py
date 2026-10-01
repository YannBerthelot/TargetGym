"""Control-room rendering for the unstable CSTR.

The schematic is cstr's reactor: contents tinted by bulk temperature, a
jacket tinted by the lagged jacket temperature. What this task adds to the
picture is the edge the controller works under: the reactor temperature is
drawn against the 365 K trip and against the point-of-no-return line at the
current concentration, a few kelvin above every target.
"""

import numpy as np

from target_gym import render_kit as rk
from target_gym.pc_gym.unstable_cstr.env import live_target

HISTORY_KEYS = ("t", "C_a", "target", "T", "T_j", "error")

# Contents colour scale (K): the extinguished branch at the cold end, the trip
# at the hot end.
_T_COLD, _T_HOT = 320.0, 365.0
# Gauge spans: C_a from the lowest untripped concentration to the feed, T from
# the extinguished branch to just past the trip.
_CA_LO, _CA_HI = 0.25, 1.0
_T_LO, _T_HI = 310.0, 370.0


def _draw_reactor(ax, state, params, pnr):
    C_a = float(state.C_a)
    T = float(state.T)
    T_j = float(state.T_j)

    vx, vy, vw, vh = 0.26, 0.16, 0.46, 0.62
    jacket_pad = 0.055

    # Jacket: deep blue when cold (strong cooling), pale once warmed toward the
    # reactor. Tinted by the jacket temperature itself, which lags the command.
    jc_frac = rk.clamp01((T_j - params.T_c_min) / (params.T_c_max - params.T_c_min))
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

    # Contents, tinted by bulk temperature over the extinguished branch to trip.
    body = rk.duty_hex((T - _T_COLD) / (_T_HOT - _T_COLD), "#1a3a5c", rk.ORANGE)
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

    # Feed and product. The feed temperature shown includes the drift, which
    # the controller does not see.
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
        f"FEED  Caf {params.Caf:.2f}  Ti {params.Ti + float(state.Ti_dev):.1f}K",
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
    rk.label(ax, 0.10, 0.25, f"Tj {T_j:.1f}K", color=jc, size=7.5)

    rk.label(
        ax, cx, vy + vh + 0.14, f"T = {T:.1f} K", color=rk.TEXT, size=11, weight="bold"
    )
    rk.label(
        ax,
        cx,
        vy + vh + 0.08,
        f"PNR line {pnr:.1f} K   trip {params.T_trip:.0f} K",
        color=rk.AMBER if T < pnr else rk.RED,
        size=7.5,
    )
    rk.caption(ax, "exothermic A -> B   ·   held on the unstable middle branch")


def render_unstable_cstr(state, params, step, history):
    # The point-of-no-return line lives with the controllers, outside the
    # version stamp; imported here so the physics never depends on it.
    from target_gym.pc_gym.unstable_cstr.experts import MPC_PNR_MARGIN_K, pnr_line

    target = float(live_target(state, params))
    C_a = float(state.C_a)
    T = float(state.T)
    T_j = float(state.T_j)
    err = abs(C_a - target)

    history["t"].append(step * params.delta_t)
    history["C_a"].append(C_a)
    history["target"].append(target)
    history["T"].append(T)
    history["T_j"].append(T_j)
    history["error"].append(err)

    pnr = float(pnr_line(C_a))
    # WATCH from the MPC's soft bound on T, ALARM at the line itself.
    status = (
        rk.ALARM
        if T >= pnr or err >= 0.04
        else (rk.WATCH if err >= 0.01 or T >= pnr - MPC_PNR_MARGIN_K else rk.NOMINAL)
    )

    ca_span = _CA_HI - _CA_LO
    t_span = _T_HI - _T_LO
    tc_span = params.T_c_max - params.T_c_min
    drift_span = 3.0 * params.Ti_sigma

    gauges = [
        rk.Gauge(
            "Ca",
            f"{C_a:.4f}",
            (C_a - _CA_LO) / ca_span,
            rk.CYAN,
            target_frac=(target - _CA_LO) / ca_span,
        ),
        rk.Gauge(
            "T REACTOR",
            f"{T:.1f} K",
            (T - _T_LO) / t_span,
            rk.ORANGE,
            limit_frac=(params.T_trip - _T_LO) / t_span,
            # The white marker is the point-of-no-return line at this C_a.
            target_frac=(pnr - _T_LO) / t_span,
        ),
        rk.Gauge(
            "T JACKET",
            f"{T_j:.1f} K",
            (T_j - params.T_c_min) / tc_span,
            rk.BLUE,
        ),
        rk.Gauge(
            "FEED DRIFT",
            f"{float(state.Ti_dev):+.2f} K",
            float(np.clip(float(state.Ti_dev) / drift_span, -1.0, 1.0)),
            rk.PURPLE,
            bipolar=True,
            neg_color=rk.TEAL,
            hidden=True,
        ),
        rk.Gauge(
            "|e| Ca",
            f"{err:.2e}",
            rk.clamp01(np.log10(max(err, 1e-5) / 1e-5) / 4.0),
            rk.AMBER,
        ),
    ]

    strips = [
        rk.Strip(
            history["t"],
            [
                rk.Series(history["C_a"], "Ca", rk.CYAN, fill_to=history["target"]),
                rk.Series(
                    history["target"], "target", "white", ls="--", lw=1.0, alpha=0.6
                ),
            ],
            ylabel="Ca  [mol/L]",
        ),
        rk.Strip(
            history["t"],
            [
                rk.Series(history["T"], "T", rk.ORANGE),
                rk.Series(
                    list(pnr_line(np.asarray(history["C_a"]))),
                    "PNR line",
                    rk.AMBER,
                    ls=":",
                    lw=1.0,
                    alpha=0.8,
                ),
            ],
            ylabel="T  [K]",
            xlabel="minutes",
            lines=[(params.T_trip, rk.RED, "trip")],
        ),
    ]
    fig = rk.frame(
        title="UNSTABLE CSTR  ·  MIDDLE BRANCH",
        step=step,
        elapsed_s=step * params.delta_t * params.time_unit_seconds,
        schematic=lambda ax: _draw_reactor(ax, state, params, pnr),
        schematic_title="REACTOR",
        gauges=gauges,
        strips=strips,
        status=status,
        subtitle="open-loop unstable at every target  ·  coolant command is the input",
    )
    return rk.finish(fig), history


_render = rk.make_render_hook(render_unstable_cstr, HISTORY_KEYS)
