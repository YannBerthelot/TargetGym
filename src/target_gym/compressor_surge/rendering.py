"""Control-room rendering for the compressor.

The schematic is the compressor map at the current speed: the pressure-rise
characteristic against the flow coefficient, the surge line at its peak, the
anti-surge controller's control line to the right of it, the throttle curves
the plenum sees, and the operating point. The gauges and strips put the header
pressure beside its setpoint and the surge margin beside the trip.
"""

import numpy as np

from target_gym import render_kit as rk
from target_gym.compressor_surge.env import (
    KPA,
    characteristic,
    consumer_flow,
    demand_opening,
    live_target,
    recycle_flow,
    suction_density,
    surge_flow_per_speed,
)

HISTORY_KEYS = ("t", "dp", "p_ref", "margin", "N", "x", "m_c", "m_d")

# Map extent in (Phi, Psi), and where it sits in the schematic's unit square.
_PHI_LO, _PHI_HI = -0.05, 0.85
_PSI_LO, _PSI_HI = 0.0, 0.75
_X0, _X1, _Y0, _Y1 = 0.12, 0.95, 0.14, 0.90
# Gauge spans: header pressure over the reachable range, margin to 40 %.
_DP_LO, _DP_HI = 5.0, 36.0  # kPa
_MARGIN_HI = 0.40


def _to_axes(phi, psi):
    u = _X0 + (np.asarray(phi) - _PHI_LO) / (_PHI_HI - _PHI_LO) * (_X1 - _X0)
    v = _Y0 + (np.asarray(psi) - _PSI_LO) / (_PSI_HI - _PSI_LO) * (_Y1 - _Y0)
    return u, v


def _draw_map(ax, state, params, opening, control_line):
    """The compressor map at the current speed, in (Phi, Psi)."""
    rho = suction_density(params)
    N = float(state.N)
    U = N * params.U_r
    flow_scale = rho * params.A_c * U
    head_scale = rho * U**2

    # Axes and grid.
    ax.plot([_X0, _X1], [_Y0, _Y0], color=rk.FRAME, lw=1.2)
    ax.plot([_X0, _X0], [_Y0, _Y1], color=rk.FRAME, lw=1.2)
    for phi_tick in (0.0, 0.2, 0.4, 0.6, 0.8):
        u, _ = _to_axes(phi_tick, _PSI_LO)
        rk.label(ax, float(u), _Y0 - 0.035, f"{phi_tick:.1f}", color=rk.DIM, size=6.5)
    for psi_tick in (0.2, 0.4, 0.6):
        _, v = _to_axes(_PHI_LO, psi_tick)
        rk.label(ax, _X0 - 0.03, float(v), f"{psi_tick:.1f}", color=rk.DIM, size=6.5)
    rk.label(
        ax, 0.5 * (_X0 + _X1), _Y0 - 0.075, "flow coefficient Phi", color=rk.DIM, size=7
    )
    rk.label(
        ax,
        _X0 - 0.075,
        0.5 * (_Y0 + _Y1),
        "head Psi",
        color=rk.DIM,
        size=7,
        rotation=90,
    )

    # Characteristic, with its tangent continuation past phi_join.
    phi = np.linspace(_PHI_LO, _PHI_HI, 181)
    psi = np.asarray(characteristic(phi, params, np))
    keep = (psi >= _PSI_LO) & (psi <= _PSI_HI)
    u, v = _to_axes(phi[keep], psi[keep])
    ax.plot(u, v, color=rk.CYAN, lw=2.0, zorder=3)

    # Surge line at the peak (the trip) and the controller's control line.
    phi_surge = 2.0 * params.W
    u_s, _ = _to_axes(phi_surge, _PSI_LO)
    ax.plot([u_s, u_s], [_Y0, _Y1], color=rk.RED, lw=1.4, zorder=2)
    rk.label(
        ax, float(u_s) - 0.01, _Y1 + 0.03, "SURGE", color=rk.RED, size=7, ha="right"
    )
    u_c, _ = _to_axes(phi_surge * (1.0 + control_line), _PSI_LO)
    ax.plot([u_c, u_c], [_Y0, _Y1], color=rk.AMBER, lw=1.0, ls="--", zorder=2)
    rk.label(
        ax,
        float(u_c) + 0.01,
        _Y1 + 0.03,
        f"{100 * control_line:.0f} % control line",
        color=rk.AMBER,
        size=7,
        ha="left",
    )

    # Throttle curves: the flow each valve passes at the head, in Phi.
    psi_t = np.linspace(0.005, _PSI_HI, 120)
    dp_t = head_scale * psi_t
    m_d = np.asarray(consumer_flow(dp_t, opening, params, np))
    m_r = np.asarray(recycle_flow(dp_t, float(state.x), params, np))
    for flow, color in ((m_d, rk.GREEN), (m_d + m_r, rk.PURPLE)):
        phi_t = flow / flow_scale
        keep = phi_t <= _PHI_HI
        u, v = _to_axes(phi_t[keep], psi_t[keep])
        ax.plot(u, v, color=color, lw=1.2, ls=":", alpha=0.9, zorder=2)

    # Operating point.
    phi_op = float(state.m_c) / flow_scale
    psi_op = float(state.dp) / head_scale
    u_op, v_op = _to_axes(
        np.clip(phi_op, _PHI_LO, _PHI_HI), np.clip(psi_op, _PSI_LO, _PSI_HI)
    )
    rk.glow(ax, float(u_op), float(v_op), 0.03, color=rk.ORANGE, strength=1.2)
    ax.plot([u_op], [v_op], "o", color=rk.ORANGE, ms=7, zorder=6)
    rk.caption(
        ax,
        f"speed {100 * N:.1f} %  ·  dotted: consumer throttle (green), with recycle (purple)",
    )


def render_compressor_surge(state, params, step, history):
    # The control line lives with the controllers, outside the version stamp;
    # imported here so the physics never depends on it.
    from target_gym.compressor_surge.experts import SURGE_CONTROL_LINE

    tau = int(state.block_clock) * params.delta_t
    opening = float(
        demand_opening(state.demand_levels, tau, float(state.demand_dev), params, np)
    )
    dp = float(state.dp) / KPA
    target = float(live_target(state, params)) / KPA
    N = float(state.N)
    x = float(state.x)
    m_c = float(state.m_c)
    m_d = float(consumer_flow(float(state.dp), opening, params, np))
    margin = m_c / (float(surge_flow_per_speed(params)) * N) - 1.0
    err = abs(dp - target)

    history["t"].append(step * params.delta_t)
    history["dp"].append(dp)
    history["p_ref"].append(target)
    history["margin"].append(100.0 * margin)
    history["N"].append(100.0 * N)
    history["x"].append(100.0 * x)
    history["m_c"].append(m_c)
    history["m_d"].append(m_d)

    # WATCH inside the control line or a kPa off, ALARM near the line.
    status = (
        rk.ALARM
        if margin <= 0.5 * SURGE_CONTROL_LINE or err >= 3.0
        else (rk.WATCH if margin < SURGE_CONTROL_LINE or err >= 1.0 else rk.NOMINAL)
    )

    dp_span = _DP_HI - _DP_LO
    speed_span = params.N_max - params.N_min
    gauges = [
        rk.Gauge(
            "HEADER P",
            f"{dp:.2f} kPa",
            (dp - _DP_LO) / dp_span,
            rk.CYAN,
            target_frac=(target - _DP_LO) / dp_span,
        ),
        rk.Gauge(
            "SPEED",
            f"{100 * N:.1f} %",
            (N - params.N_min) / speed_span,
            rk.BLUE,
            # The white marker is the drive setpoint the speed lags behind.
            target_frac=(float(state.N_ramp) - params.N_min) / speed_span,
        ),
        rk.Gauge("RECYCLE", f"{100 * x:.0f} %", x, rk.PURPLE),
        rk.Gauge(
            "SURGE MARGIN",
            f"{100 * margin:+.1f} %",
            float(np.clip(margin / _MARGIN_HI, -1.0, 1.0)),
            rk.GREEN,
            bipolar=True,
            neg_color=rk.RED,
            target_frac=SURGE_CONTROL_LINE / _MARGIN_HI,
        ),
        rk.Gauge(
            "DEMAND DEV",
            f"{float(state.demand_dev):+.3f}",
            float(
                np.clip(float(state.demand_dev) / (3.0 * params.demand_sigma), -1, 1)
            ),
            rk.TEAL,
            bipolar=True,
            neg_color=rk.PINK,
            hidden=True,
        ),
    ]

    strips = [
        rk.Strip(
            history["t"],
            [
                rk.Series(history["dp"], "header", rk.CYAN, fill_to=history["p_ref"]),
                rk.Series(
                    history["p_ref"], "setpoint", "white", ls="--", lw=1.0, alpha=0.6
                ),
            ],
            ylabel="dp  [kPa]",
        ),
        rk.Strip(
            history["t"],
            [rk.Series(history["margin"], "surge margin", rk.GREEN)],
            ylabel="margin  [%]",
            xlabel="seconds",
            lines=[
                (0.0, rk.RED, "surge"),
                (100.0 * SURGE_CONTROL_LINE, rk.AMBER, "control line"),
            ],
        ),
    ]
    fig = rk.frame(
        title="COMPRESSOR SURGE  ·  HEADER PRESSURE",
        step=step,
        elapsed_s=step * params.delta_t * params.time_unit_seconds,
        schematic=lambda ax: _draw_map(ax, state, params, opening, SURGE_CONTROL_LINE),
        schematic_title="COMPRESSOR MAP",
        gauges=gauges,
        strips=strips,
        status=status,
        subtitle="speed and recycle are the inputs  ·  the plant trips at the surge line",
    )
    return rk.finish(fig), history


_render = rk.make_render_hook(render_compressor_surge, HISTORY_KEYS)
