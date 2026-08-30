"""Control-room rendering for the NREL 5 MW reference turbine.

The schematic is a rotor seen head-on: blades turn at rotor speed and feather
visibly with pitch, and the inflow arrows thicken with wind. Above rated, the
job is to spill whatever the generator cannot absorb, so pitch and wind are the
two things the picture puts side by side.
"""

import numpy as np

from target_gym import render_kit as rk
from target_gym.energy.wind_turbine.env import compute_reward

HISTORY_KEYS = ("t", "power", "target", "omega", "pitch", "wind", "reward")


def _draw_turbine(ax, state, params):
    omega_rpm = float(state.omega) * 60.0 / (2.0 * np.pi)
    pitch = float(state.pitch)
    v = float(state.v_wind)
    cx, cy, R = 0.56, 0.55, 0.30

    # Inflow.
    v_frac = rk.clamp01((v - params.v_min) / (params.v_max - params.v_min))
    for i, yy in enumerate(np.linspace(cy - R * 0.8, cy + R * 0.8, 5)):
        rk.flow_arrow(
            ax,
            0.02,
            yy,
            0.20,
            yy,
            color=rk.CYAN,
            lw=0.8 + 2.6 * v_frac,
            alpha=0.30 + 0.5 * v_frac,
            mutation=8 + 8 * v_frac,
        )
    rk.label(ax, 0.11, cy + R * 0.95, f"WIND  {v:.1f} m/s", color=rk.CYAN, size=8)

    # Tower and nacelle.
    ax.plot(
        [cx, cx], [0.04, cy], color=rk.FRAME, lw=6, solid_capstyle="round", zorder=2
    )
    ax.add_patch(
        rk.patches.FancyBboxPatch(
            (cx - 0.055, cy - 0.035),
            0.14,
            0.07,
            boxstyle=rk.patches.BoxStyle("Round", pad=0.012),
            fc=rk.PANEL,
            ec=rk.FRAME,
            lw=1.6,
            zorder=4,
        )
    )

    # Rotor. Blade angle animates with rotor position; blade *chord* narrows as
    # pitch feathers, which is what spilling power looks like.
    phase = (float(state.time) * float(state.omega) * 0.02) % (2 * np.pi)
    feather = rk.clamp01(pitch / max(params.pitch_max, 1e-9))
    blade_c = rk.duty_hex(feather, rk.TEXT, rk.AMBER)
    for k in range(3):
        a = phase + k * 2 * np.pi / 3
        tipx, tipy = cx + R * np.cos(a), cy + R * np.sin(a)
        w = 0.055 * (1.0 - 0.72 * feather) + 0.010
        px, py = -np.sin(a) * w, np.cos(a) * w
        ax.fill(
            [cx + px, cx - px, tipx],
            [cy + py, cy - py, tipy],
            color=blade_c,
            alpha=0.85,
            zorder=3,
        )
    rk.disc(ax, cx, cy, 0.028, fc=rk.WELL, ec=rk.FRAME, zorder=6)
    rk.glow(
        ax,
        cx,
        cy,
        R * 0.9,
        color=rk.CYAN,
        strength=rk.clamp01(omega_rpm / params.omega_rated_rpm) * 0.7,
        zorder=1,
    )

    rk.label(
        ax,
        cx,
        0.04 - 0.035,
        f"{omega_rpm:.2f} rpm   pitch {pitch:.1f} deg",
        color=rk.TEXT,
        size=9,
        weight="bold",
    )
    rk.label(
        ax,
        cx + R + 0.05,
        cy,
        f"R = {params.R:.0f} m",
        color=rk.DIM,
        size=7,
        rotation=90,
    )
    rk.caption(ax, "above rated: pitch spills what the generator cannot absorb")


def render_wind_turbine(state, params, step, history):
    omega_rpm = float(state.omega) * 60.0 / (2.0 * np.pi)
    power_MW = (
        params.eta_gen * params.N_gear * float(state.torque) * float(state.omega)
    ) / 1e6
    history["t"].append(step * params.delta_t)
    history["power"].append(power_MW)
    history["target"].append(float(state.target_power) / 1e6)
    history["omega"].append(omega_rpm)
    history["pitch"].append(float(state.pitch))
    history["wind"].append(float(state.v_wind))
    history["reward"].append(float(compute_reward(state, params)))

    err = abs(power_MW - float(state.target_power) / 1e6)
    status = rk.NOMINAL if err < 0.25 else (rk.WATCH if err < 0.75 else rk.ALARM)
    rated_rpm = params.omega_rated_rpm

    gauges = [
        rk.Gauge(
            "POWER",
            f"{power_MW:.2f} MW",
            power_MW / (params.P_rated / 1e6),
            rk.CYAN,
            target_frac=float(state.target_power) / params.P_rated,
        ),
        rk.Gauge(
            "TARGET",
            f"{float(state.target_power) / 1e6:.2f} MW",
            float(state.target_power) / params.P_rated,
            rk.DIM,
        ),
        rk.Gauge(
            "ROTOR",
            f"{omega_rpm:.2f} rpm",
            omega_rpm / (rated_rpm * 1.3),
            rk.GREEN,
            limit_frac=params.overspeed_factor / 1.3,
            target_frac=1.0 / 1.3,
        ),
        rk.Gauge(
            "PITCH",
            f"{float(state.pitch):.1f} deg",
            float(state.pitch) / params.pitch_max,
            rk.AMBER,
        ),
        rk.Gauge(
            "TORQUE",
            f"{float(state.torque) / 1e3:.1f} kNm",
            float(state.torque) / params.torque_max,
            rk.BLUE,
            limit_frac=1.0,
        ),
        rk.Gauge(
            "WIND",
            f"{float(state.v_wind):.1f} m/s",
            (float(state.v_wind) - params.v_min) / (params.v_max - params.v_min),
            rk.TEAL,
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
            rk.Series(history["power"], "power", rk.CYAN, fill_to=history["target"]),
            rk.Series(history["target"], "target", "white", ls="--", lw=1.0, alpha=0.6),
        ],
        ylabel="power  [MW]",
        xlabel="seconds",
    )
    fig = rk.frame(
        title="WIND  TURBINE  —  NREL  5 MW",
        step=step,
        elapsed_s=step * params.delta_t,
        schematic=lambda ax: _draw_turbine(ax, state, params),
        schematic_title="ROTOR",
        gauges=gauges,
        strips=[strip],
        status=status,
        subtitle="region 3 regulation  ·  turbulent inflow is unmeasured",
    )
    return rk.finish(fig), history


_render = rk.make_render_hook(render_wind_turbine, HISTORY_KEYS, stride=8)
