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


# NREL 5 MW blade planform, normalised as (r/R, chord/R). The reference blade
# is 61.5 m long with a 4.65 m maximum chord at about a quarter span, tapering
# to roughly 1.4 m at the tip -- so chord/R peaks near 0.074 and the blade is
# slender. The previous drawing was a triangle at chord/R = 0.37, widest at the
# root and pointed at the tip, which is the shape of a paper dart and the
# reverse of a blade.
_PLANFORM = (
    (0.04, 0.022),
    (0.12, 0.055),
    (0.25, 0.074),
    (0.45, 0.060),
    (0.65, 0.046),
    (0.85, 0.031),
    (1.00, 0.010),
)


def _draw_turbine(ax, state, params):
    omega_rpm = float(state.omega) * 60.0 / (2.0 * np.pi)
    pitch = float(state.pitch)
    v = float(state.v_wind)
    cx, cy, R = 0.56, 0.55, 0.30

    # Inflow.
    v_frac = rk.clamp01(v / (1.2 * params.v_cut_out))
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

    # Ground, so the tower stands on something rather than stopping in mid-air.
    ax.axhspan(0.0, 0.045, color=rk.PANEL, zorder=0)
    ax.axhline(0.045, color=rk.FRAME, lw=1.2, zorder=1)

    # Tower: tapered, wider at the base. A constant-width line read as a stub
    # because it was drawn in the hairline colour on a near-black ground.
    base_w, top_w = 0.020, 0.009
    ax.fill(
        [cx - base_w, cx + base_w, cx + top_w, cx - top_w],
        [0.045, 0.045, cy, cy],
        color=rk.lerp_hex(rk.FRAME, rk.DIM, 0.45),
        zorder=2,
    )

    # Nacelle, *behind* the rotor. This is a head-on view, so the nacelle is
    # mostly hidden by the hub -- the previous drawing put a rounded box 0.14
    # wide across a 0.30 rotor radius in front of everything, which is the
    # black rectangle that dominated the panel.
    ax.add_patch(
        rk.patches.FancyBboxPatch(
            (cx - 0.032, cy - 0.020),
            0.064,
            0.040,
            boxstyle=rk.patches.BoxStyle("Round", pad=0.008),
            fc=rk.PANEL,
            ec=rk.FRAME,
            lw=1.0,
            zorder=2,
        )
    )

    # Rotor. Blade angle animates with rotor position; the *projected* chord
    # narrows as pitch feathers, which is what spilling power looks like
    # head-on -- a fully feathered blade turns its edge to the viewer.
    phase = (float(state.time) * float(state.omega) * 0.02) % (2 * np.pi)
    feather = rk.clamp01(pitch / max(params.pitch_max, 1e-9))
    blade_c = rk.duty_hex(feather, rk.TEXT, rk.AMBER)
    chord_scale = (1.0 - 0.80 * feather) + 0.06
    for k in range(3):
        a = phase + k * 2 * np.pi / 3
        span = np.array([np.cos(a), np.sin(a)])
        chord = np.array([-np.sin(a), np.cos(a)])
        hub = np.array([cx, cy])
        lead, trail = [], []
        for frac, c in _PLANFORM:
            mid = hub + span * (R * frac)
            half = R * c * chord_scale
            # The leading edge is the nearly straight one; the trailing edge
            # carries the taper. Splitting the chord 35/65 about the pitch axis
            # is what gives a blade its recognisable outline.
            lead.append(mid + chord * half * 0.35)
            trail.append(mid - chord * half * 0.65)
        pts = np.array(lead + trail[::-1])
        ax.fill(pts[:, 0], pts[:, 1], color=blade_c, alpha=0.9, zorder=3)

    # Spinner over the blade roots.
    rk.disc(ax, cx, cy, 0.026, fc=rk.PANEL, ec=rk.DIM, zorder=6)
    rk.disc(ax, cx, cy, 0.010, fc=rk.WELL, ec=rk.FRAME, zorder=7)
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
        0.020,
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
            float(state.v_wind) / (1.2 * params.v_cut_out),
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


_render = rk.make_render_hook(render_wind_turbine, HISTORY_KEYS)
