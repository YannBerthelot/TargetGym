#!/usr/bin/env python
"""
Model-free PID gain tuner — relay autotuning (Astrom-Hagglund method).

Runs relay feedback experiments on each environment to extract ultimate gain
(Ku) and period (Tu), then computes PID gains from standard tuning rules
(AMIGO by default). Gains are computed at multiple operating points for
gain scheduling.

This is what industrial autotuners (Honeywell, ABB, Siemens) do on real
plants. Budget: ~5 oscillation cycles per operating point, N=8 points
per env → ~40 episodes of plant interaction. Realistic for commissioning
a real control loop.

Usage
-----
    # Tune everything with defaults
    python scripts/tune_pid.py

    # Tune specific envs
    python scripts/tune_pid.py --envs cstr first_order

    # More operating points for finer gain scheduling
    python scripts/tune_pid.py --n-points 16

    # Different tuning rule
    python scripts/tune_pid.py --tuning-rule tyreus_luyben

Plane 3D tasks
--------------
The plane3d tasks (heading, circle, figure8) use gradient-based tuning
because they have multi-loop cascaded structures (altitude PID + heading
PID + bank P) that don't decompose into relay experiments on a single
SISO loop. These are kept as-is from pid_tuning.py.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime

# Resolve project root (scripts/ lives one level below root)
ROOT = pathlib.Path(__file__).resolve().parent.parent
GAINS_FILE = ROOT / "data" / "pid_gains.json"

sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Per-environment relay tuners
# ---------------------------------------------------------------------------


def _tune_cstr(n_points: int, tuning_rule: str, **kw) -> dict:
    from target_gym import CSTR, CSTRParams
    from target_gym.experts.relay_autotune import relay_sweep

    env = CSTR(integration_method="rk4_1")
    params = CSTRParams()

    def reset_fn(key, p, t):
        _, state = env.reset_env(key, p)
        return state.replace(target_CA=t, C_a=t)

    result = relay_sweep(
        env,
        params,
        reset_fn,
        state_index=0,
        setpoint_index=2,
        target_range=tuple(params.target_CA_range),
        n_points=n_points,
        sign=-1,
        tuning_rule=tuning_rule,
        relay_amplitude=0.3,
        max_steps=5000,
    )
    mid = len(result["operating_points"]) // 2
    return {
        "Kp": round(result["Kp"][mid], 6),
        "Ki": round(result["Ki"][mid], 6),
        "Kd": round(result["Kd"][mid], 6),
        "gain_schedule": {
            "operating_points": result["operating_points"],
            "Kp": [round(k, 6) for k in result["Kp"]],
            "Ki": [round(k, 6) for k in result["Ki"]],
            "Kd": [round(k, 6) for k in result["Kd"]],
        },
        "tuning_rule": tuning_rule,
        "note": "Relay autotuning, negative-gain system (higher Tc -> lower Ca).",
    }


def _tune_first_order(n_points: int, tuning_rule: str, **kw) -> dict:
    from target_gym import FirstOrderSystem, FirstOrderParams
    from target_gym.experts.relay_autotune import relay_sweep

    env = FirstOrderSystem(integration_method="rk4_1")
    params = FirstOrderParams()

    def reset_fn(key, p, t):
        _, state = env.reset_env(key, p)
        return state.replace(target_x=t, x=t)

    result = relay_sweep(
        env,
        params,
        reset_fn,
        state_index=0,
        setpoint_index=1,
        target_range=tuple(params.target_x_range),
        n_points=n_points,
        sign=1,
        tuning_rule=tuning_rule,
        relay_amplitude=0.5,
        max_steps=5000,
    )
    mid = len(result["operating_points"]) // 2
    return {
        "Kp": round(result["Kp"][mid], 6),
        "Ki": round(result["Ki"][mid], 6),
        "Kd": round(result["Kd"][mid], 6),
        "gain_schedule": {
            "operating_points": result["operating_points"],
            "Kp": [round(k, 6) for k in result["Kp"]],
            "Ki": [round(k, 6) for k in result["Ki"]],
            "Kd": [round(k, 6) for k in result["Kd"]],
        },
        "tuning_rule": tuning_rule,
        "note": "Relay autotuning, positive-gain first-order system.",
    }


def _tune_four_tank(n_points: int, tuning_rule: str, **kw) -> dict:
    from target_gym import FourTank, FourTankParams
    from target_gym.experts.relay_autotune import relay_sweep

    env = FourTank(integration_method="rk4_1")
    params = FourTankParams()

    # Loop 1: pump1 -> h1
    def reset_fn_1(key, p, t):
        _, state = env.reset_env(key, p)
        return state.replace(target_h1=t, h1=t)

    res1 = relay_sweep(
        env,
        params,
        reset_fn_1,
        state_index=0,
        setpoint_index=4,
        target_range=tuple(params.target_h1_range),
        n_points=n_points,
        sign=1,
        tuning_rule=tuning_rule,
        relay_amplitude=0.5,
        max_steps=5000,
        action_dim=2,
        action_index=0,
    )

    # Loop 2: pump2 -> h2
    def reset_fn_2(key, p, t):
        _, state = env.reset_env(key, p)
        return state.replace(target_h2=t, h2=t)

    res2 = relay_sweep(
        env,
        params,
        reset_fn_2,
        state_index=1,
        setpoint_index=5,
        target_range=tuple(params.target_h2_range),
        n_points=n_points,
        sign=1,
        tuning_rule=tuning_rule,
        relay_amplitude=0.5,
        max_steps=5000,
        action_dim=2,
        action_index=1,
    )

    mid1 = len(res1["operating_points"]) // 2
    mid2 = len(res2["operating_points"]) // 2
    return {
        "pid1": {
            "Kp": round(res1["Kp"][mid1], 6),
            "Ki": round(res1["Ki"][mid1], 6),
            "Kd": round(res1["Kd"][mid1], 6),
        },
        "pid2": {
            "Kp": round(res2["Kp"][mid2], 6),
            "Ki": round(res2["Ki"][mid2], 6),
            "Kd": round(res2["Kd"][mid2], 6),
        },
        "gain_schedule_pid1": {
            "operating_points": res1["operating_points"],
            "Kp": [round(k, 6) for k in res1["Kp"]],
            "Ki": [round(k, 6) for k in res1["Ki"]],
            "Kd": [round(k, 6) for k in res1["Kd"]],
        },
        "gain_schedule_pid2": {
            "operating_points": res2["operating_points"],
            "Kp": [round(k, 6) for k in res2["Kp"]],
            "Ki": [round(k, 6) for k in res2["Ki"]],
            "Kd": [round(k, 6) for k in res2["Kd"]],
        },
        "tuning_rule": tuning_rule,
        "note": "Two independent relay experiments: pump1->h1, pump2->h2.",
    }


def _tune_glass_furnace(n_points: int, tuning_rule: str, **kw) -> dict:
    import jax.numpy as jnp
    from target_gym import GlassFurnace, GlassFurnaceParams
    from target_gym.experts.relay_autotune import relay_sweep
    from target_gym.glass_furnace.env import N_SETPOINTS

    env = GlassFurnace(integration_method="rk4_1")
    params = GlassFurnaceParams(m_pull_noise_std=0.0)

    def reset_fn(key, p, t):
        _, st = env.reset_env(key, p)
        return st.replace(
            target_T_crown=t,
            target_schedule=jnp.full((N_SETPOINTS,), t),
            T_crown=t,
        )

    result = relay_sweep(
        env,
        params,
        reset_fn,
        state_index=0,
        setpoint_index=2,
        target_range=tuple(params.target_T_crown_range),
        n_points=n_points,
        sign=1,
        tuning_rule=tuning_rule,
        relay_amplitude=0.3,
        max_steps=10000,
    )
    mid = len(result["operating_points"]) // 2
    return {
        "Kp": round(result["Kp"][mid], 6),
        "Ki": round(result["Ki"][mid], 6),
        "Kd": round(result["Kd"][mid], 6),
        "gain_schedule": {
            "operating_points": result["operating_points"],
            "Kp": [round(k, 6) for k in result["Kp"]],
            "Ki": [round(k, 6) for k in result["Ki"]],
            "Kd": [round(k, 6) for k in result["Kd"]],
        },
        "tuning_rule": tuning_rule,
        "note": "Relay autotuning on T_crown loop (fuel -> crown temperature).",
    }


def _tune_reactor(n_points: int, tuning_rule: str, **kw) -> dict:
    import jax.numpy as jnp
    from target_gym import Reactor, ReactorParams
    from target_gym.experts.relay_autotune import relay_sweep
    from target_gym.reactor.env import N_SETPOINTS, steady_state_xenon

    env = Reactor(integration_method="rk4_50")
    # Freeze OU demand for relay experiments (need constant setpoint).
    params = ReactorParams(demand_sigma=0.0, demand_theta=0.0)

    def reset_fn(key, p, t):
        _, st = env.reset_env(key, p)
        I_eq, Xe_eq = steady_state_xenon(t, p)
        return st.replace(
            target_n=t,
            target_schedule=jnp.full((N_SETPOINTS,), t),
            n=t,
            I_hat=I_eq,
            Xe_hat=Xe_eq,
        )

    result = relay_sweep(
        env,
        params,
        reset_fn,
        state_index=0,
        setpoint_index=3,
        target_range=tuple(params.target_n_range),
        n_points=n_points,
        sign=1,
        tuning_rule=tuning_rule,
        relay_amplitude=0.3,
        max_steps=10000,
    )
    mid = len(result["operating_points"]) // 2
    return {
        "Kp": round(result["Kp"][mid], 6),
        "Ki": round(result["Ki"][mid], 6),
        "Kd": round(result["Kd"][mid], 6),
        "gain_schedule": {
            "operating_points": result["operating_points"],
            "Kp": [round(k, 6) for k in result["Kp"]],
            "Ki": [round(k, 6) for k in result["Ki"]],
            "Kd": [round(k, 6) for k in result["Kd"]],
        },
        "tuning_rule": tuning_rule,
        "note": "Relay autotuning, positive-gain (rod withdrawal raises power).",
    }


def _tune_plane(n_points: int, tuning_rule: str, **kw) -> dict:
    """Plane 2D: relay on stick (altitude loop), power fixed at cruise."""
    from target_gym.plane.env_jax import Airplane2D
    from target_gym.plane.env import PlaneParams
    from target_gym.experts.relay_autotune import relay_sweep

    env = Airplane2D(integration_method="rk4_1")
    params = PlaneParams()

    # obs: [x_dot, z, z_dot, theta, theta_dot, gamma, target_altitude, power, stick]
    # Relay on stick (action index 1), power fixed at cruise (action index 0).
    # Positive stick -> nose up -> altitude increases -> Kp > 0.
    def reset_fn(key, p, t):
        _, state = env.reset_env(key, p)
        return state.replace(target_altitude=t, z=t)

    res_stick = relay_sweep(
        env,
        params,
        reset_fn,
        state_index=1,  # z
        setpoint_index=6,  # target_altitude
        target_range=tuple(params.target_altitude_range),
        n_points=n_points,
        sign=1,
        tuning_rule=tuning_rule,
        relay_amplitude=0.3,
        max_steps=10000,
        action_dim=2,
        action_index=1,
    )

    # Power loop: very slow, use conservative fixed gains from relay
    res_power = relay_sweep(
        env,
        params,
        reset_fn,
        state_index=1,
        setpoint_index=6,
        target_range=tuple(params.target_altitude_range),
        n_points=n_points,
        sign=1,
        tuning_rule=tuning_rule,
        relay_amplitude=0.2,
        max_steps=10000,
        action_dim=2,
        action_index=0,
    )

    mid = len(res_stick["operating_points"]) // 2
    return {
        "pid1": {
            "Kp": round(res_power["Kp"][mid], 8),
            "Ki": round(res_power["Ki"][mid], 8),
            "Kd": round(res_power["Kd"][mid], 8),
        },
        "pid2": {
            "Kp": round(res_stick["Kp"][mid], 8),
            "Ki": round(res_stick["Ki"][mid], 8),
            "Kd": round(res_stick["Kd"][mid], 8),
        },
        "gain_schedule_pid1": {
            "operating_points": res_power["operating_points"],
            "Kp": [round(k, 8) for k in res_power["Kp"]],
            "Ki": [round(k, 8) for k in res_power["Ki"]],
            "Kd": [round(k, 8) for k in res_power["Kd"]],
        },
        "gain_schedule_pid2": {
            "operating_points": res_stick["operating_points"],
            "Kp": [round(k, 8) for k in res_stick["Kp"]],
            "Ki": [round(k, 8) for k in res_stick["Ki"]],
            "Kd": [round(k, 8) for k in res_stick["Kd"]],
        },
        "tuning_rule": tuning_rule,
        "note": "MIMO relay: pid1=power loop, pid2=stick loop, both on altitude error.",
    }


# ---------------------------------------------------------------------------
# Plane3D tasks — sequential loop closing with relay autotuning.
#
# Each task has a cascaded multi-loop structure:
#   inner:  bank angle (phi) → aileron       (P-controller)
#   middle: task error → desired bank         (PID, task-specific)
#   outer:  altitude error → stick            (PID)
#   power:  fixed cruise                      (no feedback)
#
# A commissioning engineer would tune these inside-out, closing each loop
# before exciting the next one — standard Skogestad practice for cascade
# systems. Each sub-loop is treated as SISO and tuned with one relay
# experiment at a representative operating point.
# ---------------------------------------------------------------------------


def _plane3d_bank_relay(env, params, tuning_rule, cruise_action, max_steps=3000):
    """Step 1 (common): tune bank P-controller (phi → aileron) via relay."""
    import numpy as np
    from target_gym.experts.relay_autotune import relay_experiment

    mid_alt = (params.target_altitude_range[0] + params.target_altitude_range[1]) / 2

    def reset_level(key, p, _target):
        _, st = env.reset_env(key, p)
        return st.replace(target_altitude=mid_alt, z=mid_alt)

    def build_bank_action(relay_out, obs):
        return np.array([cruise_action, 0.0, float(np.clip(relay_out, -1, 1))])

    res = relay_experiment(
        env,
        params,
        reset_level,
        state_index=6,
        setpoint_index=6,
        operating_point=0.0,
        fixed_setpoint=0.0,  # desired bank = 0 (level)
        relay_amplitude=0.5,
        max_steps=max_steps,
        build_action=build_bank_action,
        action_bias=0.0,
        plant_sign=1,
        n_settle_cycles=1,
        n_measure_cycles=3,
    )

    if res["success"]:
        # P-only: Kp = 0.5 * Ku.  Negative because phi > desired → aileron < 0.
        Kp_bank = -0.5 * res["Ku"]
        print(f"    Ku={res['Ku']:.4f}, Tu={res['Tu']:.4f}s → Kp_bank={Kp_bank:.4f}")
    else:
        Kp_bank = -2.0
        print(f"    Bank relay failed → fallback Kp_bank={Kp_bank}")

    return Kp_bank


def _plane3d_altitude_relay(env, params, tuning_rule, cruise_action, max_steps=10000):
    """Step (common): tune altitude PID (z → stick) via relay at level flight."""
    import numpy as np
    from target_gym.experts.relay_autotune import relay_experiment, TUNING_RULES

    rule_fn = TUNING_RULES[tuning_rule]
    mid_alt = (params.target_altitude_range[0] + params.target_altitude_range[1]) / 2

    def reset_at_alt(key, p, _target):
        _, st = env.reset_env(key, p)
        return st.replace(target_altitude=mid_alt, z=mid_alt)

    def build_alt_action(relay_out, obs):
        return np.array([cruise_action, float(np.clip(relay_out, -1, 1)), 0.0])

    res = relay_experiment(
        env,
        params,
        reset_at_alt,
        state_index=2,
        setpoint_index=10,  # z, target_altitude
        operating_point=mid_alt,
        relay_amplitude=0.3,
        max_steps=max_steps,
        build_action=build_alt_action,
        action_bias=0.0,
        plant_sign=1,
        n_settle_cycles=2,
        n_measure_cycles=3,
    )

    if res["success"]:
        Kp, Ki, Kd = rule_fn(res["Ku"], res["Tu"])
        print(
            f"    Ku={res['Ku']:.6f}, Tu={res['Tu']:.2f}s → "
            f"Kp={Kp:.6f}, Ki={Ki:.6f}, Kd={Kd:.6f}"
        )
    else:
        Kp, Ki, Kd = 0.0005, 1e-5, 0.001
        print("    Altitude relay failed → fallback gains")

    return Kp, Ki, Kd


def _tune_plane3d_heading(n_points: int, tuning_rule: str, **kw) -> dict:
    """Sequential relay: bank → heading → altitude."""
    import numpy as np
    from target_gym.plane3d.env_jax import Plane3DHeading
    from target_gym.experts.relay_autotune import relay_experiment, TUNING_RULES

    env = Plane3DHeading(integration_method="rk4_1")
    params = env.default_params
    rule_fn = TUNING_RULES[tuning_rule]
    cruise_action = 0.6 * 2.0 - 1.0  # power 0.6 → action [-1, 1]

    # ── Step 1: bank inner loop ──
    print("  [1/3] Bank relay (phi → aileron)...")
    Kp_bank = _plane3d_bank_relay(env, params, tuning_rule, cruise_action)

    # ── Step 2: heading loop (ψ → desired_bank → [bank] → aileron) ──
    print("  [2/3] Heading relay (ψ → desired_bank, bank loop closed)...")
    mid_alt = (params.target_altitude_range[0] + params.target_altitude_range[1]) / 2

    def reset_heading(key, p, _target):
        _, st = env.reset_env(key, p)
        return st.replace(target_altitude=mid_alt, z=mid_alt, target_heading=0.0)

    def build_heading_action(relay_out, obs):
        """relay_out = desired bank angle (rad); inner loop converts to aileron."""
        phi = float(obs[6])
        aileron = float(np.clip(Kp_bank * (phi - relay_out), -1, 1))
        return np.array([cruise_action, 0.0, aileron])

    def wrap_error(measured, setpoint):
        return float(
            np.arctan2(np.sin(setpoint - measured), np.cos(setpoint - measured))
        )

    hdg_res = relay_experiment(
        env,
        params,
        reset_heading,
        state_index=9,
        setpoint_index=11,  # psi, target_heading
        operating_point=0.0,
        relay_amplitude=0.10,  # ±0.10 rad desired bank (~6°)
        max_steps=5000,
        build_action=build_heading_action,
        error_fn=wrap_error,
        action_bias=0.0,
        plant_sign=1,
        n_settle_cycles=2,
        n_measure_cycles=3,
    )

    if hdg_res["success"]:
        Kp_hdg, Ki_hdg, Kd_hdg = rule_fn(hdg_res["Ku"], hdg_res["Tu"])
        print(
            f"    Ku={hdg_res['Ku']:.4f}, Tu={hdg_res['Tu']:.2f}s → "
            f"Kp={Kp_hdg:.4f}, Ki={Ki_hdg:.6f}, Kd={Kd_hdg:.4f}"
        )
    else:
        Kp_hdg, Ki_hdg, Kd_hdg = 0.5, 0.0, 0.0
        print("    Heading relay failed → fallback gains")

    # ── Step 3: altitude loop (z → stick, level flight) ──
    print("  [3/3] Altitude relay (z → stick)...")
    Kp_alt, Ki_alt, Kd_alt = _plane3d_altitude_relay(
        env,
        params,
        tuning_rule,
        cruise_action,
    )

    return {
        "alt": {"Kp": round(Kp_alt, 8), "Ki": round(Ki_alt, 8), "Kd": round(Kd_alt, 8)},
        "hdg": {"Kp": round(Kp_hdg, 6), "Ki": round(Ki_hdg, 6), "Kd": round(Kd_hdg, 6)},
        "bank": {"Kp": round(Kp_bank, 6)},
        "power": 0.6,
        "note": "Sequential relay autotuning: bank → heading → altitude.",
    }


def _tune_plane3d_circle(n_points: int, tuning_rule: str, **kw) -> dict:
    """Sequential relay: bank → radial → altitude."""
    import numpy as np
    from target_gym.plane3d.env_jax import Plane3DCircle
    from target_gym.experts.relay_autotune import relay_experiment, TUNING_RULES

    env = Plane3DCircle(integration_method="rk4_1")
    params = env.default_params
    rule_fn = TUNING_RULES[tuning_rule]
    cruise_action = 0.6 * 2.0 - 1.0
    target_bank_rad = float(np.deg2rad(15.0))
    max_bank_rad = float(np.deg2rad(25.0))

    # ── Step 1: bank inner loop ──
    print("  [1/3] Bank relay (phi → aileron)...")
    Kp_bank = _plane3d_bank_relay(env, params, tuning_rule, cruise_action)

    # ── Step 2: radial loop (distance error → bank correction → [bank] → aileron) ──
    print("  [2/3] Radial relay (dist error → bank correction, bank loop closed)...")
    mid_alt = (params.target_altitude_range[0] + params.target_altitude_range[1]) / 2
    mid_rad = (params.target_radius_range[0] + params.target_radius_range[1]) / 2

    def reset_circle(key, p, _target):
        _, st = env.reset_env(key, p)
        # Position on the circle (east of center, heading north = tangent)
        return st.replace(
            target_altitude=mid_alt,
            z=mid_alt,
            target_radius=mid_rad,
            x=st.target_x + mid_rad,
            y=st.target_y,
        )

    def measure_radial_dist(obs):
        rel_x, rel_y = float(obs[11]), float(obs[12])
        return float(np.sqrt(rel_x**2 + rel_y**2))

    def build_radial_action(relay_out, obs):
        """relay_out = bank correction (rad); add to nominal turn bank."""
        phi = float(obs[6])
        desired_bank = float(
            np.clip(
                target_bank_rad + relay_out,
                -max_bank_rad,
                max_bank_rad,
            )
        )
        aileron = float(np.clip(Kp_bank * (phi - desired_bank), -1, 1))
        return np.array([cruise_action, 0.0, aileron])

    rad_res = relay_experiment(
        env,
        params,
        reset_circle,
        state_index=11,
        setpoint_index=13,  # unused when measure_fn provided
        operating_point=mid_rad,
        measure_fn=measure_radial_dist,
        fixed_setpoint=mid_rad,
        relay_amplitude=0.08,  # ±0.08 rad bank correction (~5°)
        max_steps=5000,
        build_action=build_radial_action,
        action_bias=0.0,
        plant_sign=-1,  # more bank → smaller radius (neg gain)
        n_settle_cycles=2,
        n_measure_cycles=3,
    )

    if rad_res["success"]:
        Kp_rad, Ki_rad, Kd_rad = rule_fn(rad_res["Ku"], rad_res["Tu"])
        print(
            f"    Ku={rad_res['Ku']:.6f}, Tu={rad_res['Tu']:.2f}s → "
            f"Kp={Kp_rad:.6f}, Ki={Ki_rad:.8f}, Kd={Kd_rad:.6f}"
        )
    else:
        Kp_rad, Ki_rad, Kd_rad = 1e-5, 0.0, 0.0
        print("    Radial relay failed → fallback gains")

    # ── Step 3: altitude loop ──
    print("  [3/3] Altitude relay (z → stick)...")
    Kp_alt, Ki_alt, Kd_alt = _plane3d_altitude_relay(
        env,
        params,
        tuning_rule,
        cruise_action,
    )

    return {
        "alt": {"Kp": round(Kp_alt, 8), "Ki": round(Ki_alt, 8), "Kd": round(Kd_alt, 8)},
        "rad": {"Kp": round(Kp_rad, 8), "Ki": round(Ki_rad, 8), "Kd": round(Kd_rad, 8)},
        "bank": {"Kp": round(Kp_bank, 6)},
        "power": 0.6,
        "target_bank_deg": 15.0,
        "note": "Sequential relay autotuning: bank → radial → altitude.",
    }


def _tune_plane3d_figure8(n_points: int, tuning_rule: str, **kw) -> dict:
    """Sequential relay: bank → altitude, then grid search for Kp_hdg.

    The figure-8 PID tracks the twisted 3D lemniscate.  Obs layout:
      [... psi(9), target_alt(10), target_radius(11),
       nearest_dx(12), nearest_dy(13), nearest_dz(14), tangent_heading(15), ...]
    Altitude PID drives nearest_dz → 0.  Heading blends tangent with correction.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from target_gym.plane3d.env_jax import Plane3DFigureEight

    env = Plane3DFigureEight(integration_method="rk4_1")
    params = env.default_params
    cruise_action = 0.6 * 2.0 - 1.0
    dt = float(params.delta_t)

    # ── Step 1: bank inner loop ──
    print("  [1/3] Bank relay (phi → aileron)...")
    Kp_bank = _plane3d_bank_relay(env, params, tuning_rule, cruise_action)

    # ── Step 2: altitude loop (relay on z → stick) ──
    print("  [2/3] Altitude relay (z → stick)...")
    Kp_alt, Ki_alt, Kd_alt = _plane3d_altitude_relay(
        env,
        params,
        tuning_rule,
        cruise_action,
    )

    # ── Step 3: heading gains via grid search ──
    print("  [3/3] Grid search for heading gains (Kp_hdg)...")
    hdg_candidates = [0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    max_bank_rad = float(np.deg2rad(25.0))
    n_eval = 2000
    best_Kp_hdg, best_reward = 0.5, -float("inf")
    key = jax.random.PRNGKey(42)

    for Kp_hdg in hdg_candidates:
        _, state = env.reset_env(key, params)
        alt_int, alt_prev = 0.0, 0.0
        total_r = 0.0

        for t in range(n_eval):
            obs = env.get_obs(state, params)

            # Altitude PID on nearest_dz (obs[14])
            alt_err = float(obs[14])  # curve_z - aircraft_z
            alt_int += alt_err * dt
            alt_d = (alt_err - alt_prev) / dt
            stick = float(
                np.clip(
                    Kp_alt * alt_err + Ki_alt * alt_int + Kd_alt * alt_d,
                    -1,
                    1,
                )
            )
            if abs(stick) >= 1.0:
                alt_int -= alt_err * dt
            alt_prev = alt_err

            # Heading: blend tangent (obs[15]) with correction (obs[12:14])
            nearest_dx = float(obs[12])
            nearest_dy = float(obs[13])
            tangent_hdg = float(obs[15])
            psi = float(obs[9])
            target_radius = float(obs[11])

            lat_dist = float(np.sqrt(nearest_dx**2 + nearest_dy**2 + 1e-6))
            blend = min(lat_dist / (0.05 * max(target_radius, 1.0)), 1.0)
            corr_hdg = float(np.arctan2(nearest_dy, nearest_dx))
            bx = blend * np.cos(corr_hdg) + (1.0 - blend) * np.cos(tangent_hdg)
            by = blend * np.sin(corr_hdg) + (1.0 - blend) * np.sin(tangent_hdg)
            desired_hdg = float(np.arctan2(by, bx))

            hdg_err = float(
                np.arctan2(np.sin(desired_hdg - psi), np.cos(desired_hdg - psi))
            )
            desired_bank = float(np.clip(Kp_hdg * hdg_err, -max_bank_rad, max_bank_rad))
            phi = float(obs[6])
            aileron = float(np.clip(Kp_bank * (phi - desired_bank), -1, 1))

            action = jnp.array([cruise_action, stick, aileron])
            obs, state, reward, done, _ = env.step_env(key, state, action, params)
            total_r += float(reward)
            if bool(done):
                total_r += (n_eval - t - 1) * (-1.0)
                break

        mean_r = total_r / n_eval
        print(f"    Kp_hdg={Kp_hdg:.2f}  mean_reward={mean_r:.4f}")
        if mean_r > best_reward:
            best_reward = mean_r
            best_Kp_hdg = Kp_hdg

    print(f"    → best Kp_hdg = {best_Kp_hdg:.2f}")

    return {
        "alt": {"Kp": round(Kp_alt, 8), "Ki": round(Ki_alt, 8), "Kd": round(Kd_alt, 8)},
        "hdg": {"Kp": round(best_Kp_hdg, 6), "Ki": 0.0, "Kd": 0.0},
        "bank": {"Kp": round(Kp_bank, 6)},
        "power": 0.6,
        "note": "Sequential relay (bank, altitude) + grid search (Kp_hdg). Twisted 3D lemniscate.",
    }


# ---------------------------------------------------------------------------
# Registry: env_name -> (tuner_fn, display_name)
# ---------------------------------------------------------------------------

TUNERS = {
    "cstr": (_tune_cstr, "CSTR"),
    "first_order": (_tune_first_order, "FirstOrder"),
    "four_tank": (_tune_four_tank, "FourTank"),
    "glass_furnace": (_tune_glass_furnace, "GlassFurnace"),
    "reactor": (_tune_reactor, "Reactor"),
    "plane": (_tune_plane, "Airplane2D"),
    "plane3d_heading": (_tune_plane3d_heading, "Plane3DHeading"),
    "plane3d_circle": (_tune_plane3d_circle, "Plane3DCircle"),
    "plane3d_figure8": (_tune_plane3d_figure8, "Plane3DFigureEight"),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Tune PID gains via relay autotuning and save to data/pid_gains.json"
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        choices=list(TUNERS),
        default=list(TUNERS),
        metavar="ENV",
        help="Environments to tune (default: all). Choices: " + ", ".join(TUNERS),
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=8,
        help="Number of operating points for gain scheduling (default: 8).",
    )
    parser.add_argument(
        "--tuning-rule",
        choices=["amigo", "tyreus_luyben", "ziegler_nichols"],
        default="amigo",
        help="PID tuning rule applied to Ku/Tu (default: amigo).",
    )
    args = parser.parse_args()

    # Load existing gains file so we only overwrite the envs we're tuning
    if GAINS_FILE.exists():
        with open(GAINS_FILE) as f:
            gains = json.load(f)
    else:
        gains = {}

    gains["_meta"] = {
        "description": (
            "PID gains tuned via relay autotuning (Astrom-Hagglund). "
            "Re-generate with: python scripts/tune_pid.py"
        ),
        "tuned_at": datetime.now().isoformat(timespec="seconds"),
        "methodology": (
            "Relay feedback experiment: inject relay signal, measure sustained "
            "oscillation to extract ultimate gain (Ku) and period (Tu), then "
            "compute PID gains via tuning rules. Gain-scheduled across N "
            "operating points spanning the target range."
        ),
        "tuning_rule": args.tuning_rule,
    }

    # When running with default envs (no --envs), skip envs already present
    # in the gains file.  Explicit --envs always forces re-tuning.
    explicit_envs = "--envs" in sys.argv
    for env_name in args.envs:
        if not explicit_envs and env_name in gains and env_name != "_meta":
            print(
                f"\n── {env_name}: already tuned (skipped). "
                f"Use --envs {env_name} or clear cache to re-tune."
            )
            continue
        tuner_fn, display = TUNERS[env_name]
        print(f"\n{'='*60}")
        print(f"Tuning {display}  (n_points={args.n_points}, rule={args.tuning_rule})")
        print("=" * 60)
        gains[env_name] = tuner_fn(
            n_points=args.n_points,
            tuning_rule=args.tuning_rule,
        )
        # Print summary
        gs = gains[env_name].get("gain_schedule")
        if gs:
            print(f"  Gain schedule: {len(gs['operating_points'])} points")
        print(f"  Midpoint Kp={gains[env_name].get('Kp', 'N/A')}")

        # Save after every env so partial results survive interruptions
        GAINS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(GAINS_FILE, "w") as f:
            json.dump(gains, f, indent=2)
        print(f"  Saved to {GAINS_FILE}")

    print(f"\nAll done. Gains written to {GAINS_FILE}")


if __name__ == "__main__":
    main()
