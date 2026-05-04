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


def _tune_four_tank_search(n_points: int, tuning_rule: str, **kw) -> dict:
    """Grid search over (Kp, Ki) per pump-tank loop. Replaces relay autotuning
    for FourTank: the plant is a pure integrator (water level integrates
    inflow), so under bang-bang relay it drifts monotonically with no
    zero-crossings, giving Astrom-Hagglund nothing to extract Ku/Tu from.

    Per operating point ``t`` (target_h1 = target_h2 = t), search runs in
    two alternating refinement passes (loop1 ↔ loop2) so cross-coupling
    (γ1=γ2=0.2 → 80% of each pump goes cross-tank) is accounted for. The
    objective is cumulative env reward on a full 500-step rollout.

    ``tuning_rule`` is ignored (kept for signature compatibility); printed
    in the output dict as ``"grid_search_max_reward"``.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from target_gym import FourTank, FourTankParams

    env = FourTank(integration_method="rk4_1")
    params = FourTankParams()
    targets = np.linspace(
        params.target_h1_range[0], params.target_h1_range[1], n_points
    )
    Kps = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    Kis = jnp.array([0.0, 0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
    max_steps = int(params.max_steps_in_episode)

    def _pi_with_antiwindup(Kp, Ki, err, integ_prev):
        """Match the discrete PI semantics of pid.py:pid_step (which is
        what the expert actually runs). Anti-windup: don't grow the
        integral while the output is saturated."""
        candidate_int = integ_prev + err  # dt=1
        u_raw = Kp * err + Ki * candidate_int
        u_clip = jnp.clip(u_raw, -1.0, 1.0)
        new_integ = jnp.where(u_raw == u_clip, candidate_int, integ_prev)
        return u_clip, new_integ

    def simulate(Kp1, Ki1, Kp2, Ki2, t, key):
        rk = key
        _, state = env.reset_env(rk, params)
        state = state.replace(target_h1=t, target_h2=t)

        def body(carry, _):
            state, int1, int2, key = carry
            obs = env.get_obs(state)
            err1 = obs[4] - obs[0]
            err2 = obs[5] - obs[1]
            u1, new_int1 = _pi_with_antiwindup(Kp1, Ki1, err1, int1)
            u2, new_int2 = _pi_with_antiwindup(Kp2, Ki2, err2, int2)
            action = jnp.stack([u1, u2])
            key, sk = jax.random.split(key)
            _, new_state, r, _, _ = env.step(sk, state, action, params)
            return (new_state, new_int1, new_int2, key), r

        init = (state, jnp.float32(0.0), jnp.float32(0.0), rk)
        _, rewards = jax.lax.scan(body, init, xs=None, length=max_steps)
        return rewards.sum()

    grid_Kp, grid_Ki = jnp.meshgrid(Kps, Kis, indexing="ij")
    flat_Kp, flat_Ki = grid_Kp.flatten(), grid_Ki.flatten()

    def search_loop1(kp2, ki2, t, key):
        s = jax.vmap(lambda kp, ki: simulate(kp, ki, kp2, ki2, t, key))(
            flat_Kp, flat_Ki
        )
        idx = jnp.argmax(s)
        return flat_Kp[idx], flat_Ki[idx], s[idx]

    def search_loop2(kp1, ki1, t, key):
        s = jax.vmap(lambda kp, ki: simulate(kp1, ki1, kp, ki, t, key))(
            flat_Kp, flat_Ki
        )
        idx = jnp.argmax(s)
        return flat_Kp[idx], flat_Ki[idx], s[idx]

    Kp1_list, Ki1_list, Kp2_list, Ki2_list = [], [], [], []
    for t in targets:
        key = jax.random.PRNGKey(int(round(float(t) * 10000)))
        kp1, ki1, _ = search_loop1(
            15.0, 2.0, float(t), key
        )  # seed loop2 at hand-picked
        kp2, ki2, _ = search_loop2(float(kp1), float(ki1), float(t), key)
        kp1, ki1, _ = search_loop1(float(kp2), float(ki2), float(t), key)
        kp2, ki2, score = search_loop2(float(kp1), float(ki1), float(t), key)
        kp1, ki1, kp2, ki2 = float(kp1), float(ki1), float(kp2), float(ki2)
        score = float(score)
        print(
            f"    t={t:.3f}: loop1=(Kp={kp1:6.2f}, Ki={ki1:5.2f})  "
            f"loop2=(Kp={kp2:6.2f}, Ki={ki2:5.2f})  "
            f"cum_reward={score:5.1f}/{max_steps}"
        )
        Kp1_list.append(kp1)
        Ki1_list.append(ki1)
        Kp2_list.append(kp2)
        Ki2_list.append(ki2)

    # The midpoint (top-level pid1/pid2) keys are what the
    # FunctionalExpertPolicy actually consumes — gain scheduling is
    # currently not wired into FourTank's expert_policy. So pick the
    # midpoint by maximizing the average reward across the eval
    # distribution (uniform random targets, matching env.reset_env),
    # not just by taking the geometric centre of the operating-point
    # grid. This is the gain pair that AjaxExperiments actually sees.
    n_eval_seeds = 32
    eval_keys = jax.random.split(jax.random.PRNGKey(424242), n_eval_seeds)

    def avg_reward(Kp1, Ki1, Kp2, Ki2):
        # Per seed, env.reset_env samples a fresh random target. Match
        # that by re-resetting per seed and using the env-drawn targets
        # rather than a fixed t.
        def one_seed(k):
            _, st = env.reset_env(k, params)

            def body(carry, _):
                state, int1, int2, key = carry
                obs = env.get_obs(state)
                err1 = obs[4] - obs[0]
                err2 = obs[5] - obs[1]
                u1, new_int1 = _pi_with_antiwindup(Kp1, Ki1, err1, int1)
                u2, new_int2 = _pi_with_antiwindup(Kp2, Ki2, err2, int2)
                action = jnp.stack([u1, u2])
                key, sk = jax.random.split(key)
                _, new_state, r, _, _ = env.step(sk, state, action, params)
                return (new_state, new_int1, new_int2, key), r

            init = (st, jnp.float32(0.0), jnp.float32(0.0), k)
            _, rewards = jax.lax.scan(body, init, xs=None, length=max_steps)
            return rewards.sum()

        return jax.vmap(one_seed)(eval_keys).mean()

    # Joint 4D grid search for the midpoint. 8x8x8x8 = 4096 candidates,
    # each averaged over 32 random seeds. Memory-fits on CPU (no real
    # state, just 4 floats per candidate).
    g1, g2, g3, g4 = jnp.meshgrid(Kps, Kis, Kps, Kis, indexing="ij")
    flat = (g1.flatten(), g2.flatten(), g3.flatten(), g4.flatten())
    print(
        f"  Joint 4D midpoint search ({flat[0].size} candidates x "
        f"{n_eval_seeds} seeds, optimizing for eval distribution)..."
    )
    scores_mid = jax.vmap(avg_reward)(*flat)
    scores_mid = jax.device_get(scores_mid)
    best = int(np.argmax(scores_mid))
    kp1_mid = float(flat[0][best])
    ki1_mid = float(flat[1][best])
    kp2_mid = float(flat[2][best])
    ki2_mid = float(flat[3][best])
    print(
        f"  best midpoint: loop1=(Kp={kp1_mid:.3f}, Ki={ki1_mid:.3f})  "
        f"loop2=(Kp={kp2_mid:.3f}, Ki={ki2_mid:.3f})  "
        f"avg_reward={float(scores_mid[best]):.2f}/{max_steps} "
        f"({100*float(scores_mid[best])/max_steps:.1f}%)"
    )

    return {
        "pid1": {"Kp": round(kp1_mid, 6), "Ki": round(ki1_mid, 6), "Kd": 0.0},
        "pid2": {"Kp": round(kp2_mid, 6), "Ki": round(ki2_mid, 6), "Kd": 0.0},
        "gain_schedule_pid1": {
            "operating_points": [float(t) for t in targets],
            "Kp": [round(k, 6) for k in Kp1_list],
            "Ki": [round(k, 6) for k in Ki1_list],
            "Kd": [0.0] * len(targets),
        },
        "gain_schedule_pid2": {
            "operating_points": [float(t) for t in targets],
            "Kp": [round(k, 6) for k in Kp2_list],
            "Ki": [round(k, 6) for k in Ki2_list],
            "Kd": [0.0] * len(targets),
        },
        "tuning_rule": "grid_search_max_reward",
        "note": (
            "Sequential grid search over (Kp, Ki) maximizing cumulative env "
            "reward, alternating loop1/loop2 refinements. FourTank is a "
            "pure integrator -> relay autotuning fails (no zero-crossings)."
        ),
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


def _plane3d_altitude_relay(
    env,
    params,
    tuning_rule,
    cruise_action,
    n_points=8,
    max_steps=10000,
):
    """Sweep altitude PID (z → stick) across target_altitude_range.

    Matches the 2D Airplane2D methodology (relay_sweep, pick midpoint gain).
    """
    import numpy as np
    from target_gym.experts.relay_autotune import relay_sweep

    def reset_at_alt(key, p, target):
        _, st = env.reset_env(key, p)
        return st.replace(target_altitude=float(target), z=float(target))

    def build_alt_action(relay_out, obs):
        return np.array([cruise_action, float(np.clip(relay_out, -1, 1)), 0.0])

    res = relay_sweep(
        env,
        params,
        reset_at_alt,
        state_index=2,
        setpoint_index=10,
        target_range=tuple(params.target_altitude_range),
        n_points=n_points,
        sign=1,
        tuning_rule=tuning_rule,
        relay_amplitude=0.3,
        max_steps=max_steps,
        action_dim=3,
        action_index=1,
        action_bias=0.0,
        build_action=build_alt_action,
    )
    mid = len(res["operating_points"]) // 2
    Kp, Ki, Kd = res["Kp"][mid], res["Ki"][mid], res["Kd"][mid]
    if Kp is None:
        Kp, Ki, Kd = 0.0005, 1e-5, 0.001
        print("    All altitude relays failed → fallback gains")
    return Kp, Ki, Kd


def _plane3d_power_relay(
    env,
    params,
    tuning_rule,
    cruise_action,
    n_points=8,
    max_steps=10000,
):
    """Sweep power PID (z → power, stick=0) across target_altitude_range.

    Mirrors the 2D Airplane2D pid1 methodology: altitude-error-to-power loop.
    """
    import numpy as np
    from target_gym.experts.relay_autotune import relay_sweep

    def reset_at_alt(key, p, target):
        _, st = env.reset_env(key, p)
        return st.replace(target_altitude=float(target), z=float(target))

    def build_power_action(relay_out, obs):
        return np.array([float(np.clip(relay_out, -1, 1)), 0.0, 0.0])

    res = relay_sweep(
        env,
        params,
        reset_at_alt,
        state_index=2,
        setpoint_index=10,
        target_range=tuple(params.target_altitude_range),
        n_points=n_points,
        sign=1,
        tuning_rule=tuning_rule,
        relay_amplitude=0.2,
        max_steps=max_steps,
        action_dim=3,
        action_index=0,
        action_bias=cruise_action,
        build_action=build_power_action,
    )
    mid = len(res["operating_points"]) // 2
    Kp, Ki, Kd = res["Kp"][mid], res["Ki"][mid], res["Kd"][mid]
    if Kp is None:
        Kp, Ki, Kd = 2e-4, 5e-6, 0.0
        print("    All power relays failed → fallback gains")
    return Kp, Ki, Kd


def _tune_plane3d_heading(n_points: int, tuning_rule: str, **kw) -> dict:
    """Sequential relay: bank → heading → altitude (stick) → altitude (power)."""
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
    print("  [3/4] Altitude stick relay (z → stick)...")
    Kp_alt, Ki_alt, Kd_alt = _plane3d_altitude_relay(
        env,
        params,
        tuning_rule,
        cruise_action,
        n_points=n_points,
    )

    # ── Step 4: altitude loop (z → power, level flight) ──
    print("  [4/4] Altitude power relay (z → power)...")
    Kp_pow, Ki_pow, Kd_pow = _plane3d_power_relay(
        env,
        params,
        tuning_rule,
        cruise_action,
        n_points=n_points,
    )

    return {
        "alt": {"Kp": round(Kp_alt, 8), "Ki": round(Ki_alt, 8), "Kd": round(Kd_alt, 8)},
        "hdg": {"Kp": round(Kp_hdg, 6), "Ki": round(Ki_hdg, 6), "Kd": round(Kd_hdg, 6)},
        "bank": {"Kp": round(Kp_bank, 6)},
        "power_pid": {
            "Kp": round(Kp_pow, 8),
            "Ki": round(Ki_pow, 8),
            "Kd": round(Kd_pow, 8),
        },
        "power": 0.6,
        "note": "Sequential relay autotuning: bank → heading → altitude stick → altitude power.",
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

    # ── Step 3: altitude loop (stick) ──
    print("  [3/4] Altitude stick relay (z → stick)...")
    Kp_alt, Ki_alt, Kd_alt = _plane3d_altitude_relay(
        env,
        params,
        tuning_rule,
        cruise_action,
        n_points=n_points,
    )

    # ── Step 4: altitude loop (power) ──
    print("  [4/4] Altitude power relay (z → power)...")
    Kp_pow, Ki_pow, Kd_pow = _plane3d_power_relay(
        env,
        params,
        tuning_rule,
        cruise_action,
        n_points=n_points,
    )

    return {
        "alt": {"Kp": round(Kp_alt, 8), "Ki": round(Ki_alt, 8), "Kd": round(Kd_alt, 8)},
        "rad": {"Kp": round(Kp_rad, 8), "Ki": round(Ki_rad, 8), "Kd": round(Kd_rad, 8)},
        "bank": {"Kp": round(Kp_bank, 6)},
        "power_pid": {
            "Kp": round(Kp_pow, 8),
            "Ki": round(Ki_pow, 8),
            "Kd": round(Kd_pow, 8),
        },
        "power": 0.6,
        "target_bank_deg": 15.0,
        "note": "Sequential relay autotuning: bank → radial → altitude stick → altitude power.",
    }


def _tune_plane3d_figure8(n_points: int, tuning_rule: str, **kw) -> dict:
    """Sequential relay: bank → heading (via relay) → altitude stick → altitude power.

    The figure-8 PID tracks the twisted 3D lemniscate. Obs layout:
      [... psi(9), target_alt(10), target_radius(11),
       nearest_dx(12), nearest_dy(13), nearest_dz(14), tangent_heading(15), ...]
    Altitude PID drives nearest_dz → 0; heading loop drives psi → tangent_heading
    (relay is tuned at a point on the curve, matching the heading-task protocol).
    """
    import numpy as np
    from target_gym.plane3d.env_jax import Plane3DFigureEight
    from target_gym.experts.relay_autotune import relay_experiment, TUNING_RULES

    env = Plane3DFigureEight(integration_method="rk4_1")
    params = env.default_params
    rule_fn = TUNING_RULES[tuning_rule]
    cruise_action = 0.6 * 2.0 - 1.0

    # ── Step 1: bank inner loop ──
    print("  [1/4] Bank relay (phi → aileron)...")
    Kp_bank = _plane3d_bank_relay(env, params, tuning_rule, cruise_action)

    # ── Step 2: heading loop (relay on desired_bank, bank closed) ──
    # Fig-8's heading tracks a tangent that rotates around the curve; to stay
    # consistent with the heading task we tune at a point where the tangent is
    # locally stationary (treat fig-8 as "follow a fixed tangent direction"
    # during the relay).
    print("  [2/4] Heading relay (ψ → desired_bank, bank loop closed)...")
    mid_alt = (params.target_altitude_range[0] + params.target_altitude_range[1]) / 2

    def reset_fig8_heading(key, p, _target):
        _, st = env.reset_env(key, p)
        return st.replace(target_altitude=mid_alt, z=mid_alt, target_heading=0.0)

    def build_heading_action(relay_out, obs):
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
        reset_fig8_heading,
        state_index=9,
        setpoint_index=15,  # tangent_heading (figure-8 obs layout)
        operating_point=0.0,
        fixed_setpoint=0.0,
        relay_amplitude=0.10,
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

    # ── Step 3: altitude loop (stick) ──
    print("  [3/4] Altitude stick relay (z → stick)...")
    Kp_alt, Ki_alt, Kd_alt = _plane3d_altitude_relay(
        env,
        params,
        tuning_rule,
        cruise_action,
        n_points=n_points,
    )

    # ── Step 4: altitude loop (power) ──
    print("  [4/4] Altitude power relay (z → power)...")
    Kp_pow, Ki_pow, Kd_pow = _plane3d_power_relay(
        env,
        params,
        tuning_rule,
        cruise_action,
        n_points=n_points,
    )

    return {
        "alt": {"Kp": round(Kp_alt, 8), "Ki": round(Ki_alt, 8), "Kd": round(Kd_alt, 8)},
        "hdg": {"Kp": round(Kp_hdg, 6), "Ki": round(Ki_hdg, 6), "Kd": round(Kd_hdg, 6)},
        "bank": {"Kp": round(Kp_bank, 6)},
        "power_pid": {
            "Kp": round(Kp_pow, 8),
            "Ki": round(Ki_pow, 8),
            "Kd": round(Kd_pow, 8),
        },
        "power": 0.6,
        "note": "Sequential relay autotuning: bank → heading → altitude stick → altitude power.",
    }


# ---------------------------------------------------------------------------
# Registry: env_name -> (tuner_fn, display_name)
# ---------------------------------------------------------------------------

TUNERS = {
    "cstr": (_tune_cstr, "CSTR"),
    "first_order": (_tune_first_order, "FirstOrder"),
    "four_tank": (_tune_four_tank_search, "FourTank"),
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
