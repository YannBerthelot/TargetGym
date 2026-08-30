"""Closed-loop baseline contract for the 2D plane.

Unlike ``test_flight_scenarios.py`` (open-loop, gain-independent), these tests
exercise the shipped PID expert. They are the regression net for the claim the
README makes -- that the PID is a usable expert demonstrator -- and they are
marked ``slow`` because each runs a full 1500-step episode.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.experts.pid import make_plane_cascaded_pid
from target_gym.plane.env import PlaneParams
from target_gym.plane.env_jax import Airplane2D

pytestmark = pytest.mark.slow

STEPS = 1500
TOLERANCE_M = 150.0


def _run_pid(seed, steps=STEPS):
    env = Airplane2D()
    params = PlaneParams(max_steps_in_episode=steps)
    pid = make_plane_cascaded_pid()
    pid.reset()
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset_env(key, params)
    start, target = float(state.z), float(state.target_altitude)
    alts, aoas = [], []
    _jstep = jax.jit(env.step_env)
    for _ in range(steps):
        obs, state, _, terminated, _ = _jstep(key, state, jnp.asarray(pid(obs)), params)
        alts.append(float(state.z))
        aoas.append(float(np.rad2deg(state.alpha)))
        if bool(terminated):
            break
    return {
        "start": start,
        "target": target,
        "alts": np.array(alts),
        "aoa_max": max(aoas),
        "final_error": abs(alts[-1] - target),
    }


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_pid_reaches_target_altitude(seed):
    """The shipped PID expert must reach and hold its target altitude."""
    r = _run_pid(seed)
    assert r["final_error"] < TOLERANCE_M, (
        f"seed {seed}: {r['start']:.0f} -> target {r['target']:.0f}, "
        f"ended {r['alts'][-1]:.0f} (error {r['final_error']:.0f} m, "
        f"AoA max {r['aoa_max']:.1f} deg)"
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_pid_stays_below_stall(seed):
    """A well-behaved expert should not need to fly the aircraft into a stall."""
    r = _run_pid(seed)
    # The cascaded autopilot's alpha protection caps the commanded pitch at
    # gamma + alpha_max, so it should keep a wide margin, not merely squeak in.
    assert r["aoa_max"] < 0.75 * PlaneParams().aoa_stall, (
        f"AoA reached {r['aoa_max']:.1f} deg, within 25% of the "
        f"{PlaneParams().aoa_stall} deg stall"
    )


def test_airframe_can_reach_the_hardest_target_open_loop():
    """Seed 0's target is reachable on thrust alone.

    Proves the seed-0 xfail above is a controller limitation, not an
    aerodynamic one -- the distinction that determines where the fix belongs.
    """
    env = Airplane2D()
    params = PlaneParams(max_steps_in_episode=STEPS)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key, params)
    target = float(state.target_altitude)
    action = jnp.array([1.0, 0.0])  # full power, neutral elevator
    peak, aoa_max = -np.inf, -np.inf
    _jstep = jax.jit(env.step_env)
    for _ in range(STEPS):
        obs, state, _, terminated, _ = _jstep(key, state, action, params)
        peak = max(peak, float(state.z))
        aoa_max = max(aoa_max, float(np.rad2deg(state.alpha)))
        if bool(terminated):
            break
    assert peak >= target, f"peak {peak:.0f} m < target {target:.0f} m"
    assert aoa_max < PlaneParams().aoa_stall
