"""The lead's autopilot engages in trim (owner's decision, 2026-11-24).

Contract, from charter B5 (an established start: the episode opens on a plant
already flying, not on a transient) and the owner's decision on the aircraft
five-task review: the lead spawns in level trim, and its TECS takes up the trim
it finds on its first step, as the A320 tasks' autopilot does
(``plane3d.experts._engaged``). So on a straight leg in still air the lead holds
its altitude from the first step.

Before the decision the lead's TECS started on the airliner's trim constants
and sank 18 to 22 m in its first 5 s (patrol, formation's reference lead) or 11
to 15 m (wake_surf's A320 lead).

The bound is the 3D heading task's spec for the same autopilot, 2 m
(tests/patrol/test_formation_hold.py, ``ALTITUDE_M``), here held on every step
of the first 30 s rather than on the mean after the burn-in.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import target_gym as tg

ALTITUDE_M = 2.0
STEPS = 30


@functools.cache
def _worst_lead_altitude_error(name, seed):
    env, params = tg.make(name)
    params = params.replace(
        turbulence_sigma=0.0,
        lead_turn_rate_range=(0.0, 0.0),
        sensor_noise_scale=0.0,
    )
    expert = tg.make_expert(name).host()
    step = jax.jit(env.step_env)
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset_env(key, params)
    worst = 0.0
    for _ in range(STEPS):
        key, sub = jax.random.split(key)
        obs, state, _, _, _ = step(sub, state, expert(np.asarray(obs)), params)
        error = jnp.abs(state.lead.target_altitude - state.lead.z)
        worst = max(worst, float(error))
    return worst


@pytest.mark.parametrize("name", ["patrol", "wake_surf", "formation"])
@pytest.mark.parametrize("seed", [100, 101])
def test_the_lead_holds_its_level_from_the_first_step(name, seed):
    worst = _worst_lead_altitude_error(name, seed)
    assert worst <= ALTITUDE_M, f"lead {worst:.1f} m off its level in {STEPS} s"
