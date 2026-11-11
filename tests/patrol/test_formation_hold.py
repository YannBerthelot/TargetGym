"""The formation hold: the expert of both patrol tasks and the formation.

Flight-tested formation autopilots (AFIT's, NASA's F/A-18 Autonomous
Formation Flight) close the slot error in the lead's frame, along track on
speed, across it on course and vertically on height, over the aircraft's own
autopilot. No deployed autopilot does (patrol/PHYSICS.md, D1), so the expert is
a reference, not a bound: it need not meet the tasks' objectives (the owner,
2026-11-10). These tests hold it, on held-out seeds its tuning never saw, to
what makes it a credible reference flown on the scored configuration.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import target_gym as tg
from target_gym.eval import scored_burn_in

HELD_OUT = tuple(range(200, 210))

# The contract:
# - no trip: no crash, no collision, no slot lost, on every seed;
# - the formation's lead holds its altitude within 2 m, mean after the burn-in,
#   the 3D heading task's spec for the same autopilot
#   (tests/plane3d/test_plane3d_guidance.py).
# The slot error against AFF's 10 ft and the lead's heading against the AHRS's
# 0.5 deg are the tasks' objectives, which a reference need not meet: steady
# flight holds 10 ft (tests/patrol/test_patrol_env.py, ``TestExpert``), and
# ``scripts/tune_formation_hold.py --check`` prints both over these seeds.
ALTITUDE_M = 2.0


@functools.cache
def _setup(name):
    env, params = tg.make(name)

    def altitude_error(state):
        return jnp.abs(state.lead.target_altitude - state.lead.z)

    return env, params, jax.jit(env.step_env), jax.jit(altitude_error)


@functools.cache
def _episode(name, seed):
    """One held-out episode of the shipped expert on the scored configuration:
    ``(tripped, the lead's mean altitude error after the scored burn-in)``."""
    env, params, step, altitude_error = _setup(name)
    expert = tg.make_expert(name).host()
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset_env(key, params)
    tripped, alt = False, []
    for _ in range(int(params.max_steps_in_episode)):
        key, sub = jax.random.split(key)
        obs, state, _, _, info = step(sub, state, expert(np.asarray(obs)), params)
        tripped = tripped or bool(info["tripped"])
        alt.append(float(altitude_error(state)))
    return tripped, float(np.mean(alt[scored_burn_in(name, params) :]))


@pytest.mark.large
@pytest.mark.parametrize("name", ["patrol", "patrol_range_bearing", "formation"])
@pytest.mark.parametrize("seed", HELD_OUT)
def test_the_expert_never_trips(name, seed):
    tripped, _ = _episode(name, seed)
    assert not tripped


@pytest.mark.large
@pytest.mark.parametrize("seed", HELD_OUT)
def test_the_formation_lead_holds_its_altitude(seed):
    _, alt = _episode("formation", seed)
    assert alt <= ALTITUDE_M, f"mean altitude error {alt:.2f} m"
