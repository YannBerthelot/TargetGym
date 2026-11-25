"""The compressor's setpoint ramps, surge-line offset and valve travel.

Contract, from docs/proposals/industrial-fixes.md (PR 6) and these figures:

* Ours: a setpoint move is a 10 s linear ramp, and the top of the 20 to 28 kPa
  range stays inside what the 3 %/s drive can follow.
* Ours: each episode's surge line sits 0 to 5 % right of the map's, hidden.
* Mirsky et al. (2015) p.23, Wilson and Sheldon (2006) Table 1: the valve opens
  in 2 s, 0.05 per 0.1 s step, and closes in 10 s, 0.01 per step.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.compressor_surge.env import (
    live_target,
)
from target_gym.compressor_surge.experts import DEFAULT_GAINS, settled_point
from target_gym.registry import REGISTRY
from target_gym.runners.runners import baseline_policy

SPEC = REGISTRY["compressor_surge"]
ENV, P = SPEC.make_env(), SPEC.make_test_params()
KEY = jax.random.PRNGKey(0)


def _reset(seed=0):
    return ENV.reset_env(jax.random.PRNGKey(seed), P)[1]


def _raw(speed, valve):
    return np.array([2 * (speed - P.N_min) / (P.N_max - P.N_min) - 1, 2 * valve - 1])


@pytest.mark.parametrize(
    "clock, kpa",  # 20 to 28 kPa from t = 30 s at 0.8 kPa/s
    [(299, 20.0), (300, 20.0), (325, 22.0), (350, 24.0), (400, 28.0), (550, 28.0)],
)
def test_a_setpoint_ramps_linearly_between_its_levels(clock, kpa):
    levels = jnp.array([20e3, 28e3, 28e3, 28e3])
    s = _reset().replace(setpoint_levels=levels, block_clock=clock)
    assert float(live_target(s, P)) / 1e3 == pytest.approx(kpa, abs=1e-3)


def test_the_largest_ramp_is_inside_what_the_drive_can_follow():
    """Speed at the control line for 20 and 28 kPa, a third of it recycled."""
    speed = [settled_point(p, 0.35, DEFAULT_GAINS, P)[2] for p in (20e3, 28e3)]
    assert speed[1] - speed[0] < P.drive_rate * P.setpoint_ramp_s


def _zone_after_a_move(q, seed, to):
    """The shipped expert's zone cost over the block after the setpoint moves
    from 24 kPa to ``to`` (Pa), on seed ``seed``: the move is built into the
    schedule, as the task's own setpoint is constant [C3, p. 4]."""
    key = jax.random.PRNGKey(seed)
    _, s = ENV.reset_env(key, q)
    s = s.replace(setpoint_levels=jnp.asarray([24e3, to, to, to], jnp.float32))
    obs, policy, step = ENV.get_obs(s, q), baseline_policy(SPEC, "pid", q), _STEP
    block, zone = int(q.setpoint_block_steps), 0.0
    for t in range(2 * block):
        action = jnp.asarray(policy(np.asarray(obs)), jnp.float32)
        obs, s, _, _, _ = step(jax.random.fold_in(key, t), s, action, q)
        if t >= block:
            zone += float(ENV.reward_terms(s, q)["zone"])
    return zone


_STEP = jax.jit(ENV.step_env)


@pytest.mark.large
def test_ramps_take_the_cost_out_of_the_setpoint_moves():
    """97-99 % of the expert's cost fell in the 10 s after a step (review), so
    the ranking read the slew; ramps were added for that.

    Re-measured on the current plant (2026-11-25, owner: keep the behaviour).
    The task's setpoint is constant now, so the move is built here: 24 to 28
    and 24 to 20 kPa at the second block, seeds 200 and 201, the shipped
    expert, the setpoint's cost being the ``zone`` term (it was
    ``tracking``). Stepped, the moves cost 5600, 3600, 2300 and 3300; ramped
    over 10 s, 1700, 1400, 0 and 0: a fifth in all (21 %). Down, the ramped
    move never leaves the zone. Up it still does, since the drive's 3 %/s
    takes longer than the ramp to reach the higher speed."""
    stepped = P.replace(setpoint_ramp_s=0.1)
    cost = {
        (to, ramped): sum(_zone_after_a_move(q, s, to) for s in (200, 201))
        for to in (28e3, 20e3)
        for ramped, q in ((False, stepped), (True, P))
    }
    total = {r: cost[(28e3, r)] + cost[(20e3, r)] for r in (False, True)}
    assert total[True] < 0.25 * total[False]
    assert cost[(20e3, True)] == 0.0 < cost[(20e3, False)]


def test_the_applied_action_is_where_the_actuators_went():
    s = _reset()
    np.testing.assert_allclose(s.actuator_held, _raw(s.N_ramp, s.x), rtol=1e-5)
    new, info = (lambda o: (o[1], o[4]))(ENV.step_env(KEY, s, _raw(1.05, 1.0), P))
    np.testing.assert_allclose(
        info["applied_action"], _raw(new.N_ramp, new.x), rtol=1e-5
    )
