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

from target_gym import eval as E
from target_gym.compressor_surge.env import (
    check_is_terminal,
    compute_next_state,
    live_target,
)
from target_gym.compressor_surge.experts import DEFAULT_GAINS, settled_point
from target_gym.compressor_surge.oracle import compressor_model
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


@pytest.mark.parametrize("seed", range(3))
def test_the_first_move_ramps_from_the_resets_own_pressure(seed):
    """The reset spans 18 to 32 kPa against levels of 20 to 28, so a target that
    opened on level 0 made every episode start with a step. Halfway through the
    10 s ramp it is halfway between them."""
    s = _reset(seed)
    level, dp = float(s.setpoint_levels[0]), float(s.dp)
    assert float(live_target(s, P)) == pytest.approx(dp, rel=1e-5)
    s50 = s.replace(block_clock=50)
    assert float(live_target(s50, P)) == pytest.approx((level + dp) / 2, rel=1e-5)
    s100 = s.replace(block_clock=100)
    assert float(live_target(s100, P)) == pytest.approx(level, rel=1e-5)


def test_the_largest_ramp_is_inside_what_the_drive_can_follow():
    """Speed at the control line for 20 and 28 kPa, a third of it recycled."""
    speed = [settled_point(p, 0.35, DEFAULT_GAINS, P)[2] for p in (20e3, 28e3)]
    assert speed[1] - speed[0] < P.drive_rate * P.setpoint_ramp_s


@pytest.mark.large
def test_ramps_take_the_cost_out_of_the_setpoint_moves():
    """97-99 % of the expert's cost fell in the 10 s after a step (review), so
    the ranking read the slew. Past the opening, the same schedules ramped cost
    under a tenth."""
    cost = [
        sum(
            E.run_episode(SPEC, q, baseline_policy(SPEC, "pid", q), seed=s)
            .terms["tracking"][150:]
            .sum()
            for s in (200, 201)
        )
        for q in (P.replace(setpoint_ramp_s=0.1), P)
    ]
    assert cost[1] < 0.1 * cost[0]


def test_each_episode_draws_its_own_hidden_surge_line():
    keys = jax.random.split(KEY, 64)
    offsets = np.asarray(jax.vmap(lambda k: ENV.reset_env(k, P))(keys)[1].surge_offset)
    assert offsets.min() >= 0.0 and offsets.max() <= P.surge_offset_max
    assert offsets.max() - offsets.min() > 0.8 * P.surge_offset_max
    s = _reset()
    np.testing.assert_array_equal(
        ENV.get_obs(s.replace(surge_offset=0.0), P.replace(sensor_noise_scale=0.0)),
        ENV.get_obs(s.replace(surge_offset=0.05), P.replace(sensor_noise_scale=0.0)),
    )


@pytest.mark.parametrize(
    "phi, offset, trips",  # the map's line is at 2W = 0.5, 0.525 at +5 %
    [(0.51, 0.0, False), (0.49, 0.0, True), (0.51, 0.05, True), (0.53, 0.05, False)],
)
def test_the_trip_reads_the_episodes_line(phi, offset, trips):
    s = _reset().replace(phi_min=phi, surge_offset=offset)
    assert bool(check_is_terminal(s, P)[0]) is trips


@pytest.mark.parametrize("valve, moved", [(1.0, 0.05), (0.0, 0.01)])
def test_valve_travel_is_charged_above_the_positioners_dead_band(valve, moved):
    s = _reset()
    new = ENV.step_env(KEY, s, _raw(float(s.N_ramp), valve), P)[1]
    assert float(new.valve_travel) == pytest.approx(moved, rel=1e-3)
    charged = moved / 0.0025 - 1.0  # weight 1: over the 0.25 % dead band
    assert float(ENV.reward_terms(new, P)["wear"]) == pytest.approx(charged, rel=1e-3)


def test_the_applied_action_is_where_the_actuators_went():
    s = _reset()
    np.testing.assert_allclose(s.actuator_held, _raw(s.N_ramp, s.x), rtol=1e-5)
    new, info = (lambda o: (o[1], o[4]))(ENV.step_env(KEY, s, _raw(1.05, 1.0), P))
    np.testing.assert_allclose(
        info["applied_action"], _raw(new.N_ramp, new.x), rtol=1e-5
    )


@pytest.mark.parametrize("clock", [50, 350])  # the opening ramp and a later one
def test_the_oracles_model_reads_the_ramp_the_line_and_the_travel(clock):
    """Mid-ramp at the widest line, one exact model step from the reset is the
    env's: the same target, the same wear and the same distance to the trip."""
    s = _reset().replace(block_clock=clock, surge_offset=0.05)
    u, m = jnp.asarray(_raw(1.0, 0.8)), compressor_model(ENV, P)
    d = {"task": m.data(s), "widths": (0.0, 0.0)}
    _, cost, y, _, _ = m.step(d, m.state_x(s), u, 0)
    new = compute_next_state(u, s, P, KEY)[0]
    target = (float(live_target(new, P)) - float(new.dp)) / 1e3 / P.e_floor
    assert float(cost) == pytest.approx(target**2, rel=1e-3)
    assert float(y[1]) * P.travel_free == pytest.approx(
        float(new.valve_travel), rel=1e-3
    )
    assert float(y[2]) - 0.5 == pytest.approx(float(new.phi_min) - 0.525, abs=1e-5)
