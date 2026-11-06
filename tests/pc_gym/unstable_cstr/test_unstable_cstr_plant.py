"""The thermowell, the valve and the ramp of ``unstable_cstr``
(docs/proposals/industrial-fixes.md, PR 4).

Contract, from the plan and PHYSICS.md's sourced figures, not from the code:

- ``T`` reads through a 10 s first-order thermowell. ``C_a`` and ``T_j`` stay
  current. At 3 s steps a unit step reads 1 - exp(-0.3 k) after k steps.
- The coolant command passes a 2 s valve lag and an 8 %/s slew limit. At 3 s
  steps the valve closes 1 - exp(-1.5) = 0.7769 of its gap, and moves at most
  0.08 * 3 = 0.24 of its span, 0.48 raw.
- A reset and a restart leave the valve where the jacket is, so holding moves
  nothing.
- A target leaves its level on the step that enters its block and moves at
  1/30 mol/L per minute: the largest switch, 0.10, takes three residence times,
  so 60 steps of 3 s, and 1/600 mol/L per step. It never passes the new level.
- The point of no return is the plant's own: full cooling goes through the valve.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.base import remember
from target_gym.pc_gym.unstable_cstr.env import (
    draw_schedule,
    live_target,
    steady_coolant,
    steady_temperature,
)
from target_gym.pc_gym.unstable_cstr.env_jax import UnstableCSTR
from target_gym.pc_gym.unstable_cstr.experts import point_of_no_return_batch
from target_gym.registry import REGISTRY

KEY = jax.random.PRNGKey(0)
LEVELS = (0.50, 0.60, 0.55, 0.45, 0.50, 0.60)
#: mol/L per step: 0.10 over 60 steps.
STEP = 1.0 / 600.0


@pytest.fixture(scope="module")
def env():
    return UnstableCSTR()


@pytest.fixture(scope="module")
def params(env):
    return env.default_params.replace(sensor_noise_scale=0.0)


def _raw(T_j, params):
    return 2.0 * (T_j - params.T_c_min) / (params.T_c_max - params.T_c_min) - 1.0


def test_the_reactor_thermowell_reads_as_a_first_order_sensor(env, params):
    _, state = env.reset_env(KEY, params)
    T = float(state.T)
    hot = state.replace(T=state.T + 1.0, C_a=state.C_a + 0.01, T_j=state.T_j + 1.0)
    read = []
    for _ in range(3):
        state = remember(env, state, hot, params)
        read.append(float(env.get_obs(state, params)[1]) - T)
    # 1e-4 K: float32 resolves 3e-5 K at 350 K.
    np.testing.assert_allclose(read, [0.25918, 0.45119, 0.59343], atol=1e-4)
    obs = env.get_obs(state, params)
    assert float(obs[0]) == float(hot.C_a)
    assert float(obs[2]) == float(hot.T_j)


@pytest.mark.parametrize(
    "gap, moved",
    [
        (0.1, 0.1 * 0.77687),  # the lag alone
        (-0.1, -0.1 * 0.77687),
        (1.0, 0.48),  # the lag would take 0.777: the slew limit binds
        (-1.0, -0.48),
    ],
)
def test_the_valve_lags_small_moves_and_slews_large_ones(env, params, gap, moved):
    _, state = env.reset_env(KEY, params)
    state = state.replace(actuator_held=jnp.zeros(1))
    *_, info = env.step_env(KEY, state, jnp.array([gap]), params)
    assert float(info["applied_action"][0]) == pytest.approx(moved, abs=1e-5)


def test_a_command_past_the_valves_reach_moves_it_as_one_at_the_reach(env, params):
    """0.48 raw of slew over the 0.7769 the lag closes: a reach of 0.6179, which
    is where the oracle bounds its moves (T1r)."""
    _, state = env.reset_env(KEY, params)
    state = state.replace(actuator_held=jnp.full(1, -0.5))
    at = env.step_env(KEY, state, jnp.array([-0.5 + 0.48 / 0.77687]), params)[1]
    past = env.step_env(KEY, state, jnp.array([1.0]), params)[1]
    np.testing.assert_allclose(past.actuator_held, at.actuator_held, rtol=1e-5)


@pytest.mark.medium
def test_the_oracle_keeps_its_moves_inside_the_valves_reach():
    spec = REGISTRY["unstable_cstr"]
    model = spec.make_mpc(spec.make_env(), spec.make_test_params()).model
    held, reach = -0.5, 0.48 / 0.77687
    x = jnp.array([0.5, 350.0, 300.0, held])

    def least(move):
        return float(model.move_rows(x, jnp.array([held + move]), 1).min())

    assert least(0.99 * reach) > 0.0 > least(1.01 * reach)
    assert least(-0.99 * reach) > 0.0 > least(-1.01 * reach)


def test_a_reset_and_a_restart_leave_the_valve_at_the_jacket(env, params):
    _, state = env.reset_env(KEY, params)
    raw = _raw(float(state.T_j), params)
    assert float(state.actuator_held[0]) == pytest.approx(raw, abs=1e-5)
    *_, info = env.step_env(KEY, state, jnp.array([raw]), params)
    assert float(info["applied_action"][0]) == pytest.approx(raw, abs=1e-5)

    tripping = state.replace(T=jnp.float32(params.T_trip + 5.0))
    _, restart, *_, info = env.step_env(KEY, tripping, jnp.array([raw]), params)
    assert bool(info["tripped"])
    assert float(restart.actuator_held[0]) == pytest.approx(
        _raw(float(restart.T_j), params), abs=1e-5
    )


@pytest.mark.parametrize(
    "clock, level",
    [
        (0, 0.50),
        (199, 0.50),
        (200, 0.50 + STEP),
        (259, 0.60),
        (260, 0.60),
        (399, 0.60),
        (400, 0.60 - STEP),
        (429, 0.55),
        (430, 0.55),
        (599, 0.55),
        (659, 0.45),
        (5000, 0.60),
    ],
)
def test_the_target_ramps_from_level_to_level(env, params, clock, level):
    _, state = env.reset_env(KEY, params)
    state = state.replace(
        target_levels=jnp.asarray(LEVELS, jnp.float32), block_clock=jnp.int32(clock)
    )
    assert float(live_target(state, params)) == pytest.approx(level, abs=1e-6)


def test_every_schedule_moves_at_the_ramp_and_reaches_each_level(env, params):
    """Three residence times over the largest switch, whatever the seed."""
    assert 3.0 * params.V / params.q / params.delta_t == pytest.approx(60.0)
    _, state = env.reset_env(KEY, params)
    clocks = jnp.arange(params.max_steps_in_episode)
    for seed in range(4):
        levels = draw_schedule(jax.random.PRNGKey(seed), params)
        moving = state.replace(target_levels=levels)
        path = jax.vmap(lambda c: live_target(moving.replace(block_clock=c), params))(
            clocks
        )
        assert float(jnp.abs(jnp.diff(path)).max()) <= STEP * (1.0 + 1e-3)
        np.testing.assert_allclose(path[199::200], levels, atol=1e-6)


def test_the_point_of_no_return_is_reached_through_the_valve(env, params):
    """Either side of it, full cooling through the env's whole step catches the
    plant or does not. 0.02 K is 700 float32 steps at 350 K."""
    level = 0.50
    calm = params.replace(Ti_sigma=0.0)
    T_j = float(steady_coolant(level, 0.0, calm, np))
    edge = float(point_of_no_return_batch(level, T_j, 0.0, calm)[0])
    assert edge > float(steady_temperature(level, calm, np))

    def trips(T):
        _, state = env.reset_env(KEY, calm)
        state = state.replace(
            C_a=jnp.float32(level),
            T=jnp.float32(T),
            T_j=jnp.float32(T_j),
            Ti_dev=jnp.float32(0.0),
            actuator_held=jnp.full(1, _raw(T_j, calm), jnp.float32),
        )

        def body(s, _):
            _, s2, *_, info = env.step_env(KEY, s, jnp.array([-1.0]), calm)
            return s2, info["tripped"]

        return bool(jax.lax.scan(body, state, None, length=200)[1].any())

    assert trips(edge + 0.02) and not trips(edge - 0.02)
