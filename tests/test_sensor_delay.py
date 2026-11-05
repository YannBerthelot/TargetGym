"""Instruments that report late (docs/proposals/industrial-fixes.md, PR 3).

Contract: a reading with dead time ``d`` and interval ``p``, both in steps, read
at step ``t``, reports the true value at step ``p * (t // p) - d``, and no value
from before the reset. Its noise is held with it. The value is the reading's own,
so what is derived from it inherits the age. ``sensor_noise_scale`` 0 removes the
noise and keeps the age, and ``true_obs`` is what the plant is doing now.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.base import FieldReading, measured_state, remember, true_obs
from target_gym.benchmark import CallablePolicy, run_policy, task_info
from target_gym.registry import REGISTRY

STEP = float(REGISTRY["first_order"].make_test_params().delta_t)


def _first_order(readings, scale=0.0):
    spec = REGISTRY["first_order"]
    env = spec.env_factory()  # an instance of our own: readings are set on it
    env.readings = readings
    return env, spec.make_test_params().replace(sensor_noise_scale=scale)


def _trace(readings, steps=12, scale=0.0):
    """``(true, read)`` of ``first_order``'s state under full action, reset first."""
    env, params = _first_order(readings, scale)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key, params)
    true, read = [float(state.x)], [float(obs[0])]
    for _ in range(steps):
        obs, state, *_ = env.step_env(key, state, jnp.array([1.0]), params)
        true.append(float(state.x))
        read.append(float(obs[0]))
    return np.array(true), np.array(read)


def _reading(dead=0, every=0, **kwargs):
    return FieldReading(
        "x",
        sigma=kwargs.pop("sigma", 0.0),
        dead_time=dead * STEP,
        interval=every * STEP,
        sampled=every > 0,
        **kwargs,
    )


@pytest.mark.parametrize(
    "dead, every, age",
    [
        (0, 0, lambda t: t),
        (3, 0, lambda t: max(t - 3, 0)),
        (0, 4, lambda t: 4 * (t // 4)),
        (2, 4, lambda t: max(4 * (t // 4) - 2, 0)),
    ],
)
def test_a_reading_reports_the_value_of_its_last_update_less_its_dead_time(
    dead, every, age
):
    true, read = _trace((_reading(dead, every),))
    np.testing.assert_allclose(read, [true[age(t)] for t in range(len(true))])


def test_a_lagged_reading_rises_as_a_first_order_sensor():
    """A true value that steps by 1 reads ``1 - exp(-k dt / tau)`` after k steps.
    With ``tau`` of 2 steps, that is 0.3935, 0.6321 and 0.7769."""
    env, params = _first_order((_reading(lag=2 * STEP),))
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    state = state.replace(x=jnp.zeros_like(state.x))
    state = state.replace(sensor_history=jnp.zeros_like(state.sensor_history))
    one = state.replace(x=jnp.ones_like(state.x))
    read = []
    for _ in range(3):
        state = remember(env, state, one, params)
        read.append(float(env.get_obs(state, params)[0]))
    np.testing.assert_allclose(read, [0.39347, 0.63212, 0.77687], atol=1e-4)


def test_a_sampled_reading_holds_its_noise_with_its_value():
    _, read = _trace((_reading(0, 4, sigma=0.01),), scale=1.0)
    assert len(set(read[:4])) == 1
    assert len(set(read[4:8])) == 1
    assert read[3] != read[4]


def test_scale_zero_takes_the_noise_and_leaves_the_age():
    true, read = _trace((_reading(3, sigma=0.01),), scale=0.0)
    np.testing.assert_allclose(read[3:], true[:-3])


def test_true_obs_is_the_plant_now():
    env, params = _first_order((_reading(3),))
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    for _ in range(5):
        _, state, *_ = env.step_env(key, state, jnp.array([1.0]), params)
    assert float(true_obs(env, state, params)[0]) == float(state.x)
    assert float(env.get_obs(state, params)[0]) != float(state.x)


def test_a_late_reading_is_a_pure_function_of_the_state():
    env, params = _first_order((_reading(2, 4, sigma=0.01),), scale=1.0)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    _, state, *_ = env.step_env(key, state, jnp.array([1.0]), params)
    np.testing.assert_array_equal(
        env.get_obs(state, params), env.get_obs(state, params)
    )


def test_late_readings_run_under_jit_vmap_and_scan():
    env, params = _first_order((_reading(2, 4),))

    def episode(key):
        _, state = env.reset_env(key, params)

        def step(state, _):
            obs, state, *_ = env.step_env(key, state, jnp.array([1.0]), params)
            return state, obs[0]

        return jax.lax.scan(step, state, None, length=6)[1]

    out = jax.jit(jax.vmap(episode))(jax.random.split(jax.random.PRNGKey(0), 3))
    assert out.shape == (3, 6)


def test_a_task_that_rebuilds_its_state_keeps_its_history():
    """``plane`` builds its next state field by field, which once lost the
    state's instruments, so the history cannot rely on being carried."""
    spec = REGISTRY["plane"]
    env = spec.env_factory()
    params = spec.make_test_params().replace(sensor_noise_scale=0.0)
    env.readings = (FieldReading("z", sigma=0.0, dead_time=3 * float(params.delta_t)),)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    true = [float(state.z)]
    for _ in range(6):
        _, state, *_ = env.step_env(key, state, jnp.array([0.5, 0.2]), params)
        true.append(float(state.z))
    assert float(measured_state(env, state, params).z) == pytest.approx(true[-4])


def test_a_restart_refills_the_instruments_from_the_restarted_plant():
    """An analyser reading the plant that tripped would report a plant that is
    gone."""
    env, params = _first_order((_reading(3),))
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    far = state.replace(x=jnp.asarray(1e6, state.x.dtype))
    obs, state, *_, info = env.step_env(key, far, jnp.array([0.0]), params)
    assert bool(info["tripped"])
    assert float(obs[0]) == float(state.x)


def test_the_measured_reward_reads_the_same_late_copy():
    """A plant computes its reward from what it reads, so with a late reading
    the signal is the cost of the late value. ``reward`` stays the true cost."""
    env, params = _first_order((_reading(3),))
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    for _ in range(5):
        _, state, reward, _, info = env.step_env(key, state, jnp.array([1.0]), params)
    late = measured_state(env, state, params)
    assert float(info["measured_reward"]) == pytest.approx(
        float(env.compute_reward(late, params))
    )
    assert float(info["measured_reward"]) != pytest.approx(float(reward))


def test_the_band_kpi_reads_the_plant_and_not_the_late_reading():
    """The band is half the way to the target, so the plant enters it during
    the episode, and a reading 5 steps late would enter it 5 steps after."""
    env, params = _first_order((_reading(5),))
    obs, _ = env.reset_env(jax.random.PRNGKey(0), params)
    start = np.asarray(obs)
    band = abs(start[1] - start[0]) / 2.0
    task = dataclasses.replace(
        task_info("first_order", env, params, 1), e_floor=np.array([[band]])
    )
    direction = np.sign(start[1] - start[0])
    policy = CallablePolicy(
        lambda task: lambda obs, state: np.array([direction]), name="full", oracle=False
    )
    out = run_policy(
        policy,
        env,
        params,
        task,
        jax.random.PRNGKey(0),
        record={"x": lambda s, p, t: s.x},
    )
    error = np.abs(out["records"]["x"] - out["target"][:, :, 0])
    assert out["in_band"].any() and not out["in_band"].all()
    np.testing.assert_array_equal(out["in_band"], error <= band)


def test_the_column_reads_its_analysers_late_and_its_tray_now():
    """The pilot: a chromatograph on each product, 5 min dead and a 5 min cycle
    at 1 min steps, so step ``t`` reads step ``5 * (t // 5) - 5``. The tray
    thermocouple is on-line."""
    spec = REGISTRY["distillation"]
    env = spec.make_env()
    params = spec.make_test_params().replace(sensor_noise_scale=0.0)
    key = jax.random.PRNGKey(1)
    obs, state = env.reset_env(key, params)
    top, bottom, read = [float(state.x[-1])], [float(state.x[0])], [np.asarray(obs)]
    action = state.actuator_held + jnp.array([0.02, -0.02])  # a gentle move
    for _ in range(14):
        obs, state, *_, info = env.step_env(key, state, action, params)
        assert not bool(info["tripped"])
        top.append(float(state.x[-1]))
        bottom.append(float(state.x[0]))
        read.append(np.asarray(obs))
    aged = [max(5 * (t // 5) - 5, 0) for t in range(15)]
    np.testing.assert_allclose([r[0] for r in read], [top[a] for a in aged])
    np.testing.assert_allclose([r[1] for r in read], [bottom[a] for a in aged])
    assert read[-1][6] == pytest.approx(float(true_obs(env, state, params)[6]))
