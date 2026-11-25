import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnax.environments import spaces

import target_gym as tg
from target_gym.pc_gym.four_tank.env_jax import FourTank
from target_gym.plane.env import PlaneParams
from target_gym.plane.env_jax import Airplane2D
from target_gym.registry import REGISTRY
from target_gym.wrapper import GymnasiumEnv

max_steps_for_video = 100


@pytest.mark.parametrize(
    "jax_env_class, env_params, dummy_action",
    [
        (
            Airplane2D,
            PlaneParams(max_steps_in_episode=max_steps_for_video),
            lambda obs: np.array([0.8, 0.0]),
        ),
    ],
)
def test_wrapper_reset_step_render(jax_env_class, env_params, dummy_action, tmp_path):
    env = GymnasiumEnv(jax_env_class(), env_params, render_mode="rgb_array")

    # Reset environment
    obs, info = env.reset(seed=0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape

    # Step environment
    obs2, reward, done, truncated, info = env.step(dummy_action(obs))
    assert isinstance(obs2, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    # Render environment
    frame = env.render()
    assert frame is None or isinstance(frame, np.ndarray)

    # Save video (dummy short rollout)
    video_path = env.save_video(
        select_action=dummy_action,
        folder=tmp_path,
        episode_index=0,
        FPS=1,
        format="mp4",
        seed=0,
    )
    assert os.path.exists(video_path)


@pytest.mark.parametrize(
    "jax_env_class",
    [Airplane2D],
)
def test_wrapper_reset_random_consistency(jax_env_class):
    """Resetting with same seed produces same observation"""
    env1 = GymnasiumEnv(jax_env_class())
    obs1, _ = env1.reset(seed=42)

    env2 = GymnasiumEnv(jax_env_class())
    obs2, _ = env2.reset(seed=42)

    np.testing.assert_allclose(obs1, obs2, rtol=1e-6)


@pytest.mark.parametrize(
    "jax_env_class, dummy_action",
    [
        (Airplane2D, lambda obs: np.array([0.8, 0.0])),
    ],
)
def test_wrapper_multi_step(jax_env_class, dummy_action):
    """Run multiple steps without crashing"""
    env = GymnasiumEnv(jax_env_class())

    obs, _ = env.reset(seed=0)
    for _ in range(5):
        obs, reward, done, truncated, info = env.step(dummy_action(obs))
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)


def test_wrapper_draws_a_fresh_key_per_reset_and_step():
    """One key reused everywhere gave every unseeded episode the same start
    (and every trip restart the same draw). A seed still makes a run repeat."""
    env = GymnasiumEnv(Airplane2D(), PlaneParams(turbulence_sigma=2.0))

    def rollout(seed):
        obs = [env.reset(seed=seed)[0]]
        for _ in range(5):
            obs.append(env.step(np.array([0.8, 0.0]))[0])
        return np.stack(obs)

    np.testing.assert_array_equal(rollout(0), rollout(0))
    a, _ = env.reset()
    b, _ = env.reset()
    assert not np.array_equal(a, b)


def test_wrapper_accepts_the_scalar_action_bounds_gymnax_allows():
    """``spaces.Box(-1.0, 1.0, (2,))`` is a legal gymnax space, and the first
    thing a new environment is likely to write. No registered task uses it, so
    conformance cannot see this."""

    class ScalarBounds(FourTank):
        def action_space(self, params=None):
            return spaces.Box(-1.0, 1.0, (2,), dtype=jnp.float32)

    space = GymnasiumEnv(ScalarBounds()).action_space
    np.testing.assert_array_equal(space.low, [-1.0, -1.0])
    np.testing.assert_array_equal(space.high, [1.0, 1.0])


# Contract (``make_gymnasium``'s docstring: the task behind the classic stateful
# API, stepped on the host): a library steps it one call at a time for as long as
# it trains, so a step must cost a step. Once a step has run, another one builds
# no new program: JAX keeps every program it builds, and a wrapper that built
# one per step grew by 22 to 50 MB a step on the reactor until the process died
# (2026-11-23). ``jax.log_compiles`` is JAX's own report of what it builds.
_LOOPED = ("plane", "reactor", "cement_kiln")  # 1, 1 and 5 sub-step loops


def _programs_built(run):
    import logging

    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logger = logging.getLogger("jax")
    logger.addHandler(handler)
    try:
        with jax.log_compiles():
            run()
    finally:
        logger.removeHandler(handler)
    return [r.getMessage() for r in records]


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(n, marks=[] if n in _LOOPED else [pytest.mark.medium])
        for n in REGISTRY
    ],
)
def test_a_gymnasium_step_builds_no_program_once_one_has_run(name):
    env = tg.make_gymnasium(name)
    action = np.zeros(env.action_space.shape, np.float32)
    env.reset(seed=0)
    for _ in range(2):
        env.step(action)

    def three_more():
        for _ in range(3):
            env.step(action)

    assert _programs_built(three_more) == []


@pytest.mark.medium
def test_a_longer_video_builds_no_more_programs_than_a_short_one(tmp_path):
    """Same contract for ``save_video``, which steps a whole episode on the host:
    what it builds does not grow with the episode's length."""

    def built(steps):
        env, params = tg.make("reactor", max_steps_in_episode=steps)
        zero = np.zeros(env.action_space(params).shape, np.float32)
        policy = lambda obs: zero  # noqa: E731
        run = lambda: env.save_video(policy, 0, params, folder=tmp_path)  # noqa: E731
        return len(_programs_built(run))

    built(3)  # the renderer's and the step's own programs, built once
    assert built(8) == built(4)
