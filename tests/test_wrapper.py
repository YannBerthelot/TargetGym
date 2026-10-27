import os

import numpy as np
import pytest

from target_gym.plane.env import PlaneParams
from target_gym.plane.env_jax import Airplane2D
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
