import os

import numpy as np
import pytest

from target_gym.bicycle.env import BikeParams
from target_gym.bicycle.env_jax import RandlovBicycle
from target_gym.car.env import CarParams
from target_gym.car.env_jax import Car2D
from target_gym.plane.env import PlaneParams
from target_gym.plane.env_jax import Airplane2D
from target_gym.wrapper import gym_wrapper_factory

max_steps_for_video = 100


@pytest.mark.parametrize(
    "jax_env_class, env_params, dummy_action",
    [
        (
            Airplane2D,
            PlaneParams(max_steps_in_episode=max_steps_for_video),
            lambda obs: np.array([0.8, 0.0]),
        ),
        (
            Car2D,
            CarParams(max_steps_in_episode=max_steps_for_video),
            lambda obs: np.array(0.5),
        ),
        (
            RandlovBicycle,
            BikeParams(max_steps_in_episode=max_steps_for_video),
            lambda obs: np.array([0.2, 0.0]),
        ),
    ],
)
def test_wrapper_reset_step_render(jax_env_class, env_params, dummy_action, tmp_path):
    # Create Gym wrapper class
    GymEnv = gym_wrapper_factory(jax_env_class)
    env = GymEnv(render_mode="rgb_array", env_params=env_params)

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
    [Airplane2D, Car2D, RandlovBicycle],
)
def test_wrapper_reset_random_consistency(jax_env_class):
    """Resetting with same seed produces same observation"""
    GymEnv = gym_wrapper_factory(jax_env_class)
    env1 = GymEnv()
    obs1, _ = env1.reset(seed=42)

    env2 = GymEnv()
    obs2, _ = env2.reset(seed=42)

    np.testing.assert_allclose(obs1, obs2, rtol=1e-6)


@pytest.mark.parametrize(
    "jax_env_class, dummy_action",
    [
        (Airplane2D, lambda obs: np.array([0.8, 0.0])),
        (Car2D, lambda obs: np.array(0.5)),
        (RandlovBicycle, lambda obs: np.array([0.2, 0.0])),
    ],
)
def test_wrapper_multi_step(jax_env_class, dummy_action):
    """Run multiple steps without crashing"""
    GymEnv = gym_wrapper_factory(jax_env_class)
    env = GymEnv()

    obs, _ = env.reset(seed=0)
    for _ in range(5):
        obs, reward, done, truncated, info = env.step(dummy_action(obs))
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
