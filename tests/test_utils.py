import os
import tempfile
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.utils import (
    compute_episode_returns_vectorized,
    compute_norm_from_coordinates,
    convert_frames_from_gym_to_wandb,
    convert_raw_action_to_range,
    run_n_steps,
    save_video,
)


# -------------------------------
# Test compute_norm_from_coordinates
# -------------------------------
def test_compute_norm_from_coordinates():
    vec = jnp.array([3.0, 4.0])
    norm = compute_norm_from_coordinates(vec)
    assert jnp.isclose(norm, 5.0)


# -------------------------------
# Test convert_frames_from_gym_to_wandb
# -------------------------------
def test_convert_frames_from_gym_to_wandb():
    frames = np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8)
    converted = convert_frames_from_gym_to_wandb(frames)
    assert converted.shape == (5, 3, 64, 64)
    # Check that the pixel data is preserved
    assert np.allclose(converted[:, :, :, :], frames.transpose(0, 3, 1, 2))


# -------------------------------
# Test convert_raw_action_to_range
# -------------------------------
def test_convert_raw_action_to_range():
    raw = jnp.array([-1.0, 0.0, 1.0])
    min_action = jnp.array([0.0, 0.0, 0.0])
    max_action = jnp.array([2.0, 4.0, 6.0])
    rescaled = convert_raw_action_to_range(raw, min_action, max_action)
    expected = jnp.array([0.0, 2.0, 6.0])
    assert jnp.allclose(rescaled, expected)


# -------------------------------
# Test compute_episode_returns_vectorized
# -------------------------------
def test_compute_episode_returns_vectorized_simple():
    rewards = jnp.array([1, 1, 1, 1])
    dones = jnp.array([0, 1, 0, 1])
    returns = compute_episode_returns_vectorized(rewards, dones)
    # first episode: 1+1=2, second episode: 1+1=2
    assert jnp.allclose(returns, jnp.array([2.0, 2.0]))


def test_compute_episode_returns_vectorized_truncated():
    rewards = jnp.array([1, 1, 1])
    dones = jnp.array([0, 0, 0])  # truncated, no done
    returns = compute_episode_returns_vectorized(rewards, dones)
    assert jnp.allclose(returns, jnp.array([3.0]))


class DummyState:
    """Simple dummy state for testing save_video."""

    def __init__(self):
        self.t = 0  # timestep counter


class DummyParams:
    """Dummy parameters with max_steps_in_episode"""

    def __init__(self, max_steps_in_episode=1):
        self.max_steps_in_episode = max_steps_in_episode


class DummyEnv:
    """Mock environment for save_video testing"""

    default_params = DummyParams()

    def reset(self, key=None, params=None):
        self.state = DummyState()
        return np.zeros((2,)), self.state

    def step(self, key, state, action, params=None):
        # Increment timestep to simulate episode progress
        state.t += 1
        done = state.t >= self.default_params.max_steps_in_episode
        reward = 1.0
        return np.zeros((2,)), state, reward, done, {}

    def render(self, screen=None, state=None, params=None, frames=None, clock=None):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        if frames is not None:
            frames.append(frame)
        return frames, screen, clock


@pytest.fixture
def env():
    return DummyEnv()


@pytest.fixture
def dummy_action():
    def _action(obs):
        return np.zeros_like(obs)

    return _action


def test_save_video_creates_file(env, dummy_action, monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        original_join = os.path.join  # save original
        monkeypatch.setattr(
            "target_gym.utils.os.path.join",
            lambda *args: original_join(
                tmpdir, *args[1:]
            ),  # skip the first arg (folder)
        )

        video_path = save_video(
            env=env, select_action=dummy_action, folder=tmpdir, episode_index=0, FPS=1
        )

        expected_path = os.path.join(tmpdir, "episode_000.mp4")
        assert video_path == expected_path
        assert os.path.exists(video_path)


# -------------------------------
# Test convert_raw_action_to_range
# -------------------------------
def test_convert_raw_action_to_range_basic():
    raw = jnp.array([-1.0, 0.0, 1.0])
    min_action = jnp.array([0.0, 2.0, -1.0])
    max_action = jnp.array([1.0, 4.0, 1.0])

    rescaled = convert_raw_action_to_range(raw, min_action, max_action)
    expected = jnp.array([0.0, 3.0, 1.0])
    assert jnp.allclose(rescaled, expected)


# -------------------------------
# Dummy environment for run_n_steps
# -------------------------------
class DummyJaxEnv:
    def __init__(self, max_steps=5):
        self.max_steps = max_steps

    def reset_env(self, key, params):
        obs = jnp.zeros((2,))
        state = jnp.array(0)  # simple scalar state
        return obs, state

    def step_env(self, key, state, action, params):
        reward = 1.0
        done = state >= self.max_steps - 1
        next_state = state + 1
        info = {}
        obs = jnp.zeros((2,))
        return obs, next_state, reward, done, info


def expected_mean_return(n_steps, max_steps):
    num_full_eps = n_steps // max_steps
    last_steps = n_steps % max_steps
    returns = [max_steps] * num_full_eps
    if last_steps:
        returns.append(last_steps)
    return sum(returns) / len(returns)


def dummy_policy(obs):
    return jnp.zeros_like(obs)


# -------------------------
# Test run_n_steps
# -------------------------
@pytest.mark.parametrize("n_steps,max_steps", [(3, 2), (5, 2), (6, 3), (10, 4)])
def test_run_n_steps_dummy(n_steps, max_steps):
    env = DummyJaxEnv(max_steps=max_steps)
    params = None

    mean_return = run_n_steps(env, dummy_policy, params, n_steps=n_steps, seed=0)
    expected = expected_mean_return(n_steps, max_steps)

    assert jnp.isclose(
        mean_return, expected
    ), f"Mean return {mean_return} != expected {expected}"
