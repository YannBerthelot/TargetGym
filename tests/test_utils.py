import os
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.utils import (
    compute_norm_from_coordinates,
    convert_raw_action_to_range,
)
from target_gym.video import convert_frames_from_gym_to_wandb, save_video


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


class DummyState:
    """Simple dummy state for testing save_video."""

    def __init__(self):
        self.time = 0  # timestep counter


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
        state.time += 1
        done = state.time >= self.default_params.max_steps_in_episode
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


def test_save_video_creates_file(env, dummy_action):
    with tempfile.TemporaryDirectory() as tmpdir:
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
# Test stationary_innovation
# -------------------------------
def test_stationary_innovation_keeps_the_process_at_its_stated_spread():
    """x' = decay * x + s * xi with xi ~ N(0, 1) has variance
    decay**2 * var + s**2. At the stated spread that must give the spread back:
    s**2 = std**2 * (1 - decay**2). By hand, std 3 and decay 0.8:
    s = 3 * sqrt(0.36) = 1.8, and 0.64 * 9 + 3.24 = 9."""
    from target_gym.utils import stationary_innovation

    s = float(stationary_innovation(3.0, 0.8))
    assert s == pytest.approx(1.8, rel=1e-6)
    assert 0.8**2 * 3.0**2 + s**2 == pytest.approx(3.0**2, rel=1e-6)
    # No decay left, no memory: every draw is a fresh one at the full spread.
    assert float(stationary_innovation(3.0, 0.0)) == pytest.approx(3.0)
