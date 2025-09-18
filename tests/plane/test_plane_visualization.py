import os
import shutil
import tempfile

import jax
import numpy as np
import pytest
from gymnasium.utils.save_video import save_video
from PIL import Image

from target_gym.plane.env_gymnasium import Airplane2D as GymAirplane2D
from target_gym.plane.env_jax import Airplane2D as JaxAirplane2D
from target_gym.plane.env_jax import EnvParams, EnvState


def are_images_similar(
    img1: np.ndarray, img2: np.ndarray, threshold: float = 0.95
) -> bool:
    """
    Compare two images and return True if they are similar enough.
    Uses normalized cross-correlation for comparison.
    """
    if img1.shape != img2.shape:
        return False

    # Normalize images
    img1_norm = (img1 - img1.mean()) / img1.std()
    img2_norm = (img2 - img2.mean()) / img2.std()

    # Compute correlation
    correlation = np.sum(img1_norm * img2_norm) / img1.size

    return correlation > threshold


def test_basic_render():
    """Test that both environments can render without errors"""
    # JAX environment
    jax_env = JaxAirplane2D()
    key = jax.random.PRNGKey(0)
    obs, state = jax_env.reset(key)
    frames, screen, clock = jax_env.render(
        None, state, jax_env.default_params, [], None
    )
    assert len(frames) == 1
    assert isinstance(frames[0], np.ndarray)
    assert screen is not None
    assert clock is not None

    # Gym environment
    gym_env = GymAirplane2D(render_mode="rgb_array")
    gym_env.reset()
    frame = gym_env.render()
    assert isinstance(frame, np.ndarray)
    assert gym_env.screen is not None
    assert gym_env.clock is not None


def test_render_initialization():
    """Test that both environments initialize rendering components similarly"""
    jax_env = JaxAirplane2D()
    gym_env = GymAirplane2D(render_mode="rgb_array")

    # Reset environments
    key = jax.random.PRNGKey(0)
    _, jax_state = jax_env.reset(key)
    gym_env.reset()

    # First render to initialize
    jax_env.render(None, jax_state, jax_env.default_params, [], None)
    gym_env.render()

    # Check cloud positions match (they use same seed)
    assert len(jax_env.cloud_positions) == len(gym_env.cloud_positions)
    for jax_cloud, gym_cloud in zip(jax_env.cloud_positions, gym_env.cloud_positions):
        assert jax_cloud == gym_cloud

    # Check screen dimensions match
    assert jax_env.screen_width == gym_env.screen_width
    assert jax_env.screen_height == gym_env.screen_height


def test_render_same_state():
    """Test that both environments render identical states similarly"""
    # Create environments
    jax_env = JaxAirplane2D()
    gym_env = GymAirplane2D(render_mode="rgb_array")

    # Get a state from gym environment and copy to jax
    gym_obs, gym_info = gym_env.reset()
    gym_state = gym_env.state
    key = jax.random.PRNGKey(0)
    jax_state = type(gym_state)(
        x=float(gym_state.x),
        x_dot=float(gym_state.x_dot),
        z=float(gym_state.z),
        z_dot=float(gym_state.z_dot),
        theta=float(gym_state.theta),
        theta_dot=float(gym_state.theta_dot),
        alpha=float(gym_state.alpha),
        gamma=float(gym_state.gamma),
        m=float(gym_state.m),
        power=float(gym_state.power),
        stick=float(gym_state.stick),
        fuel=float(gym_state.fuel),
        t=int(gym_state.t),
        target_altitude=float(gym_state.target_altitude),
    )

    # First render to initialize
    frames, _, _ = jax_env.render(None, jax_state, jax_env.default_params, [], None)
    gym_frame = gym_env.render()

    # Compare frames
    assert are_images_similar(frames[0], gym_frame)

    # Check position histories
    assert jax_env.positions_history == gym_env.positions_history


def test_render_trajectory():
    """Test that both environments render trajectories similarly"""
    jax_env = JaxAirplane2D()
    gym_env = GymAirplane2D(render_mode="rgb_array")

    # Initialize with identical states
    gym_obs, gym_state = gym_env.reset()
    key = jax.random.PRNGKey(0)

    jax_state = EnvState(
        **{f: getattr(gym_state, f) for f in gym_state.__dataclass_fields__}
    )

    # Run environments with identical actions
    jax_frames = []
    gym_frames = []
    jax_screen = None
    jax_clock = None

    for _ in range(10):
        # Take same action
        action = (0.8, 0.0)  # power, stick

        # Step environments
        jax_obs, jax_state, reward, terminated, info = jax_env.step(
            key, jax_state, action, jax_env.default_params
        )
        gym_obs, reward, terminated, truncated, info = gym_env.step(action)

        gym_state = gym_env.state

        # Get frames
        jax_frame_list, jax_screen, jax_clock = jax_env.render(
            jax_screen, jax_state, jax_env.default_params, jax_frames, jax_clock
        )
        gym_frame = gym_env.render()

        # Store frames
        gym_frames.append(gym_frame)

        # Verify states remain similar
        assert np.allclose(jax_state.x, gym_state.x, rtol=1e-2)
        assert np.allclose(jax_state.z, gym_state.z, rtol=1e-2)
        assert np.allclose(jax_state.theta, gym_state.theta, rtol=1e-2)

    # Compare frames
    for jax_frame, gym_frame in zip(jax_frame_list, gym_frames):
        assert are_images_similar(jax_frame, gym_frame)


def test_save_renders():
    """Test that both environments can save renders to disk"""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    params = EnvParams(max_steps_in_episode=500)
    try:
        # JAX environment
        jax_env = JaxAirplane2D()
        seed = 0

        file = jax_env.save_video(
            lambda x: (0.8, 0.1), seed, folder=temp_dir, episode_index=0, params=params
        )
        assert os.path.exists(file)

        # Gym environment
        gym_env = GymAirplane2D(render_mode="rgb_array_list")
        gym_env.reset()
        frames = []
        for _ in range(10):
            obs, _, done, _, _ = gym_env.step((0.8, 0.1), params=params)
            frames = gym_env.render()
            if done:
                break

        save_video(
            frames, temp_dir, fps=gym_env.metadata["render_fps"], episode_index=0
        )
        assert os.path.exists(os.path.join(temp_dir, "rl-video-episode-0.mp4"))

    finally:
        # Cleanup: remove temporary directory and its contents
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__])
