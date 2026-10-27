from functools import partial

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from target_gym import make_gymnasium

Airplane2D = partial(make_gymnasium, "plane")


def test_can_create_env():
    """Test that environment can be created and reset"""
    env = Airplane2D()
    assert isinstance(env, gym.Env)
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (10,)  # Verify observation space shape


def test_can_step_env():
    """Test that environment can be stepped with random actions"""
    env = Airplane2D()
    obs, info = env.reset()

    # Test a few steps
    for _ in range(5):
        action = (np.random.uniform(0, 1), np.random.uniform(-15, 15))
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (jnp.ndarray, np.ndarray, float, np.float64))
        assert isinstance(terminated, (jnp.ndarray, bool, np.bool_, jnp.bool_))
        assert isinstance(truncated, (jnp.ndarray, bool, np.bool_, jnp.bool_))
        if terminated or truncated:
            obs, info = env.reset()


def test_passes_gymnasium_env_checker():
    """Gymnasium's own API check, the contract SB3 and other non-JAX
    libraries rely on. Rendering is checked by tests/test_wrapper.py."""
    check_env(Airplane2D(), skip_render_check=True)


def test_sb3_ppo_can_learn():
    """Test that environment works with SB3's PPO. SB3 and torch are the
    opt-in ``agent`` group (``uv sync --group agent``), so this skips without
    them."""
    PPO = pytest.importorskip("stable_baselines3").PPO
    # Create environment
    env = Airplane2D()

    # Create and train PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        n_steps=1024,
        batch_size=64,
        verbose=0,
    )

    # Train for a few steps to ensure everything works
    model.learn(total_timesteps=2048)

    # Test model can predict actions
    obs, info = env.reset()
    action, _states = model.predict(obs, deterministic=True)
    assert isinstance(action, (tuple, list, np.ndarray))
    assert len(action) == 2
    assert isinstance(action[0].item(), (float, int, np.int_, np.float64))  # power
    assert isinstance(action[1].item(), (float, int, np.int_, np.float64))  # stick

    # Test action bounds
    for _ in range(10):
        action, _states = model.predict(obs, deterministic=False)
        assert -1 <= action[0] <= 1  # power should be between 0 and 1
        assert -1 <= action[1] <= 1  # stick should be between -15 and 15 degrees


if __name__ == "__main__":
    pytest.main([__file__])
