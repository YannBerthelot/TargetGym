import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import pytest
from stable_baselines3 import PPO
from target_gym.plane.env_gymnasium import Airplane2D


def test_can_create_env():
    """Test that environment can be created and reset"""
    env = Airplane2D()
    assert isinstance(env, gym.Env)
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (9,)  # Verify observation space shape


def test_can_step_env():
    """Test that environment can be stepped with random actions"""
    env = Airplane2D()
    obs, info = env.reset()

    # Test a few steps
    for _ in range(5):
        action = (np.random.uniform(0, 1), np.random.uniform(-15, 15))
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (jnp.ndarray, np.ndarray, float, np.float_))
        assert isinstance(terminated, (jnp.ndarray, bool, np.bool_, jnp.bool_))
        assert isinstance(truncated, (jnp.ndarray, bool, np.bool_, jnp.bool_))
        if terminated or truncated:
            obs, info = env.reset()


# @pytest.mark.skip
def test_sb3_ppo_can_learn():
    """Test that environment works with SB3's PPO"""
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
    assert isinstance(action[0].item(), (float, int, np.int_, np.float_))  # power
    assert isinstance(action[1].item(), (float, int, np.int_, np.float_))  # stick

    # Test action bounds
    for _ in range(10):
        action, _states = model.predict(obs, deterministic=False)
        assert 0 <= action[0] / 10 <= 1  # power should be between 0 and 1
        assert -1 <= action[1] <= 1  # stick should be between -15 and 15 degrees


if __name__ == "__main__":
    pytest.main([__file__])
