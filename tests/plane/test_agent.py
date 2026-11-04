from functools import partial

import numpy as np
import pytest

from target_gym import make_gymnasium

Airplane2D = partial(make_gymnasium, "plane")


@pytest.mark.agent
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
