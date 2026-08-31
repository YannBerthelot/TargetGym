import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.plane.env import PlaneParams, PlaneState, compute_reward
from target_gym.plane.env_jax import Airplane2D as JaxEnv


def test_init():
    """Constructing the environment must give a usable spec, not just not raise."""
    env = JaxEnv()
    params = env.default_params
    assert env.action_space(params).shape == (2,)
    assert env.observation_space(params).shape == env.obs_shape


def test_reset():
    env = JaxEnv()
    key = jax.random.PRNGKey(seed=42)
    obs, env_state = env.reset(key)
    assert obs.shape == env.obs_shape


def test_compute_reward():
    env = JaxEnv()
    key = jax.random.PRNGKey(seed=42)
    obs, env_state = env.reset(key)
    env_params = PlaneParams()
    reward = compute_reward(state=env_state, params=env_params)
    assert isinstance(reward, (jnp.ndarray, np.ndarray))
    assert 1 >= reward >= 0


def test_sample_action():
    env = JaxEnv()
    key = jax.random.PRNGKey(seed=42)
    obs, env_state = env.reset(key)
    env_params = PlaneParams()
    action = env.action_space(env_params).sample(key)
    assert -1 <= action[0] <= 1
    assert -1 <= action[1] <= 1
    assert action.shape == (2,)


def test_step():
    env = JaxEnv()
    key = jax.random.PRNGKey(seed=42)
    obs, state = env.reset(key)
    action = (
        1.0,
        0,
    )  # Sample a valid action (e.g., maximum throttle and no pitch change)
    # Perform the step transition.
    n_obs, new_state, reward, terminated, truncated, _ = env.step(key, state, action)
    assert new_state.x > state.x
    assert new_state.x_dot == pytest.approx(state.x_dot, rel=0.1)
    assert new_state.z < state.z
    assert new_state.z_dot < state.z_dot
    assert new_state.power >= state.power
    assert new_state.time == state.time + 1
    assert new_state.alpha > state.alpha
    assert new_state.gamma < state.gamma


def test_is_terminal():
    env = JaxEnv()
    key = jax.random.PRNGKey(seed=42)
    obs, state = env.reset(key)
    env_params = PlaneParams()
    terminal_state = PlaneState(
        x=0,
        x_dot=0,
        z=env_params.max_alt + 0.01,
        z_dot=0,
        theta_dot=0,
        theta=0,
        alpha=0,
        gamma=0,
        m=0,
        power=0,
        stick=0,
        fuel=0,
        time=0,
        target_altitude=0,
    )
    assert env.is_terminated(terminal_state, env_params)
    terminal_state = PlaneState(
        x=0,
        x_dot=0,
        z=env_params.max_alt + 0.01,
        z_dot=0,
        theta_dot=0,
        theta=0,
        alpha=0,
        gamma=0,
        m=0,
        power=0,
        stick=0,
        fuel=0,
        time=0,
        target_altitude=0,
    )
    assert env.is_terminated(terminal_state, env_params)
    terminal_state = PlaneState(
        x=0,
        x_dot=0,
        z=env_params.max_alt + 0.01,
        z_dot=0,
        theta_dot=0,
        theta=0,
        alpha=0,
        gamma=0,
        m=0,
        power=0,
        stick=0,
        fuel=0,
        time=0,
        target_altitude=0,
    )
    assert env.is_terminated(terminal_state, env_params)


def test_render():
    pass


# def test_environments_compatible():
#     """Test that both environments produce similar results"""
#     jax_env = JaxEnv()
#     gym_env = GymEnv()

#     # Reset environments
#     key = jax.random.PRNGKey(0)
#     gym_obs, gym_state = gym_env.reset(seed=0)
#     jax_obs, jax_state = gym_obs, gym_state  # jax_env.reset(key)

#     # Test same action in both environments
#     action = (0.8, 0.0)  # power, stick

#     # JAX step
#     jax_obs, jax_next_state, jax_reward, jax_terminated, _ = jax_env.step(
#         key, jax_state, action, jax_env.default_params
#     )
#     jax_truncated = jax_next_state.t >= jax_env.default_params.max_steps_in_episode

#     # Gym step

#     gym_obs, gym_reward, gym_terminated, gym_truncated, _ = gym_env.step(action)
#     gym_next_state = gym_env.state

#     # Compare results
#     assert np.allclose(jax_next_state.x, gym_next_state.x, rtol=1e-2)
#     assert np.allclose(jax_next_state.z, gym_next_state.z, rtol=1e-2)
#     assert np.allclose(jax_reward, gym_reward, rtol=1e-2)
#     assert jax_terminated == gym_terminated
#     assert jax_truncated == gym_truncated


# ---------------------------------------------------------------------------
# Tracking reward shape (PHYSICS.md section 5)
# ---------------------------------------------------------------------------


def _reward_at(error_m: float, params=None):
    """Reward for being ``error_m`` above the target, away from any boundary."""
    from target_gym.plane.env import PlaneParams, PlaneState, compute_reward

    p = params or PlaneParams()
    target = 5000.0
    state = PlaneState(
        time=1,
        x=0.0,
        x_dot=200.0,
        z=target + error_m,
        z_dot=0.0,
        theta=0.0,
        theta_dot=0.0,
        alpha=0.0,
        gamma=0.0,
        m=p.initial_mass,
        power=0.5,
        stick=0.0,
        fuel=p.initial_fuel_quantity,
        target_altitude=target,
        gust_x=0.0,
        gust_z=0.0,
    )
    return float(compute_reward(state, p))


def test_reward_is_one_at_the_target():
    assert _reward_at(0.0) == pytest.approx(1.0)


def test_every_halving_of_the_error_is_worth_the_same():
    """The property the benchmark exists to measure.

    TargetGym asks whether a learned policy can hold a setpoint better than a
    PID. That question is only askable if the reward keeps paying for precision
    all the way down: a reward that saturates once the error is "small enough"
    scores a policy holding 1 m the same as one holding 10 m, and the comparison
    it was built to make becomes invisible.

    A logarithmic reward gives the same increment for every halving, so the gain
    from 200 m to 100 m and from 2 m to 1 m are within a few percent.
    """
    gains = [
        _reward_at(c) - _reward_at(a)
        for a, c in [(1600, 800), (400, 200), (100, 50), (25, 12.5), (6.25, 3.125)]
    ]
    assert min(gains) > 0.75 * max(
        gains
    ), "reward per halving is not scale-free across three decades: " + ", ".join(
        f"{g:.4f}" for g in gains
    )


def test_precision_stops_paying_below_the_resolution_floor():
    """Beneath the floor the controller would be chasing measurement noise.

    The floor is what keeps the reward bounded -- an unfloored log diverges at
    zero error -- and it is set to a physical resolution rather than a chosen
    tolerance.
    """
    from target_gym.plane.env import PlaneParams

    floor = PlaneParams().precision_floor
    above = _reward_at(floor * 4) - _reward_at(floor * 8)
    below = _reward_at(floor / 8) - _reward_at(floor / 4)
    assert (
        below < above / 2
    ), f"reward still paying below the floor: {below:.5f} vs {above:.5f} above it"


def test_reward_still_orders_states_far_from_target():
    """An agent that has drifted must still be able to tell it is drifting.

    A Gaussian on a band is identically zero out here, leaving nothing but the
    terminal penalty to follow home.
    """
    far, further = _reward_at(800.0), _reward_at(1600.0)
    assert far > further > 0.0, f"no ordering far out: {far:.3e} vs {further:.3e}"
