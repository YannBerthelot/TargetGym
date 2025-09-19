# env_jax.py
from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np

from target_gym.utils import run_n_steps, save_video

try:
    from gymnax.environments import environment, spaces
except Exception:
    # Minimal fallback types for static checking if gymnax isn't present.
    from typing import Any

    class spaces:
        Box = object

    class environment:
        class Environment:
            pass


from target_gym.bicycle.env import (
    EnvParams,
    EnvState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    get_obs,
)
from target_gym.bicycle.rendering import _render


class RandlovBicycle(environment.Environment[EnvState, EnvParams]):
    """
    JAX-compatible Randløv bicycle environment implementing equations from the paper.
    Continuous actions: [-1,1]^2 -> [Torque, Displacement].
    Observations: [omega, omega_dot, omega_ddot, theta, theta_dot]
    """

    render_bicycle = classmethod(_render)
    screen_width = 600
    screen_height = 400

    def __init__(self, integration_method: str = "rk4_1"):
        self.integration_method = integration_method
        self.obs_shape = (5,)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: jnp.ndarray,
        params: EnvParams = None,
    ):
        """
        Perform one env step (JAX friendly).
        Returns (obs, new_state, reward, done, info)
        """
        if params is None:
            params = self.default_params

        new_state, metrics = compute_next_state(
            action, state, params, integration_method=self.integration_method
        )
        reward = compute_reward(new_state, params)
        terminated, truncated = check_is_terminal(new_state, params)
        done = terminated | truncated

        obs = self.get_obs(new_state, params=params)
        return (
            obs,
            new_state,
            reward,
            done,
            {"last_state": new_state, "metrics": metrics},
        )

    def get_obs(self, state: EnvState, params: EnvParams = None):
        """Observation vector per Randløv et al.: [omega, omega_dot, omega_ddot, theta, theta_dot]."""
        if params is None:
            params = self.default_params
        return get_obs(state, params=params)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        terminated, truncated = check_is_terminal(state, params)
        return terminated | truncated

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams = None
    ) -> Tuple[jnp.ndarray, EnvState]:
        """
        Reset the environment using JAX random keys.
        Starts the bicycle upright, centered, heading along +x.
        """
        if params is None:
            params = self.default_params

        # initialize tyre contact points near origin (front/back separated by l)
        zero = jnp.zeros(())
        # stochastic displacement: -1 to 1 scaled by max_disp, s=2cm

        max_initial_lean = jnp.deg2rad(1.0)  # 2 degrees

        init_omega = jax.random.uniform(key, minval=-1, maxval=1) * max_initial_lean
        state = EnvState(
            omega=init_omega,
            omega_dot=zero,
            theta=zero,
            theta_dot=zero,
            psi=zero,
            x_f=zero,
            y_f=zero,
            x_b=zero - params.l,
            y_b=zero,
            last_d=zero,
            t=0,
            torque=zero,
            displacement=zero,
        )
        obs = self.get_obs(state, params=params)
        return obs, state

    def action_space(self, params: EnvParams | None = None):
        """Continuous torque and displacement in [-1, 1]^2."""
        return spaces.Box(
            low=jnp.array([-1.0, -1.0]),
            high=jnp.array([1.0, 1.0]),
            shape=(2,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: EnvParams | None = None):
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: EnvParams | None = None):
        """Box space describing flattened EnvState (11 fields)."""
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, (11,), dtype=jnp.float32)

    def save_video(
        self,
        select_action: Callable[[jnp.ndarray], jnp.ndarray],
        key: chex.PRNGKey,
        params=None,
        folder="videos",
        episode_index=0,
        FPS=60,
        format="mp4",
    ):
        return save_video(
            self,
            select_action,
            folder,
            episode_index,
            FPS,
            params,
            seed=key,
            format=format,
        )

    def render(self, screen, state: EnvState, params: EnvParams, frames, clock):
        """
        JAX-compatible rendering wrapper
        """
        frames, screen, clock = self.render_bicycle(
            screen, state, params, frames, clock
        )
        return frames, screen, clock


def linear_policy(
    obs: jnp.ndarray,
    k_omega: float,
    k_omega_dot: float,
    k_theta: float,
    k_theta_dot: float,
    k_omega_dot_torque: float,
) -> jnp.ndarray:
    """
    Linear feedback policy with tunable gains.

    obs = [omega, omega_dot, omega_ddot, theta, theta_dot]
    """
    omega, omega_dot, omega_ddot, theta, theta_dot = obs

    # Steering displacement (a_d)
    a_d = -(k_omega * omega + k_omega_dot * omega_dot + k_theta * theta)

    # Torque (a_T)
    a_T = -(k_theta_dot * theta_dot + k_omega_dot_torque * omega_dot)

    return jnp.clip(jnp.array([a_T, a_d]), -1.0, 1.0)


def make_policy_from_config(config):
    return lambda obs: linear_policy(
        obs,
        k_omega=config["k_omega"],
        k_omega_dot=config["k_omega_dot"],
        k_theta=config["k_theta"],
        k_theta_dot=config["k_theta_dot"],
        k_omega_dot_torque=config["k_omega_dot_torque"],
    )


sweep_config = {
    "method": "bayes",  # Bayesian optimization
    "metric": {
        "name": "mean_return",
        "goal": "maximize",
    },
    "parameters": {
        "k_omega": {"distribution": "uniform", "min": 0.0, "max": 5.0},
        "k_omega_dot": {"distribution": "uniform", "min": 0.0, "max": 5.0},
        "k_theta": {"distribution": "uniform", "min": 0.0, "max": 5.0},
        "k_theta_dot": {"distribution": "uniform", "min": 0.0, "max": 5.0},
        "k_omega_dot_torque": {"distribution": "uniform", "min": 0.0, "max": 5.0},
    },
}

if __name__ == "__main__":
    env = RandlovBicycle()
    seed = 42
    env_params = EnvParams(max_steps_in_episode=1_000, use_goal=True)
    mean_return = run_n_steps(env, linear_policy, env_params, n_steps=10_000, seed=0)
    print(mean_return)
    action = (0.1, -1.0)
    env.save_video(
        lambda o: linear_policy(o),
        seed,
        folder="videos",
        episode_index=0,
        params=env_params,
        format="gif",
    )
