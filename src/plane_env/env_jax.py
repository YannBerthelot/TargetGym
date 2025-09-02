from typing import Callable

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from plane_env.env import (
    check_is_terminal,
    compute_next_state,
    compute_norm_from_coordinates,
    compute_reward,
    get_env_classes,
    save_video,
)
from plane_env.rendering import _render

EnvState, EnvParams, EnvMetrics = get_env_classes(use_jax=True)


class Airplane2D(environment.Environment[EnvState, EnvParams]):
    """
    JAX-compatible 2D airplane environment.
    """

    render_plane = classmethod(_render)
    screen_width = 600
    screen_height = 400
    max_steps = 1_000

    def __init__(self):
        self.obs_shape = (6,)
        self.positions_history = []

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=self.max_steps)

    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: jnp.ndarray,
        params: EnvParams = None,
    ):
        """
        Performs step transitions using JAX, returns observation, new state, reward, done, info
        """
        if params is None:
            params = self.default_params
        power, stick = action
        stick = jnp.deg2rad(stick * 15)  # radians

        new_state, metrics = compute_next_state(power, stick, state, params, xp=jnp)
        reward = compute_reward(new_state, params, xp=jnp)
        terminated, truncated = check_is_terminal(new_state, params, xp=jnp)

        obs = self.get_obs(new_state)
        return obs, new_state, reward, terminated, truncated, {"metrics": metrics}

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        return check_is_terminal(state, params, xp=jnp)

    def reset(self, key: chex.PRNGKey, params: EnvParams = None):
        """
        Reset the environment using JAX random keys
        """
        if params is None:
            params = EnvParams(max_steps_in_episode=Airplane2D.max_steps)
        key, altitude_key, target_key = jax.random.split(key, 3)

        initial_x = 0.0
        initial_z = jax.random.uniform(
            altitude_key,
            minval=params.initial_altitude_range[0],
            maxval=params.initial_altitude_range[1],
        )
        initial_z_dot = params.initial_z_dot
        initial_x_dot = params.initial_x_dot
        initial_theta = jnp.deg2rad(params.initial_theta)
        initial_gamma = jnp.arcsin(
            initial_z_dot
            / (
                compute_norm_from_coordinates(
                    jnp.array([initial_x_dot, initial_z_dot + 1e-6])
                )
            )
        )
        initial_alpha = initial_theta - initial_gamma
        initial_m = params.initial_mass + params.initial_fuel_quantity
        initial_power = params.initial_power
        initial_stick = jnp.deg2rad(params.initial_stick)
        initial_fuel = params.initial_fuel_quantity

        target_altitude = jax.random.uniform(
            target_key,
            minval=params.target_altitude_range[0],
            maxval=params.target_altitude_range[1],
        )

        state = EnvState(
            x=initial_x,
            x_dot=initial_x_dot,
            z=initial_z,
            z_dot=initial_z_dot,
            theta=initial_theta,
            theta_dot=jnp.deg2rad(params.initial_theta_dot),
            alpha=initial_alpha,
            gamma=initial_gamma,
            m=initial_m,
            power=initial_power,
            stick=initial_stick,
            fuel=initial_fuel,
            t=0,
            target_altitude=target_altitude,
        )

        obs = self.get_obs(state)
        return obs, state

    def get_obs(self, state: EnvState):
        """
        Observation vector
        """
        return jnp.stack(
            [
                state.z,
                state.x_dot,
                state.z_dot,
                state.theta,
                state.gamma,
                state.target_altitude,
            ]
        )

    def render(self, screen, state: EnvState, params: EnvParams, frames, clock):
        """
        JAX-compatible rendering wrapper
        """
        frames, screen, clock = self.render_plane(screen, state, params, frames, clock)
        return frames, screen, clock

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

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Box(
            jnp.array([0.0, -1.0]), jnp.array([1.0, 1.0]), (2,), dtype=jnp.float32
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, (10,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        high = jnp.array(
            [
                params.x_threshold * 2,
                jnp.finfo(jnp.float32).max,
                params.theta_threshold_radians * 2,
                jnp.finfo(jnp.float32).max,
            ]
        )
        return spaces.Dict(
            {
                "x": spaces.Box(-high[0], high[0], (), jnp.float32),
                "x_dot": spaces.Box(-high[1], high[1], (), jnp.float32),
                "theta": spaces.Box(-high[2], high[2], (), jnp.float32),
                "theta_dot": spaces.Box(-high[3], high[3], (), jnp.float32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )


if __name__ == "__main__":
    env = Airplane2D()
    seed = 42
    env_params = EnvParams(max_steps_in_episode=1_000)
    action = (0.8, 0.0)
    env.save_video(
        lambda o: action,
        seed,
        folder="videos",
        episode_index=0,
        params=env_params,
        format="gif",
    )
