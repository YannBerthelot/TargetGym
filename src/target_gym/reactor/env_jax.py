import os
from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment, spaces

from target_gym.reactor.env import (
    N_GROUPS,
    N_SETPOINTS,
    ReactorParams,
    ReactorState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    get_obs,
    get_target_from_schedule,
    steady_state_precursors,
    steady_state_xenon,
)
from target_gym.reactor.rendering import _render
from target_gym.utils import save_video


class Reactor(environment.Environment[ReactorState, ReactorParams]):
    """
    Nuclear reactor (point kinetics + thermal feedback).

    Observation (4,): [n, T_coolant, rho_ext_norm, target_n]
    Action      (1,): [rho_ext_norm] in [-1, 1] (control rod position)
    """

    render_reactor = classmethod(_render)
    screen_width = 700
    screen_height = 900

    # obs = [n, T_coolant, rho_ext_norm, target_n]
    obs_value_index: int = 0  # n (neutron density / normalised power)
    obs_target_index: int = 3  # target_n

    def __init__(self, integration_method: str = "rk4_50"):
        self.obs_shape = (4,)
        self.integration_method = integration_method

    @property
    def default_params(self) -> ReactorParams:
        return ReactorParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: ReactorState,
        action: jnp.ndarray,
        params: ReactorParams = None,
    ):
        if params is None:
            params = self.default_params

        rho_raw = action
        if not isinstance(action, float):
            rho_raw = action.reshape(())

        new_state, metrics = compute_next_state(
            rho_raw, state, params, integration_method=self.integration_method
        )

        reward = compute_reward(new_state, params, xp=jnp)
        terminated, truncated = check_is_terminal(new_state, params, xp=jnp)
        done = terminated | truncated

        obs = self.get_obs(new_state)
        return (
            obs,
            new_state,
            reward,
            done,
            {"last_state": new_state},
        )

    def get_obs(self, state: ReactorState, params: ReactorParams = None):
        if params is None:
            params = self.default_params
        return get_obs(state, params=params)

    def is_terminal(self, state: ReactorState, params: ReactorParams) -> jnp.ndarray:
        return check_is_terminal(state, params)

    def reset_env(
        self, key: chex.PRNGKey, params: ReactorParams = None
    ) -> Tuple[jnp.ndarray, ReactorState]:
        if params is None:
            params = self.default_params

        key, n_key, target_key, demand_key = jax.random.split(key, 4)

        initial_n = jax.random.uniform(
            n_key,
            minval=params.initial_n_range[0],
            maxval=params.initial_n_range[1],
        )
        # Initial demand drawn from target range; OU process evolves it from here.
        initial_target = jax.random.uniform(
            target_key,
            minval=params.target_n_range[0],
            maxval=params.target_n_range[1],
        )
        # Legacy schedule field (kept for backward compatibility, unused by OU).
        demand_mu = 0.5 * (params.target_n_range[0] + params.target_n_range[1])
        target_schedule = jnp.full((N_SETPOINTS,), demand_mu)

        # Precursors and xenon/iodine start at steady-state for the initial
        # neutron density. Without this, there would be a huge transient in
        # the first few seconds.
        initial_C = steady_state_precursors(initial_n, params)
        initial_I_hat, initial_Xe_hat = steady_state_xenon(initial_n, params)

        state = ReactorState(
            time=0,
            n=initial_n,
            C=initial_C,
            T_fuel=jnp.asarray(params.initial_T_fuel, dtype=jnp.float32),
            T_coolant=jnp.asarray(params.initial_T_coolant, dtype=jnp.float32),
            I_hat=jnp.asarray(initial_I_hat, dtype=jnp.float32),
            Xe_hat=jnp.asarray(initial_Xe_hat, dtype=jnp.float32),
            target_n=initial_target,
            target_schedule=target_schedule,
            demand_key=demand_key,
            rho_ext=jnp.zeros((), dtype=jnp.float32),
        )

        obs = self.get_obs(state)
        return obs, state

    def action_space(self, params: ReactorParams | None = None) -> spaces.Box:
        return spaces.Box(
            low=jnp.array([-1.0]),
            high=jnp.array([1.0]),
            shape=(1,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: ReactorParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: ReactorParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(
            -inf, inf, len(ReactorState.__dataclass_fields__), dtype=jnp.float32
        )

    def save_video(
        self,
        select_action: Callable[[jnp.ndarray], jnp.ndarray],
        seed: int,
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
            seed=seed,
            format=format,
        )

    @property
    def expert_policy(self):
        """Tuned gain-scheduled PID controller for neutron density tracking."""
        from target_gym.experts.pid import make_reactor_stateful_gs_pid

        return make_reactor_stateful_gs_pid()

    def render(self, screen, state: ReactorState, params: ReactorParams, frames, clock):
        frames, screen, clock = self.render_reactor(
            screen, state, params, frames, clock
        )
        return frames, screen, clock


if __name__ == "__main__":
    env = Reactor()
    seed = 42
    env_params = ReactorParams(max_steps_in_episode=200)
    os.makedirs("videos/reactor", exist_ok=True)
    env.save_video(
        lambda o: np.random.uniform(-1, 1),
        seed,
        folder="videos/reactor",
        episode_index=0,
        params=env_params,
        format="gif",
    )
