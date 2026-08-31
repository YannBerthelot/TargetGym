import os
from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment, spaces

from target_gym.pc_gym.ph_neutralization.env import (
    PHParams,
    PHState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    get_obs,
    solve_pH,
    steady_state_invariants,
)
from target_gym.pc_gym.ph_neutralization.rendering import _render
from target_gym.utils import save_video


class PHNeutralization(environment.Environment[PHState, PHParams]):
    """pH neutralisation CSTR (reaction-invariant formulation).

    Observation (3,): [pH, q3_pct, target_pH]
    Action      (1,): base flow, raw in [-1, 1] -> [q3_min, q3_max]
    """

    render_ph = classmethod(_render)
    screen_width = 700
    screen_height = 800

    obs_value_index: int = 0  # pH
    tracked_names: tuple = ("pH",)
    obs_target_index: int = 2  # target_pH

    def __init__(self, integration_method: str = "rk4_2"):
        self.obs_shape = (3,)
        self.integration_method = integration_method

    @property
    def default_params(self) -> PHParams:
        return PHParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: PHState,
        action: jnp.ndarray,
        params: PHParams = None,
    ):
        if params is None:
            params = self.default_params
        action_raw = action
        if not isinstance(action, float):
            action_raw = action.reshape(())

        new_state, _ = compute_next_state(
            action_raw, state, params, key, integration_method=self.integration_method
        )
        reward = compute_reward(new_state, params, xp=jnp)
        # gymnax >= 1.0 owns truncation; step_env reports natural termination only.
        terminated, _ = check_is_terminal(new_state, params, xp=jnp)
        return (
            self.get_obs(new_state),
            new_state,
            reward,
            terminated,
            {"last_state": new_state},
        )

    def get_obs(self, state: PHState, params: PHParams = None):
        if params is None:
            params = self.default_params
        return get_obs(state, params=params)

    def is_terminated(self, state: PHState, params: PHParams) -> jnp.ndarray:
        """Natural termination only; the time limit is gymnax's ``is_truncated``."""
        terminated, _ = check_is_terminal(state, params)
        return terminated

    def reset_env(
        self, key: chex.PRNGKey, params: PHParams = None
    ) -> Tuple[jnp.ndarray, PHState]:
        if params is None:
            params = self.default_params

        key, q3_key, target_key, buffer_key = jax.random.split(key, 4)

        # Start on the steady state for the sampled flows: a CSTR left running
        # is at equilibrium, and starting off it would spend the first several
        # residence times relaxing rather than being controlled.
        q3 = jax.random.uniform(
            q3_key,
            minval=params.initial_q3_range[0],
            maxval=params.initial_q3_range[1],
        )
        q2 = jnp.clip(
            params.q2_nominal + params.q2_noise_std * jax.random.normal(buffer_key),
            params.q2_min,
            params.q2_max,
        )
        target_pH = jax.random.uniform(
            target_key,
            minval=params.target_pH_range[0],
            maxval=params.target_pH_range[1],
        )
        Wa, Wb = steady_state_invariants(q3, q2, params)

        state = PHState(
            time=0,
            Wa=Wa,
            Wb=Wb,
            q2=q2,
            pH=solve_pH(Wa, Wb, params),
            q3=q3,
            target_pH=target_pH,
        )
        return self.get_obs(state), state

    def action_space(self, params: PHParams | None = None) -> spaces.Box:
        return spaces.Box(
            low=jnp.array([-1.0]), high=jnp.array([1.0]), shape=(1,), dtype=jnp.float32
        )

    def observation_space(self, params: PHParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: PHParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(
            -inf, inf, len(PHState.__dataclass_fields__), dtype=jnp.float32
        )

    def make_pid(self):
        """Return a ready-to-use StatefulPID for pH tracking."""
        from target_gym.experts.pid import make_ph_stateful_pid

        return make_ph_stateful_pid()

    def make_mpc(self, params=None, **kwargs):
        """Return a CasADi MPC oracle for pH tracking."""
        from target_gym.experts.mpc import make_ph_mpc

        if params is None:
            params = self.default_params
        return make_ph_mpc(self, params, **kwargs)

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

    def render(self, screen, state: PHState, params: PHParams, frames, clock):
        frames, screen, clock = self.render_ph(screen, state, params, frames, clock)
        return frames, screen, clock


if __name__ == "__main__":
    env = PHNeutralization()
    os.makedirs("videos/ph_neutralization", exist_ok=True)
    env.save_video(
        lambda o: np.random.uniform(-1, 1),
        42,
        folder="videos/ph_neutralization",
        episode_index=0,
        params=PHParams(max_steps_in_episode=300),
        format="gif",
    )
