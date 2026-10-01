from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from target_gym.base import canonical_reset, failure_kernel
from target_gym.pc_gym.unstable_cstr.env import (
    UnstableCSTRParams,
    UnstableCSTRState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_reward_terms,
    draw_schedule,
    get_obs,
    steady_coolant,
    steady_temperature,
)
from target_gym.pc_gym.unstable_cstr.rendering import _render
from target_gym.utils import save_video


class UnstableCSTR(environment.Environment[UnstableCSTRState, UnstableCSTRParams]):
    """Exothermic CSTR held on its unstable middle steady state.

    Action (1,): coolant command, raw in [-1, 1] -> [T_c_min, T_c_max] K.
    """

    render_car = classmethod(_render)
    screen_width = 600
    screen_height = 400

    # obs = [C_a, T, T_j, live target]
    obs_value_index: int = 0  # C_a (concentration)
    tracked_names: tuple = ("C_a (mol/L)",)
    obs_target_index: int = 3  # live target

    def __init__(self, integration_method: str = "rk4_1"):
        self.obs_shape = (4,)
        self.positions_history = []
        self.integration_method = integration_method

    @property
    def default_params(self) -> UnstableCSTRParams:
        return UnstableCSTRParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def reward_terms(self, state, params):
        return compute_reward_terms(state, params)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: UnstableCSTRState,
        action: jnp.ndarray,
        params: UnstableCSTRParams = None,
    ):
        """
        Performs step transitions using JAX, returns observation, new state, reward, done, info
        """
        if params is None:
            params = self.default_params

        T_c = action
        if not isinstance(action, float):
            T_c = action.reshape(())

        new_state, metrics = compute_next_state(
            T_c, state, params, key, integration_method=self.integration_method
        )

        # A trip is part of the kernel: the step is charged the trip cost and
        # the plant restarts at once from a fresh draw (new schedule, new
        # drift, block clock 0) on the same window clock; ``terminated`` is
        # never raised (``base.failure_kernel``).
        new_state, tripped, scored = failure_kernel(self, key, state, new_state, params)
        # Scored on the proposal, against the live target of the advanced
        # clock: the step that enters a block is scored against its level,
        # which the returned observation also shows.
        reward = compute_reward(scored, params, xp=jnp)
        return (
            self.get_obs(new_state, params),
            new_state,
            reward,
            jnp.zeros((), dtype=bool),
            {"last_state": new_state, "tripped": tripped},
        )

    def get_obs(self, state: UnstableCSTRState, params: UnstableCSTRParams = None):
        """
        Observation vector
        """
        if params is None:
            params = (
                self.default_params
            )  # NOTE: gymnax's get_obs takes no params; this environment's
            # observation needs them, so the signature accepts them and falls
            # back to the defaults when a caller follows the gymnax shape.
        return get_obs(state, params=params)

    def is_terminated(
        self, state: UnstableCSTRState, params: UnstableCSTRParams
    ) -> jnp.ndarray:
        """Natural termination only; the time limit is gymnax's ``is_truncated``."""
        terminated, _ = check_is_terminal(state, params)
        return terminated

    @canonical_reset
    def reset_env(
        self, key: chex.PRNGKey, params: UnstableCSTRParams = None
    ) -> Tuple[jnp.ndarray, UnstableCSTRState]:
        """A warm start near the first level, with the jacket already at the
        steady coolant for the drawn drift."""
        if params is None:
            params = self.default_params

        schedule_key, C_a_key, T_key, Ti_key = jax.random.split(key, 4)
        target_levels = draw_schedule(schedule_key, params)
        first = target_levels[0]

        # The initial drift is the stationary law clipped at
        # +-Ti_dev_reset_clip standard deviations, so no reset starts past the
        # point of no return.
        Ti_dev = params.Ti_sigma * jnp.clip(
            jax.random.normal(Ti_key),
            -params.Ti_dev_reset_clip,
            params.Ti_dev_reset_clip,
        )
        C_a = first + jax.random.uniform(
            C_a_key,
            minval=-params.initial_CA_offset,
            maxval=params.initial_CA_offset,
        )
        T = steady_temperature(first, params) + jax.random.uniform(
            T_key,
            minval=-params.initial_T_offset,
            maxval=params.initial_T_offset,
        )
        # The clip never binds for |Ti_dev| <= 6 K (derived); it keeps the
        # jacket inside the actuator's range if the band or the drift is ever
        # widened.
        T_j = jnp.clip(
            steady_coolant(first, Ti_dev, params), params.T_c_min, params.T_c_max
        )

        state = UnstableCSTRState(
            time=0,
            C_a=C_a,
            T=T,
            T_j=T_j,
            Ti_dev=Ti_dev,
            target_levels=target_levels,
            block_clock=0,
        )

        obs = self.get_obs(state, params)
        return obs, state

    def action_space(self, params: UnstableCSTRParams | None = None) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(
            low=jnp.array([-1.0]),
            high=jnp.array([1.0]),
            shape=(1,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: UnstableCSTRParams) -> spaces.Box:
        """Observation space of the environment."""
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: UnstableCSTRParams) -> spaces.Box:
        """State space of the environment."""
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(
            -inf, inf, len(UnstableCSTRState.__dataclass_fields__), dtype=jnp.float32
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

    def render(
        self,
        screen,
        state: UnstableCSTRState,
        params: UnstableCSTRParams,
        frames,
        clock,
    ):
        """
        JAX-compatible rendering wrapper
        """
        frames, screen, clock = self.render_car(screen, state, params, frames, clock)
        return frames, screen, clock

    def make_pid(self):
        """Return the cascade PID (outer PI on C_a, inner PD on T)."""
        from target_gym.pc_gym.unstable_cstr.experts import make_unstable_cstr_pid

        return make_unstable_cstr_pid()

    def make_mpc(self, params=None, **kwargs):
        """Return the CasADi NMPC oracle, which falls back to the cascade PID."""
        from target_gym.pc_gym.unstable_cstr.experts import make_unstable_cstr_mpc

        if params is None:
            params = self.default_params
        return make_unstable_cstr_mpc(self, params, **kwargs)
