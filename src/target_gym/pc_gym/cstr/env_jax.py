from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from target_gym.base import (
    FieldReading,
    actuate,
    canonical_reset,
    finish_step,
    measured_obs,
)
from target_gym.expert_access import RegistryExperts
from target_gym.pc_gym.cstr.env import (
    CSTRParams,
    CSTRState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_reward_terms,
    get_obs,
    steady_state,
)
from target_gym.pc_gym.cstr.rendering import _render
from target_gym.video import SavesVideo


class CSTR(RegistryExperts, SavesVideo, environment.Environment[CSTRState, CSTRParams]):
    """Jacketed exothermic CSTR holding its temperature with cooling water.

    Observation (4,): [T, T_j, F_w_pct, T_setpoint]
    Action      (1,): cooling water, raw in [-1, 1] -> [Fw_min, Fw_max]
    """

    registry_name: str = "cstr"

    render_car = classmethod(_render)
    screen_width = 600
    screen_height = 400

    # obs = [T, T_j, F_w_pct, T_setpoint]
    obs_value_index: int = 0  # T
    tracked_names: tuple = ("T (K)",)
    obs_target_index: int = 3  # the setpoint

    # Readings, in ``delta_t``'s minutes (PHYSICS.md, Readings): the reactor's
    # Pt100 in a thin-walled thermowell, 63 % in about 13 s [S10, Fig. 2]; the
    # jacket outlet, about 2 min late [S6].
    readings: tuple = (
        FieldReading("T", sigma=0.04, lag=13.0 / 60.0),
        FieldReading("T_j", sigma=0.04, dead_time=2.0),
    )

    def __init__(self, integration_method: str = "rk4_1"):
        self.obs_shape = (4,)
        self.positions_history = []
        self.integration_method = integration_method

    @property
    def default_params(self) -> CSTRParams:
        return CSTRParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def reward_terms(self, state, params, tripped=None):
        return compute_reward_terms(state, params, tripped=tripped)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: CSTRState,
        action: jnp.ndarray,
        params: CSTRParams = None,
    ):
        """
        Performs step transitions using JAX, returns observation, new state, reward, done, info
        """
        if params is None:
            params = self.default_params

        action, state = actuate(self, state, jnp.reshape(action, (1,)), params)

        new_state, metrics = compute_next_state(
            action.reshape(()),
            state,
            params,
            integration_method=self.integration_method,
        )

        # A trip is part of the kernel: the step is charged the trip cost and
        # the plant restarts at once, or is lost (``terminated``) with
        # ``params.continuing = False`` (``base.finish_step``).
        new_state, reward, terminated, info = finish_step(
            self, key, state, new_state, params
        )
        return self.get_obs(new_state, params), new_state, reward, terminated, info

    @measured_obs
    def get_obs(self, state: CSTRState, params: CSTRParams = None):
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

    def is_terminated(self, state: CSTRState, params: CSTRParams) -> jnp.ndarray:
        """Natural termination only; the time limit is gymnax's ``is_truncated``."""
        terminated, _ = check_is_terminal(state, params)
        return terminated

    @canonical_reset
    def reset_env(
        self, key: chex.PRNGKey, params: CSTRParams = None
    ) -> Tuple[jnp.ndarray, CSTRState]:
        """
        Reset the environment using JAX random keys
        """
        if params is None:
            params = self.default_params

        feed_key, event_key = jax.random.split(key)

        # Established operation (charter rule 17): the reactor on the steady
        # state that holds its setpoint for the feed it meets today, the
        # cooling water where that needs it.
        T_feed = jax.random.uniform(
            feed_key, minval=params.feed_T_range[0], maxval=params.feed_T_range[1]
        )
        T = jnp.asarray(params.T_setpoint)
        C_a, T_j, F_w = steady_state(T, T_feed, params)
        lo, hi = params.event_window
        state = CSTRState(
            time=0,
            C_a=C_a,
            T=T,
            target_T=T,
            F_w=jnp.clip(F_w, params.Fw_min, params.Fw_max),
            T_j=T_j,
            T_feed=T_feed,
            event_at=jax.random.uniform(event_key, minval=lo, maxval=hi),
        )

        obs = self.get_obs(state)
        return obs, state

    def initial_action(self, state, params):
        """The raw action that holds the reset's cooling water."""
        span = params.Fw_max - params.Fw_min
        return jnp.reshape(2.0 * (state.F_w - params.Fw_min) / span - 1.0, (1,))

    def action_space(self, params: CSTRParams | None = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Box(
            low=jnp.array([-1.0]),
            high=jnp.array([1.0]),
            shape=(1,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: CSTRParams) -> spaces.Box:
        """Observation space of the environment."""
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: CSTRParams) -> spaces.Box:
        """Observation space of the environment."""
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(
            -inf, inf, len(CSTRState.__dataclass_fields__), dtype=jnp.float32
        )

    def render(self, screen, state: CSTRState, params: CSTRParams, frames, clock):
        """
        JAX-compatible rendering wrapper
        """
        frames, screen, clock = self.render_car(screen, state, params, frames, clock)
        return frames, screen, clock
