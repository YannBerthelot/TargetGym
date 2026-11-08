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
from target_gym.pc_gym.unstable_cstr.env import (
    UnstableCSTRParams,
    UnstableCSTRState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_reward_terms,
    get_obs,
    steady_state,
)
from target_gym.pc_gym.unstable_cstr.rendering import _render
from target_gym.video import SavesVideo


class UnstableCSTR(
    RegistryExperts,
    SavesVideo,
    environment.Environment[UnstableCSTRState, UnstableCSTRParams],
):
    """LDPE autoclave held at its unstable steady state by initiator flow.

    Observation (3,): [T, I_f_pct, T_setpoint]
    Action      (1,): initiator in the feed, raw in [-1, 1] -> [I_f_min, I_f_max]
    """

    registry_name: str = "unstable_cstr"

    render_car = classmethod(_render)
    screen_width = 600
    screen_height = 400

    obs_value_index: int = 0  # T
    tracked_names: tuple = ("T (K)",)
    obs_target_index: int = 2  # the setpoint

    # The zone thermocouple (PHYSICS.md, Readings), in ``delta_t``'s seconds:
    # 1 K scatter and a 2 s lag (ours, family).
    readings: tuple = (FieldReading("T", sigma=0.5, lag=2.0),)

    def __init__(self, integration_method: str = "rk4_4"):
        self.obs_shape = (3,)
        self.integration_method = integration_method

    @property
    def default_params(self) -> UnstableCSTRParams:
        return UnstableCSTRParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def reward_terms(self, state, params, tripped=None):
        return compute_reward_terms(state, params, tripped=tripped)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: UnstableCSTRState,
        action: jnp.ndarray,
        params: UnstableCSTRParams = None,
    ):
        if params is None:
            params = self.default_params
        action, state = actuate(self, state, jnp.reshape(action, (1,)), params)
        new_state, _ = compute_next_state(
            action.reshape(()),
            state,
            params,
            integration_method=self.integration_method,
        )
        new_state, reward, terminated, info = finish_step(
            self, key, state, new_state, params
        )
        return self.get_obs(new_state, params), new_state, reward, terminated, info

    @measured_obs
    def get_obs(self, state: UnstableCSTRState, params: UnstableCSTRParams = None):
        if params is None:
            params = self.default_params
        return get_obs(state, params=params)

    def is_terminated(self, state, params) -> jnp.ndarray:
        """Natural termination only; the time limit is gymnax's ``is_truncated``."""
        terminated, _ = check_is_terminal(state, params)
        return terminated

    def initial_action(self, state, params):
        """The raw action that holds the reset's initiator feed."""
        span = params.I_f_max - params.I_f_min
        return jnp.reshape(2.0 * (state.I_f - params.I_f_min) / span - 1.0, (1,))

    @canonical_reset
    def reset_env(
        self, key: chex.PRNGKey, params: UnstableCSTRParams = None
    ) -> Tuple[jnp.ndarray, UnstableCSTRState]:
        """Established operation (charter rule 17): the reactor on its steady
        state at the setpoint, the initiator where that needs it, as the PID
        holds it [L1]. The scenario's event comes later."""
        if params is None:
            params = self.default_params
        T = jnp.asarray(params.T_setpoint)
        I, M, I_f = steady_state(T, params)
        lo, hi = params.event_window
        state = UnstableCSTRState(
            time=0,
            I=I,
            M=M,
            T=T,
            target_T=T,
            I_f=I_f,
            event_at=jax.random.uniform(key, minval=lo, maxval=hi),
        )
        return self.get_obs(state, params), state

    def action_space(self, params: UnstableCSTRParams | None = None) -> spaces.Box:
        return spaces.Box(
            low=jnp.array([-1.0]), high=jnp.array([1.0]), shape=(1,), dtype=jnp.float32
        )

    def observation_space(self, params: UnstableCSTRParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: UnstableCSTRParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(
            -inf, inf, len(UnstableCSTRState.__dataclass_fields__), dtype=jnp.float32
        )

    def render(self, screen, state, params, frames, clock):
        frames, screen, clock = self.render_car(screen, state, params, frames, clock)
        return frames, screen, clock
