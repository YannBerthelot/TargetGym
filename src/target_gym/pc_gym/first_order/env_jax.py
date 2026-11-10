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
from target_gym.pc_gym.first_order.env import (
    FirstOrderParams,
    FirstOrderState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_reward_terms,
    get_obs,
)
from target_gym.pc_gym.first_order.rendering import _render
from target_gym.video import SavesVideo


class FirstOrderSystem(
    RegistryExperts,
    SavesVideo,
    environment.Environment[FirstOrderState, FirstOrderParams],
):
    """First-order lag: the simplest target MDP in the suite.

    Action (1,): [input], raw in [-1, 1] -> [u_min, u_max]
    """

    #: What a trip's restart carries over (``base.failure_kernel``): the outside
    #: world and the records, never the plant (docs/proposals/long-strata.md, S2).
    OUTSIDE_WORLD = ("target_x",)
    #: Ready to run as a continuing plant (``target_gym.stream``): it starts and
    #: restarts on target, and ``start_event`` changes the target. Not opted in
    #: (``STREAM_READY``) while the shared conformance suite runs 2 events an
    #: hour, 0.008 events in this 0.05 s plant's 2 N steps; the stream's scenario
    #: (tests/pc_gym/first_order/test_first_order_stream.py) is not declared on
    #: the registry yet, since tests/test_scenarios.py uses this task as its
    #: example of one with no declared suite.

    registry_name: str = "first_order"

    render_car = classmethod(_render)
    screen_width = 600
    screen_height = 400

    # obs = [x, target_x]
    obs_value_index: int = 0  # x
    tracked_names: tuple = ("x",)
    obs_target_index: int = 1  # target_x

    # An abstract lag has no instrument: a smart transmitter's figure, by
    # analogy (PHYSICS.md section on readings).
    readings: tuple = (FieldReading("x", sigma=1.1e-3),)

    def __init__(self, integration_method: str = "rk4_1"):
        self.obs_shape = (2,)
        self.integration_method = integration_method

    @property
    def default_params(self) -> FirstOrderParams:
        return FirstOrderParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def reward_terms(self, state, params, tripped=None):
        return compute_reward_terms(state, params, tripped=tripped)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: FirstOrderState,
        action: jnp.ndarray,
        params: FirstOrderParams = None,
    ):
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
    def get_obs(self, state: FirstOrderState, params: FirstOrderParams = None):
        if params is None:
            params = self.default_params
        return get_obs(state, params=params)

    def is_terminated(
        self, state: FirstOrderState, params: FirstOrderParams
    ) -> jnp.ndarray:
        """Natural termination only; the time limit is gymnax's ``is_truncated``."""
        terminated, _ = check_is_terminal(state, params)
        return terminated

    @canonical_reset
    def reset_env(
        self, key: chex.PRNGKey, params: FirstOrderParams = None
    ) -> Tuple[jnp.ndarray, FirstOrderState]:
        if params is None:
            params = self.default_params

        key, x_key, target_key = jax.random.split(key, 3)

        initial_x = jax.random.uniform(
            x_key,
            minval=params.initial_x_range[0],
            maxval=params.initial_x_range[1],
        )
        target_x = jax.random.uniform(
            target_key,
            minval=params.target_x_range[0],
            maxval=params.target_x_range[1],
        )

        initial_x = jnp.where(params.start_on_target > 0.0, target_x, initial_x)
        state = FirstOrderState(time=0, x=initial_x, target_x=target_x, u=0.0)
        obs = self.get_obs(state)
        return obs, state

    def restart_plant(self, key: chex.PRNGKey, state, params: FirstOrderParams):
        """After a trip the plant restarts as ``reset_env`` would; a stream's
        (``start_on_target``) on the target it is held to then, which carries on
        (``OUTSIDE_WORLD``)."""
        fresh = self.reset_env(key, params)[1]
        on = params.start_on_target > 0.0
        return fresh.replace(x=jnp.where(on, state.target_x, fresh.x))

    def start_event(self, state: FirstOrderState, key, params: FirstOrderParams):
        """The stream's event hook (docs/proposals/long-strata.md, S1): the
        target changes now, a fresh draw over ``target_x_range``, as a window's
        reset draws it."""
        lo, hi = params.target_x_range
        return state.replace(target_x=jax.random.uniform(key, minval=lo, maxval=hi))

    def action_space(self, params: FirstOrderParams | None = None) -> spaces.Box:
        return spaces.Box(
            low=jnp.array([-1.0]),
            high=jnp.array([1.0]),
            shape=(1,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: FirstOrderParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: FirstOrderParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(
            -inf, inf, len(FirstOrderState.__dataclass_fields__), dtype=jnp.float32
        )

    def render(
        self, screen, state: FirstOrderState, params: FirstOrderParams, frames, clock
    ):
        frames, screen, clock = self.render_car(screen, state, params, frames, clock)
        return frames, screen, clock
