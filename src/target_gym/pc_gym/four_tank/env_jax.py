from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from target_gym.base import canonical_reset, finish_step
from target_gym.expert_access import RegistryExperts
from target_gym.pc_gym.four_tank.env import (
    FourTankParams,
    FourTankState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_reward_terms,
    get_obs,
)
from target_gym.pc_gym.four_tank.rendering import _render
from target_gym.video import SavesVideo


class FourTank(
    RegistryExperts, SavesVideo, environment.Environment[FourTankState, FourTankParams]
):
    """Quadruple-tank rig; the obvious loop pairing is non-minimum phase.

    Action (2,): [pump 1 voltage, pump 2 voltage], raw in [-1, 1] -> [v_min, v_max]
    """

    registry_name: str = "four_tank"

    render_car = classmethod(_render)
    screen_width = 600
    screen_height = 400

    # obs = [h1, h2, h3, h4, target_h1, target_h2]
    obs_value_index: tuple = (0, 1)  # h1, h2
    tracked_names: tuple = ("h1 (m)", "h2 (m)")
    obs_target_index: tuple = (4, 5)  # target_h1, target_h2

    def __init__(self, integration_method: str = "rk4_1"):
        self.obs_shape = (6,)
        self.integration_method = integration_method

    @property
    def default_params(self) -> FourTankParams:
        return FourTankParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def reward_terms(self, state, params):
        return compute_reward_terms(state, params)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: FourTankState,
        action: jnp.ndarray,
        params: FourTankParams = None,
    ):
        if params is None:
            params = self.default_params

        action = action.reshape((2,))

        new_state, metrics = compute_next_state(
            action, state, params, integration_method=self.integration_method
        )

        # A trip is part of the kernel: the step is charged the trip cost and
        # the plant restarts at once, or is lost (``terminated``) with
        # ``params.continuing = False`` (``base.finish_step``).
        new_state, reward, terminated, info = finish_step(
            self, key, state, new_state, params
        )
        return self.get_obs(new_state), new_state, reward, terminated, info

    def get_obs(self, state: FourTankState, params: FourTankParams = None):
        if params is None:
            params = self.default_params
        return get_obs(state, params=params)

    def is_terminated(
        self, state: FourTankState, params: FourTankParams
    ) -> jnp.ndarray:
        """Natural termination only; the time limit is gymnax's ``is_truncated``."""
        terminated, _ = check_is_terminal(state, params)
        return terminated

    @canonical_reset
    def reset_env(
        self, key: chex.PRNGKey, params: FourTankParams = None
    ) -> Tuple[jnp.ndarray, FourTankState]:
        if params is None:
            params = self.default_params

        key, h1_key, h2_key, h3_key, h4_key, t1_key, t2_key = jax.random.split(key, 7)

        h1 = jax.random.uniform(
            h1_key, minval=params.initial_h1_range[0], maxval=params.initial_h1_range[1]
        )
        h2 = jax.random.uniform(
            h2_key, minval=params.initial_h2_range[0], maxval=params.initial_h2_range[1]
        )
        h3 = jax.random.uniform(
            h3_key, minval=params.initial_h3_range[0], maxval=params.initial_h3_range[1]
        )
        h4 = jax.random.uniform(
            h4_key, minval=params.initial_h4_range[0], maxval=params.initial_h4_range[1]
        )
        target_h1 = jax.random.uniform(
            t1_key, minval=params.target_h1_range[0], maxval=params.target_h1_range[1]
        )
        target_h2 = jax.random.uniform(
            t2_key, minval=params.target_h2_range[0], maxval=params.target_h2_range[1]
        )

        state = FourTankState(
            time=0,
            h1=h1,
            h2=h2,
            h3=h3,
            h4=h4,
            target_h1=target_h1,
            target_h2=target_h2,
            v1=0.0,
            v2=0.0,
        )
        obs = self.get_obs(state)
        return obs, state

    def action_space(self, params: FourTankParams | None = None) -> spaces.Box:
        return spaces.Box(
            low=jnp.array([-1.0, -1.0]),
            high=jnp.array([1.0, 1.0]),
            shape=(2,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: FourTankParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: FourTankParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(
            -inf, inf, len(FourTankState.__dataclass_fields__), dtype=jnp.float32
        )

    def render(
        self, screen, state: FourTankState, params: FourTankParams, frames, clock
    ):
        frames, screen, clock = self.render_car(screen, state, params, frames, clock)
        return frames, screen, clock
