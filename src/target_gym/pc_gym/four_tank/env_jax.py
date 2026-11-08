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

    # Guided wave radar on each tank (PHYSICS.md section on readings).
    readings: tuple = tuple(
        FieldReading(level, sigma=5e-4, step=1e-3) for level in ("h1", "h2", "h3", "h4")
    )

    def __init__(self, integration_method: str = "rk4_1"):
        self.obs_shape = (6,)
        self.integration_method = integration_method

    @property
    def default_params(self) -> FourTankParams:
        return FourTankParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def reward_terms(self, state, params, tripped=None):
        return compute_reward_terms(state, params, tripped=tripped)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: FourTankState,
        action: jnp.ndarray,
        params: FourTankParams = None,
    ):
        if params is None:
            params = self.default_params

        action, state = actuate(self, state, action.reshape((2,)), params)

        new_state, metrics = compute_next_state(
            action, state, params, integration_method=self.integration_method
        )

        # A trip is part of the kernel: the step is charged the trip cost and
        # the plant restarts at once, or is lost (``terminated``) with
        # ``params.continuing = False`` (``base.finish_step``).
        new_state, reward, terminated, info = finish_step(
            self, key, state, new_state, params
        )
        return self.get_obs(new_state, params), new_state, reward, terminated, info

    @measured_obs
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

        keys = jax.random.split(key, 7)

        # Established operation (charter rule 17): both pumps at the operating
        # point and the four tanks on its steady state [S4, pp. 5-6], solved
        # from the mass balances [S1, eq. (1)]: each outflow a_i sqrt(2 g h_i)
        # equals its inflow.
        p = params
        v = p.v_nominal
        q3 = (1.0 - p.gamma2) * p.k2 * v
        q4 = (1.0 - p.gamma1) * p.k1 * v
        q1 = p.gamma1 * p.k1 * v + q3
        q2 = p.gamma2 * p.k2 * v + q4
        level = lambda q, a: (q / a) ** 2 / (2.0 * p.g)  # noqa: E731
        h1, h2, h3, h4 = (
            level(q1, p.a1),
            level(q2, p.a2),
            level(q3, p.a3),
            level(q4, p.a4),
        )
        nominal = jnp.stack([h1, h2])
        span = p.max_steps_in_episode

        state = FourTankState(
            time=0,
            h1=h1,
            h2=h2,
            h3=h3,
            h4=h4,
            target_h1=h1,
            target_h2=h2,
            v1=v,
            v2=v,
            sp_tank=jax.random.randint(keys[0], (), 0, 2),
            sp_sign=jnp.where(jax.random.bernoulli(keys[1], 0.5, (3,)), 1.0, -1.0),
            leak_start=jax.random.randint(keys[2], (), 0, span - p.leak_duration),
            leak_tank=jax.random.randint(keys[3], (), 0, 2),
            cup_time=jax.random.randint(keys[4], (), 0, span),
            cup_tank=jax.random.randint(keys[5], (), 0, 2),
            nominal_h=nominal,
        )
        # The pumps start where the operating point holds them.
        u = 2.0 * (v - p.v_min) / (p.v_max - p.v_min) - 1.0
        state = state.replace(actuator_held=jnp.full((2,), u))
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
