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
from target_gym.pc_gym.ph_neutralization.env import (
    PHParams,
    PHState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_reward_terms,
    get_obs,
    solve_pH,
    steady_base_flow,
    steady_state_invariants,
)
from target_gym.pc_gym.ph_neutralization.rendering import _render
from target_gym.video import SavesVideo


class PHNeutralization(
    RegistryExperts, SavesVideo, environment.Environment[PHState, PHParams]
):
    """pH neutralisation CSTR (reaction-invariant formulation).

    Observation (4,): [pH, q3_pct, setpoint_pH, excursion_min]
    Action      (1,): base flow, raw in [-1, 1] -> [q3_min, q3_max]
    """

    registry_name: str = "ph_neutralization"

    render_ph = classmethod(_render)
    screen_width = 700
    screen_height = 800

    obs_value_index: int = 0  # pH
    tracked_names: tuple = ("pH",)
    obs_target_index: int = 2  # target_pH

    # Glass electrode, provisional, 10 s lag and 10 s dead time, in seconds as
    # ``delta_t`` is (PHYSICS.md section on readings).
    readings: tuple = (
        FieldReading("pH", sigma=0.01, step=0.01, lag=10.0, dead_time=10.0),
    )

    def __init__(self, integration_method: str = "rk4_2"):
        self.obs_shape = (4,)
        self.integration_method = integration_method

    @property
    def default_params(self) -> PHParams:
        return PHParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def reward_terms(self, state, params, tripped=None):
        return compute_reward_terms(state, params, tripped=tripped)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: PHState,
        action: jnp.ndarray,
        params: PHParams = None,
    ):
        if params is None:
            params = self.default_params
        action, state = actuate(self, state, jnp.reshape(action, (1,)), params)

        new_state, _ = compute_next_state(
            action.reshape(()),
            state,
            params,
            key,
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
    def get_obs(self, state: PHState, params: PHParams = None):
        if params is None:
            params = self.default_params
        return get_obs(state, params=params)

    def is_terminated(self, state: PHState, params: PHParams) -> jnp.ndarray:
        """Natural termination only; the time limit is gymnax's ``is_truncated``."""
        terminated, _ = check_is_terminal(state, params)
        return terminated

    @canonical_reset
    def reset_env(
        self, key: chex.PRNGKey, params: PHParams = None
    ) -> Tuple[jnp.ndarray, PHState]:
        if params is None:
            params = self.default_params

        key, times_key, levels_key, buffer_key = jax.random.split(key, 4)

        # Established operation (charter rule 17): the tank on the steady state
        # that holds the set point under the buffer flow it meets, as a loop in
        # automatic leaves it. The acid upsets come later, unannounced.
        q2 = jnp.clip(
            params.q2_nominal + params.q2_noise_std * jax.random.normal(buffer_key),
            params.q2_min,
            params.q2_max,
        )
        target_pH = jnp.asarray(params.setpoint_pH, jnp.float32)
        q3 = jnp.clip(
            steady_base_flow(target_pH, q2, params), params.q3_min, params.q3_max
        )
        upset_times = jnp.sort(
            jax.random.randint(times_key, (2,), 0, params.max_steps_in_episode)
        )
        upset_levels = jnp.concatenate(
            [
                jnp.ones((1,)),
                1.0
                + params.acid_upset_size
                * jax.random.uniform(levels_key, (2,), minval=-1.0, maxval=1.0),
            ]
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
            upset_times=upset_times,
            upset_levels=upset_levels,
            excursion=jnp.zeros(()),
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

    def render(self, screen, state: PHState, params: PHParams, frames, clock):
        frames, screen, clock = self.render_ph(screen, state, params, frames, clock)
        return frames, screen, clock
