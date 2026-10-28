from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from target_gym.base import canonical_reset, finish_step
from target_gym.boiler_drum.env import (
    BoilerDrumParams,
    BoilerDrumState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_reward_terms,
    get_obs,
    steady_state,
)
from target_gym.boiler_drum.rendering import _render
from target_gym.expert_access import RegistryExperts
from target_gym.video import SavesVideo


class BoilerDrum(
    RegistryExperts,
    SavesVideo,
    environment.Environment[BoilerDrumState, BoilerDrumParams],
):
    """Natural-circulation drum boiler with shrink-and-swell.

    Observation (7): [level, pressure, q_steam, fuel_pct, feed_pct,
                      target_level, target_pressure]
    Action      (2): [fuel, feedwater], raw in [-1, 1]
    """

    registry_name: str = "boiler_drum"

    render_boiler = classmethod(_render)
    screen_width = 700
    screen_height = 900

    obs_value_index: tuple = (0, 1)  # level, pressure
    tracked_names: tuple = ("drum level (m)", "drum pressure (bar)")
    obs_target_index: tuple = (5, 6)  # target_level, target_pressure

    def __init__(self, integration_method: str = "rk4_1"):
        self.obs_shape = (7,)
        self.integration_method = integration_method

    @property
    def default_params(self) -> BoilerDrumParams:
        return BoilerDrumParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def reward_terms(self, state, params):
        return compute_reward_terms(state, params)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: BoilerDrumState,
        action: jnp.ndarray,
        params: BoilerDrumParams = None,
    ):
        if params is None:
            params = self.default_params

        new_state, _ = compute_next_state(
            action, state, params, key, integration_method=self.integration_method
        )
        # A trip is part of the kernel: the step is charged the trip cost and
        # the plant restarts at once, or is lost (``terminated``) with
        # ``params.continuing = False`` (``base.finish_step``).
        new_state, reward, terminated, info = finish_step(
            self, key, state, new_state, params
        )
        return self.get_obs(new_state), new_state, reward, terminated, info

    def get_obs(self, state: BoilerDrumState, params: BoilerDrumParams = None):
        if params is None:
            params = self.default_params
        return get_obs(state, params=params)

    def is_terminated(
        self, state: BoilerDrumState, params: BoilerDrumParams
    ) -> jnp.ndarray:
        """Natural termination only; the time limit is gymnax's ``is_truncated``."""
        terminated, _ = check_is_terminal(state, params)
        return terminated

    @canonical_reset
    def reset_env(
        self, key: chex.PRNGKey, params: BoilerDrumParams = None
    ) -> Tuple[jnp.ndarray, BoilerDrumState]:
        if params is None:
            params = self.default_params

        key, level_key, target_key, load_key = jax.random.split(key, 4)

        # Start from the balanced operating point. A boiler that began with its
        # riser voidage inconsistent with its firing rate would spend the first
        # minute of every episode relaxing, which is an artefact rather than a
        # control problem.
        V_wt, m_sr, m_sd, level_ref, Q = steady_state(params)

        target_pressure = jax.random.uniform(
            target_key,
            minval=params.target_pressure_range[0],
            maxval=params.target_pressure_range[1],
        )
        level0 = jax.random.uniform(
            level_key,
            minval=params.initial_level_range[0],
            maxval=params.initial_level_range[1],
        )
        # Offset the inventory so the gauge reads ``level0`` at t = 0.
        V_wt = V_wt + level0 * params.A_d
        q_steam = params.q_steam_nominal + params.q_steam_noise_std * jax.random.normal(
            load_key
        )

        state = BoilerDrumState(
            time=0,
            pressure=jnp.asarray(params.p_nominal),
            V_wt=V_wt,
            m_sr=m_sr,
            m_sd=m_sd,
            level=jnp.asarray(level0),
            q_steam=q_steam,
            Q_fuel=jnp.asarray(Q),
            # Feedwater matched to the steam actually leaving, not to nominal.
            # Setting it to nominal while sampling steam at +/-8 kg/s opened
            # every episode with a mass imbalance of about 7% on a level that
            # does not self-regulate: uncorrected, that walks the drum roughly
            # 43 cm over the episode against a 25 cm trip. The plant now starts
            # genuinely balanced, which is what the comment above already
            # claimed, and the OU steam demand supplies the control problem
            # from there.
            q_feed=q_steam,
            target_pressure=target_pressure,
            level_ref=jnp.asarray(level_ref),
        )
        return self.get_obs(state), state

    def action_space(self, params: BoilerDrumParams | None = None) -> spaces.Box:
        return spaces.Box(
            low=jnp.array([-1.0, -1.0]),
            high=jnp.array([1.0, 1.0]),
            shape=(2,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: BoilerDrumParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: BoilerDrumParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(
            -inf, inf, len(BoilerDrumState.__dataclass_fields__), dtype=jnp.float32
        )

    def render(
        self, screen, state: BoilerDrumState, params: BoilerDrumParams, frames, clock
    ):
        frames, screen, clock = self.render_boiler(screen, state, params, frames, clock)
        return frames, screen, clock
