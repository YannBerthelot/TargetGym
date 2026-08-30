import os
from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment, spaces

from target_gym.boiler_drum.env import (
    BoilerDrumParams,
    BoilerDrumState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    drum_level,
    get_obs,
    steady_state,
)
from target_gym.boiler_drum.rendering import _render
from target_gym.utils import save_video


class BoilerDrum(environment.Environment[BoilerDrumState, BoilerDrumParams]):
    """Natural-circulation drum boiler with shrink-and-swell.

    Observation (7): [level, pressure, q_steam, fuel_pct, feed_pct,
                      target_level, target_pressure]
    Action      (2): [fuel, feedwater], raw in [-1, 1]
    """

    render_boiler = classmethod(_render)
    screen_width = 700
    screen_height = 900

    obs_value_index: tuple = (0, 1)  # level, pressure
    obs_target_index: tuple = (5, 6)  # target_level, target_pressure

    def __init__(self, integration_method: str = "rk4_1"):
        self.obs_shape = (7,)
        self.integration_method = integration_method

    @property
    def default_params(self) -> BoilerDrumParams:
        return BoilerDrumParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

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
            pressure=jnp.asarray(params.p_nominal, dtype=jnp.float32),
            V_wt=V_wt,
            m_sr=m_sr,
            m_sd=m_sd,
            level=jnp.asarray(level0, dtype=jnp.float32),
            q_steam=q_steam,
            Q_fuel=jnp.asarray(Q, dtype=jnp.float32),
            q_feed=jnp.asarray(params.q_steam_nominal, dtype=jnp.float32),
            target_pressure=target_pressure,
            level_ref=jnp.asarray(level_ref, dtype=jnp.float32),
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

    def make_pid(self):
        """Return a ready-to-use three-element level + pressure controller."""
        from target_gym.experts.pid import make_boiler_drum_stateful_pid

        return make_boiler_drum_stateful_pid()

    def make_mpc(self, params=None, **kwargs):
        from target_gym.experts.mpc import make_boiler_drum_mpc

        if params is None:
            params = self.default_params
        return make_boiler_drum_mpc(self, params, **kwargs)

    @property
    def expert_policy(self):
        from target_gym.experts.pid import (
            FunctionalExpertPolicy,
            make_boiler_drum_pid,
            mimo_pid_step,
        )

        pid_params, zero_state = make_boiler_drum_pid()
        return FunctionalExpertPolicy(pid_params, zero_state, mimo_pid_step)

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
        self, screen, state: BoilerDrumState, params: BoilerDrumParams, frames, clock
    ):
        frames, screen, clock = self.render_boiler(screen, state, params, frames, clock)
        return frames, screen, clock


if __name__ == "__main__":
    env = BoilerDrum()
    os.makedirs("videos/boiler_drum", exist_ok=True)
    env.save_video(
        lambda o: np.random.uniform(-1, 1, size=(2,)),
        42,
        folder="videos/boiler_drum",
        episode_index=0,
        params=BoilerDrumParams(max_steps_in_episode=300),
        format="gif",
    )
