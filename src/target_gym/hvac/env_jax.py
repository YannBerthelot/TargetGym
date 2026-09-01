import os
from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment, spaces

from target_gym.base import canonical_reset
from target_gym.hvac.env import (
    HVACParams,
    HVACState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    get_obs,
    internal_gain,
    outdoor_temperature,
    scheduled_setpoint,
    solar_gain,
    solve_air_and_surface,
    split_gains,
)
from target_gym.hvac.rendering import _render
from target_gym.utils import save_video


class BuildingHVAC(environment.Environment[HVACState, HVACParams]):
    """Single-zone building HVAC (ISO 13790 5R1C).

    Observation (7,): [T_air, T_out, heat_pct, solar_norm, sin_h, cos_h, target_T]
    Action      (1,): commanded heating power, raw in [-1, 1] -> [0, Q_heat_max]
    """

    render_hvac = classmethod(_render)
    screen_width = 700
    screen_height = 900

    obs_value_index: int = 0  # T_air
    tracked_names: tuple = ("zone air temperature (deg C)",)
    obs_target_index: int = 6  # target_T

    def __init__(self, integration_method: str = "rk4_2"):
        self.obs_shape = (7,)
        self.integration_method = integration_method

    @property
    def default_params(self) -> HVACParams:
        return HVACParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: HVACState,
        action: jnp.ndarray,
        params: HVACParams = None,
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

    def get_obs(self, state: HVACState, params: HVACParams = None):
        if params is None:
            params = self.default_params
        return get_obs(state, params=params)

    def is_terminated(self, state: HVACState, params: HVACParams) -> jnp.ndarray:
        """Natural termination only; the time limit is gymnax's ``is_truncated``."""
        terminated, _ = check_is_terminal(state, params)
        return terminated

    @canonical_reset
    def reset_env(
        self, key: chex.PRNGKey, params: HVACParams = None
    ) -> Tuple[jnp.ndarray, HVACState]:
        if params is None:
            params = self.default_params

        key, temp_key, sp_key, weather_key = jax.random.split(key, 4)

        # The zone starts thermally settled: mass and air at the same
        # temperature. A building whose mass disagreed with its air by several
        # degrees at t=0 would spend the first day of every episode relaxing,
        # which is an artefact rather than a control problem.
        T_initial = jax.random.uniform(
            temp_key,
            minval=params.initial_T_range[0],
            maxval=params.initial_T_range[1],
        )
        setpoint_occupied = jax.random.uniform(
            sp_key,
            minval=params.setpoint_occupied_range[0],
            maxval=params.setpoint_occupied_range[1],
        )
        weather_dev = params.T_out_noise_std * jax.random.normal(weather_key)

        T_out = outdoor_temperature(0, weather_dev, params)
        phi_ia, phi_st, _ = split_gains(
            internal_gain(0, params), solar_gain(0, params), params
        )
        T_air, T_surface = solve_air_and_surface(
            T_initial, T_out, 0.0, phi_st, phi_ia, params
        )

        state = HVACState(
            time=0,
            T_mass=T_initial,
            Q_emitter=jnp.zeros(()),
            T_air=T_air,
            T_surface=T_surface,
            T_out=T_out,
            weather_dev=weather_dev,
            target_T=scheduled_setpoint(0, setpoint_occupied, params),
            setpoint_occupied=setpoint_occupied,
            Q_command=jnp.zeros(()),
        )
        return self.get_obs(state), state

    def action_space(self, params: HVACParams | None = None) -> spaces.Box:
        return spaces.Box(
            low=jnp.array([-1.0]),
            high=jnp.array([1.0]),
            shape=(1,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: HVACParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: HVACParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(
            -inf, inf, len(HVACState.__dataclass_fields__), dtype=jnp.float32
        )

    def make_pid(self):
        """Return a ready-to-use StatefulPID for zone temperature control."""
        from target_gym.experts.pid import make_hvac_stateful_pid

        return make_hvac_stateful_pid()

    def make_mpc(self, params=None, **kwargs):
        """Return a CasADi MPC oracle for zone temperature control."""
        from target_gym.experts.mpc import make_hvac_mpc

        if params is None:
            params = self.default_params
        return make_hvac_mpc(self, params, **kwargs)

    @property
    def expert_policy(self):
        from target_gym.experts.pid import (
            FunctionalExpertPolicy,
            make_hvac_pid,
            pid_step,
        )

        pid_params, zero_state = make_hvac_pid()
        return FunctionalExpertPolicy(pid_params, zero_state, pid_step)

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

    def render(self, screen, state: HVACState, params: HVACParams, frames, clock):
        frames, screen, clock = self.render_hvac(screen, state, params, frames, clock)
        return frames, screen, clock


if __name__ == "__main__":
    env = BuildingHVAC()
    os.makedirs("videos/hvac", exist_ok=True)
    env.save_video(
        lambda o: np.random.uniform(-1, 1),
        42,
        folder="videos/hvac",
        episode_index=0,
        params=HVACParams(max_steps_in_episode=288),
        format="gif",
    )
