import os
from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment, spaces

from target_gym.reactor.env import (
    N_GROUPS,
    N_SETPOINTS,
    ReactorParams,
    ReactorState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    get_obs,
    get_target_from_schedule,
    steady_state_precursors,
    steady_state_xenon,
)
from target_gym.reactor.rendering import _render
from target_gym.utils import save_video


# Number of physics sub-steps per control step. Constant so JIT can treat it
# as static in `lax.scan(length=...)`. Change here only — `env.py` reads it
# via a delayed import in `check_is_terminal` / `get_target_from_schedule`.
CONTROL_PERIOD: int = 10


class Reactor(environment.Environment[ReactorState, ReactorParams]):
    """
    Nuclear reactor (point kinetics + thermal feedback).

    Observation (4,): [n, T_coolant, rho_ext_norm, target_n]
    Action      (1,): [rho_ext_norm] in [-1, 1] (control rod position)
    """

    render_reactor = classmethod(_render)
    screen_width = 700
    screen_height = 900
    # Number of physics sub-steps per env-step. Exposed so external
    # rollout code (eval scripts) can convert max_steps_in_episode
    # (physics units) to env-step counts.
    control_period: int = CONTROL_PERIOD

    # obs = [n, T_coolant, rho_ext_norm, target_n]
    obs_value_index: int = 0  # n (neutron density / normalised power)
    obs_target_index: int = 3  # target_n

    def __init__(self, integration_method: str = "rk4_50"):
        self.obs_shape = (4,)
        self.integration_method = integration_method

    @property
    def default_params(self) -> ReactorParams:
        return ReactorParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: ReactorState,
        action: jnp.ndarray,
        params: ReactorParams = None,
    ):
        if params is None:
            params = self.default_params

        rho_raw = action
        if not isinstance(action, float):
            rho_raw = action.reshape(())

        # Action is held constant for `control_period` physics sub-steps. Reward
        # is summed across the sub-steps; we freeze the state on termination so
        # the scan can still run for a fixed length under jit.
        def sub_step(carry, _):
            state, cum_reward, done = carry
            candidate, _metrics = compute_next_state(
                rho_raw, state, params, integration_method=self.integration_method
            )
            r = compute_reward(candidate, params, xp=jnp)
            term, trunc = check_is_terminal(candidate, params, xp=jnp)
            new_done = term | trunc
            # Freeze state once done; still accumulate the final-step reward.
            next_state = jax.tree.map(
                lambda a, b: jnp.where(done, a, b), state, candidate
            )
            next_reward = cum_reward + jnp.where(done, 0.0, r)
            next_done = done | new_done
            return (next_state, next_reward, next_done), None

        (new_state, reward, done), _ = jax.lax.scan(
            sub_step,
            (state, jnp.float32(0.0), jnp.bool_(False)),
            xs=None,
            length=CONTROL_PERIOD,
        )

        obs = self.get_obs(new_state)
        return (
            obs,
            new_state,
            reward,
            done,
            {"last_state": new_state},
        )

    def get_obs(self, state: ReactorState, params: ReactorParams = None):
        if params is None:
            params = self.default_params
        return get_obs(state, params=params)

    def is_terminal(self, state: ReactorState, params: ReactorParams) -> jnp.ndarray:
        return check_is_terminal(state, params)

    def reset_env(
        self, key: chex.PRNGKey, params: ReactorParams = None
    ) -> Tuple[jnp.ndarray, ReactorState]:
        if params is None:
            params = self.default_params

        key, n_key, target_key, demand_key = jax.random.split(key, 4)

        initial_n = jax.random.uniform(
            n_key,
            minval=params.initial_n_range[0],
            maxval=params.initial_n_range[1],
        )
        # Initial demand drawn from target range; OU process evolves it from here.
        initial_target = jax.random.uniform(
            target_key,
            minval=params.target_n_range[0],
            maxval=params.target_n_range[1],
        )
        # Legacy schedule field (kept for backward compatibility, unused by OU).
        demand_mu = 0.5 * (params.target_n_range[0] + params.target_n_range[1])
        target_schedule = jnp.full((N_SETPOINTS,), demand_mu)

        # Precursors and xenon/iodine start at steady-state for the initial
        # neutron density. Without this, there would be a huge transient in
        # the first few seconds.
        initial_C = steady_state_precursors(initial_n, params)
        initial_I_hat, initial_Xe_hat = steady_state_xenon(initial_n, params)

        state = ReactorState(
            time=0,
            n=initial_n,
            C=initial_C,
            T_fuel=jnp.asarray(params.initial_T_fuel, dtype=jnp.float32),
            T_coolant=jnp.asarray(params.initial_T_coolant, dtype=jnp.float32),
            I_hat=jnp.asarray(initial_I_hat, dtype=jnp.float32),
            Xe_hat=jnp.asarray(initial_Xe_hat, dtype=jnp.float32),
            target_n=initial_target,
            target_schedule=target_schedule,
            demand_key=demand_key,
            rho_ext=jnp.zeros((), dtype=jnp.float32),
        )

        obs = self.get_obs(state)
        return obs, state

    def action_space(self, params: ReactorParams | None = None) -> spaces.Box:
        return spaces.Box(
            low=jnp.array([-1.0]),
            high=jnp.array([1.0]),
            shape=(1,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: ReactorParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: ReactorParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(
            -inf, inf, len(ReactorState.__dataclass_fields__), dtype=jnp.float32
        )

    @property
    def expert_policy(self):
        from target_gym.experts.pid import FunctionalExpertPolicy, make_reactor_pid, pid_step
        params, zero_state = make_reactor_pid()
        return FunctionalExpertPolicy(params, zero_state, pid_step)

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

    def render(self, screen, state: ReactorState, params: ReactorParams, frames, clock):
        frames, screen, clock = self.render_reactor(
            screen, state, params, frames, clock
        )
        return frames, screen, clock


if __name__ == "__main__":
    env = Reactor()
    seed = 42
    env_params = ReactorParams(max_steps_in_episode=2000)  # 2000 physics = 200 control steps
    os.makedirs("videos/reactor", exist_ok=True)
    env.save_video(
        lambda o: np.random.uniform(-1, 1),
        seed,
        folder="videos/reactor",
        episode_index=0,
        params=env_params,
        format="gif",
    )
