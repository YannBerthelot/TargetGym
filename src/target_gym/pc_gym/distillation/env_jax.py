import os
from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment, spaces

from target_gym.base import canonical_reset
from target_gym.pc_gym.distillation.env import (
    N_STAGES,
    DistillationParams,
    DistillationState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    get_obs,
)
from target_gym.pc_gym.distillation.rendering import _render
from target_gym.utils import save_video


class DistillationColumn(
    environment.Environment[DistillationState, DistillationParams]
):
    """Binary distillation column, Skogestad's "Column A".

    Observation (6,): [yD, xB, L_pct, V_pct, target_yD, target_xB]
    Action      (2,): [L_raw, V_raw] in [-1, 1] -> reflux and boilup
    """

    render_column = classmethod(_render)
    screen_width = 700
    screen_height = 900

    obs_value_index: int = 0  # yD
    tracked_names: tuple = ("yD (mole fraction)",)
    obs_target_index: int = 4  # target_yD

    def __init__(self, integration_method: str = "rk4_16"):
        self.obs_shape = (6,)
        self.integration_method = integration_method

    @property
    def default_params(self) -> DistillationParams:
        return DistillationParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: DistillationState,
        action: jnp.ndarray,
        params: DistillationParams = None,
    ):
        if params is None:
            params = self.default_params
        action = jnp.atleast_1d(jnp.asarray(action)).reshape(-1)

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

    def get_obs(self, state: DistillationState, params: DistillationParams = None):
        if params is None:
            params = self.default_params
        return get_obs(state, params=params)

    def is_terminated(self, state, params) -> jnp.ndarray:
        """Natural termination only; the time limit is gymnax's ``is_truncated``."""
        terminated, _ = check_is_terminal(state, params)
        return terminated

    @canonical_reset
    def reset_env(
        self, key: chex.PRNGKey, params: DistillationParams = None
    ) -> Tuple[jnp.ndarray, DistillationState]:
        if params is None:
            params = self.default_params

        key, l_key, top_key, bot_key, feed_key = jax.random.split(key, 5)

        L = jax.random.uniform(
            l_key,
            minval=params.initial_L_range[0],
            maxval=params.initial_L_range[1],
        )
        V = L + 0.5  # nominal distillate D = 0.5 = F/2
        zF = jnp.clip(
            params.zF_nominal + params.zF_noise_std * jax.random.normal(feed_key),
            params.zF_min,
            params.zF_max,
        )
        target_yD = jax.random.uniform(
            top_key,
            minval=params.target_yD_range[0],
            maxval=params.target_yD_range[1],
        )
        target_xB = jax.random.uniform(
            bot_key,
            minval=params.target_xB_range[0],
            maxval=params.target_xB_range[1],
        )

        # Start from the converged nominal profile rather than an arbitrary
        # ramp: a running column is at steady state, and with a ~194 min
        # dominant time constant an arbitrary start would spend the whole
        # episode relaxing instead of being controlled.
        x = _NOMINAL_PROFILE

        state = DistillationState(
            time=0, x=x, zF=zF, L=L, V=V, target_yD=target_yD, target_xB=target_xB
        )
        return self.get_obs(state), state

    def action_space(self, params: DistillationParams | None = None) -> spaces.Box:
        return spaces.Box(
            low=jnp.array([-1.0, -1.0]),
            high=jnp.array([1.0, 1.0]),
            shape=(2,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: DistillationParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: DistillationParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(
            -inf, inf, len(DistillationState.__dataclass_fields__), dtype=jnp.float32
        )

    def make_pid(self):
        """Return a ready-to-use MIMO PID (LV pairing) for dual composition control."""
        from target_gym.experts.pid import make_distillation_stateful_pid

        return make_distillation_stateful_pid()

    def make_mpc(self, params=None, **kwargs):
        """Return a gradient MPC oracle for dual composition control."""
        from target_gym.experts.mpc import make_distillation_mpc

        if params is None:
            params = self.default_params
        return make_distillation_mpc(self, params, **kwargs)

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

    def render(self, screen, state, params, frames, clock):
        frames, screen, clock = self.render_column(screen, state, params, frames, clock)
        return frames, screen, clock


def _compute_nominal_profile() -> jnp.ndarray:
    """Converged composition profile at the nominal operating point.

    Computed once at import (a few thousand steps of the same dynamics) rather
    than hard-coded, so it stays consistent if the column parameters change.
    """
    params = DistillationParams().replace(zF_noise_std=0.0)
    raw = jnp.array(
        [
            2.0 * (2.706 - params.L_min) / (params.L_max - params.L_min) - 1.0,
            2.0 * (3.206 - params.V_min) / (params.V_max - params.V_min) - 1.0,
        ]
    )
    state = DistillationState(
        time=0,
        x=jnp.linspace(0.01, 0.99, N_STAGES),
        zF=params.zF_nominal,
        L=2.706,
        V=3.206,
        target_yD=0.99,
        target_xB=0.01,
    )
    key = jax.random.PRNGKey(0)

    def body(carry, _):
        s, _ = compute_next_state(raw, carry, params, key)
        return s, None

    state, _ = jax.lax.scan(body, state, xs=None, length=4000)
    return state.x


_NOMINAL_PROFILE = jax.jit(_compute_nominal_profile)()


if __name__ == "__main__":
    env = DistillationColumn()
    os.makedirs("videos/distillation", exist_ok=True)
    env.save_video(
        lambda o: np.random.uniform(-1, 1, size=(2,)),
        42,
        folder="videos/distillation",
        episode_index=0,
        params=DistillationParams(max_steps_in_episode=200),
        format="gif",
    )
