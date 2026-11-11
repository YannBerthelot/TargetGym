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
from target_gym.pc_gym.distillation.env import (
    N_STAGES,
    TEMPERATURE_STAGE,
    DistillationParams,
    DistillationState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_reward_terms,
    get_obs,
    steady_liquid,
)
from target_gym.pc_gym.distillation.rendering import _render
from target_gym.video import SavesVideo

_CHROMATOGRAPH = dict(interval=5.0, dead_time=5.0, sampled=True)


class DistillationColumn(
    RegistryExperts,
    SavesVideo,
    environment.Environment[DistillationState, DistillationParams],
):
    """Binary distillation column, Skogestad's "Column A".

    Observation (8,): [yD, xB, L_pct, V_pct, spec_yD, spec_xB, T_15, F]
    Action      (2,): [L_raw, V_raw] in [-1, 1] -> reflux and boilup
    """

    registry_name: str = "distillation"

    render_column = classmethod(_render)
    screen_width = 700
    screen_height = 900

    obs_value_index: tuple = (0, 1)  # yD, xB
    tracked_names: tuple = ("yD (mole fraction)", "xB (mole fraction)")
    obs_target_index: tuple = (4, 5)  # target_yD, target_xB

    # A chromatograph on each product and an RTD on the tray (PHYSICS.md section
    # on readings). ``interval`` and ``dead_time`` are in ``delta_t``'s unit, so
    # 5 is 5 minutes: a cycle reports what was sampled one cycle earlier.
    readings: tuple = (
        FieldReading("x", index=0, sigma=4e-4, **_CHROMATOGRAPH),
        FieldReading("x", index=TEMPERATURE_STAGE - 1, sigma=3e-3),
        FieldReading("x", index=-1, sigma=4e-4, **_CHROMATOGRAPH),
    )

    def __init__(self, integration_method: str = "rk4_16"):
        self.obs_shape = (8,)
        self.integration_method = integration_method

    @property
    def default_params(self) -> DistillationParams:
        return DistillationParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def reward_terms(self, state, params, tripped=None):
        return compute_reward_terms(state, params, tripped=tripped)

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
        action, state = actuate(self, state, action, params)

        new_state, _ = compute_next_state(
            action, state, params, key, integration_method=self.integration_method
        )
        # A trip is part of the kernel: the step is charged the trip cost and
        # the plant restarts at once, or is lost (``terminated``) with
        # ``params.continuing = False`` (``base.finish_step``).
        new_state, reward, terminated, info = finish_step(
            self, key, state, new_state, params
        )
        return self.get_obs(new_state, params), new_state, reward, terminated, info

    def initial_action(self, state, params):
        """The raw action that holds the reset plant's own flows."""
        return jnp.stack(
            [
                2.0 * (state.L - params.L_min) / (params.L_max - params.L_min) - 1.0,
                2.0 * (state.V - params.V_min) / (params.V_max - params.V_min) - 1.0,
            ]
        )

    @measured_obs
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

        keys = jax.random.split(key, 4)

        # Established operation (charter rule 17): the column at rest at its
        # design point [D2], the converged profile at the nominal flows, with
        # the products on their specs. The feed's two steps come later.
        lo, hi = params.step_window
        state = DistillationState(
            time=0,
            x=_NOMINAL_PROFILE,
            liquid=steady_liquid(params.L_nominal, params),
            zF=jnp.asarray(params.zF_nominal),
            L=jnp.asarray(params.L_nominal),
            V=jnp.asarray(params.L_nominal + 0.5 * params.F),
            target_yD=jnp.asarray(params.yD_spec),
            target_xB=jnp.asarray(params.xB_spec),
            F=jnp.asarray(params.F),
            feed_step_at=jax.random.randint(keys[0], (), lo, hi),
            feed_sign=jnp.where(jax.random.bernoulli(keys[1]), 1.0, -1.0),
            zF_step_at=jax.random.randint(keys[2], (), lo, hi),
            zF_sign=jnp.where(jax.random.bernoulli(keys[3]), 1.0, -1.0),
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

    def render(self, screen, state, params, frames, clock):
        frames, screen, clock = self.render_column(screen, state, params, frames, clock)
        return frames, screen, clock


def _compute_nominal_profile() -> jnp.ndarray:
    """Converged composition profile at the nominal operating point.

    Computed once at import (a few thousand steps of the same dynamics) rather
    than hard-coded, so it stays consistent if the column parameters change.
    """
    params = DistillationParams()
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
