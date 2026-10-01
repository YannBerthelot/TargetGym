from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from target_gym.base import canonical_reset, failure_kernel
from target_gym.compressor_surge.env import (
    CompressorSurgeParams,
    CompressorSurgeState,
    _substeps,
    characteristic,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_reward_terms,
    demand_opening,
    draw_schedule,
    equilibrium_phi,
    get_obs,
    suction_density,
)
from target_gym.compressor_surge.rendering import _render
from target_gym.utils import save_video


class CompressorSurge(
    environment.Environment[CompressorSurgeState, CompressorSurgeParams]
):
    """Compressor feeding a header, with surge as the trip.

    Action (2,): [speed command, recycle command], raw in [-1, 1] -> [N_min, N_max] of rated and [0, 1] opening.
    """

    render_car = classmethod(_render)
    screen_width = 600
    screen_height = 400

    # obs = [dp (kPa), m_c, m_d, N (%), N_ramp (%), x (%), live setpoint (kPa)]
    obs_value_index: int = 0  # header pressure
    tracked_names: tuple = ("header pressure (kPa)",)
    obs_target_index: int = 6  # live setpoint

    def __init__(self, integration_method: str = "rk4_10"):
        # "rk4_N": N substeps per step, each one actuator rate-limit update and
        # one RK4 step of delta_t / N. Checked here so a bad string fails at
        # construction rather than at the first traced step.
        _substeps(integration_method)
        self.obs_shape = (7,)
        self.positions_history = []
        self.integration_method = integration_method

    @property
    def default_params(self) -> CompressorSurgeParams:
        return CompressorSurgeParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def reward_terms(self, state, params):
        return compute_reward_terms(state, params)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: CompressorSurgeState,
        action: jnp.ndarray,
        params: CompressorSurgeParams = None,
    ):
        """
        Performs step transitions using JAX, returns observation, new state, reward, done, info
        """
        if params is None:
            params = self.default_params

        action = jnp.reshape(jnp.asarray(action), (2,))
        new_state, metrics = compute_next_state(
            action, state, params, key, integration_method=self.integration_method
        )

        # A trip is part of the kernel: the step is charged the trip cost and
        # the plant restarts at once from a fresh draw (new schedule, new
        # deviation, block clock 0) on the same window clock; ``terminated``
        # is never raised (``base.failure_kernel``).
        new_state, tripped, scored = failure_kernel(self, key, state, new_state, params)
        # Scored on the proposal, against the live setpoint of the advanced
        # clock: the step that enters a block is scored against its level,
        # which the returned observation also shows.
        reward = compute_reward(scored, params, xp=jnp)
        return (
            self.get_obs(new_state, params),
            new_state,
            reward,
            jnp.zeros((), dtype=bool),
            {"last_state": new_state, "tripped": tripped},
        )

    def get_obs(
        self, state: CompressorSurgeState, params: CompressorSurgeParams = None
    ):
        """
        Observation vector
        """
        if params is None:
            params = (
                self.default_params
            )  # NOTE: gymnax's get_obs takes no params; this environment's
            # observation needs them, so the signature accepts them and falls
            # back to the defaults when a caller follows the gymnax shape.
        return get_obs(state, params=params)

    def is_terminated(
        self, state: CompressorSurgeState, params: CompressorSurgeParams
    ) -> jnp.ndarray:
        """Natural termination only; the time limit is gymnax's ``is_truncated``."""
        terminated, _ = check_is_terminal(state, params)
        return terminated

    @canonical_reset
    def reset_env(
        self, key: chex.PRNGKey, params: CompressorSurgeParams = None
    ) -> Tuple[jnp.ndarray, CompressorSurgeState]:
        """A fresh schedule, the initial deviation drawn from its stationary law
        clipped at ``demand_dev_reset_clip`` sd, and the plant at its
        equilibrium at ``N_reset`` and ``x_reset`` for the drifted first
        opening, with the schedule clock at 0. The reset is off target by
        design (PHYSICS.md, task design)."""
        if params is None:
            params = self.default_params

        schedule_key, dev_key = jax.random.split(key)
        setpoint_levels, demand_levels = draw_schedule(schedule_key, params)
        clip = params.demand_dev_reset_clip
        demand_dev = params.demand_sigma * jnp.clip(
            jax.random.normal(dev_key), -clip, clip
        )
        opening = demand_opening(demand_levels, 0.0, demand_dev, params)
        phi = equilibrium_phi(
            params.N_reset, params.x_reset, opening, params, params.reset_newton_iters
        )
        rho = suction_density(params)
        U = params.N_reset * params.U_r

        state = CompressorSurgeState(
            time=0,
            m_c=rho * params.A_c * U * phi,
            dp=rho * U**2 * characteristic(phi, params),
            N=params.N_reset,
            N_ramp=params.N_reset,
            x=params.x_reset,
            demand_dev=demand_dev,
            phi_min=phi,
            setpoint_levels=setpoint_levels,
            demand_levels=demand_levels,
            block_clock=0,
        )

        obs = self.get_obs(state, params)
        return obs, state

    def action_space(self, params: CompressorSurgeParams | None = None) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(
            low=jnp.array([-1.0, -1.0]),
            high=jnp.array([1.0, 1.0]),
            shape=(2,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: CompressorSurgeParams) -> spaces.Box:
        """Observation space of the environment."""
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: CompressorSurgeParams) -> spaces.Box:
        """State space of the environment."""
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(
            -inf,
            inf,
            len(CompressorSurgeState.__dataclass_fields__),
            dtype=jnp.float32,
        )

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
        self,
        screen,
        state: CompressorSurgeState,
        params: CompressorSurgeParams,
        frames,
        clock,
    ):
        """
        JAX-compatible rendering wrapper
        """
        frames, screen, clock = self.render_car(screen, state, params, frames, clock)
        return frames, screen, clock

    def make_pid(self):
        """Return the classic pair: a pressure PI on speed and an anti-surge PI
        on the recycle."""
        from target_gym.compressor_surge.experts import make_compressor_surge_pid

        return make_compressor_surge_pid()

    def make_mpc(self, params=None, **kwargs):
        """Return the CasADi NMPC oracle, which falls back to the PID pair."""
        from target_gym.compressor_surge.experts import make_compressor_surge_mpc

        if params is None:
            params = self.default_params
        return make_compressor_surge_mpc(self, params, **kwargs)
