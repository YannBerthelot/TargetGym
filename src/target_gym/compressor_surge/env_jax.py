from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from target_gym.base import (
    FieldReading,
    actuate,
    canonical_dtypes,
    canonical_reset,
    finish_step,
    measured_obs,
)
from target_gym.compressor_surge.env import (
    VALVE_HISTORY,
    CompressorSurgeParams,
    CompressorSurgeState,
    _substeps,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_reward_terms,
    demand_opening,
    draw_schedule,
    established_point,
    get_obs,
    renew_schedule,
    schedule_ramp,
)
from target_gym.compressor_surge.rendering import _render
from target_gym.expert_access import RegistryExperts
from target_gym.video import SavesVideo

#: Salt for the machine's draw at the reset, apart from the schedule's.
MACHINE = 0x4D414348
#: The tenths ``machine_stratum`` picks from (ruling F, 2026-11-24).
MACHINE_STRATA = 10


class CompressorSurge(
    RegistryExperts,
    SavesVideo,
    environment.Environment[CompressorSurgeState, CompressorSurgeParams],
):
    """Compressor feeding a header; three counted surges trip it.

    Action (2,): [speed command, recycle command], raw in [-1, 1] -> [N_min, N_max] of rated and [0, 1] opening.
    """

    #: What a trip's restart carries over (``base.failure_kernel``): the outside
    #: world, which depends on the seed and the time only, never the plant
    #: (docs/proposals/long-strata.md, S2).
    OUTSIDE_WORLD = (
        "demand_dev",
        "setpoint_levels",
        "demand_levels",
        "block_clock",
        "demand_ramp",
        "line_shift",
    )
    #: The plant's history a restart keeps too: the surge counter's times and
    #: the last step's count, which the policy writes.
    RECORDS = ("surge_times", "surges")
    #: Runs as a continuing plant (``target_gym.stream``): its demand schedule
    #: renews, its clocks count on the schedule, and ``start_event`` moves the
    #: demand now.
    STREAM_READY = True

    registry_name: str = "compressor_surge"

    render_car = classmethod(_render)
    screen_width = 600
    screen_height = 400

    # obs = [dp (kPa), m_c, m_d, N (%), N_ramp (%), x (%), live setpoint (kPa),
    #        surge count]
    obs_value_index: int = 0  # header pressure
    tracked_names: tuple = ("header pressure (kPa)",)
    obs_target_index: int = 6  # live setpoint

    # PHYSICS.md section on readings. The delivered flow is computed, so its
    # meter reads through ``demand_dev``. 1 rpm at an assumed 10,000 rpm.
    readings: tuple = (
        # First-order transmitter lags [C10, pp. 29-30]: flow dP at 100 to
        # 150 ms, 200 ms at most; 300 to 400 ms acceptable on pressure.
        FieldReading("dp", sigma=9.17, lag=0.3),
        FieldReading("m_c", sigma=0.021, lag=0.15),
        FieldReading("demand_dev", sigma=8.4e-4),
        FieldReading("N", sigma=5.8e-5, step=1e-4),
        FieldReading("x", sigma=1.5e-3),
    )

    def __init__(self, integration_method: str = "rk4_10"):
        # "rk4_N": N substeps per step, each one actuator rate-limit update and
        # one RK4 step of delta_t / N. Checked here so a bad string fails at
        # construction rather than at the first traced step.
        _substeps(integration_method)
        self.obs_shape = (8,)
        self.positions_history = []
        self.integration_method = integration_method

    @property
    def default_params(self) -> CompressorSurgeParams:
        return CompressorSurgeParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def reward_terms(self, state, params, tripped=None):
        return compute_reward_terms(state, params, tripped=tripped)

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
        action, state = actuate(self, state, action, params)
        proposal, metrics = compute_next_state(
            action, state, params, key, integration_method=self.integration_method
        )

        # A trip is part of the kernel: the step is charged the trip cost and
        # the plant restarts at once from a fresh draw (new schedule, new
        # deviation, block clock 0) on the same window clock, or is lost (``terminated``) with
        # ``params.continuing = False`` (``base.finish_step``).
        # Scored on the proposal, against the live setpoint of the advanced
        # clock: the step that enters a block is scored against its level,
        # which the returned observation also shows.
        new_state, reward, terminated, info = finish_step(
            self, key, state, proposal, params
        )
        # The plant's own actuators set the rates, so the applied action is
        # where they went.
        info["applied_action"] = self.initial_action(proposal, params)
        return self.get_obs(new_state, params), new_state, reward, terminated, info

    def initial_action(self, state, params):
        """The raw action that holds the drive setpoint and valve where they are."""
        return jnp.stack(
            [
                2.0 * (state.N_ramp - params.N_min) / (params.N_max - params.N_min)
                - 1.0,
                2.0 * state.x - 1.0,
            ]
        )

    @measured_obs
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
        clipped at ``demand_dev_reset_clip`` sd, and the plant in established
        operation for the drifted first opening (charter B5,
        ``established_point``): at rest at the header setpoint, on or right of
        the ``reset_margin`` control line, with the schedule clock at 0."""
        if params is None:
            params = self.default_params

        schedule_key, dev_key = jax.random.split(key)
        setpoint_levels, demand_levels = draw_schedule(schedule_key, params)
        clip = params.demand_dev_reset_clip
        demand_dev = params.demand_sigma * jnp.clip(
            jax.random.normal(dev_key), -clip, clip
        )
        opening = demand_opening(demand_levels, 0.0, demand_dev, params)
        # The machine, on a key of its own so that the schedule's and the
        # deviation's draws keep their values.
        # A stratum puts the same draw inside its tenth of the range.
        draw = jax.random.uniform(
            jax.random.fold_in(key, MACHINE), minval=-1.0, maxval=1.0
        )
        stratum = params.machine_stratum
        spread = 2.0 * (stratum + 0.5 * (draw + 1.0)) / MACHINE_STRATA - 1.0
        line_shift = params.line_shift_max * jnp.where(stratum < 0, draw, spread)
        m_c, dp, N0, x0, phi = established_point(opening, params, line_shift)

        state = CompressorSurgeState(
            time=0,
            m_c=m_c,
            dp=dp,
            N=N0,
            N_ramp=N0,
            x=x0,
            demand_dev=demand_dev,
            phi_min=phi,
            setpoint_levels=setpoint_levels,
            demand_levels=demand_levels,
            block_clock=0,
            valve_cmds=jnp.full((VALVE_HISTORY,), x0),
            demand_ramp=jnp.asarray(params.demand_ramp_s, jnp.float32),
            line_shift=line_shift,
        )

        obs = self.get_obs(state, params)
        return obs, state

    def params_for_seed(self, params: CompressorSurgeParams, seed: int):
        """The protocol's parameters for seed ``seed``: its machine from tenth
        ``seed mod 10`` of the shift range (``machine_stratum``). The protocol
        calls it through ``benchmark.seed_params`` wherever it resets a seed."""
        return params.replace(machine_stratum=int(seed) % MACHINE_STRATA)

    def restart_plant(self, key, state, params: CompressorSurgeParams):
        """A trip's restart (``base.failure_kernel``): the plant back in
        established operation for the demand the outside world carries on
        with, which is ``state``'s, not a fresh draw's, and on ``state``'s
        machine: a trip does not move its surge line. A machine restarted on
        another demand's balance would meet its consumers off its header zone,
        or left of its line."""
        fresh = self.reset_env(key, params)[1]
        opening = demand_opening(
            state.demand_levels,
            state.block_clock * params.delta_t,
            state.demand_dev,
            params,
            ramp=schedule_ramp(state, params),
        )
        m_c, dp, N0, x0, phi = established_point(opening, params, state.line_shift)
        return canonical_dtypes(
            fresh.replace(
                m_c=m_c,
                dp=dp,
                N=N0,
                N_ramp=N0,
                x=x0,
                phi_min=phi,
                valve_cmds=jnp.full((VALVE_HISTORY,), x0),
                actuator_held=self.initial_action(
                    fresh.replace(N_ramp=N0, x=x0), params
                ),
            )
        )

    def start_event(self, state, key, params: CompressorSurgeParams):
        """The stream's event hook (docs/proposals/long-strata.md, S1): the
        consumers' demand moves now, from the opening it holds, onto a fresh
        schedule drawn from ``key`` in the scenario's range."""
        return renew_schedule(state, params, key)

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
