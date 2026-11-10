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
from target_gym.boiler_drum.env import (
    BoilerDrumParams,
    BoilerDrumState,
    _base_plan,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_reward_terms,
    get_obs,
    load_plan,
    plan_seconds,
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

    Observation (12): [level, pressure, q_steam, fuel_pct, feed_pct,
                       target_level, target_pressure, load_now, load_planned,
                       s_to_ramp, fuel_cap_pct, feed_cap_pct]
    Action      (2): [fuel, feedwater], raw in [-1, 1]
    """

    #: What a trip's restart carries over (``base.failure_kernel``): the outside
    #: world and the records, never the plant (docs/proposals/long-strata.md, S2).
    OUTSIDE_WORLD = (
        "q_demand",
        "target_pressure",
        "level_ref",
        "load_from",
        "load_to",
        "ramp_start",
        "ramp_rate",
        "event_time",
        "event_rate",
        "standby_delay",
        "event_kind",
        "event_until",
        "plan_t0",
    )
    #: Runs as a continuing plant (``target_gym.stream``): its load changes and
    #: upsets start through ``start_event``, a runback returns, and the plan's
    #: clock is rebased at each event.
    STREAM_READY = True

    registry_name: str = "boiler_drum"

    render_boiler = classmethod(_render)
    screen_width = 700
    screen_height = 900

    obs_value_index: tuple = (0, 1)  # level, pressure
    tracked_names: tuple = ("drum level (m)", "drum pressure (bar)")
    obs_target_index: tuple = (5, 6)  # target_level, target_pressure

    # Guided-wave radar, a pressure transmitter and a vortex meter (PHYSICS.md,
    # Readings).
    readings: tuple = (
        FieldReading("level", sigma=5e-4),
        FieldReading("pressure", sigma=0.022),
        FieldReading("q_steam", sigma=0.047),
    )

    def __init__(self, integration_method: str = "rk4_1"):
        self.obs_shape = (12,)
        self.integration_method = integration_method

    @property
    def default_params(self) -> BoilerDrumParams:
        return BoilerDrumParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def reward_terms(self, state, params, tripped=None):
        return compute_reward_terms(state, params, tripped=tripped)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: BoilerDrumState,
        action: jnp.ndarray,
        params: BoilerDrumParams = None,
    ):
        if params is None:
            params = self.default_params

        action, state = actuate(self, state, action.reshape((2,)), params)

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

    @measured_obs
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

        # Established operation (rule 17): the unit holds a load on its plan,
        # balanced at normal water level, and the plan ramps once to another
        # load at an industry ramp rate (PHYSICS.md, Task; REALITY.md sec. 4).
        k_from, k_to, k_rate, k_start, k_ev, k_back, k_wait = jax.random.split(key, 7)
        nominal = params.q_steam_nominal
        lo, hi = params.load_range
        load_from = nominal * jax.random.uniform(k_from, minval=lo, maxval=hi)
        load_to = nominal * jax.random.uniform(k_to, minval=lo, maxval=hi)
        rate_pct = jax.random.uniform(
            k_rate,
            minval=params.ramp_rate_range[0],
            maxval=params.ramp_rate_range[1],
        )
        ramp_start = jax.random.uniform(
            k_start,
            minval=params.ramp_start_range[0],
            maxval=params.ramp_start_range[1],
        )
        # A stream starts on a flat plan: its load changes are events.
        load_to = jnp.where(params.window_ramp, load_to, load_from)
        ramp_start = jnp.where(params.window_ramp, ramp_start, 0.0)
        event_time = jnp.where(
            params.event_kind > 0,
            jax.random.uniform(
                k_ev,
                minval=params.event_time_range[0],
                maxval=params.event_time_range[1],
            ),
            1e9,
        )
        back_pct = jax.random.uniform(
            k_back,
            minval=params.runback_rate_range[0],
            maxval=params.runback_rate_range[1],
        )
        standby = jax.random.uniform(
            k_wait,
            minval=params.standby_delay_range[0],
            maxval=params.standby_delay_range[1],
        )
        V_wt, alpha_r, V_sd, level_ref, Q = steady_state(
            params, params.p_nominal, load_from
        )
        zero = jnp.zeros_like(load_from)
        state = BoilerDrumState(
            time=0,
            pressure=jnp.asarray(params.p_nominal) + zero,
            V_wt=V_wt,
            alpha_r=alpha_r,
            V_sd=V_sd,
            level=zero,
            q_steam=load_from,  # at p_nominal the valve passes its plan
            q_demand=load_from,
            Q_fuel=Q,
            q_feed=load_from,
            target_pressure=jnp.asarray(params.p_nominal) + zero,
            level_ref=jnp.asarray(level_ref) + zero,
            load_from=load_from,
            load_to=load_to,
            ramp_start=ramp_start,
            ramp_rate=rate_pct / 100.0 * nominal / 60.0,
            event_time=event_time + zero,
            event_rate=back_pct / 100.0 * nominal / 60.0 + zero,
            standby_delay=standby + zero,
            time_high=zero,
            time_low=zero,
            event_kind=jnp.asarray(params.event_kind, jnp.int32),
            event_until=event_time + params.event_hold + zero,
            plan_t0=jnp.zeros((), jnp.int32),
        )
        return self.get_obs(state), state

    def start_event(
        self, state: BoilerDrumState, key, params: BoilerDrumParams
    ) -> BoilerDrumState:
        """The stream's event hook (docs/proposals/long-strata.md, S1): the
        scenario's event starts now, drawn from its ranges.

        - ``event_kind`` 0: a planned load change. From where the plan stands it
          ramps to a load in ``load_range`` at a rate in ``ramp_rate_range``,
          announced ``ramp_start_range`` ahead, as the window's one ramp is.
        - ``event_kind`` 1 or 2: an ID fan runback or a feed pump trip, as the
          window's, that returns ``event_hold`` later.

        The plan's clock is rebased on now, the ramp in progress going on from
        where it stands. While a runback is still holding the load down, the
        unit master takes no new event: the draw is absorbed (ours).
        """
        p = params
        now = plan_seconds(state, p)
        base = _base_plan(state, now)
        busy = (
            (state.event_kind > 0)
            & (now >= state.event_time)
            & ((now < state.event_until) | (load_plan(state, now, p) < base))
        )
        rebased = state.replace(
            plan_t0=jnp.asarray(state.time, jnp.int32),
            load_from=base,
            ramp_start=jnp.maximum(state.ramp_start - now, 0.0),
        )
        k_to, k_rate, k_start, k_back, k_wait = jax.random.split(key, 5)
        nominal = p.q_steam_nominal

        def draw(k, bounds):
            return jax.random.uniform(k, minval=bounds[0], maxval=bounds[1])

        ramp = rebased.replace(
            load_to=nominal * draw(k_to, p.load_range),
            ramp_rate=draw(k_rate, p.ramp_rate_range) / 100.0 * nominal / 60.0,
            ramp_start=draw(k_start, p.ramp_start_range),
            event_kind=jnp.zeros((), jnp.int32),
            event_time=jnp.full((), 1.0e9),
            event_until=jnp.full((), 1.0e9),
        )
        upset = rebased.replace(
            event_kind=jnp.asarray(p.event_kind, jnp.int32),
            event_time=jnp.zeros(()),
            event_rate=draw(k_back, p.runback_rate_range) / 100.0 * nominal / 60.0,
            standby_delay=draw(k_wait, p.standby_delay_range),
            event_until=jnp.asarray(p.event_hold, jnp.float32),
        )
        started = jax.tree_util.tree_map(
            lambda u, r: jnp.where(p.event_kind > 0, u, r), upset, ramp
        )
        return jax.tree_util.tree_map(
            lambda old, new: jnp.where(busy, old, new), state, started
        )

    def initial_action(self, state: BoilerDrumState, params: BoilerDrumParams):
        """The raw action that keeps the reset's firing and feedwater."""
        return jnp.array(
            [
                2.0 * state.Q_fuel / params.Q_max - 1.0,
                2.0 * state.q_feed / params.q_feed_max - 1.0,
            ]
        )

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
