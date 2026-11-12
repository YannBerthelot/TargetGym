from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from target_gym import reward as R
from target_gym.base import (
    FieldReading,
    actuate,
    applied_action,
    canonical_dtypes,
    canonical_reset,
    charged_trip_cost,
    failure_kernel,
    measured_obs,
    measured_state,
    remember,
)
from target_gym.expert_access import RegistryExperts
from target_gym.reactor.env import (
    BORON_METER_STEPS,
    BORON_QUEUE,
    OFFSET_HOURS,
    ReactorParams,
    ReactorState,
    balance_power,
    burnt,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_reward_terms,
    flux_tilt,
    get_obs,
    moderator_tilt,
    plan_seconds,
    planned_load,
    rod_balance,
    rod_tilt,
    rod_worth,
    roll_offset_clock,
    scheduled_load,
    steady_state_precursors,
    steady_state_temperatures,
    steady_state_xenon,
    tavg_program,
    timed_out,
    xenon_after_shutdown,
)
from target_gym.reactor.rendering import _render
from target_gym.video import SavesVideo

#: Salt for a refuelling's draws, apart from the step's own key.
_REFUEL = 0x52454655

# Number of physics sub-steps per control step. Constant so JIT can treat it
# as static in `lax.scan(length=...)`. Change here only — `env.py` reads it
# via ``Reactor.control_period``.
CONTROL_PERIOD: int = 6


class Reactor(
    RegistryExperts, SavesVideo, environment.Environment[ReactorState, ReactorParams]
):
    """
    Nuclear reactor: a two-point core with rods and boron.

    Observation (13,): [n, T_hot, rod position, load, T_avg, T_ref, delta_I, boron
    (20 min late), hours to the ramp down, hours to the ramp up, xenon's power, the
    day's ramp rate (%/min), the day's minimum load]
    Action      (2,): [rod demand, boron flow] in [-1, 1]
    """

    #: What a trip's restart carries over (``base.failure_kernel``): the outside
    #: world and the records, never the plant (docs/proposals/long-strata.md, S2).
    #: The turbine's load and its Tref are not: after a restart the turbine reloads
    #: (``restart_plant``), so they follow the plant's own history.
    OUTSIDE_WORLD = (
        "load_wobble",
        "ramp_down_at",
        "ramp_up_at",
        "demand_key",
        "physics_time",
        "plan_t0",
        "plan_rate",
        "plan_depth",
    )
    #: Runs as a continuing plant (``target_gym.stream``): each load-follow day
    #: starts through ``start_event``, carries its rate and depth and ends, the
    #: plan's clock is rebased at each day, and the offset's allowance looks back
    #: 24 h.
    STREAM_READY = True

    registry_name: str = "reactor"

    render_reactor = classmethod(_render)
    screen_width = 700
    screen_height = 900
    # Number of physics sub-steps per env-step. ``max_steps_in_episode`` and
    # ``state.time`` count env steps; ``state.physics_time`` counts sub-steps.
    # Exposed so tooling can convert between the two (a control step lasts
    # ``delta_t * control_period`` seconds).
    control_period: int = CONTROL_PERIOD

    # obs = [n, T_hot, rod position, load, T_avg, T_ref, delta_I, boron, to_down, to_up]
    obs_value_index: int = 4  # Tavg (K)
    tracked_names: tuple = ("Tavg (K)",)
    obs_target_index: int = 5  # Tref, the program at this load (K)

    # Ex-core ion chambers and hot- and cold-leg RTDs (PHYSICS.md, Readings). The
    # flux figure is its upper bound at full power, held at every power.
    readings: tuple = (
        FieldReading("n", sigma=1.9e-4),
        FieldReading("T_coolant", sigma=0.045),
        FieldReading("T_cold", sigma=0.045),
        # Two ex-core sections, each as noisy as the flux reading.
        FieldReading("delta_I", sigma=2.7e-4),
    )

    def __init__(self, integration_method: str = "tr_bdf2_4"):
        self.obs_shape = (13,)
        self.integration_method = integration_method

    @property
    def default_params(self) -> ReactorParams:
        # The scored configuration: env steps of 30 s (``control_period``
        # physics sub-steps each) over the 24 h of a load-follow day.
        return ReactorParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def reward_terms(self, state, params, tripped=None):
        """Cost terms at one physics sub-step. ``step_env`` returns their exact
        sum over the control period in ``info["reward_terms"]``."""
        return compute_reward_terms(state, params, tripped=tripped)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: ReactorState,
        action: jnp.ndarray,
        params: ReactorParams = None,
    ):
        if params is None:
            params = self.default_params

        action, state = actuate(self, state, jnp.reshape(action, (2,)), params)
        action = jnp.clip(action, -1.0, 1.0)
        state = roll_offset_clock(state, params, CONTROL_PERIOD)
        # The boron command joins the queue once a control step.
        state = state.replace(
            boron_queue=jnp.concatenate([action[1:], state.boron_queue[:-1]])
        )
        # A controller gets one reading a step, so every sub-step's measured
        # cost carries the draw of the observation this step returns.
        read_at = state.time + 1

        # Action is held constant for `control_period` physics sub-steps. Reward
        # is accumulated across the sub-steps; we freeze the state on natural
        # termination so the scan can still run for a fixed length under jit.
        #
        # Only *natural* termination is checked inside the scan. The time
        # limit is an env-step count (``state.time``, advanced once below), so
        # it cannot fire mid-period; gymnax's ``is_truncated`` reads it off the
        # returned state. Checking truncation on the sub-step clock is what
        # used to freeze this plant at a tenth of its episode.
        def sub_step(carry, _):
            state, cum_reward, cum_measured, cum_terms, terminated = carry
            candidate, _metrics = compute_next_state(
                action, state, params, integration_method=self.integration_method
            )
            r = compute_reward(candidate, params, xp=jnp)
            terms = compute_reward_terms(candidate, params, xp=jnp)
            term = self.is_terminated(candidate, params)
            measured = measured_state(self, candidate.replace(time=read_at), params)
            m = R.total(self.reward_terms(measured, params, tripped=term))
            # Freeze the plant once terminated; still accumulate the final-step
            # reward. The `~terminated` guard records only the first crossing:
            # once frozen, `term` keeps firing on the held state. The outside
            # world runs on through the period, so a policy that trips meets the
            # load and the wobble one that did not would (long-strata S2).
            world = {f: getattr(candidate, f) for f in self.OUTSIDE_WORLD}
            next_state = jax.tree.map(
                lambda a, b: jnp.where(terminated, a, b),
                state.replace(**world),
                candidate,
            )
            live = jnp.logical_not(terminated)
            next_reward = cum_reward + jnp.where(live, r, 0.0)
            next_measured = cum_measured + jnp.where(live, m, 0.0)
            next_terms = {
                k: cum_terms[k] + jnp.where(live, v, 0.0) for k, v in terms.items()
            }
            next_terminated = terminated | term
            carry = next_state, next_reward, next_measured, next_terms, next_terminated
            return carry, None

        # The accumulators take the state's precision: float32 as scored, and
        # float64 under ``enable_x64`` for a planner that rolls the plant so.
        zero = jnp.zeros((), jnp.result_type(state.n))
        zero_terms = {k: zero for k in compute_reward_terms(state, params, xp=jnp)}
        (new_state, reward, measured, terms, terminated), _ = jax.lax.scan(
            sub_step,
            (state, zero, zero, zero_terms, jnp.bool_(False)),
            xs=None,
            length=CONTROL_PERIOD,
        )
        # One environment step has elapsed, whatever the physics clock did.
        new_state = new_state.replace(
            time=state.time + 1,
            boron_meter=jnp.concatenate(
                [new_state.boron[None], new_state.boron_meter[:-1]]
            ),
        )
        new_state = remember(self, state, new_state, params)
        standby = timed_out(new_state, params)
        # A SCRAM is part of the kernel: the step that leaves the envelope is
        # charged the trip cost and the plant restarts at once, or is lost
        # with ``params.continuing = False`` (``base.failure_kernel``). On that step the cost is the trip cost
        # and nothing else, whatever the sub-steps summed to.
        new_state, tripped, _, terminated = failure_kernel(
            self, key, state, new_state, params
        )
        new_state = self.refuel(key, new_state, params)
        restart = jnp.where(standby, params.standby_steps, params.restart_steps)
        trip = restart * params.failure_cost
        # The trip's dollars are version 2's. Version 1 has no trip cost, as in the
        # shared step (``base``): its reward stays the legacy score of the state.
        charged = jnp.logical_and(tripped, params.reward_version != 1)
        reward = jnp.where(charged, -trip, reward)
        measured = jnp.where(charged, -trip, measured)
        terms = {k: jnp.where(tripped, 0.0, v) for k, v in terms.items()}
        terms["failure"] = jnp.where(tripped, trip, terms["failure"])

        # Mean over the sub-steps rather than the sum. The action is held for
        # ``CONTROL_PERIOD`` physics sub-steps, and summing made one environment
        # step of this plant worth several times a step of any other -- its
        # per-step reward peaked near 4.7 where every other environment caps
        # around 1.0, which is misleading the moment returns are read across
        # environments. Dividing by a positive constant leaves the optimal policy
        # and every within-environment comparison untouched.
        #
        # Terminating mid-period still costs: only the sub-steps before the stop
        # contribute to the sum, so the mean falls with them.
        #
        # Version 2 sums instead: its terms are dollars per physics sub-step, so
        # the sum is the cost of the whole 30 s control step, the unit every
        # other plant's step reward is in.
        obs = self.get_obs(new_state, params)
        reward = jnp.where(params.reward_version == 1, reward / CONTROL_PERIOD, reward)
        # The version-2 terms summed over the sub-steps, exactly as the reward
        # is, for ``target_gym.eval``'s tracking / running split.
        return (
            obs,
            new_state,
            reward,
            terminated,
            {
                "last_state": new_state,
                "reward_terms": terms,
                "tripped": tripped,
                "trip_cost": charged_trip_cost(tripped, params),
                # Version 1 has no terms, so there it is the true reward.
                "measured_reward": R.select(params.reward_version, reward, measured),
                "applied_action": applied_action(self, state, params),
            },
        )

    @measured_obs
    def get_obs(self, state: ReactorState, params: ReactorParams = None):
        if params is None:
            params = self.default_params
        return get_obs(state, params=params)

    def is_terminated(self, state: ReactorState, params: ReactorParams) -> jnp.ndarray:
        """Natural termination only; the time limit is gymnax's ``is_truncated``."""
        terminated, _ = check_is_terminal(state, params)
        return terminated

    @canonical_reset
    def reset_env(
        self, key: chex.PRNGKey, params: ReactorParams = None
    ) -> Tuple[jnp.ndarray, ReactorState]:
        if params is None:
            params = self.default_params

        keys = jax.random.split(key, 6)
        key, down_key, hold_key, demand_key, history_key, delay_key = keys

        # The day: when the load starts down, and how long it holds.
        down_at = jax.random.uniform(
            down_key,
            minval=params.ramp_down_range[0],
            maxval=params.ramp_down_range[1],
        )
        hold = jax.random.uniform(
            hold_key, minval=params.hold_range[0], maxval=params.hold_range[1]
        )
        up_at = down_at + (1.0 - params.load_min) / params.load_ramp_rate + hold
        load = scheduled_load(0.0, down_at, up_at, params)

        # Xenon and iodine start at the equilibrium of a recent power level, and
        # everything else is lined out for full load on the program, with the
        # rods where that leaves the core critical.
        n_recent = jax.random.uniform(
            history_key,
            minval=params.initial_xenon_power_range[0],
            maxval=params.initial_xenon_power_range[1],
        )
        I_hat, Xe_hat = steady_state_xenon(n_recent, params)
        # Burnup: today's rate, and the dilution earlier days left undone, which
        # lines the day out that far below its program (D9). Folded keys keep the
        # other draws as they were.
        rate = jax.random.uniform(
            jax.random.fold_in(key, 7),
            minval=params.burnup_range[0],
            maxval=params.burnup_range[1],
        )
        carried = rate * jax.random.uniform(
            jax.random.fold_in(key, 8),
            minval=params.carried_days_range[0],
            maxval=params.carried_days_range[1],
        )
        # Yesterday's axial xenon: one end-of-day pair from the table (finding 3).
        tilts = jnp.asarray(params.initial_tilts, dtype=jnp.float32)
        row = jax.random.randint(jax.random.fold_in(key, 9), (), 0, tilts.shape[0])
        follow = (
            jax.random.uniform(jax.random.fold_in(key, 10)) < params.initial_tilt_share
        )
        I_tilt0 = jnp.where(follow, tilts[row, 0], 0.0)
        Xe_tilt0 = jnp.where(follow, tilts[row, 1], 0.0)
        # K: at a fixed power the fuel cools with the coolant, so both feedbacks.
        shift = (
            carried * params.boron_worth / (params.alpha_coolant + params.alpha_fuel)
        )
        T_fuel, T_hot, T_cold, T_sg = (
            T - shift for T in steady_state_temperatures(load, params)
        )
        # Critical on boron: the rods stay parked, whatever xenon asks for.
        needed = rod_balance(T_fuel, 0.5 * (T_hot + T_cold), Xe_hat, params)
        boron = params.boron_ref + needed / params.boron_worth
        delay = jax.random.randint(
            delay_key,
            (),
            params.boron_delay_range[0],
            params.boron_delay_range[1] + 1,
        )

        # The draw's precision: float32 as scored, float64 under enable_x64.
        dtype = down_at.dtype
        as_state = lambda x: jnp.asarray(x, dtype=dtype)  # noqa: E731
        state = ReactorState(
            time=0,
            physics_time=0,
            n=as_state(load),
            C=steady_state_precursors(as_state(load), params),
            T_fuel=as_state(T_fuel),
            T_coolant=as_state(T_hot),
            T_cold=as_state(T_cold),
            T_sg=as_state(T_sg),
            I_hat=as_state(I_hat),
            Xe_hat=as_state(Xe_hat),
            I_tilt=as_state(I_tilt0),
            Xe_tilt=as_state(Xe_tilt0),
            target_n=as_state(load),
            T_ref=as_state(tavg_program(load, params)),
            load_wobble=as_state(0.0),
            ramp_down_at=down_at,
            ramp_up_at=up_at,
            demand_key=demand_key,
            rod_in=as_state(params.rod_park),
            rho_ext=as_state(rod_worth(params.rod_park, params)),
            rho_ext_cmd=as_state(0.0),
            boron=as_state(boron),
            delta_I=as_state(
                load
                * flux_tilt(
                    Xe_tilt0,
                    Xe_hat,
                    load,
                    rod_tilt(params.rod_park, params)
                    + moderator_tilt(T_hot, T_cold, params),
                    params,
                )
            ),
            boron_queue=jnp.zeros(BORON_QUEUE, dtype),
            boron_delay=delay,
            rod_moved=as_state(0.0),
            effluent=as_state(0.0),
            xenon_power=as_state(n_recent),
            burn_rate=as_state(rate / 86400.0),
            boron_meter=jnp.full(BORON_METER_STEPS + 1, boron, dtype),
            plan_t0=jnp.zeros((), jnp.int32),
            plan_rate=as_state(params.load_ramp_rate),
            plan_depth=as_state(1.0 - params.load_min),
            ao_hours=jnp.zeros(OFFSET_HOURS, dtype),
            ao_mark=as_state(0.0),
            burn_t0=jnp.zeros((), jnp.int32),
        )

        obs = self.get_obs(state)
        return obs, state

    def start_event(
        self, state: ReactorState, key: chex.PRNGKey, params: ReactorParams
    ) -> ReactorState:
        """The stream's event hook (docs/proposals/long-strata.md, S1): the
        dispatch announces a load-follow day. The load starts down a time drawn
        in ``day_notice_range`` from now, at the scenario's rate to its minimum,
        holds a time drawn in ``hold_range``, and comes back at the same rate, as
        the window's one day does. The day carries its rate and depth, and the
        plan's clock is rebased on now.

        One day's program at a time: a day drawn while one is under way, from its
        announcement until its return ramp has ended, is absorbed (ours).
        """
        p = params
        now = plan_seconds(state.physics_time, state, p)
        end = state.ramp_up_at + state.plan_depth / state.plan_rate
        busy = now < end
        k_down, k_hold = jax.random.split(key)
        down = jax.random.uniform(
            k_down, minval=p.day_notice_range[0], maxval=p.day_notice_range[1]
        )
        hold = jax.random.uniform(
            k_hold, minval=p.hold_range[0], maxval=p.hold_range[1]
        )
        rate = jnp.asarray(p.load_ramp_rate, down.dtype)
        depth = jnp.asarray(1.0 - p.load_min, down.dtype)
        day = state.replace(
            plan_t0=jnp.asarray(state.physics_time, jnp.int32),
            ramp_down_at=down,
            ramp_up_at=down + depth / rate + hold,
            plan_rate=rate,
            plan_depth=depth,
        )
        return jax.tree_util.tree_map(
            lambda old, new: jnp.where(busy, old, new), state, day
        )

    def restart_plant(
        self, key: chex.PRNGKey, state: ReactorState, params: ReactorParams
    ) -> ReactorState:
        """A trip's restart (long-strata R8; owner, 2026-11-11 and 2026-11-12): the
        plant comes back at reduced power once the lump's dead time has passed, and
        the turbine reloads.

        The trip's lump prices the dead time only: ``restart_steps`` after a scram,
        ``standby_steps`` after a timed limit's hot standby. The plant restarts at
        ``restart_load``, or the plan's load if lower, lined out on its program with
        the rods parked, and critical on boron with the burnup since the last
        refuelling diluted, as operators line a restart out. Iodine and xenon, means
        and tilts, are what the dead time at zero power left
        (``xenon_after_shutdown``). The turbine then takes at most ``pickup``, which
        climbs at the conditioning ramps until it meets the plan
        (``env.compute_next_state``); the plan it does not take is charged as lost
        output in tracking. The turbine's load and Tref start at the restart load.
        The core's burn rate, boron delay and cycle point are the plant's own and
        stay. The xenon reading (obs 10) is the power whose balance is that xenon,
        1 above full power's (``balance_power``). The rest is as at reset, and so is
        the xenon of a solve that overflowed.
        """
        p = params
        _, fresh = self.reset_env(key, p)
        steps = jnp.where(timed_out(state, p), p.standby_steps, p.restart_steps)
        seconds = steps * (p.delta_t * CONTROL_PERIOD)
        # A solve that overflowed is a lost plant with no xenon to carry: it
        # restarts from the reset's balance instead.
        species = ("I_hat", "Xe_hat", "I_tilt", "Xe_tilt")
        lost = ~jnp.all(jnp.isfinite(jnp.stack([getattr(state, f) for f in species])))
        I_hat, Xe_hat, I_tilt, Xe_tilt = (
            jnp.where(lost, getattr(fresh, f), getattr(state, f)) for f in species
        )
        I_hat, Xe_hat = xenon_after_shutdown(I_hat, Xe_hat, seconds, p)
        I_tilt, Xe_tilt = xenon_after_shutdown(I_tilt, Xe_tilt, seconds, p)
        plan = planned_load(plan_seconds(state.physics_time, state, p), state)
        load = jnp.minimum(plan, p.restart_load)
        T_fuel, T_hot, T_cold, T_sg = steady_state_temperatures(load, p)
        needed = rod_balance(T_fuel, 0.5 * (T_hot + T_cold), Xe_hat, p)
        boron = p.boron_ref + needed / p.boron_worth - burnt(state, p)
        dtype = fresh.n.dtype
        as_state = lambda x: jnp.asarray(x, dtype=dtype)  # noqa: E731
        eps = rod_tilt(p.rod_park, p) + moderator_tilt(T_hot, T_cold, p)
        hot = fresh.replace(
            n=as_state(load),
            C=steady_state_precursors(as_state(load), p),
            T_fuel=as_state(T_fuel),
            T_coolant=as_state(T_hot),
            T_cold=as_state(T_cold),
            T_sg=as_state(T_sg),
            I_hat=as_state(I_hat),
            Xe_hat=as_state(Xe_hat),
            I_tilt=as_state(I_tilt),
            Xe_tilt=as_state(Xe_tilt),
            delta_I=as_state(load * flux_tilt(Xe_tilt, Xe_hat, load, eps, p)),
            boron=as_state(boron),
            boron_meter=jnp.full(BORON_METER_STEPS + 1, boron, dtype),
            xenon_power=as_state(balance_power(Xe_hat, p)),
            burn_rate=state.burn_rate,
            boron_delay=state.boron_delay,
            burn_t0=state.burn_t0,
            pickup=as_state(load),
            target_n=as_state(load + state.load_wobble),
            T_ref=as_state(tavg_program(load + state.load_wobble, p)),
        )
        return canonical_dtypes(hot)

    def refuel(self, key: chex.PRNGKey, state: ReactorState, params: ReactorParams):
        """A stream's refuelling (owner, 2026-11-11): every ``renewal_days`` the
        core is renewed at its cycle point as at reset, uncharged, the outside
        world carrying on; burnup counts again from there. A window (0) never
        refuels."""
        p = params
        days = jnp.asarray(p.renewal_days)
        per = jnp.maximum(jnp.round(days * 86400.0 / (p.delta_t * CONTROL_PERIOD)), 1)
        time = jnp.asarray(state.time, jnp.int32)
        due = (days > 0) & (time > 0) & (time % per.astype(jnp.int32) == 0)
        _, fresh = self.reset_env(jax.random.fold_in(key, _REFUEL), p)
        # The core is renewed, not the turbine: its load, Tref and reload carry on.
        turbine = ("target_n", "T_ref", "pickup")
        world = {f: getattr(state, f) for f in (*self.OUTSIDE_WORLD, *turbine)}
        fresh = fresh.replace(
            time=state.time,
            sensor_key=state.sensor_key,
            burn_t0=jnp.asarray(state.physics_time, jnp.int32),
            **world,
        )
        return jax.tree_util.tree_map(
            lambda new, old: jnp.where(due, new, old), fresh, state
        )

    def action_space(self, params: ReactorParams | None = None) -> spaces.Box:
        return spaces.Box(
            low=-jnp.ones(2),
            high=jnp.ones(2),
            shape=(2,),
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

    def render(self, screen, state: ReactorState, params: ReactorParams, frames, clock):
        frames, screen, clock = self.render_reactor(
            screen, state, params, frames, clock
        )
        return frames, screen, clock
