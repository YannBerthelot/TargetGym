"""The oracle (the MPC slot) for the HVAC task, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem, by the
smoothing protocol's inventory (docs/proposals/oracle-smoothing-protocol.md,
section 5.4), run by ``target_gym.oracles.controller.Oracle``:

- model: the env's ``compute_velocity`` and integrator on the mass and slab
  core (``velocity``), the move a supply water temperature inside the valve's
  range (T1); the weather's deviation, updated before the step as the env
  does: on its noise-free mean path in the registry's receding slot, and on
  the episode's own path for the ceiling (``ceiling_data``), which knows the
  weather ahead as a ceiling may; the clock, sun and gains along the plan;
- objective: the heat pump's electricity at the tariff and the comfort
  hinges through epigraph rows on every occupied state (T2): the room's lost
  work past each band edge as the published curve's terms in the distance past
  it (``curve_terms``, ``comfort_rows``; owner, 2026-11-28), and the floor's
  kelvin-hours above its limit at its price;
- window (owner, 2026-11-25, the review's finding 1): the registry's oracle
  plans on the task's scored window alone (``scenarios.hvac_evaluation``), as
  the task scores its lines, so the ceiling bounds the scored cost. A step
  outside it costs the plan nothing. The evaluation's edge charge (owner,
  2026-11-27, the RL research review's H2) is a fourth hinge, once: the
  store's value at the window's start (``env.stored_value``, the dearest heat
  price) less its value at the window's end, or at the plan's end where that
  comes first. The plan carries the start's value as a third state, read as
  the plan passes the window's start, or given as ``start_store`` for a plan
  that starts inside it (the receding oracle reads it as it passes). The
  hinge is a surrogate: the env never charges it, the evaluation does, so
  K8's replay compares the reward's cost alone (``replay_cost``), inside the
  window, nothing outside;
- smoothed site (T5): the slab heat's two edges in ``slab_heat``, heating
  only (``max(T_sup - T_core, 0)``) and the source's size [13], which enter the
  dynamics and so cannot be epigraph rows: K2's softplus at the solve's width
  on both, the exact edges at width 0,
  where the planned cost of a plan is the env's
  (``tests/oracles/test_thermal_energy_oracles.py``). Water at the valve's
  lowest setting, which the env counts as off, is below the core, so it gives
  no heat either way;
- no limit trips (T3) and no dropped guard (T4).

Frost protection is not planned: the band keeps the room far above it.

The observation-only MPC (``observation_only_mpc``; v1 item 5, as the owner
read it on 2026-11-28) is this planner fed the shared estimator's belief
(``estimation.Estimator``) in place of the true state, with one change to its
objective: comfort is priced at its expected cost under the weather's forecast
error, not on the mean path. A plan on the mean path holds the room on the band's
edge, where the weather it did not forecast leaves it below half the time, and the
slab cannot correct within the day. The room's error at each lead is the plant's
own weather innovations to come, carried to the room by the building at a held
supply (``room_spread``); the expected kelvin-hours outside the band are those of
a normal error of that spread (``expected_excess``). This is the chance-constrained
MPC of building climate control (Oldewurtel et al., Energy and Buildings 45, 2012,
15-27) written as an expected cost; nothing in it is tuned.
"""

import math

import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial

from target_gym.hvac.env import (
    PERFORMANCE_CURVE,
    WEATHER_OU_THETA,
    drift_over,
    electricity_price,
    floor_surface_temperature,
    heat_pump_cop,
    internal_gain,
    is_occupied,
    outdoor_temperature,
    shortfall_price,
    slab_conductances,
    solar_gain,
    solve_air_and_surface,
    source_size,
    split_gains,
    steady_state,
    stored_heat,
    stored_value,
    zone_conductances,
)
from target_gym.hvac.scenarios import hvac_evaluation
from target_gym.integration import integrate_dynamics
from target_gym.oracles.controller import (
    Oracle,
    TaskModel,
    enable_x64,
    float64,
    raw_of,
    scored_run,
    softplus_clip,
)
from target_gym.oracles.shooting import Hinge
from target_gym.utils import ou_decay, stationary_innovation


def heat(T_supply, T_core, p, width):
    """``slab_heat`` with its two edges smoothed at ``width`` (K): heating only,
    and no more than the source's size."""
    H_water = slab_conductances(p)["H_water"]
    most = source_size(p) / H_water
    return H_water * softplus_clip(T_supply - T_core, 0.0, most, width)


def velocity(position, action, time, weather_dev, width, p):
    """``compute_velocity`` with the slab heat of ``heat``."""
    T_mass, T_core = position[0], position[1]
    c, s = zone_conductances(p), slab_conductances(p)
    T_out = outdoor_temperature(time, weather_dev, p)
    phi_ia, phi_st, phi_m = split_gains(internal_gain(time, p), solar_gain(time, p), p)
    T_air, T_surface = solve_air_and_surface(T_mass, T_core, T_out, phi_st, phi_ia, p)
    dT_mass = (
        c["H_tr_ms"] * (T_surface - T_mass) + c["H_tr_em"] * (T_out - T_mass) + phi_m
    ) / c["C_m"]
    dT_core = (heat(action, T_core, p, width) + s["H_slab"] * (T_air - T_core)) / s[
        "C_slab"
    ]
    return jnp.stack([dT_mass, dT_core]), None


def expected_power(d, spread, power: int):
    """E[max(d + spread Z, 0) ** power] for a standard normal Z, power 1 to 3:
    the expected excess over a limit, and its square and cube, of a quantity
    whose mean exceeds the limit by ``d`` and whose error has ``spread``: the
    normal's partial moments. Smooth in ``d``; ``spread`` must be positive."""
    z = d / spread
    up, dens = jax.scipy.stats.norm.cdf(z), spread * jax.scipy.stats.norm.pdf(z)
    if power == 1:
        return d * up + dens
    if power == 2:
        return (d * d + spread * spread) * up + d * dens
    if power == 3:
        return (d**3 + 3.0 * d * spread * spread) * up + (
            d * d + 2.0 * spread**2
        ) * dens
    raise ValueError(f"power is 1, 2 or 3, not {power}")


def expected_excess(d, spread):
    """E[max(d + spread Z, 0)]: ``expected_power`` of the first power."""
    return expected_power(d, spread, 1)


def curve_terms(p) -> tuple[tuple, tuple]:
    """The reward's lost work past each band edge (``env.work_lost``) as powers
    of the distance d past it: ``P(edge) - P(edge -+ d) = sum_k w_k d^k``, the
    curve's Taylor terms at the edge, exact for its cubic. ``(below, above)``,
    each ``(w_1, w_2, w_3)``, shares of an hour's work per kelvin to the k."""
    c1, c2, c3, _ = PERFORMANCE_CURVE
    low, high = float(p.band_low), float(p.band_high)

    def slope(T):
        return c1 + 2.0 * c2 * T + 3.0 * c3 * T * T

    below = (slope(low), -(c2 + 3.0 * c3 * low), c3)
    above = (-slope(high), -(c2 + 3.0 * c3 * high), -c3)
    return below, above


def comfort_rows(p) -> tuple[tuple, tuple]:
    """The planner's hinge weights for the lost work past each edge, from
    ``curve_terms``: below, the three; above, the first two, since the cubic's
    term there is negative and a hinge cannot carry it, which overstates the
    fall by 0.0000623 d^3 of an hour's work, 0.5 % of it 1 K past the edge and
    1.8 % 2 K past it, a room this heating task never reaches. The span the curve is read on (15 to 35 C) is left out: no plan
    goes there."""
    below, above = curve_terms(p)
    return below, above[:2]


def room_spread(p, steps: int):
    """The room temperature's forecast error, K, at each lead of 1 to ``steps``
    task steps, for a planner on the weather deviation's mean path: the plant's own
    innovations to come (``WEATHER_OU_THETA``, ``T_out_noise_std``), each decaying
    as the deviation does, carried to the room by the building at a held supply,
    open loop. The network is linear at a held supply while the slab is heated, so
    with ``h(m)`` the room's response m steps after a unit innovation the error at
    lead k has variance ``eta^2 sum_{m<k} h(m)^2``, ``eta`` the innovation's spread.
    ``p`` are the plant's params, noise included. The supply is the one that holds
    the band's low edge at the outdoor mean; at any heating supply the response is
    the same."""
    decay = ou_decay(WEATHER_OU_THETA, float(p.delta_t), xp=math)
    eta = stationary_innovation(float(p.T_out_noise_std), decay)
    T_m, T_c, _, T_sup = steady_state(p, float(p.T_out_mean), float(p.band_low))

    def run(bump):
        def step(x, t):
            dev = bump * decay**t
            x1, _ = integrate_dynamics(
                positions=x, delta_t=p.delta_t, method="exp_1",
                compute_velocity=partial(
                    velocity, action=T_sup, time=t, weather_dev=dev, width=0.0, p=p
                ),
                drift=drift_over,
            )  # fmt: skip
            now = t + 1
            phi_ia, phi_st, _ = split_gains(
                internal_gain(now, p), solar_gain(now, p), p
            )
            T_air, _ = solve_air_and_surface(
                x1[0], x1[1], outdoor_temperature(now, dev, p), phi_st, phi_ia, p
            )
            return x1, T_air

        return jax.lax.scan(step, jnp.stack([T_m, T_c]), jnp.arange(steps))[1]

    h = run(1.0) - run(0.0)
    return eta * jnp.sqrt(jnp.cumsum(h**2))


def scored_window() -> tuple[int, int]:
    """The task's scored steps, ``(first, end)``."""
    ev = hvac_evaluation()
    return ev["unscored_steps"], ev["unscored_steps"] + ev["scored_steps"]


def hvac_model(env, p, window=None, start_store=None, spread=None) -> TaskModel:
    """The plan's model; ``window`` ``(first, end)`` costs only the steps in it
    and the store's shortfall at its end (every step and no edge when None).
    ``start_store()`` gives the store's value at the window's start, in euros,
    for a plan that starts inside the window. ``spread``, the room's forecast
    error at each lead (``room_spread``), prices comfort at its expected cost under
    that error in place of the hinges on the mean path; the floor's error is the
    room's share of it."""
    hours = p.delta_t / 3600.0
    work = float(p.work_price * hours)  # an occupied step's work, euros
    floor_weight = float(p.comfort_price * hours)
    below, above = curve_terms(p)
    rows_below, rows_above = comfort_rows(p)
    # The deviation's mean per step, a plain float whatever the run's precision.
    decay = ou_decay(WEATHER_OU_THETA, float(p.delta_t), xp=math)
    price = shortfall_price(p)

    def value(T_mass, T_core):
        """The store's value, as the evaluation reads it (``env.stored_value``)."""
        return price * stored_heat(T_mass, T_core, p)

    def step(d, x, u, t):
        start, deviation, path = d["task"]
        widths = jnp.asarray(d["widths"])
        width = widths[0] if widths.shape[0] else jnp.zeros((), x.dtype)
        T_sup = p.T_supply_min + 0.5 * (u[0] + 1.0) * (p.T_supply_max - p.T_supply_min)
        if path is None:
            dev = deviation * decay ** (t + 1)
        else:  # the episode's own weather, t + 1 steps after the solve's start
            dev = path[jnp.minimum(t, path.shape[0] - 1)]
        x1, _ = integrate_dynamics(
            positions=x[:2], delta_t=p.delta_t, method=env.integration_method,
            compute_velocity=partial(
                velocity, action=T_sup, time=start + t, weather_dev=dev,
                width=width, p=p,
            ),
            drift=drift_over,
        )  # fmt: skip
        now = start + t + 1
        phi_ia, phi_st, _ = split_gains(internal_gain(now, p), solar_gain(now, p), p)
        T_air, _ = solve_air_and_surface(
            x1[0], x1[1], outdoor_temperature(now, dev, p), phi_st, phi_ia, p
        )
        T_floor = floor_surface_temperature(x1[1], T_air, p)
        occupied = is_occupied(now, p)
        if window is None:
            inside, held, edge = True, x[2], -jnp.ones((), x.dtype)
        else:
            inside = (start + t >= window[0]) & (start + t < window[1])
            # The store at the window's start, read as the plan passes it, and
            # its shortfall at the window's end, or at the plan's where that
            # comes first.
            stored = value(x1[0], x1[1])
            held = jnp.where(now <= window[0], stored, x[2])
            end = d.get("end")
            last = window[1] if end is None else jnp.minimum(window[1], start + end)
            edge = jnp.where((now == last) & (now > window[0]), held - stored, -1.0)
        outside = jnp.stack(
            [p.band_low - T_air, T_air - p.band_high, T_floor - p.floor_surface_max]
        )
        # The env scores the step's heat on the core it ends at, at the COP of
        # the water the source made (``env.supply_made``).
        q = heat(T_sup, x1[1], p, width)
        made = jnp.minimum(T_sup, x1[1] + q / slab_conductances(p)["H_water"])
        kwh = q / heat_pump_cop(made, p) / 1000.0 * hours
        cost = electricity_price(start + t, p) * kwh
        if spread is None:
            rows = jnp.stack(
                [outside[0]] * len(rows_below) + [outside[1]] * len(rows_above)
                + [outside[2]]
            )  # fmt: skip
            y = jnp.where(occupied & inside, rows, -1.0)
        else:
            s = spread[jnp.minimum(t, spread.shape[0] - 1)]
            floor_share = slab_conductances(p)["floor_share"]
            lost = sum(
                w * expected_power(outside[0], s, k)
                for k, w in enumerate(below, start=1)
            ) + sum(
                w * expected_power(outside[1], s, k)
                for k, w in enumerate(above, start=1)
            )
            floor = expected_excess(outside[2], (1.0 - floor_share) * s)
            comfort = work * lost + floor_weight * floor
            cost = cost + jnp.where(occupied, comfort, 0.0)
            y = jnp.zeros((0,), x.dtype)
        x1 = jnp.stack([x1[0], x1[1], held])
        y = jnp.concatenate([y, jnp.reshape(edge, (1,))])
        return x1, jnp.where(inside, cost, 0.0), y, T_air[None], ()

    def state_x(s):
        """The plan's start: the two nodes and the store's value at the window's
        start, the value now where the plan starts at or before it."""
        now = value(s.T_mass, s.T_core)
        if window is None or int(s.time) <= window[0]:
            return jnp.stack([s.T_mass, s.T_core, now])
        if start_store is None or start_store() is None:
            raise ValueError(
                f"hvac: a plan from step {int(s.time)}, inside the scored window "
                f"from step {window[0]}, needs the store's value at the window's "
                "start (start_store)"
            )
        return jnp.stack([s.T_mass, s.T_core, jnp.asarray(start_store(), now.dtype)])

    def replay_cost(s, s1, r):
        """What K8's replay compares of an env step: its reward's cost inside
        the window, nothing outside. The edge's shortfall is the evaluation's
        charge, never the env's, so the plan pays it as a surrogate hinge."""
        if window is None:
            return -r
        inside = (s.time >= window[0]) & (s.time < window[1])
        return jnp.where(inside, -r, 0.0)

    hinges = ()
    if spread is None:
        hinges = (
            *(Hinge(weight=work * w, power=k) for k, w in enumerate(rows_below, 1)),
            *(Hinge(weight=work * w, power=k) for k, w in enumerate(rows_above, 1)),
            Hinge(weight=floor_weight),
        )
    # G10: what an error of half the resolution costs in one step, at the
    # band's low edge, where the heated room sits: the curve's fall over that
    # half step, all three of its terms (curve_terms is exact for the cubic).
    half = 0.5 * float(p.precision_floor)
    tie = work * sum(w * half**k for k, w in enumerate(below, start=1))
    return TaskModel(
        nx=3, nu=1, step=step, state_x=state_x,
        data=lambda s: (s.time, s.weather_dev, None),
        hold=lambda s: raw_of(s.T_supply, p.T_supply_min, p.T_supply_max)[None],
        tracked=lambda s: s.T_air[None],
        resolution=(float(p.precision_floor),),
        # The comfort rows at the curve's work price, and the store's shortfall in
        # euros at the dearest heat price.
        hinges=hinges + (Hinge(weight=1.0, surrogate=True),),
        tie_budget=tie,
        replay_cost=replay_cost,
    )  # fmt: skip


def ceiling_data(env, p, state, step_keys, run_as=scored_run):
    """The data of a plan from ``state`` over the steps of ``step_keys``: the
    weather's deviation after every step, from the episode's step keys. The
    deviation does not depend on the moves, so it is read off a run of the env
    at any move (here the reset's supply). The run is the scored one's, in
    float32 (``scored_run``; a test of the
    model on x64 data passes its own ``run_as``): under x64 the weather draws other values.
    """

    def run(state, p, step_keys):
        def advance(s, k):
            s1 = env.step_env(k, s, jnp.zeros(1, jnp.float32), p)[1]
            return s1, s1.weather_dev

        return jax.lax.scan(advance, state, step_keys)[1]

    path = jnp.asarray(run_as(run, state, p, step_keys), jnp.result_type(state.T_core))
    return state.time, state.weather_dev, path


class WindowOracle(Oracle):
    """The shared solver's receding oracle on a windowed ``hvac_model``. It reads
    the store's value as the run passes the window's start, so a plan solved
    inside the window charges its end against it (``start_store``)."""

    def __init__(self, env, params, name, window, spread=None):
        self.window, self._start = window, None

        def build(env, p):
            return hvac_model(
                env, p, window=window, start_store=lambda: self._start,
                spread=None if spread is None else jnp.asarray(spread, jnp.float64),
            )  # fmt: skip

        super().__init__(env, params, name, build)

    def reset(self):
        super().reset()
        self._start = None

    def step(self, obs, state):
        if int(state.time) == self.window[0]:
            with enable_x64():
                self._start = float(stored_value(float64(state), self.params))
        return super().step(obs, state)


def make_hvac_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``hvac_model``, windowed on
    the task's scored window."""
    return WindowOracle(env, params, "hvac", scored_window())


#: What the observation-only MPC's estimator tracks: the building's two nodes, the
#: room and outdoor air the sensors read, and the weather's deviation, which the
#: day-ahead forecast shows.
OBSERVED_FIELDS = ("T_mass", "T_core", "T_air", "T_out", "weather_dev")
#: The room, outdoor and core sensors and the forecast (``env.get_obs``).
OBSERVED_CHANNELS = (0, 1, 3, 9)


def observation_only_mpc(env, params, plan):
    """The MPC on the true model that reads the observation only (v1 item 5), the
    line ``scripts/record_v1.py`` records: the window planner, pricing comfort at
    its expected cost under the weather's forecast error (``room_spread``), fed the
    shared estimator's belief (``estimation.ObservationOnly``). ``params`` is the
    scenario's configuration, noise included, whose noise the estimator and the
    spread read; ``plan`` the planner's (its noise zeroed, ``plan_params``), on
    which it plans and the estimator predicts. A host policy:
    ``policy.reset()``, then ``policy(obs)``."""
    from target_gym.estimation import Estimator, ObservationOnly

    # The spread at every lead a plan can reach: the episode bounds the horizon.
    steps = int(params.max_steps_in_episode)
    oracle = WindowOracle(
        env, plan, "hvac", scored_window(), spread=room_spread(params, steps)
    )
    estimator = Estimator(
        env, params, OBSERVED_FIELDS, channels=OBSERVED_CHANNELS, model=plan
    )
    return ObservationOnly(oracle, estimator)
