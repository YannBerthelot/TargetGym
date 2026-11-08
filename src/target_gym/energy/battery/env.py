"""
Grid battery: a Megapack-sized LFP unit selling a frequency regulation service
under GB dynamic-service rules, steered only by its declared baseline.

See ``PHYSICS.md`` in this directory for every number's source, and
``REALITY.md`` for the plant record each part rests on.

Model
-----
A Tesla Megapack 2 XL, 2-hour configuration (1927 kW, 3854 kWh, 92.0 % round
trip including auxiliaries, REALITY.md sec. 3), built of EVE LF280K LFP cells
(3.2 V, 280 Ah, 2.5 to 3.65 V). Stored energy follows CAISO's market model::

    E(t + dt) = E(t) - dt (P_dis + eta P_ch)          (CAISO-SOC eq. 1)

with ``eta`` the round-trip efficiency, and the BMS refuses power past the
cells' voltage window and temperature ranges (protected limits, rule 14).

The property: nobody measures the state of charge. The BMS reports its own
estimate, the true value plus an initial error, an Ah-counting drift and a
temperature shift of the OCV curve, none of which it can correct on the flat
LFP plateau (ARX-LFP, MPS-ACCURE).

Task
----
One 4 h EFA block of Dynamic Regulation. The unit delivers its Operational
Baseline plus the contracted response to a regulation signal; it may not
under-deliver. Its only lever is the baseline, changeable for a settlement
period until 60 min before it starts (NESO-ST para. 6.11 to 6.13, NESO-GUIDE
p. 36). It is paid for availability per 30 min settlement period, lost when
the reported stored energy misses the period's requirement at its start or
when the unit fails to deliver for 0.1 % of the period (NESO-ST Schedule 3).
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym import reward as R
from target_gym.base import EnvParams, EnvState, actuator_field
from target_gym.utils import convert_raw_action_to_range

#: Settlement periods a baseline array holds: one day, so a long run (the 24 h
#: neutrality check) has a period for every step. A 4 h block uses 8.
N_PERIODS = 48


@struct.dataclass
class BatteryParams(EnvParams):
    # ---- Time (PHYSICS.md, Episode) ----
    # Data every second for regulation (PNNL-ESS Foreword); one EFA block of
    # 4 h (NESO-SOE Table 1).
    delta_t: float = 1.0
    max_steps_in_episode: int = 14400

    # ---- Unit: Megapack 2 XL, 2-hour (TESLA-MP2XL) ----
    power_max: float = 1.927e6  # W, charge and discharge
    energy_capacity: float = 3854.0e3 * 3600.0  # J
    round_trip_efficiency: float = 0.920  # full depth, with auxiliaries, 25 degC

    # ---- Cell: EVE LF280K (EVE-LF280K) ----
    cell_capacity_Ah: float = 280.0
    cell_nominal_V: float = 3.2
    cell_resistance: float = 0.25e-3  # ohm, AC 1 kHz, item 5
    V_cell_min: float = 2.5
    V_cell_max: float = 3.65
    T_charge_min: float = 0.0  # degC
    T_charge_max: float = 55.0
    T_discharge_min: float = -20.0
    T_discharge_max: float = 55.0
    # LFP plateau: 1 mV of OCV is nearly 5 % SoC (ARX-LFP eq. 5).
    ocv_plateau_slope: float = 0.02  # V per unit SoC
    ocv_knee_width: float = 0.03  # SoC, ours: keeps the knees outside 20-80 %

    # ---- Cell temperature ----
    # Held by the unit's thermal management at the rating temperature; no
    # thermal model is sourced (PHYSICS.md, deviations).
    T_cell_start: float = 25.0  # degC

    # ---- BMS estimate (ARX-LFP, MPS-ACCURE) ----
    soc_error_std: float = 0.075  # +-15 % read as 2 sigma
    soc_drift_rate: float = 0.10 / (14.0 * 86400.0)  # 1/s: 10 % after 14 days
    soc_temp_coeff: float = 0.0051  # per degC, on the 25 degC curve
    T_bms_curve: float = 25.0  # degC

    # ---- Service: Dynamic Regulation (NESO-SOE, NESO-ST, NESO-GUIDE) ----
    reserve_fraction: float = 0.4  # DR, of the offered quantity
    volume_hours: float = 1.0  # DR's contracted response energy, h
    recovery_fraction: float = 0.2  # of the volume per settlement period
    period_seconds: float = 1800.0
    gate_closure_seconds: float = 3600.0
    unavailable_fraction: float = 0.001  # of a period, Schedule 3
    #: Below this a delivered power counts as the required one (W): rounding
    #: of float32 power, ours.
    shortfall_tolerance: float = 1.0

    # ---- Regulation signal, p.u. of the contracted power ----
    signal_interval: float = 2.0  # s, PJM's scan rate (PJM-M12 sec. 4.3.2)
    signal_tau: float = 300.0  # s, ours
    signal_std: float = 0.4  # p.u., ours
    neutrality_tau: float = 1800.0  # s, ours: the pull that keeps it neutral

    # ---- Prices (rule 23), GBP ----
    #: Day-ahead price of a normal period, and the value of stored energy: the
    #: GB 2024 mean (Electricity Maps, REALITY.md sec. 7).
    price_base: float = 74.0  # GBP/MWh
    #: One price event per block, ``price_event_periods`` periods long (one
    #: hourly price), its level uniform in this range and its first period
    #: uniform in ``price_event_first``. The normal block's range is the base
    #: price itself, so it has no event (scenarios.py).
    price_event_range: Tuple[float, float] = (74.0, 74.0)  # GBP/MWh
    price_event_periods: int = 2
    price_event_first: Tuple[int, int] = (3, 6)  # ours: inside the block
    #: Dynamic Regulation's availability price, Jan 2024 average (Modo).
    availability_price: float = 4.36  # GBP/MW/h

    # ---- Long one-way deviation (scenarios.py) ----
    #: Response, p.u., held one way: DR's answer to a 50 to 100 mHz mean
    #: deviation (ENTSOE-LER sec. 1, NESO-GUIDE p. 4). 0 in a normal block.
    bias_pu_range: Tuple[float, float] = (0.0, 0.0)
    bias_duration_range: Tuple[float, float] = (1800.0, 5400.0)  # s, ours

    # ---- Score (rule 15), in GBP ----
    reward_version: int = 3
    #: No limit of this plant trips (PHYSICS.md, Limits), so a trip costs 0.
    restart_steps: int = 0
    failure_cost: float = 0.0
    # Kept for the KPI tables (benchmark._OUTPUT_FLOORS); the score does not
    # read them. The meter's scatter (PHYSICS.md, Readings).
    e_floor: float = 330.0  # W
    precision_floor: float = 330.0  # W


#: Units of the BatteryParams fields, for the generated environment pages.
PARAM_UNITS = {
    "power_max": "W",
    "energy_capacity": "J",
    "round_trip_efficiency": "-",
    "cell_capacity_Ah": "A h",
    "cell_nominal_V": "V",
    "cell_resistance": "ohm",
    "V_cell_min": "V",
    "V_cell_max": "V",
    "T_charge_min": "degC",
    "T_charge_max": "degC",
    "T_discharge_min": "degC",
    "T_discharge_max": "degC",
    "ocv_plateau_slope": "V",
    "ocv_knee_width": "-",
    "T_cell_start": "degC",
    "soc_error_std": "-",
    "soc_drift_rate": "1/s",
    "soc_temp_coeff": "1/degC",
    "T_bms_curve": "degC",
    "reserve_fraction": "-",
    "volume_hours": "h",
    "recovery_fraction": "-",
    "period_seconds": "s",
    "gate_closure_seconds": "s",
    "unavailable_fraction": "-",
    "shortfall_tolerance": "W",
    "signal_interval": "s",
    "signal_tau": "s",
    "signal_std": "-",
    "neutrality_tau": "s",
    "price_base": "GBP/MWh",
    "price_event_range": "GBP/MWh",
    "price_event_periods": "periods",
    "price_event_first": "period",
    "availability_price": "GBP/(MW h)",
    "bias_pu_range": "-",
    "bias_duration_range": "s",
    "reward_version": "-",
    "restart_steps": "steps",
    "failure_cost": "GBP",
    "e_floor": "W",
    "precision_floor": "W",
}


@struct.dataclass
class BatteryState(EnvState):
    soc: float  # true state of charge, E / capacity -- HIDDEN
    soc_error: float  # BMS estimate minus truth at reset, at T_bms_curve -- HIDDEN
    soc_drift: float  # the estimate's Ah-counting drift, 1/s -- HIDDEN
    T_cell: float  # degC
    #: The BMS's report, set by the dynamics from the true cell temperature, so
    #: a temperature reading's scatter does not enter it.
    soc_reported: float
    v_cell: float  # cell terminal voltage, V
    power: float  # delivered at the meter, W, positive = discharge
    target_power: float  # required now: baseline plus response, W
    regulation: float  # the contracted response now, W
    signal_x: float  # the signal's unclipped value, p.u. -- HIDDEN
    signal_energy: float  # integral of the signal off its mean, p.u. s -- HIDDEN
    bias_pu: float  # the long deviation's response, p.u. -- HIDDEN
    bias_start: float  # s -- HIDDEN
    bias_end: float  # s -- HIDDEN
    prices: jax.Array  # GBP/MWh per settlement period, day-ahead
    baselines: jax.Array  # W per settlement period, declared
    period_response: float  # response energy this period, J, + = export
    deficit_export: float  # export energy not yet recovered, J
    deficit_import: float  # import energy not yet recovered, J
    short_seconds: float  # seconds this period the unit fell short
    unavailable: float  # 1 once this period is lost
    availability_cost: float  # GBP of availability lost on the last step
    losses_cost: float  # GBP of energy lost on the last step
    trade_cost: float  # GBP the last step's energy cost against the base price

    actuator_held: jax.Array = actuator_field(1)


def contracted_power(params: BatteryParams):
    """The Dynamic Regulation quantity, W: the largest the unit can offer and
    still hold back the 40 % reserve for SoC recovery (NESO-SOE sec. 3)."""
    return params.power_max / (1.0 + params.reserve_fraction)


def baseline_limit(params: BatteryParams):
    """The largest baseline either way, W: the reserve."""
    return params.reserve_fraction * contracted_power(params)


def response_volume(params: BatteryParams):
    """Contracted Response Energy Volume, J: the quantity for 60 min (DR)."""
    return contracted_power(params) * params.volume_hours * 3600.0


def n_cells(params: BatteryParams):
    """Cells in the unit: its energy over one cell's 3.2 V x 280 Ah."""
    cell_J = params.cell_nominal_V * params.cell_capacity_Ah * 3600.0
    return params.energy_capacity / cell_J


def steps_per_period(params: BatteryParams):
    return jnp.maximum(jnp.round(params.period_seconds / params.delta_t), 1).astype(
        jnp.int32
    )


def period_of(time, params: BatteryParams):
    """The settlement period live at step ``time``."""
    return jnp.clip(time // steps_per_period(params), 0, N_PERIODS - 1)


def cell_ocv(soc, params: BatteryParams):
    """LFP open-circuit voltage per cell: the plateau at the cell's nominal
    3.2 V with ARX-LFP's slope, and knees that reach the voltage window at
    empty and full."""
    p = params
    s = jnp.clip(soc, 0.0, 1.0)
    w = p.ocv_knee_width
    low = p.cell_nominal_V - 0.5 * p.ocv_plateau_slope - p.V_cell_min
    high = p.V_cell_max - p.cell_nominal_V - 0.5 * p.ocv_plateau_slope
    plateau = p.cell_nominal_V + p.ocv_plateau_slope * (s - 0.5)
    return plateau - low * jnp.exp(-s / w) + high * jnp.exp(-(1.0 - s) / w)


def power_limits(soc, T_cell, params: BatteryParams):
    """What the BMS lets through at the meter, W: ``(most discharge, most
    charge)``, each >= 0.

    Power stops at the rating, at the stored energy left, at the cells'
    voltage window (2.5 to 3.65 V at the terminals) and outside each
    direction's temperature range (EVE-LF280K items 4 and 8). The unit's
    losses are lumped into ``eta`` and the cell sees the meter's power,
    divided over its cells (ours).
    """
    p = params
    N = n_cells(p)
    ocv = cell_ocv(soc, p)
    R0 = p.cell_resistance
    v_floor = N * p.V_cell_min * jnp.maximum(ocv - p.V_cell_min, 0.0) / R0
    v_roof = N * p.V_cell_max * jnp.maximum(p.V_cell_max - ocv, 0.0) / R0
    E = jnp.clip(soc, 0.0, 1.0) * p.energy_capacity
    eta = p.round_trip_efficiency
    discharge_ok = (T_cell >= p.T_discharge_min) & (T_cell <= p.T_discharge_max)
    charge_ok = (T_cell >= p.T_charge_min) & (T_cell <= p.T_charge_max)
    most = jnp.minimum(jnp.minimum(p.power_max, v_floor), E / p.delta_t)
    least = jnp.minimum(
        jnp.minimum(p.power_max, v_roof), (p.energy_capacity - E) / (eta * p.delta_t)
    )
    return jnp.where(discharge_ok, most, 0.0), jnp.where(charge_ok, least, 0.0)


def cell_current(power, soc, params: BatteryParams):
    """Cell current (A) and terminal voltage (V) for ``power`` W at the meter:
    the smaller root of ``P / N = (OCV - I R) I``."""
    p = params
    ocv = cell_ocv(soc, p)
    per_cell = power / n_cells(p)
    disc = jnp.maximum(ocv**2 - 4.0 * p.cell_resistance * per_cell, 0.0)
    current = (ocv - jnp.sqrt(disc)) / (2.0 * p.cell_resistance)
    return current, ocv - current * p.cell_resistance


def plant_step(soc, T_cell, required, params: BatteryParams):
    """One step of the unit asked for ``required`` W at the meter (positive
    discharges): ``(soc after, delivered W, cell current A, cell volts V)``."""
    p = params
    most, least = power_limits(soc, T_cell, p)
    delivered = jnp.clip(required, -least, most)
    current, v = cell_current(delivered, soc, p)
    stored = jnp.maximum(delivered, 0.0) + p.round_trip_efficiency * jnp.minimum(
        delivered, 0.0
    )
    soc_next = jnp.clip(soc - stored * p.delta_t / p.energy_capacity, 0.0, 1.0)
    return soc_next, delivered, current, v


def reported_soc(soc, soc_error, T_cell, params: BatteryParams, drifted=0.0):
    """The BMS's state of charge: truth, its error at reset, what Ah counting
    has drifted since (``drifted``), and the OCV curve's shift with
    temperature on the 25 degC curve (ARX-LFP Fig. 5)."""
    shift = params.soc_temp_coeff * (T_cell - params.T_bms_curve)
    return jnp.clip(soc + soc_error + drifted + shift, 0.0, 1.0)


def requirements(deficit_export, deficit_import, params: BatteryParams):
    """The period's energy requirement each way, J: the volume less what the
    unit delivered and has not yet recovered (NESO-ST para. 6.11(iii))."""
    volume = response_volume(params)
    return volume - deficit_export, volume - deficit_import


def misses_requirement(rep_soc, deficit_export, deficit_import, params):
    """Whether the reported stored energy misses either requirement."""
    need_out, need_in = requirements(deficit_export, deficit_import, params)
    E = rep_soc * params.energy_capacity
    return (E < need_out) | (params.energy_capacity - E < need_in)


def submitted_period(time, params: BatteryParams):
    """The first settlement period still open at step ``time``: its start is
    more than gate closure away (NESO-GUIDE p. 36)."""
    now = time * params.delta_t
    return (
        jnp.floor((now + params.gate_closure_seconds) / params.period_seconds).astype(
            jnp.int32
        )
        + 1
    )


def signal_steps(params: BatteryParams):
    return jnp.maximum(jnp.round(params.signal_interval / params.delta_t), 1).astype(
        jnp.int32
    )


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state(
    action_raw: float,
    state: BatteryState,
    params: BatteryParams,
    key: jax.Array,
    integration_method: str = "rk4_2",
):
    """``action_raw`` in [-1, 1] is the Operational Baseline, W, in
    [-baseline_limit, +baseline_limit] (positive = export), submitted for the
    first period still open. ``integration_method`` is kept for the API: the
    energy model is exact over a step."""
    p = params
    t = state.time
    contracted = contracted_power(p)
    limit = baseline_limit(p)
    baseline = convert_raw_action_to_range(action_raw, -limit, limit)
    open_period = submitted_period(t, p)
    baselines = jnp.where(
        open_period < N_PERIODS,
        state.baselines.at[jnp.clip(open_period, 0, N_PERIODS - 1)].set(baseline),
        state.baselines,
    )

    # The unit delivers what is required, unless its BMS refuses.
    soc, delivered, _, v = plant_step(state.soc, state.T_cell, state.target_power, p)
    k = period_of(t, p)
    period_s = steps_per_period(p) * p.delta_t
    short = jnp.abs(state.target_power - delivered) > p.shortfall_tolerance
    short_seconds = state.short_seconds + p.delta_t * short
    newly_short = (short_seconds >= p.unavailable_fraction * period_s) & (
        state.unavailable < 0.5
    )
    unavailable = jnp.maximum(state.unavailable, newly_short.astype(jnp.float32))
    response = state.period_response + (delivered - state.baselines[k]) * p.delta_t

    # The drift is computed from the clock, not summed step by step, which
    # float32 would bias by about 1e-4 over a block.
    t1 = t + 1
    drifted = state.soc_drift * (t1 * p.delta_t)

    # Energy at the period's price; stored energy valued at the base price.
    # The trade term is offset by the most any step could win against the base
    # price, so it is >= 0, and the offset cancels out of the gap (rule 21).
    price = state.prices[k]
    to_MWh = p.delta_t / 3.6e9
    losses_cost = (
        p.price_base
        * (1.0 - p.round_trip_efficiency)
        * jnp.maximum(-delivered, 0.0)
        * to_MWh
    )
    trade_cost = (
        (p.price_base - price) * delivered + jnp.abs(price - p.price_base) * p.power_max
    ) * to_MWh

    # A new settlement period: the energy accounts roll, and the reported
    # stored energy is checked against the new requirement.
    boundary = (t1 % steps_per_period(p)) == 0
    recovery = p.recovery_fraction * response_volume(p)
    d_out = jnp.maximum(state.deficit_export - recovery, 0.0) + jnp.maximum(
        response, 0.0
    )
    d_in = jnp.maximum(state.deficit_import - recovery, 0.0) + jnp.maximum(
        -response, 0.0
    )
    rep = reported_soc(soc, state.soc_error, state.T_cell, p, drifted)
    fails = misses_requirement(rep, d_out, d_in, p)
    newly_checked = boundary & fails
    deficit_export = jnp.where(boundary, d_out, state.deficit_export)
    deficit_import = jnp.where(boundary, d_in, state.deficit_import)
    response = jnp.where(boundary, 0.0, response)
    short_seconds = jnp.where(boundary, 0.0, short_seconds)
    unavailable = jnp.where(boundary, fails.astype(jnp.float32), unavailable)
    availability_cost = (
        (newly_short.astype(jnp.float32) + newly_checked.astype(jnp.float32))
        * (contracted / 1.0e6 * period_s / 3600.0)
        * p.availability_price
    )

    # The regulation signal: an OU process every ``signal_interval``, pulled
    # back onto its own integral so that it stays energy neutral. The draw is
    # folded with the step, so a constant key still gives a fresh one.
    now_s = t * p.delta_t
    mean = jnp.where(
        (now_s >= state.bias_start) & (now_s < state.bias_end), state.bias_pu, 0.0
    )
    signal_energy = state.signal_energy + (
        jnp.clip(state.signal_x, -1.0, 1.0) - mean
    ) * (p.delta_t)
    after = t1 * p.delta_t
    mean_next = jnp.where(
        (after >= state.bias_start) & (after < state.bias_end), state.bias_pu, 0.0
    )
    a = jnp.exp(-p.signal_interval / p.signal_tau)
    eps = jax.random.normal(jax.random.fold_in(key, t))
    moved = (
        a * state.signal_x
        + (1.0 - a) * (mean_next - signal_energy / p.neutrality_tau)
        + p.signal_std * jnp.sqrt(1.0 - a**2) * eps
    )
    signal_x = jnp.where(t1 % signal_steps(p) == 0, moved, state.signal_x)
    regulation = contracted * jnp.clip(signal_x, -1.0, 1.0)
    target = baselines[period_of(t1, p)] + regulation

    return (
        state.replace(
            soc=soc,
            soc_reported=rep,
            v_cell=v,
            power=delivered,
            target_power=target,
            regulation=regulation,
            signal_x=signal_x,
            signal_energy=signal_energy,
            baselines=baselines,
            period_response=response,
            deficit_export=deficit_export,
            deficit_import=deficit_import,
            short_seconds=short_seconds,
            unavailable=unavailable,
            availability_cost=availability_cost,
            losses_cost=losses_cost,
            trade_cost=trade_cost,
            time=t1,
        ),
        None,
    )


def get_obs(state: BatteryState, params: BatteryParams):
    """What the operator's EMS reads (PHYSICS.md, Observation)::

        [reported SoC, V_cell, T_cell, P_MW, required_MW, signal_pu,
         baseline_now_MW, baseline_next_MW, baseline_after_MW,
         export_requirement_MWh, import_requirement_MWh,
         seconds_to_next_period, period_lost,
         price_now, price_next, price_after, price_open]   (GBP/MWh)

    The three baselines are the ones past gate closure; the four prices are
    the day-ahead prices of this period and the next three, known ahead
    (rule 16). The true state of
    charge, the estimate's error and the signal's internals are hidden.
    """
    p = params
    k = period_of(state.time, p)
    rep = state.soc_reported
    need_out, need_in = requirements(state.deficit_export, state.deficit_import, p)
    per = steps_per_period(p)
    to_next = (per - state.time % per) * p.delta_t
    ahead = lambda i: state.baselines[jnp.minimum(k + i, N_PERIODS - 1)]  # noqa: E731
    ahead_price = lambda i: state.prices[
        jnp.minimum(k + i, N_PERIODS - 1)
    ]  # noqa: E731
    return jnp.array(
        [
            rep,
            state.v_cell,
            state.T_cell,
            state.power / 1.0e6,
            state.target_power / 1.0e6,
            state.regulation / contracted_power(p),
            ahead(0) / 1.0e6,
            ahead(1) / 1.0e6,
            ahead(2) / 1.0e6,
            need_out / 3.6e9,
            need_in / 3.6e9,
            to_next,
            state.unavailable,
            ahead_price(0),
            ahead_price(1),
            ahead_price(2),
            ahead_price(3),
        ]
    )


def check_is_terminal(state: BatteryState, params: BatteryParams, xp=jnp):
    """Nothing trips: the energy rules are priced, the cell limits protected
    (PHYSICS.md, Limits)."""
    terminated = xp.zeros_like(xp.asarray(state.soc), dtype=bool)
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward_terms(
    state: BatteryState, params: BatteryParams, xp=jnp, tripped=None
):
    """The score's terms, GBP per step, each >= 0: availability lost, energy
    lost at the base price, and the energy's cost against the base price.

    ``tripped`` overrides the trip check on ``state`` (``base.measured_reward``).
    """
    if tripped is None:
        tripped, _ = check_is_terminal(state, params, xp)
    terms = {
        "availability": state.availability_cost,
        "losses": state.losses_cost,
        "trade": state.trade_cost,
    }
    return R.with_trip(terms, tripped, R.trip_cost(params), xp)


def compute_reward(state: BatteryState, params: BatteryParams, xp=jnp):
    return R.total(compute_reward_terms(state, params, xp), xp)


def initial_state(key, params: BatteryParams) -> Tuple[BatteryState, jax.Array]:
    """Established operation at the start of an EFA block (NESO-GUIDE p. 37):
    the reported energy meets the block's requirement both ways, the earlier
    baselines are declared at zero (ours), and the signal is at its spread."""
    p = params
    k_rep, k_err, k_sign, k_sig, k_ev, k_lvl, k_b, k_bt, k_bd, k_bs = jax.random.split(
        key, 10
    )
    margin = response_volume(p) / p.energy_capacity
    rep = jax.random.uniform(k_rep, minval=margin, maxval=1.0 - margin)
    error = p.soc_error_std * jax.random.normal(k_err)
    drift = p.soc_drift_rate * jnp.where(jax.random.bernoulli(k_sign), 1.0, -1.0)
    T = jnp.asarray(p.T_cell_start, jnp.float32)
    soc = jnp.clip(rep - error - p.soc_temp_coeff * (T - p.T_bms_curve), 0.0, 1.0)
    signal_x = p.signal_std * jax.random.normal(k_sig)
    regulation = contracted_power(p) * jnp.clip(signal_x, -1.0, 1.0)
    _, v = cell_current(regulation, soc, p)
    zero = jnp.zeros(())

    # The day-ahead prices: the base, and one event of two periods.
    lo, hi = p.price_event_range
    level = jax.random.uniform(k_lvl, minval=lo, maxval=hi)
    first = jax.random.randint(
        k_ev, (), p.price_event_first[0], p.price_event_first[1] + 1
    )
    periods = jnp.arange(N_PERIODS)
    in_event = (periods >= first) & (periods < first + p.price_event_periods)
    prices = jnp.where(in_event, level, p.price_base)

    # The long one-way deviation: its size, sign, length and start.
    blo, bhi = p.bias_pu_range
    size = jax.random.uniform(k_b, minval=blo, maxval=bhi)
    bias = size * jnp.where(jax.random.bernoulli(k_bs), 1.0, -1.0)
    dlo, dhi = p.bias_duration_range
    length = jax.random.uniform(k_bd, minval=dlo, maxval=dhi)
    block = p.max_steps_in_episode * p.delta_t
    start = jax.random.uniform(
        k_bt, minval=0.0, maxval=jnp.maximum(block - length, 0.0)
    )
    return BatteryState(
        time=0,
        soc=soc,
        soc_error=error,
        soc_drift=drift,
        T_cell=T,
        soc_reported=reported_soc(soc, error, T, p),
        v_cell=v,
        power=regulation,
        target_power=regulation,
        regulation=regulation,
        signal_x=signal_x,
        signal_energy=zero,
        bias_pu=bias,
        bias_start=start,
        bias_end=start + length,
        prices=prices,
        baselines=jnp.zeros(N_PERIODS),
        period_response=zero,
        deficit_export=zero,
        deficit_import=zero,
        short_seconds=zero,
        unavailable=zero,
        availability_cost=zero,
        losses_cost=zero,
        trade_cost=zero,
    )
