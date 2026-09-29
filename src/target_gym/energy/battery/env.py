"""
Grid battery storage — equivalent-circuit Li-ion pack tracking a dispatch signal.

See ``PHYSICS.md`` in this directory for provenance, the sourced parameter
table and validation targets. Method: ``docs/PHYSICS_METHODOLOGY.md``.

Model
-----
A 2 MWh / 1 MW grid battery (a 2-hour system, the common build) as a
first-order equivalent circuit with one RC branch, a lumped thermal mass and
a capacity-fade model::

    V_term = OCV(soc) − I R0 − v_rc
    dsoc/dt   = −I / Q                          (coulomb counting)
    dv_rc/dt  = I / C1 − v_rc / (R1 C1)         (diffusion / polarisation)
    C_th dT/dt = I² R0 + v_rc I − UA (T − T_amb)
    dq_loss/dt = calendar(T) + cycle(|I|, T)    (capacity fade)

The controller commands **power**, not current, so the current follows from
``P = V_term I`` — a quadratic whose physical root is taken. That quadratic is
what makes power a genuinely harder control variable than current: the
deliverable power is bounded by ``(OCV − v_rc)² / 4R0``, and that bound moves
with state of charge.

Task
----
Track a grid dispatch signal while managing a finite energy budget. The
tension is structural: following dispatch drains charge, and running the pack
empty or full is irrecoverable within the episode.

Why it is a hard target MDP
---------------------------
* **A finite, depletable budget.** Unlike a thermal plant the battery cannot
  hold a setpoint indefinitely. Tracking now costs the ability to track later.
* **Irrecoverable end states.** Hitting the state-of-charge limits ends the
  episode; no later action recovers it.
* **Degradation is a hidden cost.** Capacity fade accumulates invisibly and is
  driven by throughput and temperature, so aggressive tracking is paid for
  later rather than immediately.
* **Efficiency depends on where you are.** Losses scale with current squared,
  and current for a given power depends on state of charge through the OCV
  curve — so the same dispatch costs more when the pack is low.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym import reward as R
from target_gym.base import EnvParams, EnvState
from target_gym.integration import integrate_dynamics
from target_gym.utils import convert_raw_action_to_range, log_scaled_reward

GAS_CONSTANT = 8.314  # J/(mol K)
KELVIN = 273.15

#: Dispatch blocks in a schedule. A grid battery is not handed a random walk:
#: it is handed a setpoint, holds it for a market interval, and is handed
#: another. Twelve blocks covers a 60 min episode at the 5 min interval below.
#:
#: The signal used to be an Ornstein-Uhlenbeck process, and it made the task
#: unmeasurable. Its one-step innovation had a standard deviation of 63.6 kW
#: against a 150 kW tracking band, which puts the *best possible* tracking
#: reward at 0.429 -- and the shipped PID scored 0.447 while the MPC scored
#: 0.430. Both controllers were sitting on an irreducible noise floor, so the
#: environment could not tell a good controller from a mediocre one, and the
#: only thing separating them was noise. A schedule is both more plausible and
#: actually measures something: the error is now the transient after each step,
#: which is what a controller is for.
N_DISPATCH_BLOCKS = 12


@struct.dataclass
class BatteryParams(EnvParams):
    # ---- Pack ----
    energy_nominal: float = 2.0e6 * 3600.0  # J (2 MWh)
    power_max: float = 1.0e6  # W (0.5 C)
    n_series: float = 192.0  # cells in series
    capacity_As: float = 2500.0 * 3600.0  # A s (2500 Ah)

    # ---- Equivalent circuit ----
    # R0 is sized for the published round-trip efficiency band; see PHYSICS.md.
    R0: float = 0.02  # ohm, pack series resistance
    R1: float = 0.01  # ohm, diffusion branch
    C1: float = 20_000.0  # F, diffusion branch

    # ---- OCV curve, per cell (NMC-like) ----
    ocv_a: float = 3.0
    ocv_b: float = 1.15
    ocv_c: float = 0.30
    ocv_d: float = 12.0
    ocv_e: float = 0.05
    ocv_f: float = 8.0
    ocv_g: float = 0.85

    # ---- Thermal ----
    # Actively cooled, as grid packs are: a passive UA gives a physically
    # absurd temperature rise at rated power.
    C_thermal: float = 9.0e6  # J/K (~10 t at 900 J/(kg K))
    UA_thermal: float = 3000.0  # W/K
    T_ambient: float = 25.0  # degC

    # ---- Degradation ----
    k_calendar: float = 3.0e-9  # 1/s at the reference temperature
    E_activation: float = 20_000.0  # J/mol
    T_ref_aging: float = 298.15  # K
    k_cycle: float = 1.5e-9  # fractional fade per unit throughput

    # ---- Operating / termination bounds ----
    soc_min: float = 0.05
    soc_max: float = 0.95
    T_max: float = 60.0  # degC
    V_cell_min: float = 2.7
    V_cell_max: float = 4.25

    # ---- Reward shaping ----
    # Error scale for the MPC's tracking term, not read by ``compute_reward``.
    # See "Why the MPC does not minimise the reward" in docs/baselines.md.
    power_band: float = 0.15e6  # W
    precision_floor: float = 1e3  # W, revenue-grade power metering resolution
    # Upper bound on the per-step cost terms, used to keep the reward
    # non-negative: soc_comfort_weight * max((soc-0.5)^2) = 0.0203 over the
    # [0.05, 0.95] window, plus degradation_weight * fade at the cell's thermal
    # limit and full-power current = 0.0081. Worst observed in rollout: 0.0229.
    max_step_cost: float = 0.03
    # Fraction of the tracking reward the worst-case cost may discount away.
    #
    # NOT a consumption cost, despite the name, and deliberately left live when
    # the running costs were zeroed for the 0.6 line. It gates the two terms
    # below, degradation and state-of-charge comfort, and those are not a price
    # on the plant's inputs: they are what keeps the control problem well posed.
    # Without the SoC term the optimal policy follows dispatch until the pack
    # hits a limit and the episode ends, which is not a tracking task; without
    # the degradation term the pack has no reason to care about throughput,
    # which is most of what a battery controller is for. This environment has
    # no consumption cost to remove, so nothing here was zeroed.
    cost_weight: float = 0.1
    degradation_weight: float = 2.0e5  # scales fractional fade into reward units
    soc_comfort_weight: float = 0.10  # gentle pull toward mid charge

    # ---- Dispatch signal ----
    # Held for a market interval, then stepped. 300 s is the dispatch interval
    # of most wholesale real-time markets.
    dispatch_block_seconds: float = 300.0
    dispatch_range: float = 0.8e6  # W, half-range of a block's level
    # Regulation jitter on top of the held setpoint. Deliberately small against
    # the 150 kW band: it should stop the task being noise-free without
    # becoming the thing that decides the score. At 2 kW the best attainable
    # tracking reward is ~0.86 rather than the OU signal's 0.43.
    dispatch_noise_std: float = 2.0e3  # W
    initial_soc_range: Tuple[float, float] = (0.35, 0.75)

    # ---- Time discretization ----
    # 10-90 % state of charge at full power takes ~96 min, so a 60 min episode
    # at 5 s per step exercises a real fraction of the energy budget.
    delta_t: float = 5.0
    max_steps_in_episode: int = 360

    # ---- Reward (docs/reward-shaping.md; version 2), in dollars per step ----
    # Tracking: dispatch imbalance at ``imbalance_price`` per MWh, linear in
    # |error|. Floor: the dispatch target is a block level plus white noise of
    # sd ``dispatch_noise_std`` drawn after the action is chosen, so no
    # controller can hold the mean |error| below E|N(0, sd)| = sd * sqrt(2/pi)
    # = 1596 W (closed form; the shipped PID and MPC hold 5.6 and 6.4 kW,
    # `scripts/measure_hold.py`). Degradation: capacity fade above the
    # hold-phase rate (1.92e-8 of capacity per step, PID and MPC alike) at
    # ``fade_price`` per kWh of the 1692 kWh pack -- the avoidable part of
    # ageing, so the best achievable reward stays near zero. The SOC-comfort
    # term of version 1 has no owner price and is dropped (provisional): a
    # pack driven to the edge of its window pays through the dispatch it can
    # then not follow. A trip costs, per step, twice the 1 MW envelope's
    # imbalance.
    reward_version: int = 2
    e_floor: float = 1596.0  # W, E|noise|, closed form
    e_tol: float = 0.0
    tracking_exponent: float = 1.0
    imbalance_price: float = 100.0  # $/MWh
    c_hold: float = 1.92e-8  # fractional fade per step while holding
    fade_price: float = 300.0  # $/kWh of lost capacity
    pack_kWh: float = 1692.0  # capacity_As * OCV(50%) / 3.6e6
    failure_cost: float = (
        2.0 * 100.0 * 1.8 * 5.0 / 3600.0
    )  # twice the imbalance of the reachable 1.8 MW (0.8 MW target vs 1 MW power), per 5 s step
    #: Restart time priced into a trip (``reward.trip_cost``; 1 h at 5 s steps: a protection trip's reset, provisional).
    restart_steps: int = 720
    #: Tracking cost per step at the floor, in the reward's units; the NEA floor.
    rho_floor_tracking: float = 100.0 / 1.0e6 * 5.0 / 3600.0 * 1596.0
    rho_floor: float = 100.0 / 1.0e6 * 5.0 / 3600.0 * 1596.0
    #: True where e_floor is a resolution, not a measured or certified floor.
    floor_is_documented_minimum: bool = False


@struct.dataclass
class BatteryState(EnvState):
    soc: float  # state of charge (0-1)
    v_rc: float  # diffusion branch voltage (V) -- HIDDEN
    T_cell: float  # pack temperature (degC)
    q_loss: float  # cumulative fractional capacity fade -- HIDDEN

    current: float  # pack current (A), positive = discharge
    power: float  # delivered electrical power (W)
    target_power: float  # dispatch request (W)
    # The whole schedule, so a predictive controller can see the next block
    # coming. The observation exposes only the current request, which is what
    # keeps the lookahead a genuine advantage rather than a free lunch.
    dispatch_schedule: jnp.ndarray


def dispatch_block(time, params: BatteryParams, xp=jnp):
    """Which block of the schedule is live at *time*."""
    per_block = xp.maximum(params.dispatch_block_seconds / params.delta_t, 1.0)
    return xp.clip((time / per_block).astype(int), 0, N_DISPATCH_BLOCKS - 1)


def open_circuit_voltage(soc, params: BatteryParams):
    """Pack OCV. Monotone in state of charge, with the usual mid-range plateau."""
    p = params
    s = jnp.clip(soc, 0.0, 1.0)
    cell = (
        p.ocv_a
        + p.ocv_b * s
        - p.ocv_c * jnp.exp(-p.ocv_d * s)
        + p.ocv_e * jnp.tanh(p.ocv_f * (s - p.ocv_g))
    )
    return cell * p.n_series


def max_deliverable_power(soc, v_rc, params: BatteryParams):
    """Power beyond which ``P = V I`` has no real solution.

    The circuit cannot deliver more than ``(OCV - v_rc)^2 / 4 R0``: past that
    the extra current costs more in internal loss than it adds at the
    terminals. This is a real limit that tightens as the pack empties.
    """
    driving = open_circuit_voltage(soc, params) - v_rc
    return driving**2 / (4.0 * params.R0)


def current_for_power(power, soc, v_rc, params: BatteryParams):
    """Solve ``P = (OCV - v_rc) I - R0 I^2`` for the physical (smaller) root.

    Positive ``power`` discharges. Charging (negative power) always has a real
    root; discharging is capped at :func:`max_deliverable_power`.
    """
    p = params
    driving = open_circuit_voltage(soc, params) - v_rc
    limit = driving**2 / (4.0 * p.R0)
    power = jnp.clip(power, -limit * 0.999, limit * 0.999)
    disc = jnp.maximum(driving**2 - 4.0 * p.R0 * power, 0.0)
    return (driving - jnp.sqrt(disc)) / (2.0 * p.R0)


def terminal_voltage(current, soc, v_rc, params: BatteryParams):
    return open_circuit_voltage(soc, params) - current * params.R0 - v_rc


def degradation_rate(current, T_cell, params: BatteryParams):
    """Fractional capacity fade per second: calendar plus cycling.

    Calendar ageing follows an Arrhenius law in temperature; cycle ageing is
    proportional to charge throughput. Both are small per step and only matter
    cumulatively -- which is the point, since the controller pays for
    aggressive tracking later rather than now.
    """
    p = params
    T_kelvin = T_cell + KELVIN
    arrhenius = jnp.exp(
        -p.E_activation / GAS_CONSTANT * (1.0 / T_kelvin - 1.0 / p.T_ref_aging)
    )
    calendar = p.k_calendar * arrhenius
    cycling = p.k_cycle * jnp.abs(current) / 1000.0 * arrhenius
    return calendar + cycling


def compute_velocity(position, action, params: BatteryParams):
    """RHS for ``[soc, v_rc, T_cell, q_loss]``. ``action`` is power in W."""
    p = params
    soc, v_rc, T_cell = position[0], position[1], position[2]
    current = current_for_power(action, soc, v_rc, p)

    dsoc = -current / p.capacity_As
    dv_rc = current / p.C1 - v_rc / (p.R1 * p.C1)
    heat = current**2 * p.R0 + v_rc * current
    dT = (heat - p.UA_thermal * (T_cell - p.T_ambient)) / p.C_thermal
    dq = degradation_rate(current, T_cell, p)
    return jnp.array([dsoc, dv_rc, dT, dq]), None


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state(
    action_raw: float,
    state: BatteryState,
    params: BatteryParams,
    key: jax.Array,
    integration_method: str = "rk4_2",
):
    """``action_raw`` in [-1, 1] maps to power in [-power_max, +power_max].

    Positive is discharge (delivering to the grid), negative is charge.
    """
    p = params
    power_cmd = convert_raw_action_to_range(
        action_raw, min_action=-p.power_max, max_action=p.power_max
    )

    _compute_velocity = partial(compute_velocity, action=power_cmd, params=p)
    new_positions, _ = integrate_dynamics(
        positions=jnp.array([state.soc, state.v_rc, state.T_cell, state.q_loss]),
        delta_t=p.delta_t,
        compute_velocity=_compute_velocity,
        method=integration_method,
    )
    soc = jnp.clip(new_positions[0], 0.0, 1.0)
    v_rc, T_cell, q_loss = new_positions[1], new_positions[2], new_positions[3]

    current = current_for_power(power_cmd, soc, v_rc, p)
    power = terminal_voltage(current, soc, v_rc, p) * current

    # Scheduled dispatch: the level for the block that is live at the *next*
    # step, plus regulation jitter. The jitter is drawn from a key folded with
    # ``state.time`` so a caller passing a constant key -- which every rollout
    # helper here does -- still gets a genuine zero-mean process.
    noise = jax.random.normal(jax.random.fold_in(key, state.time))
    level = state.dispatch_schedule[dispatch_block(state.time + 1, p)]
    target = jnp.clip(level + p.dispatch_noise_std * noise, -p.power_max, p.power_max)

    return (
        state.replace(
            soc=soc,
            v_rc=v_rc,
            T_cell=T_cell,
            q_loss=q_loss,
            current=current,
            power=power,
            target_power=target,
            time=state.time + 1,
        ),
        None,
    )


# Deliberately not jitted. It was decorated with
# ``@partial(jax.jit, static_argnames=["params"])``, which keys the compilation
# cache on the params object: a fresh ``Params(...)`` -- what every sweep, tuner
# and MPC builds -- was a cache miss and a full recompile, measured at ~1600x the
# cost of a cached call. Callers that want it fused already jit ``step_env``,
# which traces this inline.
def get_obs(state: BatteryState, params: BatteryParams):
    """``[soc, V_cell, T_cell, P_MW, target_P_MW]``.

    A battery management system measures state of charge (estimated from
    coulomb counting and voltage), terminal voltage, temperature and power.
    The diffusion-branch voltage and the accumulated capacity fade are hidden:
    neither is directly measurable, and the fade in particular is the cost the
    controller is implicitly trading against.
    """
    v_cell = (
        terminal_voltage(state.current, state.soc, state.v_rc, params) / params.n_series
    )
    return jnp.array(
        [
            state.soc,
            v_cell,
            state.T_cell,
            state.power / 1.0e6,
            state.target_power / 1.0e6,
        ]
    )


def check_is_terminal(state: BatteryState, params: BatteryParams, xp=jnp):
    v_cell = (
        terminal_voltage(state.current, state.soc, state.v_rc, params) / params.n_series
    )
    soc_out = xp.logical_or(state.soc <= params.soc_min, state.soc >= params.soc_max)
    v_out = xp.logical_or(v_cell <= params.V_cell_min, v_cell >= params.V_cell_max)
    terminated = xp.logical_or(
        soc_out, xp.logical_or(v_out, state.T_cell >= params.T_max)
    )
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward_terms(state: BatteryState, params: BatteryParams, xp=jnp):
    """The reward's additive cost terms in $ per step (``target_gym.reward``)."""
    p = params
    per_W_step = p.imbalance_price / 1.0e6 * p.delta_t / 3600.0
    tracking = (
        per_W_step
        * p.e_floor
        * R.tracking_cost(
            state.target_power - state.power,
            p.e_floor,
            p.e_tol,
            p.tracking_exponent,
            xp,
        )
    )
    fade = degradation_rate(state.current, state.T_cell, p) * p.delta_t
    degradation = p.fade_price * p.pack_kWh * xp.maximum(fade - p.c_hold, 0.0)
    terminated, _ = check_is_terminal(state, p, xp)
    terms = {
        "tracking": tracking,
        "running": degradation,
    }
    return R.with_trip(terms, terminated, R.trip_cost(p), xp)


def compute_reward_v1(state: BatteryState, params: BatteryParams, xp=jnp):
    """Dispatch tracking, minus degradation, minus a gentle pull to mid charge.

    The state-of-charge term is deliberately weak: it should bias the
    controller toward keeping headroom in both directions without overriding
    the dispatch it is being paid to follow.
    """
    p = params
    err = xp.abs(state.target_power - state.power)
    tracking = log_scaled_reward(err, p.precision_floor, p.power_max, xp)
    fade = degradation_rate(state.current, state.T_cell, p) * p.delta_t
    headroom = (state.soc - 0.5) ** 2
    cost = p.degradation_weight * fade + p.soc_comfort_weight * headroom
    return tracking * (1.0 - jnp.clip(cost / p.max_step_cost, 0.0, 1.0) * p.cost_weight)


def compute_reward(state: BatteryState, params: BatteryParams, xp=jnp):
    return R.select(
        params.reward_version,
        compute_reward_v1(state, params, xp),
        R.total(compute_reward_terms(state, params, xp), xp),
        xp,
    )


def round_trip_efficiency(power, soc, params: BatteryParams):
    """One-way efficiency at a given power and state of charge; square for round trip."""
    current = current_for_power(power, soc, 0.0, params)
    loss = current**2 * params.R0
    return 1.0 - loss / jnp.maximum(jnp.abs(power), 1.0)
