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

from target_gym.base import EnvParams, EnvState
from target_gym.integration import integrate_dynamics
from target_gym.utils import convert_raw_action_to_range

GAS_CONSTANT = 8.314  # J/(mol K)
KELVIN = 273.15

# Ornstein-Uhlenbeck dispatch signal: mean-reversion rate (1/s). 1/theta ~ 500 s,
# so the grid's request drifts on a timescale comparable to the energy budget
# rather than flickering.
DISPATCH_OU_THETA = 2.0e-3


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
    power_band: float = 0.15e6  # W, error at which tracking reward reaches 0
    degradation_weight: float = 2.0e5  # scales fractional fade into reward units
    soc_comfort_weight: float = 0.10  # gentle pull toward mid charge

    # ---- Dispatch signal ----
    dispatch_std: float = 0.45e6  # W, OU stationary std
    initial_soc_range: Tuple[float, float] = (0.35, 0.75)

    # ---- Time discretization ----
    # 10-90 % state of charge at full power takes ~96 min, so a 60 min episode
    # at 5 s per step exercises a real fraction of the energy budget.
    delta_t: float = 5.0
    max_steps_in_episode: int = 720


@struct.dataclass
class BatteryState(EnvState):
    soc: float  # state of charge (0-1)
    v_rc: float  # diffusion branch voltage (V) -- HIDDEN
    T_cell: float  # pack temperature (degC)
    q_loss: float  # cumulative fractional capacity fade -- HIDDEN

    current: float  # pack current (A), positive = discharge
    power: float  # delivered electrical power (W)
    target_power: float  # dispatch request (W)


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

    # OU dispatch signal. Drawn from a key folded with ``state.time`` so a
    # caller passing a constant key -- which every rollout helper here does --
    # still gets a genuine zero-mean process.
    noise = jax.random.normal(jax.random.fold_in(key, state.time))
    sigma = p.dispatch_std * jnp.sqrt(2.0 * DISPATCH_OU_THETA * p.delta_t)
    target = jnp.clip(
        state.target_power
        - DISPATCH_OU_THETA * state.target_power * p.delta_t
        + sigma * noise,
        -p.power_max,
        p.power_max,
    )

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


@partial(jax.jit, static_argnames=["params"])
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


def compute_reward(state: BatteryState, params: BatteryParams, xp=jnp):
    """Dispatch tracking, minus degradation, minus a gentle pull to mid charge.

    The state-of-charge term is deliberately weak: it should bias the
    controller toward keeping headroom in both directions without overriding
    the dispatch it is being paid to follow.
    """
    p = params
    err = xp.abs(state.target_power - state.power)
    tracking = xp.clip(1.0 - err / p.power_band, 0.0, 1.0) ** 2
    fade = degradation_rate(state.current, state.T_cell, p) * p.delta_t
    headroom = (state.soc - 0.5) ** 2
    return tracking - p.degradation_weight * fade - p.soc_comfort_weight * headroom


def round_trip_efficiency(power, soc, params: BatteryParams):
    """One-way efficiency at a given power and state of charge; square for round trip."""
    current = current_for_power(power, soc, 0.0, params)
    loss = current**2 * params.R0
    return 1.0 - loss / jnp.maximum(jnp.abs(power), 1.0)
