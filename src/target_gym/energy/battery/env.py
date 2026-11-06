"""
Grid battery storage: an equivalent-circuit Li-ion pack following a regulation
signal over a window long enough to run out of energy.

See ``PHYSICS.md`` in this directory for provenance, the sourced parameter
table and validation targets. Method: ``docs/PHYSICS_METHODOLOGY.md``.

Model
-----
A 1 MW / 1.69 MWh grid battery as a first-order equivalent circuit with one RC
branch, a lumped thermal mass and a capacity-fade model, behind an inverter and
an auxiliary load::

    P_dc      = P_inv + pcs_loss P_inv^2 / power_max   (inverter)
    P_meter   = P_inv - aux_load                       (revenue meter)
    V_term    = OCV(soc) - I R0 - v_rc
    dsoc/dt   = -I / Q                                 (coulomb counting)
    dv_rc/dt  = I / C1 - v_rc / (R1 C1)                (diffusion)
    C_th dT/dt = I^2 R0 + v_rc I - UA (T - T_amb)
    dq_loss/dt = calendar(T) + ripple(|I|, T) + deep cycles

The controller commands the inverter, so the current follows from
``P_dc = V_term I``, a quadratic whose physical root is taken.

Task
----
Follow a regulation signal for 4 hours: an hourly dispatch level plus an AR(1)
regulation movement, both in MW at the meter. Following it exactly leaves the
charge window on most seeds, and leaving it trips the pack.

Why it is a hard target MDP
---------------------------
* **Energy binds.** Hourly levels of up to 0.8 MW against a 1.69 MWh pack.
* **Depth has a convex price.** A half-cycle of depth ``d`` uses ``Phi(d) / 2`` of
  the pack's life, so the next MWh of a deep swing can cost more in wear than in
  imbalance. Whether to keep following is a real choice.
* **The next block is known.** A policy can see the energy coming and the time
  left, and a trim has real work: auxiliary loads sit behind the command.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym import reward as R
from target_gym.base import EnvParams, EnvState, actuator_field
from target_gym.integration import integrate_dynamics
from target_gym.utils import convert_raw_action_to_range, log_scaled_reward

GAS_CONSTANT = 8.314  # J/(mol K)
KELVIN = 273.15

#: Dispatch blocks in a schedule, hourly: the four the 4 h window scores, then
#: the next one, so the last scored block has a next block to read too.
#:
#: The signal was once an Ornstein-Uhlenbeck process of 63.6 kW innovations against
#: a 150 kW tracking band, which pinned every controller on an irreducible noise
#: floor. The held level plus a small-innovation regulation signal keeps the
#: error a property of the controller.
N_DISPATCH_BLOCKS = 5


@struct.dataclass
class BatteryParams(EnvParams):
    # ---- Pack ----
    energy_nominal: float = 1692.0e3 * 3600.0  # J, 2500 Ah at the 677 V of 50 % charge
    power_max: float = 1.0e6  # W, about 0.6 C
    n_series: float = 192.0  # cells in series
    capacity_As: float = 2500.0 * 3600.0  # A s (2500 Ah)

    # ---- Between the pack and the meter (PHYSICS.md, section 3) ----
    pcs_loss: float = 0.03  # of throughput at rated power, inverter
    aux_load: float = 15e3  # W, cooling and controls, drawn behind the meter

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
    k_calendar: float = 6.34e-10  # 1/s at the reference temperature, 2 %/yr
    E_activation: float = 20_000.0  # J/mol
    T_ref_aging: float = 298.15  # K
    # Ripple: throughput priced at the stress of a cycle as deep as the hysteresis
    # below, Phi(0.05) / (2 * 0.05) of life per capacity of charge moved, over
    # 9000 kA s, in fade. Anything deeper is a half-cycle, charged by depth.
    k_cycle: float = 2.66e-10  # fractional fade per kA s of throughput
    # Life lost to one cycle of depth u: stress_alpha u^stress_beta.
    stress_alpha: float = 5.24e-4
    stress_beta: float = 2.03
    life_fade: float = 0.2  # fraction of capacity lost at end of life
    cycle_hysteresis: float = 0.05  # state of charge, smaller swings are ripple

    # ---- Operating / termination bounds ----
    soc_min: float = 0.05
    soc_max: float = 0.95
    T_max: float = 60.0  # degC
    V_cell_min: float = 2.7
    V_cell_max: float = 4.25

    # ---- Reward shaping ----
    # Error scale of the retired gradient MPC's tracking term. Nothing reads it
    # since the oracle became a feedforward of the scheduled level (oracle
    # audit, 2026-10); kept so the parameters, and the version stamp, do not
    # change.
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
    # on the plant's inputs. The SoC term was meant to stop the optimal policy
    # following dispatch into a limit, though a policy that follows the
    # schedule exactly stays inside the window over the 30 min episode (oracle
    # audit, 2026-10-01); version 2 drops it. Without the degradation term the
    # pack has no reason to care about throughput, which is most of what a
    # battery controller is for. This environment has no consumption cost to
    # remove, so nothing here was zeroed.
    cost_weight: float = 0.1
    degradation_weight: float = 2.0e5  # scales fractional fade into reward units
    soc_comfort_weight: float = 0.10  # gentle pull toward mid charge

    # ---- Dispatch signal ----
    # An hourly level, the unit regulation markets clear in, plus an AR(1)
    # regulation movement. The innovation sets the floor, and at 10 kW and 900 s
    # the signal itself wanders about 95 kW: a movement to follow that stays out
    # of the way of the energy the levels carry.
    dispatch_block_seconds: float = 3600.0
    dispatch_range: float = 0.8e6  # W, half-range of a block's level
    dispatch_noise_std: float = 1.0e4  # W, innovation of the regulation signal
    regulation_tau: float = 900.0  # s, correlation time of the regulation signal
    initial_soc_range: Tuple[float, float] = (0.35, 0.75)

    # ---- Time discretization ----
    # 10-90 % state of charge at full power takes about 75 min. 4 h at the 5 s
    # of a regulation cycle, so following the signal runs out of energy.
    delta_t: float = 5.0
    max_steps_in_episode: int = 2880

    # ---- Reward (docs/reward-shaping.md; version 2), in dollars per step ----
    # Tracking: imbalance at the meter at ``imbalance_price`` per MWh, linear in
    # |error|. Floor: the regulation signal moves after the action by an
    # innovation of sd ``dispatch_noise_std``, so no causal controller holds the
    # mean |error| below E|N(0, sd)| = sd * sqrt(2/pi) = 7979 W (closed form).
    # Wear: capacity fade above calendar ageing, which no controller avoids, at
    # ``fade_price`` per kWh of the 1692 kWh pack. That price is the replacement
    # cost spread over the fade that ends a pack's life: $300/kWh / 0.2. A trip
    # costs, per step, twice the dearest step the pack can survive.
    reward_version: int = 2
    e_floor: float = 7979.0  # W, E|innovation|, closed form
    e_tol: float = 0.0
    tracking_exponent: float = 1.0
    imbalance_price: float = 100.0  # $/MWh
    c_hold: float = 3.17e-9  # k_calendar * delta_t, fade per step at rest
    fade_price: float = 1500.0  # $/kWh of lost capacity: $300/kWh over 20 % fade
    pack_kWh: float = 1692.0  # capacity_As * OCV(50%) / 3.6e6
    # Twice the dearest step that survives, per 5 s step: the imbalance of the
    # reachable 1.8 MW (0.8 MW target vs 1 MW power), $0.25, the wear of a full
    # power step at the window's full depth, $0.20, and a half-cycle's turn, $0.30.
    failure_cost: float = 2.0 * (0.25 + 0.20 + 0.30)
    #: Restart time priced into a trip (``reward.trip_cost``; 1 h at 5 s steps: a protection trip's reset, provisional).
    restart_steps: int = 720
    #: Tracking cost per step at the floor, in the reward's units.
    rho_floor_tracking: float = 100.0 / 1.0e6 * 5.0 / 3600.0 * 7979.0
    rho_floor: float = 100.0 / 1.0e6 * 5.0 / 3600.0 * 7979.0
    #: True where e_floor is a resolution, not a measured or certified floor.
    floor_is_documented_minimum: bool = False


#: Units of the BatteryParams fields, for the generated environment pages.
PARAM_UNITS = {
    "energy_nominal": "J",
    "power_max": "W",
    "n_series": "-",
    "capacity_As": "A s",
    "pcs_loss": "-",
    "aux_load": "W",
    "R0": "ohm",
    "R1": "ohm",
    "C1": "F",
    "ocv_a": "V",
    "ocv_b": "V",
    "ocv_c": "V",
    "ocv_d": "-",
    "ocv_e": "V",
    "ocv_f": "-",
    "ocv_g": "-",
    "C_thermal": "J/K",
    "UA_thermal": "W/K",
    "T_ambient": "degC",
    "k_calendar": "1/s",
    "E_activation": "J/mol",
    "T_ref_aging": "K",
    "k_cycle": "1/(kA s)",
    "stress_alpha": "-",
    "stress_beta": "-",
    "life_fade": "-",
    "cycle_hysteresis": "-",
    "soc_min": "-",
    "soc_max": "-",
    "T_max": "degC",
    "V_cell_min": "V",
    "V_cell_max": "V",
    "power_band": "W",
    "precision_floor": "W",
    "max_step_cost": "reward",
    "cost_weight": "-",
    "degradation_weight": "reward",
    "soc_comfort_weight": "reward",
    "dispatch_block_seconds": "s",
    "dispatch_range": "W",
    "dispatch_noise_std": "W",
    "regulation_tau": "s",
    "reward_version": "-",
    "e_floor": "W",
    "e_tol": "W",
    "tracking_exponent": "-",
    "imbalance_price": "$/MWh",
    "c_hold": "1/step",
    "fade_price": "$/kWh",
    "pack_kWh": "kWh",
    "failure_cost": "reward",
    "restart_steps": "steps",
    "rho_floor_tracking": "reward",
    "rho_floor": "reward",
}


@struct.dataclass
class BatteryState(EnvState):
    soc: float  # state of charge (0-1)
    v_rc: float  # diffusion branch voltage (V) -- HIDDEN
    T_cell: float  # pack temperature (degC)
    q_loss: float  # cumulative fractional capacity fade -- HIDDEN

    current: float  # pack current (A), positive = discharge
    power: float  # power at the meter (W)
    target_power: float  # regulation signal (W)
    regulation: float  # the signal's AR(1) part (W) -- HIDDEN
    # The whole schedule. The observation shows the next block and how far off
    # it is, so a controller reads what the oracle reads one block ahead.
    dispatch_schedule: jnp.ndarray
    # The half-cycle under way: where it turned, and how far it has reached.
    soc_anchor: float  # -- HIDDEN
    soc_peak: float  # -- HIDDEN
    deep_fade: float = 0.0  # fraction of capacity the last step's depth used

    actuator_held: jax.Array = actuator_field(1)


def dispatch_block(time, params: BatteryParams, xp=jnp):
    """Which block of the schedule is live at *time*."""
    per_block = xp.maximum(params.dispatch_block_seconds / params.delta_t, 1.0)
    return xp.clip((time / per_block).astype(int), 0, N_DISPATCH_BLOCKS - 1)


def next_block_level(time, schedule, params: BatteryParams, xp=jnp):
    """The level of the block after the one live at *time*."""
    return schedule[
        xp.minimum(dispatch_block(time, params, xp) + 1, N_DISPATCH_BLOCKS - 1)
    ]


def minutes_to_next_block(time, params: BatteryParams):
    """Minutes until the next block goes live, seen at step *time*."""
    per_block = jnp.round(params.dispatch_block_seconds / params.delta_t)
    return (per_block - time % per_block) * params.delta_t / 60.0


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


def inverter_input(power, params: BatteryParams):
    """DC power the inverter needs for ``power`` W on its AC side. Its loss comes
    from the pack on discharge and out of what reaches the pack on charge."""
    return power + params.pcs_loss * power**2 / params.power_max


def inverter_output(dc_power, params: BatteryParams):
    """AC power for ``dc_power`` W at the pack: the inverse of ``inverter_input``."""
    k = params.pcs_loss / params.power_max
    return 2.0 * dc_power / (1.0 + jnp.sqrt(1.0 + 4.0 * k * dc_power))


def cycle_stress(depth, params: BatteryParams):
    """Life lost to one cycle of ``depth``: convex, so deep cycles dominate."""
    return params.stress_alpha * jnp.abs(depth) ** params.stress_beta


def half_cycle(anchor, peak, soc, params: BatteryParams):
    """The half-cycle after a move to ``soc``: ``(anchor, peak, life used)``.

    A half-cycle of depth ``d`` costs ``Phi(d) / 2``, spread over its depth as it
    grows. A swing back from the peak ends it once it passes ``cycle_hysteresis``,
    a racetrack filter, and a new one starts from the peak. Smaller swings are
    ripple, which ``degradation_rate`` charges by throughput.
    """
    heading = jnp.sign(peak - anchor)
    extends = (soc - peak) * heading > 0.0
    turns = jnp.logical_and(
        jnp.logical_not(extends), jnp.abs(soc - peak) > params.cycle_hysteresis
    )
    new_anchor = jnp.where(turns, peak, anchor)
    new_peak = jnp.where(jnp.logical_or(extends, turns), soc, peak)
    spent = jnp.where(turns, 0.0, cycle_stress(peak - anchor, params))
    life = 0.5 * (cycle_stress(new_peak - new_anchor, params) - spent)
    return new_anchor, new_peak, life


def degradation_rate(current, T_cell, params: BatteryParams):
    """Fractional capacity fade per second, calendar plus ripple.

    Calendar ageing follows an Arrhenius law in temperature, and ripple is
    proportional to charge throughput. Deep half-cycles come from ``half_cycle``.
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
    """RHS for ``[soc, v_rc, T_cell, q_loss]``. ``action`` is the inverter's AC
    power in W."""
    p = params
    soc, v_rc, T_cell = position[0], position[1], position[2]
    current = current_for_power(inverter_input(action, p), soc, v_rc, p)

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
    """``action_raw`` in [-1, 1] maps to the inverter's AC power in
    [-power_max, +power_max].

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

    current = current_for_power(inverter_input(power_cmd, p), soc, v_rc, p)
    dc_power = terminal_voltage(current, soc, v_rc, p) * current
    power = inverter_output(dc_power, p) - p.aux_load

    anchor, peak, life = half_cycle(state.soc_anchor, state.soc_peak, soc, p)
    deep_fade = p.life_fade * life

    # The signal for the *next* step: its block's level plus the AR(1) movement.
    # The innovation is drawn from a key folded with ``state.time``, so a caller
    # passing a constant key still gets a genuine zero-mean process.
    noise = jax.random.normal(jax.random.fold_in(key, state.time))
    regulation = (
        jnp.exp(-p.delta_t / p.regulation_tau) * state.regulation
        + p.dispatch_noise_std * noise
    )
    level = state.dispatch_schedule[dispatch_block(state.time + 1, p)]
    target = jnp.clip(level + regulation, -p.power_max, p.power_max)

    return (
        state.replace(
            soc=soc,
            v_rc=v_rc,
            T_cell=T_cell,
            q_loss=q_loss + deep_fade,
            current=current,
            power=power,
            target_power=target,
            regulation=regulation,
            soc_anchor=anchor,
            soc_peak=peak,
            deep_fade=deep_fade,
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
    """``[soc, V_cell, T_cell, P_MW, target_P_MW, next_block_MW, minutes_to_it]``.

    A battery management system measures state of charge (estimated from
    coulomb counting and voltage), terminal voltage, temperature and meter power,
    and the market publishes the next block. The diffusion-branch voltage and the
    accumulated capacity fade are hidden: neither is directly measurable.
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
            next_block_level(state.time, state.dispatch_schedule, params) / 1.0e6,
            minutes_to_next_block(state.time, params),
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


def compute_reward_terms(
    state: BatteryState, params: BatteryParams, xp=jnp, tripped=None
):
    """The reward's additive cost terms in $ per step (``target_gym.reward``).

    ``tripped`` overrides the trip check on ``state``, for a measured copy,
    whose noise must not fire or hide a trip (``base.measured_reward``).
    """
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
    fade = fade + state.deep_fade
    degradation = p.fade_price * p.pack_kWh * xp.maximum(fade - p.c_hold, 0.0)
    if tripped is None:
        tripped, _ = check_is_terminal(state, p, xp)
    terms = {
        "tracking": tracking,
        "running": degradation,
    }
    return R.with_trip(terms, tripped, R.trip_cost(p), xp)


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
    dc_power = inverter_input(power, params)
    current = current_for_power(dc_power, soc, 0.0, params)
    loss = current**2 * params.R0 + dc_power - power
    return 1.0 - loss / jnp.maximum(jnp.abs(power), 1.0)
