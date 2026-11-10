"""Compressor surge: a Greitzer compression system feeding a header.

See ``PHYSICS.md`` in this directory for provenance, the parameter table and
the validation targets. Method: ``docs/PHYSICS_METHODOLOGY.md``.

Topology::

                 recycle valve x (anti-surge), back to suction
          +-----------------<|>-------------------+
          |                                       |
    suction --> [ compressor, speed N ] --duct--> [ plenum / header, dp ] --> consumer valve u --> users
                                        m_c                                    (static head dp_out)

A lumped duct of length ``L_c`` carries the compressor flow ``m_c`` into a
plenum of volume ``V_p``, whose gauge pressure ``dp`` is the header pressure the
task tracks. The compressor's pressure rise is the Moore-Greitzer cubic in the
flow coefficient Phi, scaled by the fan laws. Gas leaves the plenum through
the consumers' valve against their static head and through the recycle valve
back to suction. The drive follows a rate-limited speed setpoint through a
first-order lag; the recycle valve is rate limited, faster opening than
closing. The consumers' opening follows a hidden schedule plus an
Ornstein-Uhlenbeck deviation. The plant trips when Phi falls below the
episode's surge line, a hidden offset right of the characteristic's peak.

State (three ODEs, two actuator positions, the disturbance, the trip variable
and the schedule)
------------------------------------------------------------------------------
``m_c``              compressor (duct) mass flow (kg/s), measured
``dp``               plenum gauge pressure (Pa), measured, observed in kPa
``N``                shaft speed as a fraction of rated, measured
``N_ramp``           drive setpoint after its rate limit, measured
``x``                recycle valve position (opening 0 to 1), measured
``demand_dev``       OU deviation of the consumer opening, hidden
``phi_min``          soft minimum of Phi over the last step's substeps, hidden
``setpoint_levels``  the episode's four header-pressure levels (Pa)
``demand_levels``    the episode's six consumer-opening levels, hidden
``block_clock``      steps since the schedule started, hidden; the plant's clock
``demand_ramp``      the schedule's ramp (s), as its scenario set it, hidden
``valve_cmds``       the recent valve commands, behind the positioner's dead time
``surge_times``      the last surges' times, read as the surge count

As a continuing plant the schedule renews when its six demand blocks have run
(``renew_schedule``): five fresh blocks from the level it holds, the clock five
blocks back. The valve's and the counter's times count on ``block_clock``, so no
clock in the plant grows without bound (long-strata draft 4, S1).
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax.tree_util import Partial as partial

from target_gym import reward as R
from target_gym.base import EnvParams, EnvState, actuator_field
from target_gym.integration import integrate_dynamics
from target_gym.utils import convert_raw_action_to_range, log_scaled_reward

#: Header-pressure blocks per episode (30 s each).
N_SETPOINT_BLOCKS = 4
#: Consumer-demand blocks per episode (20 s each).
N_DEMAND_BLOCKS = 6
#: Pa per kPa. The reward and the observation read pressure in kPa.
KPA = 1000.0
#: Commands the valve's dead time remembers: enough for 0.3 s, the top of
#: [C7]'s range, at a 5 ms step.
VALVE_HISTORY = 64
#: Salt for a schedule renewal's draw, apart from the deviation's.
RENEW = 0x52454E57


@struct.dataclass
class CompressorSurgeParams(EnvParams):
    # ---- Suction air (read: ISA sea level) ----
    P01: float = 101325.0  # Pa
    T01: float = 288.15  # K
    R_gas: float = 287.05  # J/(kg K)
    gamma: float = 1.4

    # ---- Geometry ----
    #: Duct flow area (ours). Sets the flow scale rho01 A_c U_r against which
    #: the valves, the static head and the setpoints are sized.
    A_c: float = 0.10  # m^2
    #: Duct length. TUNED: sensitivity rows at 0.5x and 2x (PHYSICS.md).
    L_c: float = 4.0  # m
    #: Plenum volume. TUNED: gives B = 1.80 at rated speed, the surge value of
    #: Gravdahl's thesis; sensitivity rows at 0.5x and 2x (PHYSICS.md).
    V_p: float = 15.0  # m^3
    #: Rated tip speed (ours). Sets the pressure scale rho01 U_r^2.
    U_r: float = 200.0  # m/s

    # ---- Characteristic (read: NASA CR-3878; Gravdahl thesis Table D.1) ----
    psi_c0: float = 0.30
    H: float = 0.18
    W: float = 0.25
    #: Where the cubic hands over to its tangent line (ours), just past the
    #: largest Phi the plant reaches, 0.777 (derived, PHYSICS.md).
    phi_join: float = 0.78

    # ---- Valves (ours) ----
    k_d: float = 0.12  # kg/(s Pa^0.5), consumer valve at opening 1
    dp_out: float = 10.0e3  # Pa, the consumers' static head
    #: Recycle valve. At full opening it passes 2.20 times the surge flow at
    #: any pressure (derived), the top of the 1.8 to 2.2 sizing range Mirsky
    #: et al. (2015) p.23 give.
    k_r: float = 0.15  # kg/(s Pa^0.5)
    #: Width of the smooth one-sided square root (the check valves).
    check_width: float = 50.0  # Pa

    # ---- Actuators ----
    N_min: float = 0.70  # of rated, speed command range (ours)
    N_max: float = 1.05  # of rated
    #: Drive setpoint rate limit. TUNED (PHYSICS.md, drive-rate unit).
    drive_rate: float = 0.03  # of rated per s
    #: Speed lag behind the rate-limited drive setpoint (ours).
    tau_N: float = 1.0  # s
    #: Full open in 2 s, dead time included (read-consistent: Mirsky et al.
    #: p.23; Wilson and Sheldon Table 1).
    valve_open_rate: float = 0.5  # per s
    #: The positioner and actuator lag ahead of the stroke, "within 150-300
    #: ms" [C7]: a dead time before the valve moves. Ours, the family's lower
    #: end: 0.15 s.
    valve_lag: float = 0.15  # s
    #: Full close in 10 s, the slowest closing Mirsky et al. p.23 allow.
    #: TUNED (PHYSICS.md sensitivity table).
    valve_close_rate: float = 0.1  # per s

    # ---- Schedules (ours) ----
    #: The header pressure the consumers need. Constant: the usual target is
    #: a constant suction or discharge pressure [C3, p. 4], and no source moves
    #: it. 24 kPa, the middle of the header this plant supplies (ours).
    header_setpoint: float = 24.0e3  # Pa
    #: The header zone: within +-5 % of the setpoint the consumers are served
    #: (ours: no band found). Outside it is priced (``zone_price``).
    header_band: float = 1.2e3  # Pa
    setpoint_block_steps: int = 300  # 30 s
    #: Linear ramp at the start of every block (ours): twice what the drive
    #: needs for the largest move. Under half a block, so that the scorer still
    #: reads each move (``benchmark.target_changes``).
    setpoint_ramp_s: float = 10.0  # s
    demand_range: Tuple[float, float] = (0.35, 0.95)  # opening
    demand_block_steps: int = 200  # 20 s
    #: Linear ramp at the start of demand blocks 2 to 6. Must be shorter than
    #: a block (``scheduled_demand``).
    demand_ramp_s: float = 5.0  # s
    demand_clip: Tuple[float, float] = (0.2, 1.0)  # total opening
    #: Stationary sd of the OU deviation of the consumer opening.
    demand_sigma: float = 0.02  # opening
    demand_theta: float = 0.2  # 1/s, reversion rate (5 s correlation)
    #: Clip on the initial deviation draw only, in standard deviations.
    demand_dev_reset_clip: float = 3.0
    # ---- The counted-surge trip (charter rule 14) ----
    #: A surge is a flow reversal [C1; C4]. Centrifugal machines trip on 3
    #: counts within 10 to 20 s [C5]; Siemens shuts down at the 3rd or 4th
    #: event [C3, p. 3]. Ours inside that range: 3 in 15 s.
    surge_trip_count: int = 3
    surge_window: float = 15.0  # s

    # ---- Reset (ours) ----
    N_reset: float = 1.0  # of rated
    #: Surge-safe for every opening 0.2 to 1.0 (PHYSICS.md, reset). The speed
    #: is then the one that holds the header setpoint (established operation).
    x_reset: float = 0.35
    #: Newton iterations of the reset equilibrium. Static: it is a Python loop
    #: count, so it must not be traced. A different value simply retraces.
    reset_newton_iters: int = struct.field(pytree_node=False, default=6)
    #: Temperature of the soft minimum over substeps. Makes the trip
    #: conservative by at most T ln(substeps), 2.3e-3 in Phi at 10 substeps.
    soft_min_temperature: float = 1e-3

    delta_t: float = 0.1  # s
    time_unit_seconds: float = 1.0
    max_steps_in_episode: int = 1200  # 120 s

    # ---- Score (charter rule 15; REALITY.md, "The operator's job") ----
    # Meet the header pressure, right of the surge line, with as little
    # recycle as possible and no counted surge [C3, pp. 1-2; C5]. In kJ of
    # recycle energy, the cost the margin trades against [C3, p. 1; C1]. No
    # price was found for the rest, so each is in kJ, ours:
    # - zone_price: a second outside the header zone costs 1,000 kJ, about 16 s
    #   of the hold's recycle power, so failing the consumers never pays;
    # - surge_price: each surge costs service life and efficiency [C3, p. 1],
    #   priced at 2,000 kJ;
    # - the trip (the counted surges) is 15 min of lost service at the worst
    #   surviving step's cost per step.
    reward_version: int = 3
    energy_price: float = 1.0  # per kJ of recycle energy
    zone_price: float = 1000.0  # kJ per s outside the zone
    surge_price: float = 2000.0  # kJ per surge
    failure_cost: float = 2200.0  # kJ per step, above any surviving step
    #: Restart time priced into a trip: 15 min (ours, provisional).
    restart_steps: int = 9000
    #: Read by the benchmark's time-in-band KPI, not by the reward: the header
    #: zone's half-width, kPa.
    e_floor: float = 1.2  # kPa
    #: Version 1 only: the transmitter's reference accuracy and the envelope.
    precision_floor: float = 0.0275  # kPa
    dp_error_max: float = 20.68  # kPa


#: Units of the CompressorSurgeParams fields, for the generated environment pages.
PARAM_UNITS = {
    "P01": "Pa",
    "T01": "K",
    "R_gas": "J/(kg K)",
    "gamma": "-",
    "A_c": "m^2",
    "L_c": "m",
    "V_p": "m^3",
    "U_r": "m/s",
    "psi_c0": "-",
    "H": "-",
    "W": "-",
    "phi_join": "-",
    "k_d": "kg/(s Pa^0.5)",
    "dp_out": "Pa",
    "k_r": "kg/(s Pa^0.5)",
    "check_width": "Pa",
    "N_min": "-",
    "N_max": "-",
    "drive_rate": "1/s",
    "tau_N": "s",
    "valve_open_rate": "1/s",
    "valve_close_rate": "1/s",
    "setpoint_block_steps": "steps",
    "setpoint_ramp_s": "s",
    "demand_block_steps": "steps",
    "demand_ramp_s": "s",
    "demand_sigma": "-",
    "demand_theta": "1/s",
    "demand_dev_reset_clip": "-",
    "N_reset": "-",
    "x_reset": "-",
    "reset_newton_iters": "-",
    "soft_min_temperature": "-",
    "time_unit_seconds": "s",
    "reward_version": "-",
    "energy_price": "1/kJ",
    "zone_price": "kJ/s",
    "surge_price": "kJ",
    "valve_lag": "s",
    "header_setpoint": "Pa",
    "header_band": "Pa",
    "surge_trip_count": "-",
    "surge_window": "s",
    "e_floor": "kPa",
    "dp_error_max": "kPa",
    "failure_cost": "kJ",
    "restart_steps": "steps",
    "precision_floor": "kPa",
}


@struct.dataclass
class CompressorSurgeState(EnvState):
    m_c: float
    dp: float
    N: float
    N_ramp: float
    x: float
    #: OU deviation of the consumer opening. Hidden. The value the coming step
    #: integrates with, which the MPC reads.
    demand_dev: float
    #: Soft minimum of Phi over the last step's substeps, read by the trip.
    phi_min: float
    #: The four header-pressure levels. Only the live one is observed.
    setpoint_levels: jnp.ndarray
    #: The six consumer-opening levels. Hidden.
    demand_levels: jnp.ndarray
    #: Steps since the schedule started. Hidden. The clock the valve's and the
    #: surge counter's times count on, so it is ``time`` in a window; a renewal
    #: (``renew_schedule``) moves it and them back together.
    block_clock: int
    #: The valve command through the positioner's dead time [C7]: the recent
    #: commands with the time (s) each was issued, newest first.
    valve_cmds: jax.Array = struct.field(
        default_factory=lambda: jnp.zeros((VALVE_HISTORY,)), kw_only=True
    )
    valve_cmd_times: jax.Array = struct.field(
        default_factory=lambda: jnp.full((VALVE_HISTORY,), -1.0e9), kw_only=True
    )
    #: The times (s) of the last ``surge_trip_count`` surges, newest first; a
    #: long-past time for none. The surge counter operators read [C5].
    surge_times: jax.Array = struct.field(
        default_factory=lambda: jnp.full((3,), -1.0e9), kw_only=True
    )
    #: Surges counted over the last step, for the reward.
    surges: float = 0.0
    #: The reset's own pressure above level 0 (Pa), so that the first move ramps
    #: like the others. 0 on a state built at its target.
    setpoint_lead: float = 0.0
    #: The demand schedule's ramp (s), set with its levels from the scenario's
    #: ``demand_ramp_s``, so a ramp in progress at a switch keeps its rate. A
    #: state built by hand has none and reads the params'.
    demand_ramp: float = struct.field(default=-1.0, kw_only=True)

    actuator_held: jax.Array = actuator_field(2)


# ---------------------------------------------------------------------------
# Derived constants. Functions, never fields, so they follow the params.
# ---------------------------------------------------------------------------


def suction_density(params: CompressorSurgeParams):
    """rho01 = P01 / (R T01), kg/m^3. Plain arithmetic, so it follows the
    params' own type (a Python float, or a traced array under ``jit``)."""
    return params.P01 / (params.R_gas * params.T01)


def sound_speed(params: CompressorSurgeParams, xp=jnp):
    """a01 = sqrt(gamma R T01), m/s, at suction temperature (deviation D3)."""
    return xp.sqrt(params.gamma * params.R_gas * params.T01)


def helmholtz_omega(params: CompressorSurgeParams, xp=jnp):
    """omega_H = a01 sqrt(A_c / (V_p L_c)), rad/s."""
    return sound_speed(params, xp) * xp.sqrt(params.A_c / (params.V_p * params.L_c))


def greitzer_B(N, params: CompressorSurgeParams, xp=jnp):
    """Greitzer's B = U / (2 omega_H L_c) = (U / 2 a01) sqrt(V_p / (A_c L_c)),
    with U = N U_r."""
    return N * params.U_r / (2.0 * helmholtz_omega(params, xp) * params.L_c)


def peak_head(params: CompressorSurgeParams):
    """Psi_c at the surge line Phi = 2W: psi_c0 + 2H."""
    return params.psi_c0 + 2.0 * params.H


def surge_flow_per_speed(params: CompressorSurgeParams):
    """Compressor flow on the surge line at rated speed, rho01 A_c U_r 2W
    (kg/s); the surge flow at speed N is N times it."""
    return suction_density(params) * params.A_c * params.U_r * 2.0 * params.W


def surge_constant(params: CompressorSurgeParams):
    """K in dp = K m^2 on the surge line, Pa/(kg/s)^2. The line is the peak at
    every speed, so the fan laws put it on this parabola."""
    return peak_head(params) / (
        suction_density(params) * params.A_c**2 * (2.0 * params.W) ** 2
    )


# ---------------------------------------------------------------------------
# Component models
# ---------------------------------------------------------------------------


def characteristic(phi, params: CompressorSurgeParams, xp=jnp):
    """Psi_c(Phi): the Moore-Greitzer cubic for Phi <= ``phi_join``, and its
    tangent line beyond, so the head keeps falling past the reachable range
    without the cubic's turn."""
    y = phi / params.W - 1.0
    cubic = params.psi_c0 + params.H * (1.0 + 1.5 * y - 0.5 * y**3)
    y_j = params.phi_join / params.W - 1.0
    psi_j = params.psi_c0 + params.H * (1.0 + 1.5 * y_j - 0.5 * y_j**3)
    slope_j = 1.5 * params.H / params.W * (1.0 - y_j**2)
    line = psi_j + slope_j * (phi - params.phi_join)
    return xp.where(phi > params.phi_join, line, cubic)


def check_valve_sqrt(z, params: CompressorSurgeParams, xp=jnp):
    """sqrt(s softplus(z / s)): sqrt(z) well above zero and a smooth leak near
    it (deviation D5), with s = ``check_width``.

    The softplus is the library's stable form (a literal log(1 + exp) overflows
    in float32 above z / s = 88), and the square root is guarded by a
    ``where``: softplus underflows to 0 far below zero, where a plain sqrt has
    an infinite derivative and would put NaNs into every gradient.
    """
    s = params.check_width
    if xp is np:
        v = s * np.logaddexp(0.0, z / s)
    else:
        v = s * jax.nn.softplus(z / s)
    positive = v > 0.0
    return xp.where(positive, xp.sqrt(xp.where(positive, v, 1.0)), 0.0)


def consumer_flow(dp, opening, params: CompressorSurgeParams, xp=jnp):
    """Delivered flow (kg/s) through the consumers' valve against their
    static head."""
    return params.k_d * opening * check_valve_sqrt(dp - params.dp_out, params, xp)


def recycle_flow(dp, x, params: CompressorSurgeParams, xp=jnp):
    """Recycle flow (kg/s) from the header back to suction."""
    return params.k_r * x * check_valve_sqrt(dp, params, xp)


def recycle_power(state: CompressorSurgeState, params: CompressorSurgeParams, xp=jnp):
    """Ideal compression power spent on recycled gas, m_r dp / rho01 (W): the
    running quantity, and the consumption scripts/measure_hold.py records."""
    return (
        recycle_flow(state.dp, state.x, params, xp) * state.dp / suction_density(params)
    )


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def ramped_levels(levels, tau, block_s, ramp_s, xp=jnp):
    """A schedule at ``tau`` seconds from its start.

    Block k holds level k after a linear ramp of ``ramp_s`` from level k - 1 at
    its start; block 0 holds level 0 and the last level holds past the last
    block. Written as a sum of ramps, which is the same function while a ramp
    is shorter than a block, is continuous in ``tau`` and needs no index, so
    the RK4 stages read it at their own times.
    """
    out = levels[..., 0]
    for k in range(1, levels.shape[-1]):
        w = xp.clip((tau - k * block_s) / ramp_s, 0.0, 1.0)
        out = out + (levels[..., k] - levels[..., k - 1]) * w
    return out


def scheduled_setpoint(levels, lead, tau, params: CompressorSurgeParams, xp=jnp):
    """The header-pressure target (Pa) at ``tau`` seconds from the start: the
    blocks, opening ``lead`` above level 0 and ramping onto it."""
    block_s = params.setpoint_block_steps * params.delta_t
    ramp_s = params.setpoint_ramp_s
    opening = lead * (1.0 - xp.clip(tau / ramp_s, 0.0, 1.0))
    return ramped_levels(levels, tau, block_s, ramp_s, xp) + opening


def live_target(state: CompressorSurgeState, params: CompressorSurgeParams, xp=jnp):
    """The header-pressure target (Pa) the state is scored against and
    observes."""
    tau = state.block_clock * params.delta_t
    return scheduled_setpoint(
        state.setpoint_levels, state.setpoint_lead, tau, params, xp
    )


def scheduled_demand(
    demand_levels, tau, params: CompressorSurgeParams, xp=jnp, ramp=None
):
    """The scheduled consumer opening at ``tau`` seconds from the schedule's
    start (``ramped_levels``), at the schedule's ``ramp`` (s), the params' by
    default. Where the two agree the params' is read, so a window computes
    exactly what it did before the schedule carried its ramp."""
    block_s = params.demand_block_steps * params.delta_t
    out = ramped_levels(demand_levels, tau, block_s, params.demand_ramp_s, xp)
    if ramp is None:
        return out
    own = ramped_levels(demand_levels, tau, block_s, ramp, xp)
    return xp.where(ramp == params.demand_ramp_s, out, own)


def demand_opening(
    demand_levels, tau, dev, params: CompressorSurgeParams, xp=jnp, ramp=None
):
    """The consumers' total opening: the schedule plus the deviation, clipped."""
    lo, hi = params.demand_clip
    scheduled = scheduled_demand(demand_levels, tau, params, xp, ramp)
    return xp.clip(scheduled + dev, lo, hi)


def schedule_ramp(state: CompressorSurgeState, params: CompressorSurgeParams):
    """The demand ramp (s) the state's schedule was drawn with; the params' for
    a state built by hand."""
    return jnp.where(state.demand_ramp > 0.0, state.demand_ramp, params.demand_ramp_s)


def renew_schedule(state: CompressorSurgeState, params, key, clock=None):
    """The schedule renewed (long-strata draft 4, S1): from the opening and the
    header setpoint it holds at ``clock`` (its own, by default), five fresh
    demand blocks drawn in ``demand_range`` at ``demand_ramp_s``, the clock at
    the start of the first of them. The valve's and the counter's times move
    back with the clock, so the dynamics see no change."""
    p = params
    clock = state.block_clock if clock is None else clock
    tau = clock * p.delta_t
    held = scheduled_demand(state.demand_levels, tau, p, ramp=schedule_ramp(state, p))
    setpoint = scheduled_setpoint(state.setpoint_levels, state.setpoint_lead, tau, p)
    u_lo, u_hi = p.demand_range
    fresh = jax.random.uniform(key, (N_DEMAND_BLOCKS - 1,), minval=u_lo, maxval=u_hi)
    start = jnp.asarray(p.demand_block_steps, jnp.int32)
    back = (state.block_clock - start) * p.delta_t
    return state.replace(
        demand_levels=jnp.concatenate([jnp.reshape(held, (1,)), fresh]),
        setpoint_levels=jnp.full_like(state.setpoint_levels, setpoint),
        setpoint_lead=jnp.zeros_like(state.setpoint_lead),
        demand_ramp=jnp.asarray(p.demand_ramp_s, jnp.float32),
        block_clock=start,
        valve_cmd_times=state.valve_cmd_times - back,
        surge_times=state.surge_times - back,
    )


def draw_schedule(key, params: CompressorSurgeParams):
    """The header setpoint, constant [C3, p. 4], as its four levels, and six
    iid uniform consumer-opening levels."""
    setpoint_key, demand_key = jax.random.split(key)
    u_lo, u_hi = params.demand_range
    setpoint_levels = jnp.full((N_SETPOINT_BLOCKS,), params.header_setpoint)
    demand_levels = jax.random.uniform(
        demand_key, (N_DEMAND_BLOCKS,), minval=u_lo, maxval=u_hi
    )
    return setpoint_levels, demand_levels


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------


def compute_velocity(
    position, action, N_ramp, x, dev, demand_levels, params, ramp=None
):
    """d/dt of ``[m_c, dp, N, tau]``.

    Duct:    dm_c/dt = (A_c / L_c) (rho01 U^2 Psi_c(Phi) - dp),
             U = N U_r, Phi = m_c / (rho01 A_c U)
    Plenum:  d dp/dt = (a01^2 / V_p) (m_c - m_d - m_r)
    Speed:   dN/dt   = (N_ramp - N) / tau_N
    Clock:   dtau/dt = 1, so each RK4 stage reads the demand at its own time.

    ``action`` is unused (the actuators enter through ``N_ramp`` and ``x``)
    and kept for the shared signature.
    """
    m_c, dp, N, tau = position[0], position[1], position[2], position[3]
    rho = suction_density(params)
    U = N * params.U_r
    phi = m_c / (rho * params.A_c * U)
    dp_c = rho * U**2 * characteristic(phi, params)
    opening = demand_opening(demand_levels, tau, dev, params, ramp=ramp)
    outflow = consumer_flow(dp, opening, params) + recycle_flow(dp, x, params)
    return (
        jnp.stack(
            [
                (params.A_c / params.L_c) * (dp_c - dp),
                (params.gamma * params.R_gas * params.T01 / params.V_p)
                * (m_c - outflow),
                (N_ramp - N) / params.tau_N,
                jnp.ones_like(tau),
            ]
        ),
        None,
    )


def _substeps(integration_method: str) -> int:
    """N of ``"rk4_N"``: each substep is one rate-limit update and one RK4
    step of ``delta_t / N``."""
    method, _, n = integration_method.partition("_")
    if method != "rk4" or not n.isdigit() or int(n) < 1:
        raise ValueError(
            f"integration_method must be 'rk4_N' with N >= 1, got {integration_method!r}"
        )
    return int(n)


def equilibrium_phi(N, x, opening, params: CompressorSurgeParams, iters, xp=jnp):
    """The right-branch equilibrium flow coefficient at speed ``N``, recycle
    ``x`` and consumer opening ``opening``, by ``iters`` Newton steps from Phi
    0.70 on rho01 A_c U Phi - m_d(dp) - m_r(dp) with dp = rho01 U^2 Psi_c(Phi).

    The derivative is forward-mode autodiff of that residual, elementwise, so
    the inputs may be arrays. The iteration runs in JAX; ``xp=np`` returns a
    NumPy array.
    """
    rho = suction_density(params)
    U = N * params.U_r

    def residual(phi):
        dp = rho * U**2 * characteristic(phi, params)
        return (
            rho * params.A_c * U * phi
            - consumer_flow(dp, opening, params)
            - recycle_flow(dp, x, params)
        )

    shape = jnp.broadcast_shapes(jnp.shape(N), jnp.shape(x), jnp.shape(opening))
    dtype = jnp.result_type(N, x, opening, float)
    phi = jnp.full(shape, 0.70, dtype=dtype)
    for _ in range(iters):
        f, df = jax.jvp(residual, (phi,), (jnp.ones_like(phi),))
        phi = phi - f / df
    return np.asarray(phi) if xp is np else phi


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state(
    action: jnp.ndarray,
    state: CompressorSurgeState,
    params: CompressorSurgeParams,
    key: jax.Array,
    integration_method: str = "rk4_10",
):
    """One step: the commands, the substeps, the trip variable, the
    deviation, then the clocks.

    action : ``[speed command, recycle command]``, raw in [-1, 1] (clipped);
             raw 0 is 87.5 % speed and a half-open recycle.
    """
    n_sub = _substeps(integration_method)
    # Past its six demand blocks the schedule renews, so the demand keeps
    # moving; a window never gets there (its N is the schedule's length).
    run_out = N_DEMAND_BLOCKS * params.demand_block_steps
    renewed = renew_schedule(
        state, params, jax.random.fold_in(jax.random.fold_in(key, RENEW), state.time)
    )
    state = jax.tree_util.tree_map(
        lambda a, b: jnp.where(state.block_clock >= run_out, a, b), renewed, state
    )
    ramp = schedule_ramp(state, params)
    action = jnp.reshape(action, (2,))
    N_cmd = convert_raw_action_to_range(action[0], params.N_min, params.N_max)
    x_cmd = convert_raw_action_to_range(action[1], 0.0, 1.0)
    h = params.delta_t / n_sub
    rho = suction_density(params)
    tau_0 = state.block_clock * params.delta_t

    t0 = state.block_clock * params.delta_t
    cmds = jnp.concatenate([jnp.reshape(x_cmd, (1,)), state.valve_cmds[:-1]])
    issued = jnp.concatenate([jnp.reshape(t0, (1,)), state.valve_cmd_times[:-1]])

    def delayed(tau):
        """The command the positioner passes at ``tau``: the newest issued at
        least ``valve_lag`` before, or the oldest kept."""
        ready = issued <= tau - params.valve_lag
        k = jnp.where(jnp.any(ready), jnp.argmax(ready), VALVE_HISTORY - 1)
        return cmds[k]

    def substep(carry, j):
        m_c, dp, N, N_ramp, x = carry
        # The actuators move first, by at most one substep's reach; the valve
        # after its positioner's dead time [C7], then at its stroke rate.
        N_ramp = N_ramp + jnp.clip(
            N_cmd - N_ramp, -params.drive_rate * h, params.drive_rate * h
        )
        x = x + jnp.clip(
            delayed(t0 + (j + 1) * h) - x,
            -params.valve_close_rate * h,
            params.valve_open_rate * h,
        )
        # The deviation is held at its start-of-step value, the value the MPC
        # reads; the schedule enters through the clock ``tau``.
        velocity = partial(
            compute_velocity,
            action=None,
            N_ramp=N_ramp,
            x=x,
            dev=state.demand_dev,
            demand_levels=state.demand_levels,
            params=params,
            ramp=ramp,
        )
        new, _ = integrate_dynamics(
            positions=jnp.stack([m_c, dp, N, tau_0 + j * h]),
            delta_t=h,
            method="rk4_1",
            compute_velocity=velocity,
        )
        phi = new[0] / (rho * params.A_c * params.U_r * new[2])
        # A surge: the flow turns from forward to reverse [C1; C4].
        reversal = jnp.logical_and(m_c >= 0.0, new[0] < 0.0)
        return (new[0], new[1], new[2], N_ramp, x), (phi, reversal)

    (m_c, dp, N, N_ramp, x), (phis, reversals) = jax.lax.scan(
        substep,
        (state.m_c, state.dp, state.N, state.N_ramp, state.x),
        jnp.arange(n_sub),
    )
    surged = jnp.any(reversals)
    now = (state.block_clock + 1) * params.delta_t
    surge_times = jnp.where(
        surged,
        jnp.concatenate([jnp.reshape(now, (1,)), state.surge_times[:-1]]),
        state.surge_times,
    )

    # The trip variable: a soft minimum of Phi over the substeps. The
    # max-shifted form is needed in float32, where exp(-Phi / T) underflows
    # and the plain form returns inf and never trips.
    T = params.soft_min_temperature
    phi_min = -T * jax.nn.logsumexp(-phis / T)

    # Exact Ornstein-Uhlenbeck update, after the integration: the step used
    # the deviation it started with, and the next one uses this draw. The key
    # is folded with the pre-increment ``time``, so a caller passing a
    # constant key still gets a zero-mean process.
    a = jnp.exp(-params.demand_theta * params.delta_t)
    xi = jax.random.normal(jax.random.fold_in(key, state.time))
    demand_dev = a * state.demand_dev + params.demand_sigma * jnp.sqrt(1.0 - a**2) * xi

    # The live setpoint and demand are derived from ``block_clock``, so
    # advancing it is the whole schedule update.
    return (
        state.replace(
            m_c=m_c,
            dp=dp,
            N=N,
            N_ramp=N_ramp,
            x=x,
            demand_dev=demand_dev,
            phi_min=phi_min,
            block_clock=state.block_clock + 1,
            valve_cmds=cmds,
            valve_cmd_times=issued,
            surge_times=surge_times,
            surges=surged.astype(m_c.dtype),
            time=state.time + 1,
        ),
        None,
    )


# Deliberately not jitted (the shipped envs' reason). Decorated with
# ``@partial(jax.jit, static_argnames=["params"])``, it would key the
# compilation cache on the params object, so a fresh ``Params(...)``, which
# every sweep, tuner and MPC builds, would be a cache miss and a full recompile,
# measured at ~1600x the cost of a cached call. Callers that want it fused
# already jit ``step_env``, which traces this inline.
def get_obs(state: CompressorSurgeState, params: CompressorSurgeParams):
    """``[dp (kPa), m_c (kg/s), m_d (kg/s), N (%), N_ramp (%), x (%), live
    setpoint (kPa)]``.

    ``m_d`` is the consumers' flow at the state's clock with the deviation the
    coming step integrates with, so the total opening can be inferred; its
    split into block level and deviation, both schedules' future values, the
    clock and ``phi_min`` are hidden. The surge margin is not a channel: the
    map's is ``obs[1] / (surge_flow_per_speed * obs[3] / 100) - 1``. The last
    channel is the surge counter over its window [C5].
    """
    opening = demand_opening(
        state.demand_levels,
        state.block_clock * params.delta_t,
        state.demand_dev,
        params,
        ramp=schedule_ramp(state, params),
    )
    return jnp.array(
        [
            state.dp / KPA,
            state.m_c,
            consumer_flow(state.dp, opening, params),
            100.0 * state.N,
            100.0 * state.N_ramp,
            100.0 * state.x,
            live_target(state, params) / KPA,
            surge_count(state, params).astype(state.dp.dtype),
        ]
    )


def surge_count(state: CompressorSurgeState, params: CompressorSurgeParams, xp=jnp):
    """Surges within the last ``surge_window`` seconds, as the counter reads."""
    now = state.block_clock * params.delta_t
    return xp.sum(state.surge_times > now - params.surge_window)


def check_is_terminal(
    state: CompressorSurgeState, params: CompressorSurgeParams, xp=jnp
):
    """The counted-surge trip [C5; C3, p. 3]: ``surge_trip_count`` surges
    within ``surge_window``. A NaN fails the comparison, so a non-finite
    proposal trips. No trip on the line itself: crossing it is the start of a
    surge, which is counted when the flow reverses."""
    counted = surge_count(state, params, xp) >= params.surge_trip_count
    finite = xp.isfinite(state.m_c) & xp.isfinite(state.dp)
    terminated = xp.logical_or(counted, xp.logical_not(finite))
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward_terms(
    state: CompressorSurgeState, params: CompressorSurgeParams, xp=jnp, tripped=None
):
    """The reward's additive cost terms, each >= 0, in kJ (params, "Score").

    - ``energy``: the recycle's power over the step [C3, p. 1; C1];
    - ``zone``: the header pressure outside its zone, per second;
    - ``surge``: each surge's wear [C3, p. 1];
    - ``failure``: the counted-surge trip, through ``reward.with_trip``.

    ``tripped`` overrides the trip check on ``state``, for a measured copy,
    whose noise must not fire or hide a trip (``base.measured_reward``).
    """
    p = params
    if tripped is None:
        tripped, _ = check_is_terminal(state, p, xp)
    off = xp.abs(live_target(state, p, xp) - state.dp) > p.header_band
    terms = {
        "energy": R.priced_cost(
            recycle_power(state, p, xp) * p.delta_t / 1000.0, p.energy_price, xp
        ),
        "zone": xp.where(off, p.zone_price * p.delta_t, 0.0),
        "surge": p.surge_price * state.surges,
    }
    return R.with_trip(terms, tripped, R.trip_cost(p), xp)


def compute_reward_v1(
    state: CompressorSurgeState, params: CompressorSurgeParams, xp=jnp
):
    """Version 1 (ours): log-scaled tracking, 1 at zero error and 0 at
    ``dp_error_max``, and ``-restart_steps`` on a tripped step.

    With 0 on the tripped step, a policy that trips often scored above safe
    ones (PHYSICS.md, version-1 reward). Charging the downtime at version 1's
    best per-step score, as version 2 prices it, puts the frequent trippers
    below every safe policy. Version 1 is therefore not bounded in [0, 1] on
    this task.
    """
    terminated, _ = check_is_terminal(state, params, xp)
    tracking = log_scaled_reward(
        xp.abs(live_target(state, params, xp) - state.dp) / KPA,
        params.precision_floor,
        params.dp_error_max,
        xp,
    )
    return xp.where(terminated, -1.0 * params.restart_steps, tracking)


def compute_reward(state: CompressorSurgeState, params: CompressorSurgeParams, xp=jnp):
    return R.select(
        params.reward_version,
        compute_reward_v1(state, params, xp),
        R.total(compute_reward_terms(state, params, xp), xp),
        xp,
    )
