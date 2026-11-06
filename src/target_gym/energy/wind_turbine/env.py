"""
Wind turbine — NREL 5 MW reference turbine, collective-pitch power regulation.

See ``PHYSICS.md`` in this directory for provenance, the sourced parameter
table and validation targets. Method: ``docs/PHYSICS_METHODOLOGY.md``.

Model
-----
Single-mass drivetrain driven by rotor aerodynamics, with pitch and generator
torque as the two manipulated variables::

    J dw/dt = tau_aero - N tau_gen
    tau_aero = 0.5 rho A v^3 Cp(lambda, beta) / w        lambda = w R / v
    P_elec   = eta N tau_gen w

``J`` is the drivetrain inertia referred to the rotor, ``N`` the gearbox ratio.
Pitch and generator torque follow their demands through first-order actuators,
both rate-limited: the pitch rate binds in gusts, the torque rate on a dispatch
move.

Task
----
Track a **power setpoint** in above-rated wind -- the curtailed-operation
problem a modern turbine actually faces when the grid dispatches it below its
available power. The turbine must follow that dispatch, which moves every 25 s,
while turbulence moves the available power around it, without overspeeding the
rotor.

Why it is a hard target MDP
---------------------------
* **The rotor-effective wind is hidden.** A nacelle anemometer sits in the
  rotor's own wake and is famously unreliable, so the controller has to infer
  what the wind is doing from rotor speed and power. That is a real estimation
  problem, not an artificial restriction.
* **Two inputs, coupled through one inertia.** Pitch spills aerodynamic power;
  generator torque converts it. Both change rotor speed, and the useful
  combination depends on where the turbine sits on the Cp surface.
* **Pitch is rate-limited.** During a gust the controller cannot pitch as fast
  as the wind changes, so it must anticipate rather than react.
* **Overspeed is irrecoverable.** Losing rotor-speed regulation trips the
  turbine, and the reward cannot buy that back.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.scipy.ndimage import map_coordinates
from jax.tree_util import Partial as partial

from target_gym import reward as R
from target_gym.base import EnvParams, EnvState, actuator_field
from target_gym.energy.wind_turbine.cp_table import (
    CP_TABLE,
    PITCH_STEP,
    TSR_FIRST,
    TSR_STEP,
)
from target_gym.integration import integrate_dynamics
from target_gym.utils import convert_raw_action_to_range, log_scaled_reward

RPM_PER_RAD_S = 60.0 / (2.0 * jnp.pi)

# Ornstein-Uhlenbeck turbulence: mean-reversion rate (1/s). 1/theta ~ 20 s, a
# gust timescale rather than white noise.
WIND_OU_THETA = 0.05
#: Dispatch levels per episode (``target_block_steps`` each).
N_TARGET_BLOCKS = 4


@struct.dataclass
class WindTurbineParams(EnvParams):
    # ---- Rotor ----
    R: float = 63.0  # m, rotor radius (126 m diameter)
    rho: float = 1.225  # kg/m^3, air density

    # ---- Drivetrain ----
    J_rotor: float = 38_759_228.0  # kg m^2
    J_gen: float = 534.116  # kg m^2, generator side
    N_gear: float = 97.0  # gearbox ratio
    eta_gen: float = 0.944  # generator efficiency

    # ---- Ratings ----
    P_rated: float = 5.0e6  # W
    v_rated: float = 11.4  # m/s
    omega_rated_rpm: float = 12.1
    v_cut_out: float = 25.0  # m/s

    # ---- Actuators ----
    pitch_min: float = 0.0  # deg
    pitch_max: float = 40.0  # deg
    pitch_rate_max: float = 8.0  # deg/s -- the binding constraint in gusts
    pitch_tau: float = 0.2  # s, actuator lag
    torque_max: float = 47_400.0  # N m, generator side (~110 % of rated)
    torque_rate_max: float = 15_000.0  # N m/s (Jonkman et al. 2009, Table 7-2)
    torque_tau: float = 0.1  # s, power-electronics lag

    # ---- Wind ----
    # Comfortably above the 11.4 m/s rated wind: with 1.2 m/s turbulence a
    # mean of 12 dips below rated regularly, where the power setpoint is
    # physically unachievable and the rotor stalls no matter what the
    # controller does. 13.5 keeps the task well-posed.
    v_mean_range: Tuple[float, float] = (13.5, 20.0)
    turbulence_std: float = 1.2  # m/s, OU stationary std
    v_min: float = 4.0
    v_max: float = 28.0

    # ---- Operating / termination bounds ----
    overspeed_factor: float = 1.25  # trip above this multiple of rated speed
    #: Rotor speed below which the episode ends, as a fraction of rated. Not an
    #: arbitrary guard: as the rotor slows in above-rated wind the tip-speed
    #: ratio ``lambda = omega R / v`` collapses, and the Cp surface falls away
    #: at low lambda, so aerodynamic torque drops and the rotor slows further.
    #: That is an aerodynamic stall of the rotor and it does not recover without
    #: external drive. It also guards a singularity: ``tau_aero = P / omega``
    #: diverges as omega approaches zero, so letting it run to a stop would buy
    #: a numerical failure rather than a physical one.
    underspeed_factor: float = 0.40

    # ---- Reward shaping ----
    # Error scale for the version-1 gradient planner's tracking term, not read
    # by ``compute_reward``. See "Why the MPC does not minimise the reward" in
    # docs/baselines.md. The version-2 oracle, on the shared solver, does not
    # read it either. The version-1 reward's envelope is ``power_envelope``
    # below.
    power_band: float = 0.5e6  # W
    # The ceiling's resolution and version 1's floor: the 1.7 kW a class 0.1S
    # meter can certify at 5 MW (PHYSICS.md, Readings). The reading's own scatter
    # is the benchmark's assumed 1 kW.
    precision_floor: float = 1.7e3  # W
    power_envelope: float = 5e6  # W, rated electrical output
    pitch_activity_weight: float = 0.02  # fatigue proxy

    # ---- Targets ----
    target_power_range: Tuple[float, float] = (3.5e6, 5.0e6)
    #: The dispatch's iid levels each hold this long, 25 s (ours).
    target_block_steps: int = 100

    # ---- Time discretization ----
    # Rotor time constant J w^2 / P ~ 14 s, so dt = 0.25 s gives ~56 steps per
    # time constant. 1200 steps = 5 min.
    delta_t: float = 0.25
    max_steps_in_episode: int = 400

    # ---- Reward (docs/reward-shaping.md; version 2), in dollars per step ----
    # Tracking: an electrical imbalance settled at ``imbalance_price`` per
    # MWh, linear in |error| (p = 1). Floor: the lowest hold error a shipped
    # controller had demonstrated under the shipped OU turbulence when it was
    # set, the gradient planner's lowest per-seed mean |error| over the 300
    # hold steps of the test episode: 1.68 kW on seed 1 of five. A per-seed
    # minimum rather than a mean, so that no run of the reference controller
    # sits below it. An upper bound on the achievable floor. The oracle that
    # replaced the planner holds 1454 W on its best
    # seed and 1749 W on average, the PID 4563 W (`scripts/measure_hold.py`,
    # 400 hold steps after a 300-step burn-in, 3 seeds). 1680 W is within
    # 1.5x of that best hold (1.16x), so the floor, and with it the reward,
    # was kept; only ``rho_floor`` below follows the new hold. Fatigue:
    # pitch travel per step / pitch_max above the expert's hold-phase level
    # (``c_hold``) is charged, per unit of avoidable fraction, at
    # ``fatigue_weight`` times what tracking at the floor costs per step --
    # a documented stand-in for a maintenance model's price (provisional;
    # sweep 0.5 / 1 / 2). The $80/MWh imbalance price is the reactor's spot
    # price (provisional). An overspeed or underspeed trip is charged, for
    # every step the episode had left, twice the imbalance of the 7 MW the
    # overspeed limit and torque_max allow (the rated 5 MW is not the envelope).
    reward_version: int = 2
    e_floor: float = (
        1680.0  # W, an earlier oracle's lowest per-seed hold error (upper bound)
    )
    e_tol: float = 0.0
    tracking_exponent: float = 1.0
    imbalance_price: float = (
        100.0  # $/MWh, provisional; the one imbalance price of the priced plants
    )
    c_hold: float = 0.0023  # pitch travel per step / pitch_max, the oracle's hold
    fatigue_weight: float = 1.0  # provisional; sweep 0.5 / 1 / 2
    #: Rotor speed, as a multiple of rated, above which every step pays (ours).
    speed_band: float = 1.10
    #: Floor-widths per step per unit of speed ratio above the band (ours,
    #: provisional): 1.15 x rated costs ten floor-widths, like the planner's box.
    speed_weight: float = 200.0
    failure_cost: float = 2.0 * 100.0 * 7.0 * 0.25 / 3600.0
    #: Restart time priced into a trip (``reward.trip_cost``; 10 min at 0.25 s steps: an overspeed trip's reset and re-synchronisation, provisional).
    restart_steps: int = 2400
    #: The oracle's best hold: the tracking cost per step, in the reward's units, of
    #: the oracle's lowest per-seed hold, 1454 W (seed 0, `scripts/measure_hold.py`;
    #: (1454 / 1680)^1 = 0.865 of the cost at ``e_floor``).
    rho_floor_tracking: float = 100.0 / 1.0e6 * 0.25 / 3600.0 * 1454.0
    rho_floor: float = 100.0 / 1.0e6 * 0.25 / 3600.0 * 1454.0
    #: True where e_floor is a resolution, not a measured or certified floor.
    floor_is_documented_minimum: bool = False


#: Units of the WindTurbineParams fields, for the generated environment pages.
PARAM_UNITS = {
    "R": "m",
    "rho": "kg/m^3",
    "J_rotor": "kg m^2",
    "J_gen": "kg m^2",
    "N_gear": "-",
    "eta_gen": "-",
    "P_rated": "W",
    "v_rated": "m/s",
    "omega_rated_rpm": "rpm",
    "v_cut_out": "m/s",
    "pitch_min": "deg",
    "pitch_max": "deg",
    "pitch_rate_max": "deg/s",
    "pitch_tau": "s",
    "torque_max": "N m",
    "torque_rate_max": "N m/s",
    "torque_tau": "s",
    "target_block_steps": "steps",
    "turbulence_std": "m/s",
    "v_min": "m/s",
    "v_max": "m/s",
    "overspeed_factor": "-",
    "underspeed_factor": "-",
    "power_band": "W",
    "precision_floor": "W",
    "power_envelope": "W",
    "pitch_activity_weight": "-",
    "reward_version": "-",
    "e_floor": "W",
    "e_tol": "W",
    "tracking_exponent": "-",
    "imbalance_price": "$/MWh",
    "c_hold": "-",
    "fatigue_weight": "-",
    "speed_band": "-",
    "speed_weight": "-",
    "failure_cost": "reward",
    "restart_steps": "steps",
    "rho_floor_tracking": "reward",
    "rho_floor": "reward",
}


@struct.dataclass
class WindTurbineState(EnvState):
    omega: float  # rotor speed (rad/s)
    pitch: float  # blade pitch (deg), actuator output
    torque: float  # generator torque (N m), actuator output

    v_wind: float  # rotor-effective wind (m/s) -- HIDDEN
    v_mean: float  # episode mean wind (m/s) -- HIDDEN

    pitch_cmd: float
    torque_cmd: float
    target_power: float
    #: Electrical output (W), always ``electrical_power(omega, torque)``. A field
    #: so the meter can be its own reading (PHYSICS.md, Readings).
    power: float
    #: The dispatch's levels. Only the live one, ``target_power``, is observed.
    target_levels: jax.Array = struct.field(
        default_factory=lambda: jnp.zeros(N_TARGET_BLOCKS)
    )
    #: Steps since the schedule started. ``reset_env`` sets it to 0, so a trip's
    #: fresh draw restarts the schedule while ``time`` runs on.
    block_clock: int = 0
    #: How far the pitch moved over the last step (deg): the wear the reward reads.
    pitch_travel: float = 0.0

    actuator_held: jax.Array = actuator_field(2)


def rotor_area(params: WindTurbineParams):
    return jnp.pi * params.R**2


def omega_rated(params: WindTurbineParams):
    return params.omega_rated_rpm * 2.0 * jnp.pi / 60.0


def power_coefficient(tip_speed_ratio, pitch_deg, params: WindTurbineParams):
    """Cp(lambda, beta) from the published table (``cp_table``), bilinear
    between its nodes and held at its edges. Clipped at zero: the table goes
    negative at high pitch, where the rotor brakes (PHYSICS.md, D1).
    """
    lam, beta = jnp.broadcast_arrays(tip_speed_ratio, pitch_deg)
    cp = map_coordinates(
        jnp.asarray(CP_TABLE, lam.dtype),
        [(lam - TSR_FIRST) / TSR_STEP, beta / PITCH_STEP],
        order=1,
        mode="nearest",
    )
    return jnp.clip(cp, 0.0, 0.593)  # Betz limit as a hard ceiling


def aerodynamic_torque(omega, v_wind, pitch_deg, params: WindTurbineParams):
    """Rotor torque from the momentum balance."""
    w = jnp.maximum(omega, 1e-3)
    lam = w * params.R / jnp.maximum(v_wind, 1e-3)
    cp = power_coefficient(lam, pitch_deg, params)
    power = 0.5 * params.rho * rotor_area(params) * v_wind**3 * cp
    return power / w, power


def balance_pitch(omega, v_wind, power, params: WindTurbineParams):
    """The feather-side pitch leaving ``power`` (W) in the wind at this speed,
    to 0.1 deg: scanning up from fine pitch, the first below it."""
    grid = jnp.arange(301.0) * 0.1
    _, available = aerodynamic_torque(omega, v_wind, grid, params)
    return grid[jnp.argmax(available < power)]


def electrical_power(omega, torque, params: WindTurbineParams):
    """Generator electrical output; torque is generator-side."""
    return params.eta_gen * params.N_gear * torque * omega


def drivetrain_inertia(params: WindTurbineParams):
    """Inertia referred to the rotor shaft."""
    return params.J_rotor + params.N_gear**2 * params.J_gen


def rated_generator_torque(params: WindTurbineParams):
    """Generator torque that delivers rated power at rated speed."""
    return params.P_rated / (params.eta_gen * params.N_gear * omega_rated(params))


def compute_velocity(position, action, v_wind, params: WindTurbineParams):
    """RHS for ``[omega, pitch, torque]``. ``action`` is ``(pitch_cmd, torque_cmd)``."""
    p = params
    omega, pitch, torque = position[0], position[1], position[2]
    pitch_cmd, torque_cmd = action

    tau_aero, _ = aerodynamic_torque(omega, v_wind, pitch, p)
    domega = (tau_aero - p.N_gear * torque) / drivetrain_inertia(p)

    # Pitch actuator: first-order toward the demand, then rate-limited. The
    # rate limit is the binding constraint during a gust.
    dpitch = jnp.clip(
        (pitch_cmd - pitch) / p.pitch_tau, -p.pitch_rate_max, p.pitch_rate_max
    )
    dtorque = jnp.clip(
        (torque_cmd - torque) / p.torque_tau, -p.torque_rate_max, p.torque_rate_max
    )
    return jnp.array([domega, dpitch, dtorque]), None


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state(
    action_raw: jnp.ndarray,
    state: WindTurbineState,
    params: WindTurbineParams,
    key: jax.Array,
    integration_method: str = "rk4_2",
):
    """``action_raw`` is ``[pitch_raw, torque_raw]`` in [-1, 1]."""
    p = params
    pitch_cmd = convert_raw_action_to_range(
        action_raw[0], min_action=p.pitch_min, max_action=p.pitch_max
    )
    torque_cmd = convert_raw_action_to_range(
        action_raw[1], min_action=0.0, max_action=p.torque_max
    )

    # OU turbulence about the episode mean. Drawn from a key folded with
    # ``state.time`` so a caller passing a constant key still gets a genuine
    # zero-mean process.
    noise = jax.random.normal(jax.random.fold_in(key, state.time))
    sigma = p.turbulence_std * jnp.sqrt(2.0 * WIND_OU_THETA * p.delta_t)
    v_wind = jnp.clip(
        state.v_wind
        + WIND_OU_THETA * (state.v_mean - state.v_wind) * p.delta_t
        + sigma * noise,
        p.v_min,
        p.v_max,
    )

    _compute_velocity = partial(
        compute_velocity, action=(pitch_cmd, torque_cmd), v_wind=v_wind, params=p
    )
    new_positions, _ = integrate_dynamics(
        positions=jnp.array([state.omega, state.pitch, state.torque]),
        delta_t=p.delta_t,
        compute_velocity=_compute_velocity,
        method=integration_method,
    )
    omega = jnp.maximum(new_positions[0], 1e-3)
    pitch = jnp.clip(new_positions[1], p.pitch_min, p.pitch_max)
    torque = jnp.clip(new_positions[2], 0.0, p.torque_max)
    clock = state.block_clock + 1
    block = jnp.minimum(clock // p.target_block_steps, N_TARGET_BLOCKS - 1)

    return (
        state.replace(
            omega=omega,
            pitch=pitch,
            torque=torque,
            v_wind=v_wind,
            pitch_cmd=pitch_cmd,
            torque_cmd=torque_cmd,
            power=electrical_power(omega, torque, p),
            target_power=state.target_levels[block],
            block_clock=clock,
            pitch_travel=jnp.abs(pitch - state.pitch),
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
def get_obs(state: WindTurbineState, params: WindTurbineParams):
    """``[omega_rpm, pitch_deg, torque_pct, P_MW, target_P_MW]``.

    A turbine measures rotor speed, pitch, generator torque and electrical
    power. It does *not* reliably measure the rotor-effective wind -- a
    nacelle anemometer sits in the rotor's own wake -- so ``v_wind`` is hidden
    and must be inferred from how the rotor is behaving.
    """
    p = params
    return jnp.array(
        [
            state.omega * 60.0 / (2.0 * jnp.pi),
            state.pitch,
            100.0 * state.torque / p.torque_max,
            state.power / 1.0e6,
            state.target_power / 1.0e6,
        ]
    )


def check_is_terminal(state: WindTurbineState, params: WindTurbineParams, xp=jnp):
    w_rated = omega_rated(params)
    terminated = xp.logical_or(
        state.omega >= params.overspeed_factor * w_rated,
        state.omega <= params.underspeed_factor * w_rated,
    )
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward_terms(
    state: WindTurbineState, params: WindTurbineParams, xp=jnp, tripped=None
):
    """The reward's additive cost terms in $ per step (``target_gym.reward``).

    ``tripped`` marks a measured copy (``base.measured_reward``): its trip comes
    from the true state, and its power from the meter. Speed times torque,
    through their own sensors, would scatter about 10.6 kW against its 1 kW.
    """
    p = params
    power = electrical_power(state.omega, state.torque, p)
    if tripped is None:
        tripped, _ = check_is_terminal(state, p, xp)
    else:
        power = state.power
    per_W_step = p.imbalance_price / 1.0e6 * p.delta_t / 3600.0  # $ per W per step
    tracking = (
        per_W_step
        * p.e_floor
        * R.tracking_cost(
            state.target_power - power, p.e_floor, p.e_tol, p.tracking_exponent, xp
        )
    )
    fatigue = R.excess_cost(
        state.pitch_travel / p.pitch_max, p.c_hold, p.fatigue_weight, xp
    )
    overspeed = xp.maximum(state.omega / omega_rated(p) - p.speed_band, 0.0)
    terms = {
        "tracking": tracking,
        "running": per_W_step * p.e_floor * (fatigue + p.speed_weight * overspeed),
    }
    return R.with_trip(terms, tripped, R.trip_cost(p), xp)


def compute_reward_v1(state: WindTurbineState, params: WindTurbineParams, xp=jnp):
    """Power tracking minus a pitch-activity penalty (a fatigue proxy)."""
    power = electrical_power(state.omega, state.torque, params)
    err = xp.abs(state.target_power - power)
    tracking = log_scaled_reward(err, params.precision_floor, params.power_envelope, xp)
    # Command minus achieved, not pitch motion. Pitch follows its command
    # through a first-order actuator *and* a rate limit, so this quantity is
    # large exactly when the command is running ahead of what the actuator can
    # deliver -- it charges for *futile* commands rather than for using the
    # actuator at all, which a controller that must pitch has to do. The
    # incentive it creates is "stay inside the achievable envelope", which is
    # the same thing as "anticipate rather than react", and that is what this
    # task is about.
    #
    # The reactor's rod penalty now uses the same form, for the same reason:
    # its rods are rate-limited too, and its previous ``|rho_ext|`` charged for
    # holding the rods where the physics requires them.
    activity = xp.abs(state.pitch_cmd - state.pitch) / params.pitch_max
    return tracking * (1.0 - params.pitch_activity_weight * activity)


def compute_reward(state: WindTurbineState, params: WindTurbineParams, xp=jnp):
    return R.select(
        params.reward_version,
        compute_reward_v1(state, params, xp),
        R.total(compute_reward_terms(state, params, xp), xp),
        xp,
    )


def available_power(v_wind, omega, params: WindTurbineParams):
    """Best electrical power extractable at this wind and rotor speed."""
    lam = omega * params.R / jnp.maximum(v_wind, 1e-3)
    cp = power_coefficient(lam, 0.0, params)
    return params.eta_gen * 0.5 * params.rho * rotor_area(params) * v_wind**3 * cp
