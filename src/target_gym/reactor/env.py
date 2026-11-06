"""
Nuclear reactor: point kinetics with delayed neutrons, xenon poisoning, thermal
feedback, a steam generator driven by the turbine, and rate-limited control rods.

Physics
-------
Point-kinetics equations (PKE) for neutron density ``n`` (normalised) and the
six delayed-neutron precursor concentrations ``C_i``::

    dn/dt   = ((rho - beta) / Lambda) * n + sum_i lambda_i * C_i
    dC_i/dt = (beta_i / Lambda) * n - lambda_i * C_i        (i = 1..6)

``beta_i`` / ``lambda_i`` are the standard U-235 thermal 6-group constants.
``beta = sum beta_i ≈ 0.0065``.

Reactivity has an external part (control rod), thermal feedback, and xenon::

    rho = rho_ext
        + alpha_fuel * (T_fuel - T_fuel_ref)
        + alpha_coolant * (T_avg - T_coolant_ref)
        - rho_Xe_full * (Xe_hat - 1)

Both temperature coefficients are *negative* (Doppler + moderator), which is
what makes a PWR follow its turbine. The xenon term adds a slow, history-dependent
reactivity swing.

Xenon-135 / Iodine-135 fission product chain
---------------------------------------------
Xe-135 has the largest thermal neutron absorption cross-section of any nuclide
(~2.65 × 10⁻¹⁸ cm²). It is produced both directly from fission and indirectly
via I-135 decay (the dominant path). After a power reduction, Xe builds up
because iodine keeps decaying into xenon while burnup (which destroys Xe) drops.
This creates the classic "xenon pit" — a delayed negative reactivity insertion
that can make power recovery impossible for hours.

Using normalised concentrations (I_hat = I/I_eq, Xe_hat = Xe/Xe_eq where eq is
the equilibrium at full power n=1)::

    dI_hat/dt  = lambda_I * (n - I_hat)
    dXe_hat/dt = a * n + b * I_hat - (lambda_Xe + sigma_phi0 * n) * Xe_hat

    a = gamma_ratio * (lambda_Xe + sigma_phi0) / (1 + gamma_ratio)
    b = (lambda_Xe + sigma_phi0) / (1 + gamma_ratio)

At equilibrium (n=1): I_hat=1, Xe_hat=1, rho_Xe=0 (absorbed into reference).

Rod speed
---------
The jacks have one speed: 72 steps/min over a 228-step stroke, both ways. The
action is the *desired* rod position, and the actual rod reactivity moves toward
it at that speed::

    rho_ext = clamp(rho_ext + clip(desired - rho_ext, -step, +step))

Thermal model, with the secondary side::

    C_fuel    dT_fuel/dt = P_ref n - UA (T_fuel - T_hot)
    C_coolant dT_hot/dt  = UA (T_fuel - T_hot) - m_dot_cp (T_hot - T_cold)
    C_cold    dT_cold/dt = m_dot_cp (T_hot - T_cold) - UA_sg (T_avg - T_sg)
    C_sg      dT_sg/dt   = UA_sg (T_avg - T_sg) - P_ref load

The turbine sets ``load``. Energy balance then fixes the steady power at the load
whatever the rods do, so what the rods set is ``T_avg``, and that is the task: hold
it on its program while the turbine goes 100-50-100.

Why this is interesting for RL
------------------------------
* **Xenon is a multi-hour disturbance nobody measures.** It moves the rods' balance
  by hundreds of pcm through the day, and the load schedule decides how much.
* **The rods have a stop.** Their 1500 pcm cannot cover the whole swing after a
  mid-length hold, so Tavg sags at the return.
* **Rod travel wears the mechanism.** Holding tighter costs steps, which is why a
  real rod program has a dead band.
* **Hidden states.** Precursors, fuel temperature, iodine, xenon and the steam
  generator are unobserved.
* **Stiff + slow coupling.** Prompt neutronics (ms), precursors (s to min), thermal
  (min), xenon (hours).

Observation (partially observable)
----------------------------------
``[n, T_hot, rho_ext_norm, load, T_avg, T_ref]``.

Action
------
Scalar in ``[-1, 1]``: the *desired* control-rod reactivity. The rods move toward
it at their one speed.
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

# ---------------------------------------------------------------------------
# Standard 6-group U-235 thermal delayed-neutron data (Keepin 1965).
# ---------------------------------------------------------------------------
BETA_I = jnp.array(
    [2.15e-4, 1.424e-3, 1.274e-3, 2.568e-3, 7.48e-4, 2.73e-4],
    dtype=jnp.float32,
)
LAMBDA_I = jnp.array(
    [0.0124, 0.0305, 0.1110, 0.3010, 1.1400, 3.0100],
    dtype=jnp.float32,
)
BETA_TOT = float(BETA_I.sum())  # ≈ 0.00650
N_GROUPS = 6

# ---------------------------------------------------------------------------
# Xenon / Iodine nuclear data
# ---------------------------------------------------------------------------
# I-135 half-life: 6.57 hours → decay constant
LAMBDA_IODINE = 2.93e-5  # 1/s  (ln2 / (6.57 * 3600))
# Xe-135 half-life: 9.14 hours → decay constant
LAMBDA_XENON = 2.11e-5  # 1/s  (ln2 / (9.14 * 3600))


@struct.dataclass
class ReactorParams(EnvParams):
    # ---- Kinetics ----
    Lambda_gen: float = 1e-4  # s — effective mean neutron generation time

    # Thermal feedback coefficients (1/K). Both negative (stabilising).
    alpha_fuel: float = -3.0e-5  # Doppler (prompt, on fuel temperature)
    alpha_coolant: float = -30.0e-5  # Moderator temperature coefficient, on Tavg
    # Zero feedback at full power on program but for 40 pcm: rods sit 40 pcm in
    # there, with +540 pcm above them (PHYSICS.md, section 2).
    T_fuel_ref: float = 1064.0  # K
    T_coolant_ref: float = 580.2  # K, Tavg at full load

    # ---- External (control rod) reactivity range ----
    # A 1500 pcm bank, safely below prompt-critical (beta = 650 pcm) at the top.
    rho_ext_min: float = -0.010  # rods fully inserted
    rho_ext_max: float = 0.005  # rods fully withdrawn

    # ---- Rod speed ----
    # 72 steps/min over the 228-step stroke of that bank, in either direction:
    # a magnetic jack has one speed. That is 7.89 pcm/s.
    rod_speed: float = 72.0 / 60.0 / 228.0 * 0.015  # 1/s
    rod_stroke_steps: float = 228.0

    # ---- Thermal model: fuel, hot leg, cold leg, steam generator ----
    P_thermal_ref: float = 3.0e9  # W, nominal thermal power (3 GW PWR)
    C_fuel: float = 3.3e7  # J/K  (~100 t UO2 x c_p)
    C_coolant: float = 4.7e8  # J/K, a third of the primary, the core and hot leg
    C_cold: float = 9.3e8  # J/K, the rest, the steam generator tubes and cold leg
    UA: float = 6.6e6  # W/K  (fuel to coolant heat transfer)
    m_dot_cp: float = 9.0e7  # W/K  (mass flow x c_p of primary loop)
    UA_sg: float = 3.0e9 / 22.0  # W/K, 22 K from Tavg to the steam at full load
    C_sg: float = 1.0e9  # J/K, secondary water and tube metal
    # The Tavg program, 557 F to 584.7 F from no load to full load.
    T_avg_no_load: float = 564.8  # K
    T_avg_full_load: float = 580.2  # K

    # ---- Xenon / Iodine ----
    # sigma_phi0: Xe-135 neutron absorption rate at full power (sigma_a_Xe × Phi_0).
    # With sigma_a = 2.65e-18 cm² and Phi_0 ≈ 3e13 n/cm²/s for a 3 GW PWR.
    sigma_phi0: float = 7.95e-5  # 1/s
    # gamma_ratio: ratio of direct Xe-135 fission yield to I-135 yield.
    # gamma_Xe ≈ 0.003, gamma_I ≈ 0.061 → ratio ≈ 0.049.
    gamma_ratio: float = 0.049
    # Equilibrium xenon reactivity worth at full power. Typical PWR: 2000–3500 pcm.
    # At Xe_hat=1 (equilibrium), rho_Xe=0 (absorbed into reference conditions).
    # Deviation from equilibrium: rho_Xe = -rho_Xe_full * (Xe_hat - 1).
    rho_Xe_full: float = 0.025  # = 2500 pcm

    # ---- Termination / safety bounds ----
    n_min: float = 0.01  # near-shutdown — SCRAM
    precision_floor: float = 0.05  # K, the RTD reading's resolution
    n_max: float = 1.09  # the high-flux trip, 109 % of rated
    T_fuel_max: float = 1473.0  # K, well below UO2 melting (~3120 K)
    T_fuel_min: float = 500.0
    T_coolant_max: float = 620.0  # K, the hot leg's saturation margin
    T_avg_min: float = 556.0  # K, 541 F, the minimum temperature for criticality

    # ---- Reward shaping ----
    rod_motion_weight: float = 0.02
    # Error scale for the MPC's tracking term, not read by ``compute_reward``.
    # See "Why the MPC does not minimise the reward" in docs/baselines.md. The reward
    # has not been a Gaussian for some time; this is the planner's scale.
    reward_band: float = 0.03  # relative flux (3% = tight)

    # ---- Load: a daily 100-50-100, plus a small frequency-control wobble ----
    # Ramps at 5 %/min, the rapid daily manoeuvre of the EPRI URD, with the ramp
    # and the hold drawn at reset inside its 2 to 10 h hold.
    load_min: float = 0.5
    load_ramp_rate: float = 0.05 / 60.0  # 1/s
    ramp_down_range: Tuple[float, float] = (1800.0, 7200.0)  # s
    hold_range: Tuple[float, float] = (3.0 * 3600.0, 8.0 * 3600.0)  # s
    # The wobble is an OU process on the load: 5 min correlation, 0.5 % sd.
    demand_theta: float = 1.0 / 300.0
    demand_sigma: float = 4.08e-4

    # ---- Revenue model (visualisation only — not in reward) ----
    # 1 GWe plant at spot price. Imbalance penalty = factor × spot.
    P_electric_GW: float = 1.0
    spot_price_per_MWh: float = 80.0
    imbalance_factor: float = 3.0

    # ---- Initial state ----
    #: Recent power level the xenon inventory is in equilibrium with at reset:
    #: narrow, because a unit that load-follows daily is never quite at rest.
    initial_xenon_power_range: Tuple[float, float] = (0.9, 1.0)

    # ---- Time discretization ----
    # One control step applies the action for Reactor.control_period physics
    # sub-steps (a class-level constant on the Env, not a param, so JIT treats it
    # as static): 6 of 5 s, so 30 s. At 60 s the rod program's slowest speed
    # overshoots its dead band and the rods limit-cycle. Each sub-step takes 4
    # inner solves: past about 1.25 s a solve misses the prompt jump, and at 2.5 s
    # inserting rods raises power. ``max_steps_in_episode``
    # counts *environment* steps, as on every other plant; ``state.time`` advances
    # by 1 per ``step_env`` call, and the physics clock lives in
    # ``state.physics_time`` (sub-steps since reset).
    delta_t: float = 5.0
    max_steps_in_episode: int = 1440  # env steps of 30 s: 12 h

    # ---- Reward (docs/reward-shaping.md; version 2), in dollars per 30 s step ----
    # Tracking: Tavg against its program. Off program, steam conditions cost
    # output: ``imbalance_multiple`` is 1.25 x spot x 0.23 %/K, the Carnot slope at
    # 547 K steam and 305 K condenser, so $100/MWh on 0.23 % of 1 GWe per K is
    # $230/h per K, linear (p = 1). Rod wear: the steps the rods actually moved at
    # ``rod_step_price``. A trip is charged the restart time at twice the dearest
    # surviving step.
    reward_version: int = 2
    e_floor: float = 0.05  # K, the RTD reading's resolution
    e_tol: float = 0.0
    tracking_exponent: float = 1.0
    imbalance_multiple: float = 1.25 * 0.0023  # of spot, per K
    rod_step_price: float = 1.0  # $ per rod step, ours
    # Twice the dearest step that survives: 40 K off program, $7.67, and 36 rod
    # steps, $36, per 30 s.
    failure_cost: float = 2.0 * (7.67 + 36.0)
    #: Restart time priced into a trip (``reward.trip_cost``; 48 h at 30 s steps: a SCRAM's xenon-limited restart, provisional).
    restart_steps: int = 5760
    #: Tracking cost per step at the floor, in the reward's units.
    rho_floor_tracking: float = 100.0 * 1000.0 * 0.0023 / 3600.0 * 0.05 * 30.0
    rho_floor: float = 100.0 * 1000.0 * 0.0023 / 3600.0 * 0.05 * 30.0
    #: True where e_floor is a resolution, not a measured or certified floor.
    floor_is_documented_minimum: bool = True


#: Units of the ReactorParams fields, for the generated environment pages.
PARAM_UNITS = {
    "Lambda_gen": "s",
    "alpha_fuel": "1/K",
    "alpha_coolant": "1/K",
    "T_fuel_ref": "K",
    "T_coolant_ref": "K",
    "rho_ext_min": "-",
    "rho_ext_max": "-",
    "rod_speed": "1/s",
    "rod_stroke_steps": "steps",
    "P_thermal_ref": "W",
    "C_fuel": "J/K",
    "C_coolant": "J/K",
    "C_cold": "J/K",
    "UA": "W/K",
    "m_dot_cp": "W/K",
    "UA_sg": "W/K",
    "C_sg": "J/K",
    "T_avg_no_load": "K",
    "T_avg_full_load": "K",
    "sigma_phi0": "1/s",
    "gamma_ratio": "-",
    "rho_Xe_full": "-",
    "n_min": "-",
    "precision_floor": "K",
    "n_max": "-",
    "T_fuel_max": "K",
    "T_fuel_min": "K",
    "T_coolant_max": "K",
    "T_avg_min": "K",
    "rod_motion_weight": "-",
    "reward_band": "-",
    "load_min": "-",
    "load_ramp_rate": "1/s",
    "demand_theta": "1/s",
    "demand_sigma": "1/s^0.5",
    "P_electric_GW": "GW",
    "spot_price_per_MWh": "$/MWh",
    "imbalance_factor": "-",
    "reward_version": "-",
    "e_floor": "K",
    "e_tol": "K",
    "tracking_exponent": "-",
    "imbalance_multiple": "1/K",
    "rod_step_price": "$/step",
    "failure_cost": "reward",
    "restart_steps": "steps",
    "rho_floor_tracking": "reward",
    "rho_floor": "reward",
}


@struct.dataclass
class ReactorState(EnvState):
    n: float  # normalised neutron density (n=1 = nominal)
    C: jnp.ndarray  # (6,) delayed-neutron precursors

    T_fuel: float
    T_coolant: float  # hot leg
    T_cold: float  # cold leg
    T_sg: float  # steam generator secondary

    # Xenon / Iodine (normalised: 1.0 = equilibrium at full power)
    I_hat: float  # normalised I-135 concentration
    Xe_hat: float  # normalised Xe-135 concentration

    target_n: float  # the turbine's load
    T_ref: float  # Tref, the program at that load (K)
    load_wobble: float  # its frequency-control part
    ramp_down_at: float  # s, when the load starts down
    ramp_up_at: float  # s, when it starts back
    demand_key: jnp.ndarray  # PRNGKey for reproducible OU noise
    #: Physics sub-steps since reset (``delta_t`` seconds each). ``time`` counts
    #: environment steps; this is the clock the load schedule and its noise are
    #: keyed on, so it runs ``control_period`` times faster.
    physics_time: int

    rho_ext: float  # current (actual, rate-limited) rod reactivity
    #: What the controller *asked* the rods for this step, before the rate
    #: limit. Carried so the reward can charge for demanding motion the rods
    #: cannot deliver; see ``compute_reward``.
    rho_ext_cmd: float
    rod_moved: float = 0.0  # reactivity the rods travelled on the last sub-step

    actuator_held: jax.Array = actuator_field(1)


def steady_state_precursors(n_0: float, params: ReactorParams) -> jnp.ndarray:
    """Precursor concentrations that keep dC_i/dt = 0 at neutron density n_0."""
    return (BETA_I / (LAMBDA_I * params.Lambda_gen)) * n_0


def steady_state_xenon(n_0: float, params: ReactorParams) -> tuple[float, float]:
    """Equilibrium (I_hat, Xe_hat) at power level n_0.

    At n=1: I_hat=1, Xe_hat=1 by construction. At other power levels the
    equilibrium shifts because xenon burnup is proportional to n while decay
    is not.
    """
    I_hat_eq = n_0
    Xe_hat_eq = (
        (LAMBDA_XENON + params.sigma_phi0)
        * n_0
        / (LAMBDA_XENON + params.sigma_phi0 * n_0)
    )
    return I_hat_eq, Xe_hat_eq


def tavg_program(load, params: ReactorParams):
    """Tref: the Tavg the program asks for at ``load``."""
    span = params.T_avg_full_load - params.T_avg_no_load
    return params.T_avg_no_load + span * load


def scheduled_load(t, ramp_down_at, ramp_up_at, params: ReactorParams, xp=jnp):
    """The day's load at ``t`` seconds: full, down a ramp, held, back up a ramp."""
    depth = 1.0 - params.load_min
    down = xp.minimum(params.load_ramp_rate * xp.maximum(t - ramp_down_at, 0.0), depth)
    up = xp.minimum(params.load_ramp_rate * xp.maximum(t - ramp_up_at, 0.0), depth)
    return 1.0 - down + up


def steady_state_temperatures(load, params: ReactorParams):
    """``(T_fuel, T_hot, T_cold, T_sg)`` lined out at ``load`` on the program."""
    p = params
    power = p.P_thermal_ref * load
    T_avg = tavg_program(load, p)
    T_hot = T_avg + 0.5 * power / p.m_dot_cp
    T_cold = T_avg - 0.5 * power / p.m_dot_cp
    return T_hot + power / p.UA, T_hot, T_cold, T_avg - power / p.UA_sg


def rod_balance(T_fuel, T_avg, Xe_hat, params: ReactorParams):
    """The rod reactivity that leaves the core critical at these temperatures."""
    return -(
        params.alpha_fuel * (T_fuel - params.T_fuel_ref)
        + params.alpha_coolant * (T_avg - params.T_coolant_ref)
        - params.rho_Xe_full * (Xe_hat - 1.0)
    )


def _reactivity(rho_ext, T_fuel, T_hot, T_cold, Xe_hat, p: ReactorParams):
    return rho_ext - rod_balance(T_fuel, 0.5 * (T_hot + T_cold), Xe_hat, p)


def _slow_rates(slow, n, load, p: ReactorParams):
    """Rates of ``[T_fuel, T_hot, T_cold, T_sg, I_hat, Xe_hat]``."""
    T_f, T_h, T_c, T_s, I_h, Xe = slow
    to_coolant = p.UA * (T_f - T_h)
    around_loop = p.m_dot_cp * (T_h - T_c)
    to_steam = p.UA_sg * (0.5 * (T_h + T_c) - T_s)
    lam_sum = LAMBDA_XENON + p.sigma_phi0
    feeds = lam_sum * (p.gamma_ratio * n + I_h) / (1.0 + p.gamma_ratio)
    return jnp.array(
        [
            (p.P_thermal_ref * n - to_coolant) / p.C_fuel,
            (to_coolant - around_loop) / p.C_coolant,
            (around_loop - to_steam) / p.C_cold,
            (to_steam - p.P_thermal_ref * load) / p.C_sg,
            LAMBDA_IODINE * (n - I_h),
            feeds - (LAMBDA_XENON + p.sigma_phi0 * n) * Xe,
        ]
    )


def compute_velocity(position, action, params: ReactorParams, load=1.0):
    """
    Right-hand side of the coupled ODE system.

    ``position = [n, C_1..6, T_fuel, T_hot, T_cold, T_sg, I_hat, Xe_hat]`` (shape
    (13,)). ``action = rho_ext`` (rate-limited rod reactivity, physical units).
    """
    n = position[0]
    C = position[1 : 1 + N_GROUPS]
    slow = position[1 + N_GROUPS :]
    T_f, T_h, T_c, _, _, Xe = slow

    rho = _reactivity(action, T_f, T_h, T_c, Xe, params)
    dn_dt = ((rho - BETA_TOT) / params.Lambda_gen) * n + jnp.sum(LAMBDA_I * C)
    dC_dt = (BETA_I / params.Lambda_gen) * n - LAMBDA_I * C

    dposition = jnp.concatenate(
        [jnp.array([dn_dt]), dC_dt, _slow_rates(slow, n, load, params)]
    )
    return dposition, None


# TR-BDF2 (Bank et al. 1985): one-step composite of a trapezoidal stage and
# a BDF2 stage. L-stable + 2nd-order, so it damps stiff modes correctly at any
# step size. gamma = 2 - sqrt(2) makes both stages share the structure below.
_TRBDF2_GAMMA = 2.0 - 2.0**0.5
_TRBDF2_C1 = 1.0 / (_TRBDF2_GAMMA * (2.0 - _TRBDF2_GAMMA))
_TRBDF2_C2 = (1.0 - _TRBDF2_GAMMA) ** 2 / (_TRBDF2_GAMMA * (2.0 - _TRBDF2_GAMMA))
_TRBDF2_C3 = (1.0 - _TRBDF2_GAMMA) / (2.0 - _TRBDF2_GAMMA)


def _arrowhead_solve(d0, row, col, diag, b0, b):
    """Solve ``M x = b`` in O(n) for an arrowhead matrix M.

    M is zero except for the scalar (0,0) entry ``d0``, the first row
    ``M[0,1:] = row``, the first column ``M[1:,0] = col`` and the trailing
    diagonal ``M[i,i] = diag``. ``row``/``col``/``diag``/``b`` are (6,) arrays,
    ``d0``/``b0`` scalars. Eliminating the trailing rows against their
    diagonal reduces the system to a single scalar equation for ``x0``.
    """
    inv = 1.0 / diag
    x0 = (b0 - jnp.sum(row * b * inv)) / (d0 - jnp.sum(row * col * inv))
    x = (b - col * x0) * inv
    return x0, x


def _fast_block_trbdf2(n, C, rho, params: ReactorParams, h):
    """One TR-BDF2 step of the linear neutron-precursor block over ``h``.

    The point-kinetics matrix ``A`` (``d[n,C]/dt = A @ [n,C]``) is an
    *arrowhead* matrix, so both implicit stages — each a solve of
    ``(I - tau*A) x = b`` — are done in O(n) by ``_arrowhead_solve`` rather
    than a 7x7 factorisation. Returns ``(n_new, C_new, n_gamma)``; the
    TR-stage neutron density ``n_gamma`` is handed to the slow block.
    """
    Lam = params.Lambda_gen
    a00 = (rho - BETA_TOT) / Lam  # A[0,0]
    col_a = BETA_I / Lam  # A[1:,0]; A[0,1:] = LAMBDA_I, A[i,i] = -LAMBDA_I

    def apply_i_plus_tau_a(tau, n_, c_):
        """Return ``(I + tau*A) @ [n_, c_]``."""
        out0 = (1.0 + tau * a00) * n_ + tau * jnp.sum(LAMBDA_I * c_)
        out_c = tau * col_a * n_ + (1.0 - tau * LAMBDA_I) * c_
        return out0, out_c

    def solve_i_minus_tau_a(tau, b0, b_c):
        """Solve ``(I - tau*A) x = b``."""
        return _arrowhead_solve(
            1.0 - tau * a00,  # d0
            -tau * LAMBDA_I,  # row  M[0,1:]
            -tau * col_a,  # col  M[1:,0]
            1.0 + tau * LAMBDA_I,  # diag M[i,i]
            b0,
            b_c,
        )

    # TR stage -> t + gamma*h.
    tau1 = 0.5 * _TRBDF2_GAMMA * h
    rhs0, rhs_c = apply_i_plus_tau_a(tau1, n, C)
    n_g, C_g = solve_i_minus_tau_a(tau1, rhs0, rhs_c)

    # BDF2 stage -> t + h.
    tau2 = _TRBDF2_C3 * h
    n_new, C_new = solve_i_minus_tau_a(
        tau2,
        _TRBDF2_C1 * n_g - _TRBDF2_C2 * n,
        _TRBDF2_C1 * C_g - _TRBDF2_C2 * C,
    )
    return n_new, C_new, n_g


def _reactor_stiff_step(
    positions, rho_ext, params: ReactorParams, n_substeps: int, load=1.0
):
    """Operator-split stiff integrator for one ``params.delta_t`` step.

    The system has a stiffness ratio of ~10**6 (prompt neutrons ~15 ms vs xenon
    ~9 h), which is why explicit RK4 needs ~50 substeps just to stay numerically
    stable. This integrator splits it:

    * **Fast block** ``[n, C_1..6]``, linear in itself given the reactivity
      ``rho``, is advanced with TR-BDF2, an L-stable 2nd-order method, solved
      in O(n) via the arrowhead structure of the point-kinetics matrix. It is
      unconditionally stable, so ``n_substeps`` sets *accuracy*, not stability.
    * **Slow block**: temperatures, iodine and xenon, non-stiff (time constants
      ~1 s to hours), advanced with RK4, driven by the neutron density from the
      fast solve.

    ``rho`` depends only on the slow variables, so it is refreshed once per
    substep: cheap, and it removes the within-step reactivity-freezing error.
    """
    h = params.delta_t / n_substeps
    n = positions[0]
    C = positions[1 : 1 + N_GROUPS]
    slow0 = positions[1 + N_GROUPS :]

    def body(carry, _):
        n_, C_, slow = carry
        T_f, T_h, T_c, _T_s, _I, Xe = slow
        rho = _reactivity(rho_ext, T_f, T_h, T_c, Xe, params)
        n_new, C_new, n_g = _fast_block_trbdf2(n_, C_, rho, params, h)
        # RK4 on the non-stiff slow block; n sampled from the fast solve.
        k1 = _slow_rates(slow, n_, load, params)
        k2 = _slow_rates(slow + 0.5 * h * k1, n_g, load, params)
        k3 = _slow_rates(slow + 0.5 * h * k2, n_g, load, params)
        k4 = _slow_rates(slow + h * k3, n_new, load, params)
        slow_next = slow + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return (n_new, C_new, slow_next), None

    (n_f, C_f, slow_f), _ = jax.lax.scan(
        body, (n, C, slow0), xs=None, length=n_substeps
    )
    return jnp.concatenate([jnp.array([n_f]), C_f, slow_f]), None


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state(
    rho_raw: float,
    state: ReactorState,
    params: ReactorParams,
    integration_method: str = "tr_bdf2_4",
):
    """
    ``rho_raw`` : raw action in [-1, 1] = *desired* rod position.
    Actual rod position moves toward it at the rods' one speed.
    """
    p = params
    desired_rho = convert_raw_action_to_range(
        rho_raw, min_action=p.rho_ext_min, max_action=p.rho_ext_max
    )
    most = p.rod_speed * p.delta_t
    delta_rho = jnp.clip(desired_rho - state.rho_ext, -most, most)
    rho_ext = jnp.clip(state.rho_ext + delta_rho, p.rho_ext_min, p.rho_ext_max)

    # ── Integrate coupled ODEs ──
    positions = jnp.concatenate(
        [
            jnp.array([state.n]),
            state.C,
            jnp.array(
                [
                    state.T_fuel,
                    state.T_coolant,
                    state.T_cold,
                    state.T_sg,
                    state.I_hat,
                    state.Xe_hat,
                ]
            ),
        ]
    )
    load = state.target_n
    if integration_method.startswith("tr_bdf2"):
        # Stiff operator-split integrator (arrowhead TR-BDF2 + RK4). Accepts
        # "tr_bdf2" (default substeps) or "tr_bdf2_<N>" to set the substep
        # count explicitly. Unconditionally stable; N controls accuracy.
        tail = integration_method.split("_")[-1]
        n_substeps = int(tail) if tail.isdigit() else 4
        new_positions, metrics = _reactor_stiff_step(
            positions, rho_ext, p, n_substeps, load
        )
    else:
        _compute_velocity = partial(
            compute_velocity, action=rho_ext, params=p, load=load
        )
        new_positions, metrics = integrate_dynamics(
            positions=positions,
            delta_t=p.delta_t,
            compute_velocity=_compute_velocity,
            method=integration_method,
        )

    new_n = new_positions[0]
    new_C = new_positions[1 : 1 + N_GROUPS]
    T_fuel, T_hot, T_cold, T_sg, I_hat, Xe_hat = new_positions[1 + N_GROUPS :]

    # One physics sub-step. The env-step clock ``state.time`` is advanced by
    # ``Reactor.step_env`` once per control period, not here.
    new_physics_time = state.physics_time + 1

    # ── The load: the day's schedule, plus the OU wobble on top ──
    key_t = jax.random.fold_in(state.demand_key, new_physics_time)
    noise = jax.random.normal(key_t, dtype=jnp.float32)
    wobble = (
        state.load_wobble
        - p.demand_theta * state.load_wobble * p.delta_t
        + p.demand_sigma * jnp.sqrt(p.delta_t) * noise
    )
    day = scheduled_load(
        new_physics_time * p.delta_t, state.ramp_down_at, state.ramp_up_at, p
    )
    new_load = jnp.clip(day + wobble, 0.0, 1.0)

    return (
        state.replace(
            n=new_n,
            C=new_C,
            T_fuel=T_fuel,
            T_coolant=T_hot,
            T_cold=T_cold,
            T_sg=T_sg,
            I_hat=I_hat,
            Xe_hat=Xe_hat,
            target_n=new_load,
            T_ref=tavg_program(new_load, p),
            load_wobble=wobble,
            rho_ext=rho_ext,
            rho_ext_cmd=desired_rho,
            rod_moved=jnp.abs(rho_ext - state.rho_ext),
            physics_time=new_physics_time,
        ),
        metrics,
    )


# Deliberately not jitted. It was decorated with
# ``@partial(jax.jit, static_argnames=["params"])``, which keys the compilation
# cache on the params object: a fresh ``Params(...)`` -- what every sweep, tuner
# and MPC builds -- was a cache miss and a full recompile, measured at ~1600x the
# cost of a cached call. Callers that want it fused already jit ``step_env``,
# which traces this inline.
def get_obs(state: ReactorState, params: ReactorParams):
    """
    Partially observable: the agent sees only measurable quantities.

    ``obs = [n, T_hot, rho_ext_norm, load, T_avg, T_ref]``

    Hidden: C_i (6 precursors), T_fuel, T_sg, I_hat, Xe_hat.
    """
    span = params.rho_ext_max - params.rho_ext_min
    rho_norm = 2.0 * (state.rho_ext - params.rho_ext_min) / span - 1.0
    T_avg = 0.5 * (state.T_coolant + state.T_cold)
    return jnp.array(
        [
            state.n,
            state.T_coolant,
            rho_norm,
            state.target_n,
            T_avg,
            state.T_ref,
        ]
    )


def check_is_terminal(state: ReactorState, params: ReactorParams, xp=jnp):
    # Written so a non-finite power is outside too: a coarse explicit solve can
    # overflow in a sub-step, and that is a lost plant, not a live one.
    n_out = xp.logical_not(
        xp.logical_and(state.n > params.n_min, state.n < params.n_max)
    )
    T_fuel_out = xp.logical_or(
        state.T_fuel <= params.T_fuel_min, state.T_fuel >= params.T_fuel_max
    )
    T_coolant_out = xp.logical_or(
        0.5 * (state.T_coolant + state.T_cold) <= params.T_avg_min,
        state.T_coolant >= params.T_coolant_max,
    )
    terminated = xp.logical_or(xp.logical_or(n_out, T_fuel_out), T_coolant_out)
    # Both in environment steps (see ReactorParams.max_steps_in_episode).
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward_terms(
    state: ReactorState, params: ReactorParams, xp=jnp, tripped=None
):
    """The reward's additive cost terms in $ per physics sub-step
    (``target_gym.reward``).

    ``tripped`` overrides the trip check on ``state``, for a measured copy,
    whose noise must not fire or hide a trip (``base.measured_reward``).
    """
    p = params
    per_K_s = (
        p.imbalance_multiple * p.spot_price_per_MWh * p.P_electric_GW * 1000.0 / 3600.0
    )  # $ per second per K off program
    error = state.T_ref - 0.5 * (state.T_coolant + state.T_cold)
    tracking = (
        per_K_s
        * p.delta_t
        * p.e_floor
        * R.tracking_cost(error, p.e_floor, p.e_tol, p.tracking_exponent, xp)
    )
    span = p.rho_ext_max - p.rho_ext_min
    rod_wear = p.rod_step_price * p.rod_stroke_steps * state.rod_moved / span
    if tripped is None:
        tripped, _ = check_is_terminal(state, p, xp)
    terms = {
        "tracking": tracking,
        "running": rod_wear,
    }
    return R.with_trip(terms, tripped, R.trip_cost(p), xp)


def compute_reward_v1(state: ReactorState, params: ReactorParams, xp=jnp):
    """Flux tracking minus a small rod-motion penalty.

    Tracking is log-scaled (``utils.log_scaled_reward``): every halving of the
    error is worth the same increment, from the plant's operating envelope down
    to ``precision_floor``, the finest error an ex-core detector can resolve.

    This replaced a Gaussian on ``reward_band``, which was effectively zero --
    no gradient at all -- beyond about 10% error, and which stopped
    discriminating just where a good controller operates. See
    docs/reward-shaping.md.

    ``step_env`` applies ``control_period`` physics sub-steps per environment
    step and returns their **mean**, so one env step is on the same numeric
    scale as a step of any other environment. It used to sum them, which put
    this plant's per-step reward near 4.7 where everything else caps around 1.0
    and made returns misleading the moment they were read across environments.

    * Rod cost : small penalty for holding rods far from neutral.
    """
    error = xp.abs(state.target_n - state.n)
    tracking = log_scaled_reward(
        error, params.precision_floor, params.n_max - params.n_min, xp
    )

    # Charged for asking the rods to move faster than they can, not for where
    # they are. The previous form was ``|rho_ext| / rho_scale``, which penalises
    # holding the rods where the physics requires them: a reactor at 60% power
    # *must* sit with rods inserted, and paying for that is paying for doing the
    # job. This is the wind turbine's pitch-activity form -- command minus
    # achieved, divided by the actuator's range -- and it is large exactly when
    # the demand is running ahead of the rate limit. It rewards staying inside
    # the achievable envelope, which given asymmetric rod speeds (insertion
    # fast, withdrawal slow) is a real thing to have to plan around.
    rho_scale = xp.maximum(xp.abs(params.rho_ext_min), xp.abs(params.rho_ext_max))
    rod_penalty = (
        params.rod_motion_weight * xp.abs(state.rho_ext_cmd - state.rho_ext) / rho_scale
    )

    return tracking * (1.0 - rod_penalty)


def compute_reward(state: ReactorState, params: ReactorParams, xp=jnp):
    return R.select(
        params.reward_version,
        compute_reward_v1(state, params, xp),
        R.total(compute_reward_terms(state, params, xp), xp),
        xp,
    )


def compute_revenue_rate(state: ReactorState, params: ReactorParams):
    """
    Instantaneous revenue rate in k$/hour (for visualisation only).

    revenue  = delivered_power × spot_price
    penalty  = |deviation| × imbalance_price
    net_rate = revenue - penalty

    At perfect tracking with n=target=1.0 and 1 GWe @ $80/MWh:
    revenue = $80k/h.  A 5% error costs an extra $12k/h in imbalance fees.
    """
    P_MW = params.P_electric_GW * 1000.0 * state.n
    revenue = P_MW * params.spot_price_per_MWh  # $/h
    imbalance_price = params.spot_price_per_MWh * params.imbalance_factor
    penalty = (
        jnp.abs(state.n - state.target_n)
        * params.P_electric_GW
        * 1000.0
        * imbalance_price
    )
    net = revenue - penalty
    return net / 1000.0  # k$/h
