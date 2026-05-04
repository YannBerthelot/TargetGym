"""
Nuclear reactor — point-kinetics with delayed neutrons, xenon poisoning,
thermal feedback, and rate-limited control rods.

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
        + alpha_coolant * (T_coolant - T_coolant_ref)
        - rho_Xe_full * (Xe_hat - 1)

Both temperature coefficients are *negative* (Doppler + moderator) — this is
what makes a PWR passively self-regulating. The xenon term adds a slow,
history-dependent reactivity swing that is the dominant control challenge.

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

Rod speed limits
----------------
Real control rods have asymmetric travel speed: insertion (safety) is fast,
withdrawal is slow. The action is the *desired* rod position; the actual rod
reactivity is rate-limited toward it each timestep::

    max_increase = rod_speed_withdraw * delta_t  (withdrawal, positive direction)
    max_decrease = rod_speed_insert * delta_t     (insertion, negative direction)
    rho_ext = clamp(rho_ext + clip(desired - rho_ext, -max_decrease, max_increase))

A lumped two-node thermal model closes the feedback loop::

    C_fuel    * dT_fuel/dt    = P_ref * n  -  UA * (T_fuel - T_coolant)
    C_coolant * dT_coolant/dt = UA * (T_fuel - T_coolant)
                              - m_dot_cp * (T_coolant - T_inlet)

Why this is interesting for RL
------------------------------
* **Xenon creates a multi-hour memory.** After a power change, iodine decays
  into xenon over 6–9 hours, causing a large delayed reactivity swing. The
  agent must anticipate this — reacting only when it arrives is too late.
* **Rod authority is limited.** Maximum positive reactivity (+500 pcm) is far
  less than equilibrium xenon worth (~2500 pcm). A 20 % xenon overshoot
  exhausts all available rod authority → "xenon deadtime" where power cannot
  be raised. The agent must learn smooth power transitions to avoid this.
* **Asymmetric rod speed.** Insertion is 2× faster than withdrawal — the
  controller can shut down quickly but recovering power is slow.
* **Hidden states.** Precursors (6), fuel temperature, iodine, and xenon are
  all unobserved. The agent sees only n, T_coolant, rod position, and target.
* **Stiff + slow coupling.** Prompt neutronics (~ms), precursors (~s to min),
  thermal (~min), xenon (~hours) — four timescales the agent must handle.

Observation (partially observable)
----------------------------------
``[n, T_coolant, rho_ext_norm, target_n]`` — precursors, fuel temperature,
iodine, and xenon are hidden.

Action
------
Scalar in ``[-1, 1]`` → *desired* control-rod reactivity. Actual rod position
is rate-limited toward this value each step.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym.base import EnvParams, EnvState
from target_gym.integration import integrate_dynamics
from target_gym.utils import convert_raw_action_to_range

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

# Number of piecewise-constant power setpoints per episode. Each segment has
# equal duration; setpoints sampled uniformly from ``target_n_range`` at reset.
N_SETPOINTS = 4


@struct.dataclass
class ReactorParams(EnvParams):
    # ---- Kinetics ----
    Lambda_gen: float = 1e-4  # s — effective mean neutron generation time

    # Thermal feedback coefficients (1/K). Both negative (stabilising).
    alpha_fuel: float = -3.0e-5  # Doppler (prompt, on fuel temperature)
    alpha_coolant: float = -5.0e-5  # Moderator temperature coefficient
    T_fuel_ref: float = 900.0  # K
    T_coolant_ref: float = 580.0  # K

    # ---- External (control rod) reactivity range ----
    # rho_ext_max raised to 0.005 (500 pcm) to give enough authority to
    # partially compensate xenon swings. Still safely below prompt-critical
    # (beta ≈ 0.0065 = 650 pcm).
    rho_ext_min: float = -0.010  # rods fully inserted
    rho_ext_max: float = 0.005  # rods fully withdrawn

    # ---- Rod speed limits (reactivity per second) ----
    # Typical PWR: insertion ~40 pcm/s (safety, fast), withdrawal ~20 pcm/s.
    # This creates an asymmetry: the controller can reduce power quickly but
    # must raise it slowly — matching real operational constraints.
    rod_speed_insert: float = 4.0e-4  # 1/s — max insertion rate (negative direction)
    rod_speed_withdraw: float = 2.0e-4  # 1/s — max withdrawal rate (positive direction)

    # ---- Two-node thermal model ----
    P_thermal_ref: float = 3.0e9  # W — nominal thermal power (3 GW PWR)
    C_fuel: float = 3.3e7  # J/K  (~100 t UO2 × c_p)
    C_coolant: float = 7.0e7  # J/K  (~14 t primary coolant inventory)
    UA: float = 6.6e6  # W/K  (fuel → coolant heat transfer)
    m_dot_cp: float = 9.0e7  # W/K  (mass flow × c_p of primary loop)
    T_inlet: float = 555.0  # K  (coolant inlet / cold-leg temperature)

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
    n_max: float = 1.5  # 150 % overpower — SCRAM
    T_fuel_max: float = 1473.0  # K — well below UO2 melting (~3120 K)
    T_fuel_min: float = 500.0
    T_coolant_max: float = 620.0  # K — saturation margin
    T_coolant_min: float = 500.0

    # ---- Reward shaping ----
    rod_motion_weight: float = 0.02
    reward_band: float = 0.03  # Gaussian tracking band (3% = tight)

    # ---- Demand (Ornstein-Uhlenbeck process) ----
    # Grid demand evolves as a mean-reverting random walk — the plant must
    # follow without advance notice. OU parameters:
    #   dx = theta * (mu - x) * dt + sigma * sqrt(dt) * dW
    # theta: mean-reversion rate (1/s). 1/theta ≈ 2 h correlation time.
    # sigma: volatility.  Steady-state std ≈ sigma / sqrt(2*theta) ≈ 0.14 (14%).
    demand_theta: float = 1.5e-4
    demand_sigma: float = 2.5e-3

    # ---- Revenue model (visualisation only — not in reward) ----
    # 1 GWe plant at spot price. Imbalance penalty = factor × spot.
    P_electric_GW: float = 1.0
    spot_price_per_MWh: float = 80.0
    imbalance_factor: float = 3.0

    # ---- Initial / target ranges ----
    target_n_range: Tuple[float, float] = (0.3, 1.0)
    initial_n_range: Tuple[float, float] = (0.7, 1.0)
    initial_T_fuel: float = 900.0
    initial_T_coolant: float = 580.0

    # ---- Time discretization ----
    # Physics integrates at delta_t=1.0 s for numerical stability. One control
    # step applies the action for Reactor.control_period physics sub-steps
    # (a class-level constant on the Env, not a param, so JIT treats it as
    # static). Effective control period = delta_t * control_period seconds.
    # state.time advances by 1 per physics sub-step, so this counter is in
    # physics-step units to match what gymnax's Environment.step uses for
    # truncation (info["truncated"] = state.time >= max_steps_in_episode).
    # 24 h of simulated time = 86400 s = 86400 physics steps × 1 s/step
    # = 8640 control steps × 10 s/step.
    delta_t: float = 1.0
    max_steps_in_episode: int = 86400  # in physics steps (86400 × 1 s = 24 h, = 8640 control steps)


@struct.dataclass
class ReactorState(EnvState):
    n: float  # normalised neutron density (n=1 = nominal)
    C: jnp.ndarray  # (6,) delayed-neutron precursors

    T_fuel: float
    T_coolant: float

    # Xenon / Iodine (normalised: 1.0 = equilibrium at full power)
    I_hat: float  # normalised I-135 concentration
    Xe_hat: float  # normalised Xe-135 concentration

    target_n: float
    target_schedule: jnp.ndarray  # (N_SETPOINTS,) — legacy, unused with OU demand
    demand_key: jnp.ndarray  # PRNGKey for reproducible OU noise

    rho_ext: float  # current (actual, rate-limited) rod reactivity


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


def get_target_from_schedule(
    target_schedule: jnp.ndarray, time: int, params: ReactorParams
) -> jnp.ndarray:
    """Select the active setpoint from a piecewise-constant schedule."""
    # state.time and max_steps_in_episode are both in physics-step units.
    slot = jnp.minimum(
        (time * N_SETPOINTS) // params.max_steps_in_episode,
        N_SETPOINTS - 1,
    )
    return target_schedule[slot]


def compute_velocity(position, action, params: ReactorParams):
    """
    Right-hand side of the coupled ODE system.

    ``position = [n, C_1..6, T_fuel, T_coolant, I_hat, Xe_hat]`` (shape (11,)).
    ``action   = rho_ext`` (rate-limited rod reactivity, physical units).
    """
    n = position[0]
    C = position[1 : 1 + N_GROUPS]
    T_fuel = position[1 + N_GROUPS]
    T_coolant = position[2 + N_GROUPS]
    I_hat = position[3 + N_GROUPS]
    Xe_hat = position[4 + N_GROUPS]

    rho_ext = action

    # ── Reactivity ──
    rho_feedback = params.alpha_fuel * (
        T_fuel - params.T_fuel_ref
    ) + params.alpha_coolant * (T_coolant - params.T_coolant_ref)
    rho_xenon = -params.rho_Xe_full * (Xe_hat - 1.0)
    rho = rho_ext + rho_feedback + rho_xenon

    # ── Point kinetics ──
    dn_dt = ((rho - BETA_TOT) / params.Lambda_gen) * n + jnp.sum(LAMBDA_I * C)
    dC_dt = (BETA_I / params.Lambda_gen) * n - LAMBDA_I * C

    # ── Thermal hydraulics (two-node) ──
    P_thermal = params.P_thermal_ref * n
    Q_fuel_to_cool = params.UA * (T_fuel - T_coolant)
    Q_flow_out = params.m_dot_cp * (T_coolant - params.T_inlet)
    dT_fuel_dt = (P_thermal - Q_fuel_to_cool) / params.C_fuel
    dT_coolant_dt = (Q_fuel_to_cool - Q_flow_out) / params.C_coolant

    # ── Xenon / Iodine kinetics ──
    dI_hat_dt = LAMBDA_IODINE * (n - I_hat)

    lam_sum = LAMBDA_XENON + params.sigma_phi0
    a_coeff = params.gamma_ratio * lam_sum / (1.0 + params.gamma_ratio)
    b_coeff = lam_sum / (1.0 + params.gamma_ratio)
    dXe_hat_dt = (
        a_coeff * n + b_coeff * I_hat - (LAMBDA_XENON + params.sigma_phi0 * n) * Xe_hat
    )

    dposition = jnp.concatenate(
        [
            jnp.array([dn_dt]),
            dC_dt,
            jnp.array([dT_fuel_dt, dT_coolant_dt, dI_hat_dt, dXe_hat_dt]),
        ]
    )
    return dposition, None


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state(
    rho_raw: float,
    state: ReactorState,
    params: ReactorParams,
    integration_method: str = "rk4_50",
):
    """
    ``rho_raw`` : raw action in [-1, 1] = *desired* rod position.
    Actual rod position is rate-limited toward this value.
    """
    # ── Rate-limited rod movement ──
    desired_rho = convert_raw_action_to_range(
        rho_raw, min_action=params.rho_ext_min, max_action=params.rho_ext_max
    )
    delta_rho = desired_rho - state.rho_ext
    max_withdraw = params.rod_speed_withdraw * params.delta_t  # positive
    max_insert = params.rod_speed_insert * params.delta_t  # magnitude
    delta_rho = jnp.clip(delta_rho, -max_insert, max_withdraw)
    rho_ext = jnp.clip(
        state.rho_ext + delta_rho, params.rho_ext_min, params.rho_ext_max
    )

    # ── Integrate coupled ODEs ──
    _compute_velocity = partial(compute_velocity, action=rho_ext, params=params)

    positions = jnp.concatenate(
        [
            jnp.array([state.n]),
            state.C,
            jnp.array([state.T_fuel, state.T_coolant, state.I_hat, state.Xe_hat]),
        ]
    )
    new_positions, metrics = integrate_dynamics(
        positions=positions,
        delta_t=params.delta_t,
        compute_velocity=_compute_velocity,
        method=integration_method,
    )

    new_n = new_positions[0]
    new_C = new_positions[1 : 1 + N_GROUPS]
    new_T_fuel = new_positions[1 + N_GROUPS]
    new_T_coolant = new_positions[2 + N_GROUPS]
    new_I_hat = new_positions[3 + N_GROUPS]
    new_Xe_hat = new_positions[4 + N_GROUPS]

    new_time = state.time + 1

    # ── Ornstein-Uhlenbeck demand process ──
    demand_mu = 0.5 * (params.target_n_range[0] + params.target_n_range[1])
    key_t = jax.random.fold_in(state.demand_key, new_time)
    noise = jax.random.normal(key_t, dtype=jnp.float32)
    drift = params.demand_theta * (demand_mu - state.target_n) * params.delta_t
    diffusion = params.demand_sigma * jnp.sqrt(params.delta_t) * noise
    new_target = jnp.clip(
        state.target_n + drift + diffusion,
        params.target_n_range[0],
        params.target_n_range[1],
    )

    return (
        state.replace(
            n=new_n,
            C=new_C,
            T_fuel=new_T_fuel,
            T_coolant=new_T_coolant,
            I_hat=new_I_hat,
            Xe_hat=new_Xe_hat,
            target_n=new_target,
            rho_ext=rho_ext,
            time=new_time,
        ),
        metrics,
    )


@partial(jax.jit, static_argnames=["params"])
def get_obs(state: ReactorState, params: ReactorParams):
    """
    Partially observable: the agent sees only measurable quantities.

    ``obs = [n, T_coolant, rho_ext_norm, target_n]``

    Hidden: C_i (6 precursors), T_fuel, I_hat, Xe_hat — 9 hidden dimensions.
    """
    span = params.rho_ext_max - params.rho_ext_min
    rho_norm = 2.0 * (state.rho_ext - params.rho_ext_min) / span - 1.0
    return jnp.array([state.n, state.T_coolant, rho_norm, state.target_n])


def check_is_terminal(state: ReactorState, params: ReactorParams, xp=jnp):
    n_out = xp.logical_or(state.n <= params.n_min, state.n >= params.n_max)
    T_fuel_out = xp.logical_or(
        state.T_fuel <= params.T_fuel_min, state.T_fuel >= params.T_fuel_max
    )
    T_coolant_out = xp.logical_or(
        state.T_coolant <= params.T_coolant_min,
        state.T_coolant >= params.T_coolant_max,
    )
    terminated = xp.logical_or(xp.logical_or(n_out, T_fuel_out), T_coolant_out)
    # state.time and max_steps_in_episode are both in physics-step units.
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward(state: ReactorState, params: ReactorParams, xp=jnp):
    """
    Reward = Gaussian tracking term minus a small rod-motion penalty.

    * Tracking : ``exp(-0.5 * (error / band)^2)``  — sharp: 3% error → 0.61,
      5% → 0.25, 10% → ~0.  This reflects the economic reality where small
      deviations from grid demand incur steep imbalance penalties.
    * Rod cost : small penalty for holding rods far from neutral.
    """
    error = xp.abs(state.target_n - state.n)
    tracking = xp.exp(-0.5 * (error / params.reward_band) ** 2)

    rho_scale = xp.maximum(xp.abs(params.rho_ext_min), xp.abs(params.rho_ext_max))
    rod_penalty = params.rod_motion_weight * xp.abs(state.rho_ext) / rho_scale

    return tracking - rod_penalty


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
