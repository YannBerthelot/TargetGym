"""
Glass furnace (float-glass process) — regenerative end-port fired furnace.

See ``PHYSICS.md`` in this directory for provenance, the sourced parameter
table and the validation targets each figure of merit is checked against.

All temperatures are in degrees Celsius. Stefan-Boltzmann terms convert to
Kelvin internally because T^4 needs an absolute scale; every other term uses a
temperature *difference*, which is scale-invariant.

Layout
------
::

      air in ->[ REGEN A ]-\\                    /-[ REGEN B ]-> stack
                            \\                  /
                             +--- CROWN GAS ---+        <- T_gas   (flame)
                             |   crown refractory       <- T_crown [measured]
                             +------------------+
                             | BATCH | MELT | WORKING END |
                             | blanket T_melt |  T_work   |   (hidden)
                             +------------------+
                                 glass flows ->

The two regenerators alternate every ``reversal_period`` seconds: one preheats
incoming combustion air while the other is heated by the exhaust. Each is
modelled as two nodes (hot end / cold end) in counter-flow, because a single
lumped node cannot reproduce both a hot preheated air stream and a cool stack
-- the checker temperature gradient is what makes both possible.

State (9 ODEs)
--------------
``T_gas``      combustion-space gas / flame (fast, seconds)
``T_crown``    crown refractory (slow, hours) -- the *measured* variable
``T_melt``     glass in the melting zone (hidden)
``T_work``     glass in the working end (hidden)
``T_rA_hot``, ``T_rA_cold``   regenerator A checker nodes (hidden)
``T_rB_hot``, ``T_rB_cold``   regenerator B checker nodes (hidden)
``m_batch``    unmelted batch blanket floating on the melt (hidden)

Why the extra structure over a 3-node model
-------------------------------------------
* **Regenerators** are the defining feature of a float furnace. Without heat
  recovery the energy balance is out by roughly 2x against the published
  4-6 GJ/tonne, and the slow checker mass plus the reversal cycle -- a genuine
  history-dependent disturbance -- are missing entirely.
* **Separating flame gas from crown refractory** fixes a timescale
  conflation. Lumping them gave the measured crown temperature a single
  hours-long time constant, hiding the seconds-scale combustion response the
  controller actually sees first.
* **The batch blanket** shields the melt from radiation in proportion to how
  much unmelted batch is floating on it, coupling pull rate to melting rate
  nonlinearly.
* **Temperature-dependent glass c_p** (~1000 -> 1400 J/(kg.K)) matters for
  energy accounting across a 1000 K operating span.
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

SIGMA_SB = 5.670374419e-8  # Stefan-Boltzmann (W m^-2 K^-4)
KELVIN_OFFSET = 273.15  # °C -> K

# Number of piecewise-constant setpoints in an episode.
N_SETPOINTS = 5

# AR(1) correlation for the pull-rate disturbance. dt=30 s, rho=0.99 -> ~50 min
# memory: a drifting operating point rather than white noise.
M_PULL_AR_RHO = 0.99

#: Steps of pure transport delay between commanding a fuel flow and that heat
#: reaching the crown. At 30 s a step this is 60 s, covering the gas train, the
#: combustion-air adjustment that follows it and flame development.
#:
#: Dead time is what makes thermal control hard, and this model had none: the
#: valve reached the flame within the step and the crown thermocouple read the
#: true state instantly. Inertia is not the same thing. A first-order plant with
#: no dead time has no bandwidth limit, so a PID can be tuned arbitrarily tight
#: against it, and one duly held the crown to 0.078 K against a 10 K open-loop
#: drift -- roughly thirty times better than a real furnace is held. Dead time
#: is also the specific thing an MPC handles better than a PID, by predicting
#: through it, so its absence removed the reason to have both baselines.
FUEL_DEAD_TIME_STEPS = 2


@struct.dataclass
class GlassFurnaceParams(EnvParams):
    # ---- Combustion ----
    LHV: float = 50.0e6  # natural gas lower heating value (J/kg)
    AFR: float = 17.0  # stoichiometric air/fuel mass ratio
    excess_air: float = 0.10  # 10 % excess air (typical; sets flue O2)
    c_p_air: float = 1150.0  # J/(kg.K), hot air
    c_p_gas: float = 1200.0  # J/(kg.K), flue gas
    # Sized against the measured steady-state gain, ~1861 C per kg/s: the
    # 1427-1677 C operating band corresponds to only 0.53-0.635 kg/s, so a
    # wider action range would spend most of its span driving the furnace out
    # of bounds and leave the controller with almost no usable resolution.
    # raw=0 -> 0.59 kg/s holds ~1586 C at ~5.1 GJ/tonne; both extremes remain
    # genuinely unsurvivable, so the range still spans real failure modes.
    # Raised 2.6% when the reversal changeover started interrupting firing.
    # The burners have to deliver the same time-averaged heat despite losing
    # 40 s in every 1500 s, which is what real burners are sized to do; without
    # it, mid throttle settled at 1527 C, below the calibrated operating band.
    fuel_min: float = 0.513  # kg/s (under-fires: crown cools past the limit)
    fuel_max: float = 0.698  # kg/s (over-fires: crown exceeds the limit)
    # Fraction of combustion heat released as radiation from the flame rather
    # than carried as gas enthalpy.
    flame_rad_fraction: float = 0.55

    # ---- Regenerators (two chambers, two nodes each) ----
    C_regen_node: float = 3.0e7  # J/K per checker node
    # Per-node effectiveness; N_REGEN_NODES of them in series per chamber.
    eps_regen_node: float = 0.80
    reversal_period: float = 1500.0  # s (25 min) -- typical float furnace

    #: Firing interruption at each reversal. The burners are shut off while the
    #: reversing valves change over and the other port lights, which on a real
    #: furnace takes tens of seconds and shows as a crown temperature dip every
    #: 25 min. The changeover here used to be instantaneous and lossless, which
    #: deviation D2 in PHYSICS.md recorded: the model was missing the one
    #: periodic upset the plant actually has.
    #:
    #: Not conserved, unlike batch charging. The heat is genuinely not released
    #: during the changeover, so mean firing falls by the interruption fraction
    #: and the controller has to make it up -- which is what happens.
    reversal_dead_time: float = 40.0  # s of interrupted firing per reversal
    #: Firing during the changeover, as a fraction of commanded. Not zero:
    #: pilots stay lit, and a completely dead combustion space makes the flame
    #: balance singular for no physical gain.
    reversal_firing_floor: float = 0.05

    #: Crown thermocouple time constant, s. The element sits in a refractory
    #: sheath in the crown, so it reports a filtered version of the gas-side
    #: temperature rather than the temperature itself. Only the observation is
    #: filtered; the reward scores the true crown temperature, because what the
    #: glass sees is what matters and the instrument is the controller's problem.
    tau_thermocouple: float = 120.0

    #: Batch charging. The doser is a reciprocating pusher on a cycle, not a
    #: continuous stream: it fires for ``charge_duty`` of each ``charge_period``
    #: and is idle the rest. The mean charge rate is unchanged, so the steady
    #: state is exactly as before, but the load now has content at the charging
    #: frequency instead of only the slow drift of the pull disturbance -- and a
    #: slow smooth load is the one thing integral action cancels perfectly.
    charge_period: float = 300.0  # s, one charger cycle
    charge_duty: float = 0.30  # fraction of the cycle the doser is feeding
    charge_mass_jitter: float = 0.15  # +/- fraction, dose to dose
    U_regen: float = 0.30  # W/(m^2.K) casing loss
    A_regen: float = 300.0  # m^2 per chamber

    # ---- Glass flow ----
    m_pull: float = 5.79  # kg/s glass (~500 t/day float line)
    batch_yield: float = 0.83  # kg glass per kg batch (CO2 / volatiles loss)
    T_batch_in: float = 25.0  # °C
    dH_fusion: float = 0.8e6  # J/kg (latent + endothermic reactions)
    m_batch_full: float = 39000.0  # kg blanket giving full melt coverage
    batch_shield: float = 0.85  # radiation blocked at full coverage

    # ---- Pull-rate disturbance (AR(1) stationary std, kg/s) ----
    m_pull_noise_std: float = 0.4

    # ---- Thermal capacities (J/K) ----
    C_gas: float = 2.4e5  # combustion space gas (tau ~ 17 s)
    C_crown: float = 2.5e8  # silica crown refractory (~250 t)
    C_melt: float = 7.8e8  # ~625 t glass -> ~30 h residence
    C_work: float = 7.8e8

    # ---- Glass specific heat, c_p(T) = c_p_a + c_p_b * T[°C] ----
    c_p_glass_a: float = 900.0
    c_p_glass_b: float = 0.35

    # ---- Radiation / convection ----
    eps_rad: float = 0.8  # effective emissivity
    A_crown: float = 200.0  # m^2 gas <-> crown
    A_melt: float = 200.0  # m^2 exposed to flame + crown
    A_work: float = 10.0  # m^2 (largely shielded beyond the throat)
    h_conv: float = 30.0  # W/(m^2.K)

    # ---- Wall losses ----
    U_wall: float = 1.0  # W/(m^2.K)
    A_wall_crown: float = 200.0
    A_wall_melt: float = 200.0
    A_wall_work: float = 200.0
    # Conditioning-zone heat extraction (W/K). A float working end cools
    # glass from the ~1500 C melting temperature to the ~1100-1200 C the
    # tin bath needs, which is several MW -- crown radiation plus forced
    # cooling. Without it the working end sits hotter than the melt, which
    # is thermodynamically backwards for a zone fed *by* the melt.
    UA_work_cooling: float = 3000.0
    T_ambient: float = 25.0  # °C

    # ---- Operating / termination bounds ----
    T_crown_min: float = 1427.0  # °C, incomplete melting
    #: The error at which crown tracking is bad, in K. Used by the MPC's
    #: objective, in the same role as ``tracking_band`` on the four-tank, the
    #: distillation column and the pH loop. Sized from what the log-scaled
    #: reward actually discriminates over here: 1 K scores 0.875, 2 K scores
    #: 0.801, 4 K scores 0.709, and a furnace held 10 K off its setpoint is one
    #: nobody would call controlled.
    tracking_band: float = 10.0
    precision_floor: float = 1.0  # K, type-B thermocouple resolution at 1600 K
    T_crown_max: float = 1677.0  # °C, refractory damage
    T_glass_min: float = 900.0
    T_glass_max: float = 1727.0

    # ---- Reward shaping ----
    # Zeroed for the 0.6 line: this phase scores setpoint tracking alone.
    # A running cost is a real part of every one of these plants, but its
    # weight against tracking accuracy is a design decision this library has
    # not earned yet, and an arbitrary one turns a tracking benchmark into a
    # multi-objective problem whose Pareto point nobody chose. The glass
    # furnace showed the cost of getting it wrong: its MPC sat 6 K cold with
    # fuel at minimum 80% of the time, because a 0.1 fuel weight against a
    # quadratic tracking surrogate made that the optimum of the objective it
    # was given. The field and the term stay, so a weight can be restored
    # once there is a defensible way to set it. See docs/roadmap.md.
    fuel_cost_weight: float = 0.0  # was 0.1
    # ---- Initial / target ranges ----
    # Operationally realistic setpoint band. A float furnace is trimmed by
    # +/-10-20 C around nominal; a 150 C step means re-heating the entire glass
    # inventory, which takes days (settling time 140-300 h, measured) and is
    # untrackable inside an episode. With the old 1500-1650 range the schedule
    # spread, not the controller, decided the score: seeds whose five setpoints
    # happened to sit close together scored 4413 with 6 C mean error, while a
    # seed with a 146 C swing scored 734 with 37 C -- a 6x spread from sampling
    # alone.
    target_T_crown_range: Tuple[float, float] = (1565.0, 1610.0)
    #: How large a single trim may be. The schedule used to be five independent
    #: uniform draws from the band above, which is not how a furnace is
    #: operated: five independent draws from a 45 C band span about 30 C on
    #: average and can step 40 C between consecutive slots, while the comment
    #: on that band already says a float furnace is trimmed by 10-20 C around
    #: nominal. An operator trims from where the furnace is, in small
    #: increments, so the schedule is now a bounded walk and this is the
    #: largest step it may take.
    target_T_crown_trim: float = 6.0
    #: Where the walk starts, as a band about the middle of the operating
    #: range. Narrower than the full band, because the first setpoint is the
    #: level the furnace is already being held at, not an arbitrary one.
    target_T_crown_start_range: Tuple[float, float] = (1580.0, 1595.0)
    initial_T_crown_range: Tuple[float, float] = (1572.0, 1602.0)
    #: How far off its first setpoint the furnace may start. Drawn
    #: independently of the schedule, the crown began a median 7 K and up
    #: to 18 K away from the level it was being held at, which is not a
    #: state a running furnace is found in: it takes days to reach thermal
    #: equilibrium and is then held within a few degrees. A small offset
    #: still leaves the first slot something to correct.
    #: ``initial_T_crown_range`` bounds the result.
    initial_T_crown_offset: float = 4.0
    # Reset places the furnace on a *consistent operating point* rather than an
    # arbitrary state. Offsets are measured from the settled steady state at
    # nominal firing (crown 1587 C): a real furnace takes days to reach thermal
    # equilibrium, so starting it away from one would make the whole episode an
    # initial transient and leave nothing to control.
    initial_T_gas_offset: float = 31.0  # T_gas   - T_crown at steady state
    initial_T_melt_offset: float = -35.0  # T_melt  - T_crown
    initial_T_work_offset: float = -151.0  # T_work  - T_crown
    initial_T_regen_hot: float = 1383.0  # hot end of the settled checker stack
    initial_T_regen_cold: float = 544.0  # cold end
    initial_m_batch: float = 10753.0  # settled blanket mass

    # ---- Time discretization ----
    delta_t: float = 30.0  # s per step
    max_steps_in_episode: int = 1600  # 13.3 h at dt = 30 s

    # ---- Reward (docs/reward-shaping.md; version 2) ----
    # Floor: the shipped MPC's long-run mean |crown error| under the shipped
    # pull disturbance, 0.175 K (the lowest of three seeds: 0.192 / 0.175 / 0.228; PID 0.49-0.52) over 3600 hold steps (30 h) after a 10 800-step
    # (90 h) burn-in, 3 seeds, stationary across the window (0.209 / 0.188 K
    # halves), `scripts/measure_hold.py`; PID 0.504 K. An upper bound on the
    # achievable floor. Fuel above the hold-phase flow (0.590 kg/s, the same
    # for PID and MPC) is charged at weight 1 (provisional: no fuel price
    # supplied). The crown envelope costs (250 / 0.175)^2 = 2.0e6 per step;
    # a refractory or glass excursion twice that.
    reward_version: int = 2
    # The MPC holds 0.175 K in the shipped disturbance, below the 1 K thermocouple
    # resolution: the instrument sets the scale (a hold it cannot see is not a floor).
    e_floor: float = 1.0  # K
    e_tol: float = 0.0
    tracking_exponent: float = 2.0
    c_hold: float = 0.590  # kg/s fuel while holding (PID = MPC)
    running_weight: float = 1.0  # provisional; sweep 0.5 / 1 / 2
    failure_cost: float = (
        2.0 * (183.0 / 1.0) ** 2
    )  # reachable crown excursion, 1610 -> 1427 C
    #: Restart time priced into a trip (``reward.trip_cost``): refractory damage or
    #: glass out of range ends the campaign; a furnace heat-up schedule is about two
    #: weeks, 40 320 steps at 30 s (provisional).
    restart_steps: int = 40320
    #: The NEA reference: the lowest per-seed hold cost the shipped MPC
    #: demonstrated, in the reward's units (the MPC's 0.175 K hold in 1 K units).
    rho_floor_tracking: float = (0.175 / 1.0) ** 2
    rho_floor: float = (0.175 / 1.0) ** 2
    floor_is_documented_minimum: bool = False


@struct.dataclass
class GlassFurnaceState(EnvState):
    T_gas: float
    T_crown: float
    T_melt: float
    T_work: float
    T_rA: jnp.ndarray  # (N_REGEN_NODES,) chamber A, hot end first
    T_rB: jnp.ndarray  # (N_REGEN_NODES,) chamber B
    m_batch: float

    target_T_crown: float
    target_schedule: jnp.ndarray
    m_pull_disturbance: float

    #: What the crown thermocouple reads, lagging ``T_crown`` by
    #: ``tau_thermocouple``. This is what ``get_obs`` reports.
    T_crown_meas: float
    #: Fuel flows commanded but not yet burning, oldest first. The applied flow
    #: is ``fuel_pipeline[0]``; a new command enters at the end.
    fuel_pipeline: jnp.ndarray

    # Diagnostics for rendering / observation
    fuel_flow: float
    T_air_preheat: float
    T_stack: float


def glass_c_p(T, params: GlassFurnaceParams):
    """Soda-lime glass specific heat, linear in temperature (J/(kg.K))."""
    return params.c_p_glass_a + params.c_p_glass_b * T


def reversal_phase(time, params: GlassFurnaceParams):
    """0.0 while regenerator A preheats air, 1.0 while B does."""
    cycles = (time * params.delta_t) / params.reversal_period
    return jnp.mod(jnp.floor(cycles), 2.0)


def get_target_from_schedule(
    target_schedule: jnp.ndarray, time: int, params: GlassFurnaceParams
) -> jnp.ndarray:
    """Select the currently-active setpoint from the schedule."""
    slot = jnp.minimum(
        (time * N_SETPOINTS) // params.max_steps_in_episode,
        N_SETPOINTS - 1,
    )
    return target_schedule[slot]


N_REGEN_NODES = 4  # checker nodes per chamber, hot end first


def _regenerator_flows(T_nodes, T_gas, m_air, m_gas, is_air_side, params):
    """Counter-flow duties for one regenerator chamber's checker stack.

    ``T_nodes`` is hot-end-first. Exhaust enters at the hot end and works
    down; air enters at the cold end and works up. Each node exchanges with
    the stream passing it at per-node effectiveness ``eps_regen_node``.

    Why a *stack* rather than one lumped node: a single node must settle at
    the flow-weighted mean of the two stream temperatures it alternately
    sees, which caps air preheat near 1000 C and leaves the stack near 750 C.
    A real regenerator only achieves 1200-1400 C air with a 400-600 C stack
    because of its continuous top-to-bottom temperature gradient, and it
    takes several nodes to represent that gradient at all.

    Returns ``(Q_nodes, T_air_out, T_stack)``; ``Q_nodes`` is net heat into
    each node (negative while preheating air).
    """
    p = params
    eps = p.eps_regen_node

    def exhaust_pass(T_in, T_node):
        T_out = T_in - eps * (T_in - T_node)
        return T_out, m_gas * p.c_p_gas * (T_in - T_out)

    def air_pass(T_in, T_node):
        T_out = T_in + eps * (T_node - T_in)
        return T_out, -m_air * p.c_p_air * (T_out - T_in)

    # Exhaust: hot end -> cold end (node order as stored).
    T_e, Q_exh = jax.lax.scan(exhaust_pass, T_gas, T_nodes)
    # Air: cold end -> hot end (reversed), then flip the duties back.
    T_a, Q_air_rev = jax.lax.scan(air_pass, p.T_ambient, T_nodes[::-1])
    Q_air = Q_air_rev[::-1]

    Q_nodes = jnp.where(is_air_side, Q_air, Q_exh)
    return Q_nodes, T_a, T_e


def solve_T_gas(T_gas_guess, T_crown, T_melt, T_work, T_air, m_fuel, melt_open, params):
    """Quasi-steady flame temperature, solved algebraically.

    The combustion-space gas has a radiative time constant of
    ``C_gas / (4*eps*sigma*A*T^3)`` ~ 0.5 s -- roughly sixty times shorter than
    the 30 s control step. Integrating it explicitly is both wasteful and
    unstable (the eigenvalue is ~2 /s, far outside RK4's stability region at
    this step size, which produces NaN). Its balance is therefore solved to
    steady state each step, the standard singular-perturbation reduction:
    keep the flame as a distinct *node* -- so the crown still sees a separate,
    hotter radiating source -- without carrying it as a stiff differential
    state.

    Newton on ``Q_in(T) - Q_out(T) = 0``; the residual is smooth and monotone
    decreasing in ``T``, so a handful of iterations from the previous value
    converge tightly.
    """
    p = params
    m_air = p.AFR * (1.0 + p.excess_air) * m_fuel
    m_gas = m_fuel + m_air
    Q_in = m_fuel * p.LHV + m_air * p.c_p_air * (T_air - p.T_ambient)

    A_eff = p.A_crown + p.A_melt * melt_open + p.A_work
    T_crown_K = T_crown + KELVIN_OFFSET
    T_melt_K = T_melt + KELVIN_OFFSET
    T_work_K = T_work + KELVIN_OFFSET
    sink_K4 = (
        p.A_crown * T_crown_K**4
        + p.A_melt * melt_open * T_melt_K**4
        + p.A_work * T_work_K**4
    )
    sink_T = p.A_crown * T_crown + p.A_melt * melt_open * T_melt + p.A_work * T_work

    def body(T, _):
        T_K = T + KELVIN_OFFSET
        f = (
            Q_in
            - p.eps_rad * SIGMA_SB * (A_eff * T_K**4 - sink_K4)
            - p.h_conv * (A_eff * T - sink_T)
            - m_gas * p.c_p_gas * (T - p.T_ambient)
        )
        df = (
            -4.0 * p.eps_rad * SIGMA_SB * A_eff * T_K**3
            - p.h_conv * A_eff
            - m_gas * p.c_p_gas
        )
        return T - f / df, None

    T_gas, _ = jax.lax.scan(body, T_gas_guess, xs=None, length=6)
    return jnp.clip(T_gas, p.T_ambient, 4000.0)


def compute_velocity(
    position,
    action,
    m_pull,
    phase,
    T_gas,
    params: GlassFurnaceParams,
    position_time: float = 0.0,
):
    """Right-hand side of the coupled ODE system.

    ``position`` = [T_crown, T_melt, T_work,
                    T_rA_hot, T_rA_cold, T_rB_hot, T_rB_cold, m_batch]
    ``T_gas``    = quasi-steady flame temperature from :func:`solve_T_gas`
    ``phase``    = 0.0 -> A preheats air, 1.0 -> B preheats air
    """
    p = params
    T_crown, T_melt, T_work = position[0], position[1], position[2]
    m_batch = jnp.maximum(position[3], 0.0)
    T_rA = position[4 : 4 + N_REGEN_NODES]
    T_rB = position[4 + N_REGEN_NODES : 4 + 2 * N_REGEN_NODES]
    m_fuel = action

    m_air = p.AFR * (1.0 + p.excess_air) * m_fuel
    m_gas = m_fuel + m_air

    A_is_air = phase < 0.5
    QA_nodes, _, _ = _regenerator_flows(T_rA, T_gas, m_air, m_gas, A_is_air, p)
    QB_nodes, _, _ = _regenerator_flows(
        T_rB, T_gas, m_air, m_gas, jnp.logical_not(A_is_air), p
    )

    T_gas_K = T_gas + KELVIN_OFFSET
    T_crown_K = T_crown + KELVIN_OFFSET
    T_melt_K = T_melt + KELVIN_OFFSET
    T_work_K = T_work + KELVIN_OFFSET

    coverage = jnp.clip(m_batch / p.m_batch_full, 0.0, 1.0)
    melt_open = 1.0 - p.batch_shield * coverage

    Q_rad_gas_crown = p.eps_rad * SIGMA_SB * p.A_crown * (T_gas_K**4 - T_crown_K**4)
    Q_rad_gas_melt = (
        p.eps_rad * SIGMA_SB * p.A_melt * (T_gas_K**4 - T_melt_K**4) * melt_open
    )
    Q_rad_gas_work = p.eps_rad * SIGMA_SB * p.A_work * (T_gas_K**4 - T_work_K**4)
    Q_rad_crown_melt = (
        p.eps_rad * SIGMA_SB * p.A_melt * (T_crown_K**4 - T_melt_K**4) * melt_open
    )
    Q_rad_crown_work = p.eps_rad * SIGMA_SB * p.A_work * (T_crown_K**4 - T_work_K**4)

    Q_conv_gas_crown = p.h_conv * p.A_crown * (T_gas - T_crown)
    Q_conv_gas_melt = p.h_conv * p.A_melt * (T_gas - T_melt) * melt_open
    Q_conv_gas_work = p.h_conv * p.A_work * (T_gas - T_work)

    Q_wall_crown = p.U_wall * p.A_wall_crown * (T_crown - p.T_ambient)
    Q_wall_melt = p.U_wall * p.A_wall_melt * (T_melt - p.T_ambient)
    Q_wall_work = p.U_wall * p.A_wall_work * (T_work - p.T_ambient)
    Q_cool_work = p.UA_work_cooling * (T_work - p.T_ambient)

    # ---- Batch blanket ----
    Q_to_batch = (
        p.batch_shield
        * coverage
        * p.eps_rad
        * SIGMA_SB
        * p.A_melt
        * ((T_gas_K**4 - T_melt_K**4) + (T_crown_K**4 - T_melt_K**4))
    )
    Q_to_batch = jnp.maximum(Q_to_batch, 0.0)
    melt_rate = Q_to_batch / p.dH_fusion
    charge_rate = charge_rate_now(m_pull, position_time, p)
    dm_batch = charge_rate - melt_rate

    cp_melt = glass_c_p(T_melt, p)
    cp_work = glass_c_p(T_work, p)
    Q_adv_in_melt = m_pull * cp_melt * (p.T_batch_in - T_melt)
    Q_adv_melt_to_work = m_pull * cp_work * (T_melt - T_work)

    dT_crown = (
        Q_rad_gas_crown
        + Q_conv_gas_crown
        - Q_rad_crown_melt
        - Q_rad_crown_work
        - Q_wall_crown
    ) / p.C_crown

    dT_melt = (
        Q_rad_gas_melt
        + Q_conv_gas_melt
        + Q_rad_crown_melt
        - Q_wall_melt
        - melt_rate * p.dH_fusion
        + Q_adv_in_melt
    ) / (p.C_melt * cp_melt / p.c_p_glass_a)

    dT_work = (
        Q_rad_gas_work
        + Q_conv_gas_work
        + Q_rad_crown_work
        - Q_wall_work
        - Q_cool_work
        + Q_adv_melt_to_work
    ) / (p.C_work * cp_work / p.c_p_glass_a)

    UA_node = p.U_regen * p.A_regen / N_REGEN_NODES
    dT_rA = (QA_nodes - UA_node * (T_rA - p.T_ambient)) / p.C_regen_node
    dT_rB = (QB_nodes - UA_node * (T_rB - p.T_ambient)) / p.C_regen_node

    return (
        jnp.concatenate(
            [jnp.array([dT_crown, dT_melt, dT_work, dm_batch]), dT_rA, dT_rB]
        ),
        None,
    )


def regenerator_diagnostics(positions, T_gas, m_fuel, phase, params):
    """Air-preheat and stack temperatures for a given state."""
    p = params
    T_rA = positions[4 : 4 + N_REGEN_NODES]
    T_rB = positions[4 + N_REGEN_NODES : 4 + 2 * N_REGEN_NODES]
    m_air = p.AFR * (1.0 + p.excess_air) * m_fuel
    m_gas = m_fuel + m_air
    A_is_air = phase < 0.5
    _, T_air_A, T_stack_A = _regenerator_flows(T_rA, T_gas, m_air, m_gas, A_is_air, p)
    _, T_air_B, T_stack_B = _regenerator_flows(
        T_rB, T_gas, m_air, m_gas, jnp.logical_not(A_is_air), p
    )
    return (
        jnp.where(A_is_air, T_air_A, T_air_B),
        jnp.where(A_is_air, T_stack_B, T_stack_A),
    )


def firing_fraction(time, params: GlassFurnaceParams, xp=jnp):
    """Fraction of commanded firing actually released over this step.

    One at any time away from a reversal, ``reversal_firing_floor`` while the
    valves change over. Averaged over the step for the same reason
    :func:`charge_rate_now` is: the changeover is 40 s against a 30 s step, so
    an instantaneous gate sampled once a step would land inside or outside it
    depending on nothing physical.
    """
    period = xp.maximum(params.reversal_period, 1e-6)
    dead = xp.clip(params.reversal_dead_time, 0.0, period)
    t = xp.asarray(time, dtype=jnp.float32) * params.delta_t
    dp = xp.minimum(params.delta_t, period)
    p0 = xp.mod(t, period)

    # The changeover sits at the *end* of each half-cycle, just before the phase
    # flips, so a reset at t = 0 starts in a firing period rather than in the
    # middle of a valve changeover.
    def _overlap(lo, hi, a, b):
        return xp.maximum(xp.minimum(hi, b) - xp.maximum(lo, a), 0.0)

    off = _overlap(p0, p0 + dp, period - dead, period) + _overlap(
        p0, p0 + dp, 2.0 * period - dead, 2.0 * period
    )
    frac_off = off / dp
    return 1.0 - frac_off * (1.0 - params.reversal_firing_floor)


def charge_rate_now(m_pull, time, params: GlassFurnaceParams, xp=jnp):
    """Batch mass entering the furnace over this step, kg/s.

    The doser is a reciprocating pusher: it fires for ``charge_duty`` of each
    ``charge_period`` and is idle for the rest. What is returned is the rate
    *averaged over the step*, not the instantaneous rate, which matters because
    the integrator evaluates this once per step at a fixed time: sampling an
    instantaneous gate at ten points a cycle loses whichever pulses fall between
    samples, and a first attempt starved the furnace of a third of its batch.
    Averaging over the step conserves the mass exactly, whatever ``delta_t`` is.

    The dose mass varies cycle to cycle by ``charge_mass_jitter``, deterministic
    in the cycle index so the disturbance stays a pure function of time, as
    every disturbance in this suite is.
    """
    period = xp.maximum(params.charge_period, 1e-6)
    duty = xp.clip(params.charge_duty, 1e-3, 1.0)
    t = xp.asarray(time, dtype=jnp.float32) * params.delta_t
    dp = xp.minimum(params.delta_t / period, 1.0)
    p0 = xp.mod(t / period, 1.0)

    # Fraction of [p0, p0 + dp) that lies inside [0, duty), on the circle.
    def _overlap(lo, hi, a, b):
        return xp.maximum(xp.minimum(hi, b) - xp.maximum(lo, a), 0.0)

    on = _overlap(p0, p0 + dp, 0.0, duty) + _overlap(p0, p0 + dp, 1.0, 1.0 + duty)
    frac = on / dp

    cycle = xp.floor(t / period)
    jitter = 1.0 + params.charge_mass_jitter * xp.sin(12.9898 * cycle + 78.233)
    return (m_pull / params.batch_yield) * jitter * frac / duty


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state(
    fuel_raw: float,
    state: GlassFurnaceState,
    params: GlassFurnaceParams,
    key: jax.Array,
    integration_method: str = "rk4_2",
):
    """``fuel_raw`` in [-1, 1] maps to [fuel_min, fuel_max] kg/s."""
    m_fuel_cmd = convert_raw_action_to_range(
        fuel_raw, min_action=params.fuel_min, max_action=params.fuel_max
    )
    # Transport delay: what burns now was commanded FUEL_DEAD_TIME_STEPS ago,
    # and the reversal changeover interrupts whatever that was.
    m_fuel = state.fuel_pipeline[0] * firing_fraction(state.time, params)
    new_pipeline = jnp.concatenate([state.fuel_pipeline[1:], jnp.array([m_fuel_cmd])])

    # AR(1) pull-rate disturbance. The innovation is drawn from a key folded
    # with ``state.time`` so a caller passing a constant key (as every rollout
    # helper here does) still gets a genuine zero-mean process rather than one
    # repeated innovation ramping the disturbance to ~100x its intended scale.
    innovation = (
        jax.random.normal(jax.random.fold_in(key, state.time))
        * params.m_pull_noise_std
        * jnp.sqrt(1.0 - M_PULL_AR_RHO**2)
    )
    new_disturbance = M_PULL_AR_RHO * state.m_pull_disturbance + innovation
    m_pull_eff = jnp.maximum(params.m_pull + new_disturbance, 0.1)

    phase = reversal_phase(state.time, params)
    positions = jnp.concatenate(
        [
            jnp.array([state.T_crown, state.T_melt, state.T_work, state.m_batch]),
            state.T_rA,
            state.T_rB,
        ]
    )

    coverage = jnp.clip(state.m_batch / params.m_batch_full, 0.0, 1.0)
    melt_open = 1.0 - params.batch_shield * coverage
    T_air, _ = regenerator_diagnostics(positions, state.T_gas, m_fuel, phase, params)
    T_gas = solve_T_gas(
        state.T_gas,
        state.T_crown,
        state.T_melt,
        state.T_work,
        T_air,
        m_fuel,
        melt_open,
        params,
    )

    _compute_velocity = partial(
        compute_velocity,
        action=m_fuel,
        m_pull=m_pull_eff,
        phase=phase,
        T_gas=T_gas,
        params=params,
        position_time=state.time,
    )
    new_positions, _ = integrate_dynamics(
        positions=positions,
        delta_t=params.delta_t,
        compute_velocity=_compute_velocity,
        method=integration_method,
    )

    # Crown thermocouple: a first-order lag on the true crown temperature,
    # integrated exactly over the step rather than by Euler, so the reading does
    # not depend on delta_t dividing tau.
    alpha = 1.0 - jnp.exp(-params.delta_t / jnp.maximum(params.tau_thermocouple, 1e-6))
    new_T_meas = state.T_crown_meas + alpha * (new_positions[0] - state.T_crown_meas)

    new_time = state.time + 1
    new_target = get_target_from_schedule(state.target_schedule, new_time, params)
    T_air_new, T_stack_new = regenerator_diagnostics(
        new_positions, T_gas, m_fuel, phase, params
    )

    return (
        state.replace(
            T_gas=T_gas,
            T_crown=new_positions[0],
            T_melt=new_positions[1],
            T_work=new_positions[2],
            m_batch=jnp.maximum(new_positions[3], 0.0),
            T_rA=new_positions[4 : 4 + N_REGEN_NODES],
            T_rB=new_positions[4 + N_REGEN_NODES : 4 + 2 * N_REGEN_NODES],
            target_T_crown=new_target,
            m_pull_disturbance=new_disturbance,
            T_crown_meas=new_T_meas,
            fuel_pipeline=new_pipeline,
            fuel_flow=m_fuel,
            T_air_preheat=T_air_new,
            T_stack=T_stack_new,
            time=new_time,
        ),
        None,
    )


# Deliberately not jitted. It was decorated with
# ``@partial(jax.jit, static_argnames=["params"])``, which keys the compilation
# cache on the params object: a fresh ``Params(...)`` -- what every sweep, tuner
# and MPC builds -- was a cache miss and a full recompile, measured at ~1600x the
# cost of a cached call. Callers that want it fused already jit ``step_env``,
# which traces this inline.
def get_obs(state: GlassFurnaceState, params: GlassFurnaceParams):
    """Partially observable: only plant instrumentation is visible.

    ``[T_crown_meas, T_air_preheat, fuel_pct, reversal_phase, target_T_crown]``

    A real furnace has a crown thermocouple and a regenerator/air-preheat
    thermocouple, and the operator knows the reversal state. Glass
    temperatures, checker node temperatures, the batch blanket mass and the
    pull-rate disturbance are all hidden.

    The crown reading is the *instrument*, ``T_crown_meas``, which lags the true
    crown temperature by ``tau_thermocouple``. The reward scores the true one.
    A controller that assumes its thermocouple reads the plant is making the
    mistake the instrument exists to punish.
    """
    fuel_pct = 100.0 * state.fuel_flow / params.fuel_max
    return jnp.array(
        [
            state.T_crown_meas,
            state.T_air_preheat,
            fuel_pct,
            reversal_phase(state.time, params),
            state.target_T_crown,
        ]
    )


def check_is_terminal(state: GlassFurnaceState, params: GlassFurnaceParams, xp=jnp):
    crown_out = xp.logical_or(
        state.T_crown <= params.T_crown_min, state.T_crown >= params.T_crown_max
    )
    glass_out = xp.logical_or(
        xp.logical_or(
            state.T_melt <= params.T_glass_min, state.T_melt >= params.T_glass_max
        ),
        xp.logical_or(
            state.T_work <= params.T_glass_min, state.T_work >= params.T_glass_max
        ),
    )
    terminated = xp.logical_or(crown_out, glass_out)
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward_terms(state: GlassFurnaceState, params: GlassFurnaceParams, xp=jnp):
    """The reward's additive cost terms, each >= 0 (``target_gym.reward``)."""
    terminated, _ = check_is_terminal(state, params, xp)
    terms = {
        "tracking": R.tracking_cost(
            state.target_T_crown - state.T_crown,
            params.e_floor,
            params.e_tol,
            params.tracking_exponent,
            xp,
        ),
        "running": R.running_cost(
            state.fuel_flow, params.c_hold, params.running_weight, xp
        ),
    }
    return R.with_trip(terms, terminated, R.trip_cost(params), xp)


def compute_reward_v1(state: GlassFurnaceState, params: GlassFurnaceParams, xp=jnp):
    """Crown-temperature tracking minus a normalised fuel cost.

    Tracking is log-scaled (``utils.log_scaled_reward``): every halving of the
    error is worth the same increment, from the plant's operating envelope down
    to ``precision_floor``, the finest error its instrument can resolve. Below
    that the reward stops paying, because further "improvement" is noise.

    This replaced a clipped squared band, which was flat -- exactly zero, no
    gradient -- for any error outside the band, and which stopped
    discriminating just where a good controller operates. See
    docs/reward-shaping.md.
    """
    err = xp.abs(state.target_T_crown - state.T_crown)
    tracking = log_scaled_reward(
        err, params.precision_floor, params.T_crown_max - params.T_crown_min, xp
    )
    fuel_span = params.fuel_max - params.fuel_min
    fuel_norm = (state.fuel_flow - params.fuel_min) / fuel_span
    return tracking * (1.0 - params.fuel_cost_weight * fuel_norm)


def compute_reward(state: GlassFurnaceState, params: GlassFurnaceParams, xp=jnp):
    return R.select(
        params.reward_version,
        compute_reward_v1(state, params, xp),
        R.total(compute_reward_terms(state, params, xp), xp),
        xp,
    )


def specific_energy_consumption(state: GlassFurnaceState, params: GlassFurnaceParams):
    """Fuel energy per unit glass produced, in GJ/tonne.

    The headline efficiency figure for a float furnace, and the main
    validation target for the regenerator model (published: 4-6 GJ/tonne).
    """
    m_pull_eff = jnp.maximum(params.m_pull + state.m_pull_disturbance, 0.1)
    return (state.fuel_flow * params.LHV / m_pull_eff) / 1.0e9 * 1000.0
