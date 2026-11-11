"""
Nuclear reactor: a two-point core with iodine and xenon in each half, thermal
feedback, a steam generator driven by the turbine, rate-limited rods that enter from
the top, and soluble boron that reaches the core late.

Physics
-------
Point kinetics with six delayed-neutron groups gives the total power ``n``::

    dn/dt   = ((rho - beta) / Lambda) * n + sum_i lambda_i * C_i
    dC_i/dt = (beta_i / Lambda) * n - lambda_i * C_i        (i = 1..6)

Reactivity is the rods, the boron, the thermal feedback and the mean xenon::

    rho = rod_worth(rod_in) - rod_worth(rod_park)
        + boron_worth * (boron - boron_ref)
        + alpha_fuel * (T_fuel - T_fuel_ref)
        + alpha_coolant * (T_avg - T_coolant_ref)
        - rho_Xe_full * (Xe_hat - 1)

The axial shape is the two-point model of Onega and Kisner (Mastroianni's thesis,
1979, eqs 2.15 to 2.18). Flux, iodine and xenon are each a cosine plus a first
harmonic, the amplitudes come from integrating over the two halves of the core, and
the flux tilt ``A`` is algebraic in the xenon tilt ``B`` (neutrons settle in
seconds, xenon in hours). ``A`` is then the axial offset, (top - bottom) / total.
Each species is held as a mean and a tilt, so a node is ``mean +- tilt``::

    A = -(8 S B + 10 eps) / (N + sqrt(N^2 + 64 S B (S B + 5/4 eps)))
    N = 45 pi^3 D / (8 H^2) - S Xe_hat + 7 n Phi alpha Sigma_a

With ``eps = 0`` that is the thesis's ``v - sqrt(v^2 + 1)``. ``eps`` is an absorption
difference between the halves, projected the same way, and it comes from the rods
and from the moderator's temperature rise up the core (PHYSICS.md, section 1).

Rods move at one speed, and the action is where to put them. Boron is a feed and
bleed through the chemical and volumetric control system: the action opens a flow
now, and the core sees it ``boron_delay`` steps later.

Observation (partially observable)
----------------------------------
``[n, T_hot, rod position, load, T_avg, T_ref, delta_I, boron (20 min late),
hours to the ramp down, hours to the ramp up, the power xenon is balanced with]``.

Action
------
``[rod demand, boron flow]`` in ``[-1, 1]``. Rod demand: -1 fully in, +1 fully out.
Boron flow: -1 full dilution, +1 full boration.
"""

import math
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym import reward as R
from target_gym.base import EnvParams, EnvState, actuator_field
from target_gym.integration import integrate_dynamics
from target_gym.utils import log_scaled_reward

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

#: Longest boron transport delay the queue holds, in 30 s control steps: 15 min.
BORON_QUEUE = 30
#: The boronmeter's delay in control steps: 20 min of 30 s steps. Areva's patent
#: US 8,489,338 gives the boron meter's reading "after a significant delay, of
#: the order of some twenty minutes"; Framatome's COMBO meter claims under a
#: minute. The French family's figure is kept. The queue holds one more slot so
#: its oldest entry is exactly this many steps old.
BORON_METER_STEPS = 40

# The two-point model's numbers, Mastroianni (1979), Table III-1, used twice below.
_SIGMA_X = 2.72e-18  # cm2
_PHI_FULL = 2.02e13 / 0.75  # n/cm2/s, the test's flux was at 75 % power
_SIGMA_F = 0.14  # 1/cm
_GAMMA_X, _GAMMA_I = 0.0046, 0.061
_NU_SIGMA_F = 2.44 * 0.16  # 1/cm, PHYSICS.md section 3
_BURN = _SIGMA_X * _PHI_FULL * math.pi / 4.0  # 1/s, cosine flux on cosine xenon


@struct.dataclass
class ReactorParams(EnvParams):
    # ---- Kinetics ----
    Lambda_gen: float = 1e-4  # s — effective mean neutron generation time

    # Thermal feedback coefficients (1/K). Both negative (stabilising).
    alpha_fuel: float = -3.0e-5  # Doppler (prompt, on fuel temperature)
    # The moderator coefficient at the start of the cycle, where the core sits
    # (1690 ppm): near zero there, ours, PHYSICS.md section 1 (physics review,
    # finding 2, 2026-11-10).
    alpha_coolant: float = -12.0e-5  # on Tavg
    # The part-power axial shift's lumped coefficient (moderator and Doppler
    # redistribution), kept at its value before the MTC moved so Krsko's +5 % delta
    # I at half power holds: ours.
    alpha_tilt: float = -30.0e-5
    # Zero feedback at full power on program.
    T_fuel_ref: float = 1051.4  # K
    T_coolant_ref: float = 580.2  # K, Tavg at full load

    # ---- The rods: a 1500 pcm bank, entering from the top ----
    rho_ext_min: float = -0.015  # rods fully inserted
    rho_ext_max: float = 0.0  # rods fully withdrawn
    # 72 steps/min over a 228-step stroke, in either direction, as a fraction of
    # the stroke: a magnetic jack has one speed.
    rod_speed: float = 72.0 / 60.0 / 228.0  # 1/s
    rod_stroke_steps: float = 228.0
    rod_park: float = 3.0 / 228.0  # D bank at 225 steps, Krsko fig. 3

    # ---- Soluble boron ----
    boron_ref: float = 1690.0  # ppm, Krsko fig. 3, at full power
    # Ours, inside the family's -6.7 to -10.9 pcm/ppm: BEAVRS measures 10.8 to 10.9
    # at hot zero power and 975 to 1405 ppm (Tables 22, 23); boron is worth less at
    # full power and at 1690 ppm. PHYSICS.md section 1.
    boron_worth: float = -8.0e-5  # per ppm
    dilution_flow: float = 10.0  # kg/s, OAPS section 2.1
    boration_flow: float = 3.0  # kg/s, OAPS section 2.1
    boric_acid_ppm: float = 7800.0  # ppm, ours: a recycling unit's concentrate
    primary_mass: float = 2.5e5  # kg, ours, the same water as C_coolant + C_cold
    # 5 to 15 min, OAPS section 2.1, in 30 s control steps, drawn at reset.
    boron_delay_range: Tuple[int, int] = (10, 30)

    # ---- Two-point core: Mastroianni (1979), Table III-1 ----
    sigma_x: float = _SIGMA_X  # cm2
    # The coupling between the halves, set by the first axial harmonic's eigenvalue
    # separation: 283 pcm for Ringhals-4 in Pedersen et al.'s homogeneous model
    # (Progress in Nuclear Energy 202, table 3), so D = 283e-5 x nu Sigma_f x H^2 /
    # (3 pi^2), derived. The thesis's own 0.405 cm is a thermal-group figure that
    # couples them 12 times too weakly, 23 pcm (docs/proposals/reactor-coupling.md).
    D_diffusion: float = 283e-5 * _NU_SIGMA_F * 365.8**2 / (3.0 * math.pi**2)  # cm
    alpha_power: float = 3.64e-16  # cm2 s, prompt power feedback
    Sigma_f: float = _SIGMA_F  # 1/cm
    nu_Sigma_f: float = _NU_SIGMA_F  # 1/cm, Sigma_a equals it at criticality
    gamma_x: float = _GAMMA_X
    gamma_I: float = _GAMMA_I
    H_core: float = 365.8  # cm
    Phi_full: float = _PHI_FULL  # n/cm2/s

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
    # sigma_phi0: Xe-135 burn at full power, a cosine flux on a cosine xenon.
    sigma_phi0: float = _BURN  # 1/s
    # gamma_ratio: ratio of direct Xe-135 fission yield to I-135 yield.
    gamma_ratio: float = _GAMMA_X / _GAMMA_I
    # Equilibrium xenon reactivity worth at full power. Typical PWR: 2000-3500 pcm.
    # rho_Xe = -rho_Xe_full * (Xe_hat - 1).
    rho_Xe_full: float = 0.025  # = 2500 pcm

    # ---- Termination / safety bounds ----
    n_min: float = 0.01  # near-shutdown — SCRAM
    precision_floor: float = 0.05  # K, the RTD reading's resolution
    n_max: float = 1.09  # the high-flux trip, 109 % of rated
    T_fuel_max: float = 1473.0  # K, well below UO2 melting (~3120 K)
    T_fuel_min: float = 500.0
    # K, ours: below saturation at the nominal 2250 psia, 618.0 K (IAPWS-IF97).
    T_coolant_max: float = 610.0
    # The minimum temperature for criticality, 541 F at Salem. It is a limit with
    # a time to act, not a trip: restore it or go to hot standby. No EDF figure is
    # public, so the Westinghouse-family value is kept (the EDF units are of that
    # family), and so is Salem's 15 min, both ours and unconfirmed: the NRC copies
    # could not be opened (docs/proposals/reactor-plant-check.md, item 2).
    T_avg_min: float = 556.0  # K
    floor_allowed: float = 900.0  # s below T_avg_min before hot standby, a trip

    # ---- Reward shaping ----
    rod_motion_weight: float = 0.02
    # Error scale for the MPC's tracking term, not read by ``compute_reward``.
    # See "Why the MPC does not minimise the reward" in docs/baselines.md. The reward
    # has not been a Gaussian for some time; this is the planner's scale.
    reward_band: float = 0.03  # relative flux (3% = tight)

    # ---- Load: a daily 100-50-100, plus a small frequency-control wobble ----
    # Ramps at 2 %/min, the top of OAPS's 1 to 2 %/min for Mode A early in a
    # cycle, with the ramp and the hold drawn at reset inside the URD's 2 to 10 h.
    load_min: float = 0.5
    load_ramp_rate: float = 0.02 / 60.0  # 1/s
    ramp_down_range: Tuple[float, float] = (1800.0, 7200.0)  # s
    #: A stream refuels the core every ``renewal_days`` plant days: renewed at its
    #: cycle point as at reset, uncharged, the outside world carrying on (owner,
    #: 2026-11-11; ours, 30 days in ``scenarios.py``, which keeps the boron within
    #: about 100 ppm of the start of cycle the coefficients describe). 0, the
    #: default, never: a window is one day of the cycle.
    renewal_days: float = 0.0
    #: How far ahead the dispatch announces a load-follow day in a stream, where
    #: each day is an event (``Reactor.start_event``): the window's own notice. A
    #: stream sets ``ramp_down_range`` to the past, so it starts with no day.
    day_notice_range: Tuple[float, float] = (1800.0, 7200.0)  # s
    hold_range: Tuple[float, float] = (3.0 * 3600.0, 8.0 * 3600.0)  # s
    # The wobble is an OU process on the load: 5 min correlation, 0.5 % sd, two-sided
    # at full load too, so the scorer's median step stays the wobble's.
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
    #: Fuel burnup, ppm of boron a day: BEAVRS (MIT, measured Westinghouse 4-loop),
    #: Table 24, 2.27 ppm a day in cycle 1 and 3.34 in cycle 2. Lost through the
    #: whole day as if at full power (D9).
    burnup_range: Tuple[float, float] = (2.3, 3.3)
    #: Days of burnup not yet diluted at reset: ours, the operator dilutes at least
    #: once a day. The day starts lined out that much below its Tavg program.
    carried_days_range: Tuple[float, float] = (0.0, 1.0)
    #: Yesterday's axial iodine and xenon tilts (physics review, finding 3;
    #: PHYSICS.md section 4): on a share of days equal to load following's,
    #: yesterday was a load-follow day and one ``(I_tilt, Xe_tilt)`` pair is
    #: drawn from the reference's ends of those days (scripts/reactor_tilt_table.py,
    #: training seeds); otherwise yesterday was baseload, which ends with no tilt
    #: to speak of (under 3.1e-4).
    initial_tilts: Tuple[Tuple[float, float], ...] = (
        (-0.010964, -0.004986), (-0.007372, -0.003397),  # edf_daily
        (-0.009495, -0.006722), (-0.003252, -0.001931),  # edf_short_hold
        (-0.010433, 0.007226), (-0.009624, 0.008357),  # krsko_slow
        (-0.008858, 0.008466), (-0.01091, 0.005141),  # krsko_shallow
        (-6.8e-05, 8.3e-05), (-0.009142, -0.006335),  # fast_return
    )  # fmt: skip
    initial_tilt_share: float = 250.0 / (5.5 + 50.0 / 60.0) / 365.0  # REALITY.md 7

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
    max_steps_in_episode: int = 2880  # env steps of 30 s: a day

    # ---- Reward (docs/reward-shaping.md; version 2), in dollars per 30 s step ----
    # Tracking: Tavg against its program, and delta I outside its band. Off program, steam conditions cost
    # output: ``imbalance_multiple`` is 1.25 x spot x 0.23 %/K, the Carnot slope at
    # 547 K steam and 305 K condenser, so $100/MWh on 0.23 % of 1 GWe per K is
    # $230/h per K, linear (p = 1): $1.917 per K over a 30 s step, and
    # ``rho_floor_tracking`` over a step at the floor. Rod wear: the steps the rods
    # actually moved at ``rod_step_price``. A trip is charged the restart time at
    # twice the dearest surviving step.
    reward_version: int = 2
    e_floor: float = 0.05  # K, the RTD reading's resolution
    # The black rods' dead band: EDF's rods move only past 1.6 C off program (IAEA
    # NP-T-3.23, annex II-5.1), so Tavg costs nothing inside it.
    e_tol: float = 1.6
    tracking_exponent: float = 1.0
    imbalance_multiple: float = 1.25 * 0.0023  # of spot, per K
    rod_step_price: float = 1.0  # $ per rod step, ours
    # The band, EDF's: during load follow the axial offset stays within 0.05 of its
    # reference (IAEA NP-T-3.23, annex II-4.3). The reference is the offset at full
    # power, xenon balanced and rods out, which this core puts at 0 by construction
    # (the tilts start at 0 and the coolant's tilt is measured from full power).
    # Past it, each 1 % of rated delta I is priced as 1 % of rated output derated,
    # $1000/h on 1 GWe at $100/MWh, ours.
    ao_band: float = 0.05
    ao_ref: float = 0.0
    # Time outside the band counts at every power during load follow (EDF's scope),
    # and 1 h of it in 24 h is allowed (Westinghouse's allowance, D.C. Cook and RAOC
    # specifications, pooled in since EDF's was not found). Past it the plant must
    # derate to 50 %, priced as the derate's lost output at spot. The clock looks
    # back 24 h in whole control hours (``roll_offset_clock``), so time outside a
    # day ago stops counting; a window is one day and counts from 0.
    ao_allowed: float = 3600.0  # s in 24 h
    derate_to: float = 0.5  # of rated power
    derate_price_per_MWh: float = 80.0  # the derate's lost output, spot as for trips
    # The rod insertion limit, the family's shape: a straight line in power, 22 %
    # of the stroke in at full power to 78 % at half (KAIST's lead), 2 h to restore
    # (STS 3.1.6), then hot standby.
    ril_full: float = 0.22
    ril_half: float = 0.78
    ril_allowed: float = 7200.0  # s
    # Tavg's upper limit, the DNB parameter of STS LCO 3.4.1: at or below the COLR's
    # value, restored within 2 h, else MODE 2 within 6 h (hot standby here).
    # Braidwood's COLR: nominal Tavg at rated power 588.0 F, limit 593.1 F, so 5.1 F
    # above the full-power program at every power (from a search summary; nrc.gov
    # blocks automated reads). The only plant found with both numbers.
    dnb_margin: float = 5.1 / 1.8  # K above the full-power program
    dnb_allowed: float = 7200.0  # s
    # Effluent: every kg injected displaces a kg to treat. Boiling it off is 2.26
    # MJ/kg, 0.628 kWh, at $100/MWh, ours.
    effluent_price: float = 0.0628  # $/kg
    # A trip is lost production: 1 GWe at spot ($80/MWh, inside the family's $80 to
    # $100) over the restart, $666.67 per 30 s step down.
    failure_cost: float = 1000.0 * 80.0 * 30.0 / 3600.0
    #: A scram's restart, 48 h (ours, family 24 to 70 h: Doel-3 ran 18 h at reduced
    #: power after a 6 h scram, INIS 1ypr9-dk595; APR1400's restart windows are
    #: 22 to 70 h, KNS 20A-065).
    restart_steps: int = 5760
    #: Hot standby after a timed limit's overrun: 15 h (ours, Doel-3's 6 h down and
    #: 18 h reduced, about 15 h of full output lost; no scram review).
    standby_steps: int = 1800
    #: Tracking cost per step at the floor, in the reward's units.
    rho_floor_tracking: float = 100.0 * 1000.0 * 0.0023 / 3600.0 * 0.05 * 30.0
    rho_floor: float = 100.0 * 1000.0 * 0.0023 / 3600.0 * 0.05 * 30.0
    #: True where e_floor is a resolution, not a measured or certified floor.
    floor_is_documented_minimum: bool = True


#: Units of the ReactorParams fields, for the generated environment pages.
PARAM_UNITS = {
    "renewal_days": "d",
    "Lambda_gen": "s",
    "alpha_fuel": "1/K",
    "alpha_coolant": "1/K",
    "alpha_tilt": "1/K",
    "T_fuel_ref": "K",
    "T_coolant_ref": "K",
    "rho_ext_min": "-",
    "rho_ext_max": "-",
    "rod_speed": "1/s",
    "rod_stroke_steps": "steps",
    "rod_park": "-",
    "boron_ref": "ppm",
    "boron_worth": "1/ppm",
    "dilution_flow": "kg/s",
    "boration_flow": "kg/s",
    "boric_acid_ppm": "ppm",
    "primary_mass": "kg",
    "sigma_x": "cm^2",
    "D_diffusion": "cm",
    "alpha_power": "cm^2 s",
    "Sigma_f": "1/cm",
    "nu_Sigma_f": "1/cm",
    "gamma_x": "-",
    "gamma_I": "-",
    "H_core": "cm",
    "Phi_full": "1/cm^2/s",
    "ao_band": "-",
    "ao_ref": "-",
    "floor_allowed": "s",
    "ao_allowed": "s",
    "derate_to": "",
    "derate_price_per_MWh": "$/MWh",
    "ril_full": "",
    "ril_half": "",
    "ril_allowed": "s",
    "dnb_margin": "K",
    "dnb_allowed": "s",
    "effluent_price": "$/kg",
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
    "burnup_range": "ppm/day",
    "carried_days_range": "day",
    "initial_tilts": "-",
    "initial_tilt_share": "-",
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
    "standby_steps": "steps",
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

    # Xenon / Iodine (normalised: 1.0 = equilibrium at full power). The two halves
    # of the core are mean + tilt and mean - tilt.
    I_hat: float  # mean I-135
    Xe_hat: float  # mean Xe-135

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

    rho_ext: float  # the rods' worth where they are
    #: What the controller *asked* the rods for this step, before the rate
    #: limit, as a worth. Read only by reward version 1.
    rho_ext_cmd: float
    I_tilt: float = 0.0  # I-135, top half against the mean
    Xe_tilt: float = 0.0  # Xe-135, top half against the mean
    rod_in: float = 3.0 / 228.0  # how far in the rods are, a fraction of the stroke
    boron: float = 1690.0  # ppm in the core
    delta_I: float = 0.0  # power difference, top minus bottom, a fraction of rated
    #: Boron flow commands on their way to the core, newest first.
    boron_queue: jax.Array = struct.field(
        default_factory=lambda: jnp.zeros(BORON_QUEUE, jnp.float32)
    )
    boron_delay: int = 20  # control steps from the command to the core
    rod_moved: float = 0.0  # fraction of the stroke the rods travelled last sub-step
    effluent: float = 0.0  # kg/s the last sub-step's command injected
    below_floor: float = 0.0  # s Tavg has been under T_avg_min without a break
    ao_outside: float = 0.0  # s the offset has spent outside its band today
    ril_outside: float = 0.0  # s the rods have sat past their insertion limit
    above_dnb: float = 0.0  # s Tavg has sat above its DNB limit without a break
    xenon_power: float = 1.0  # the recent power the starting xenon is balanced with
    burn_rate: float = 0.0  # ppm/s of boron the fuel's burnup is worth today
    #: Core boron as the boronmeter has read it, newest first; the reading is the
    #: oldest, ``BORON_METER_STEPS`` control steps late.
    boron_meter: jax.Array = struct.field(
        default_factory=lambda: jnp.full(BORON_METER_STEPS + 1, 1690.0, jnp.float32)
    )

    actuator_held: jax.Array = actuator_field(2)

    #: The day's program carries its own rate (1/s) and depth (of rated load), so
    #: a load-follow day runs to its end whatever the params do meanwhile, and its
    #: clock counts physics sub-steps from ``plan_t0``: 0 in a window, rebased by a
    #: stream at each new day, so the plan stays exact however long the plant runs
    #: (docs/proposals/long-strata.md, S1).
    plan_t0: jax.Array = struct.field(
        default_factory=lambda: jnp.zeros((), jnp.int32), kw_only=True
    )
    plan_rate: float = struct.field(default=0.02 / 60.0, kw_only=True)
    plan_depth: float = struct.field(default=0.5, kw_only=True)
    #: The offset's allowance looks back 24 h (``roll_offset_clock``): seconds
    #: outside the band in each of the last 24 control hours, and ``ao_outside``
    #: as it stood when the current hour began.
    ao_hours: jax.Array = struct.field(
        default_factory=lambda: jnp.zeros(OFFSET_HOURS, jnp.float32), kw_only=True
    )
    ao_mark: float = struct.field(default=0.0, kw_only=True)
    #: The physics sub-step of the core's last refuelling, which burnup counts
    #: from (``burnt``): 0 in a window; a stream's renewal moves it.
    burn_t0: jax.Array = struct.field(
        default_factory=lambda: jnp.zeros((), jnp.int32), kw_only=True
    )


#: Hours the offset's allowance looks back over: "1 hour during the previous 24
#: hours" (REALITY.md section 4).
OFFSET_HOURS = 24


def plan_seconds(physics_time, state, params: "ReactorParams"):
    """Seconds on the load plan's clock at ``physics_time`` sub-steps: from
    ``plan_t0``, the sub-step a stream last rebased it, 0 in a window."""
    return (physics_time - state.plan_t0) * params.delta_t


def planned_load(t, state, xp=jnp):
    """The day's load at ``t`` plan seconds, with the day's own rate and depth:
    full, down a ramp, held, back up a ramp (``scheduled_load`` with the shape the
    day carries)."""
    down = xp.minimum(
        state.plan_rate * xp.maximum(t - state.ramp_down_at, 0.0), state.plan_depth
    )
    up = xp.minimum(
        state.plan_rate * xp.maximum(t - state.ramp_up_at, 0.0), state.plan_depth
    )
    return 1.0 - down + up


def roll_offset_clock(state, params: "ReactorParams", sub_steps: int):
    """The offset's clock as a rolling 24 h count. At the start of each control
    hour the hour just done is filed, and the hour filed 24 h ago leaves
    ``ao_outside``, so time outside the band a day ago stops counting. Within a
    window's first day nothing leaves, so a window counts exactly as before.
    ``sub_steps`` physics sub-steps make a control step."""
    per_hour = jnp.maximum(
        jnp.round(3600.0 / (params.delta_t * sub_steps)), 1.0
    ).astype(jnp.int32)
    t = jnp.asarray(state.time, jnp.int32)
    hour = t // per_hour
    boundary = (t % per_hour == 0) & (t > 0)
    leaving = state.ao_hours[hour % OFFSET_HOURS]
    filed = state.ao_hours.at[(hour - 1) % OFFSET_HOURS].set(
        state.ao_outside - state.ao_mark
    )
    outside = state.ao_outside - leaving
    return state.replace(
        ao_hours=jnp.where(boundary, filed, state.ao_hours),
        ao_outside=jnp.where(boundary, outside, state.ao_outside),
        ao_mark=jnp.where(boundary, outside, state.ao_mark),
    )


def xenon_after_shutdown(I, Xe, seconds, params: "ReactorParams"):
    """Iodine and xenon after ``seconds`` with no flux, in the plant's normalised
    units, for the means or the tilts alike: iodine decays, and xenon is fed by it
    and decays (the rates of ``_slow_rates`` at n = 0, solved exactly)."""
    p = params
    feed = (LAMBDA_XENON + p.sigma_phi0) / (1.0 + p.gamma_ratio)
    e_I = jnp.exp(-LAMBDA_IODINE * seconds)
    e_X = jnp.exp(-LAMBDA_XENON * seconds)
    fed = feed * I * (e_I - e_X) / (LAMBDA_XENON - LAMBDA_IODINE)
    return I * e_I, Xe * e_X + fed


def balance_power(Xe, params: "ReactorParams"):
    """The power whose balanced xenon is ``Xe`` (``steady_state_xenon`` read
    backwards), clipped to [0, 1]: a xenon above full power's balance reads 1."""
    lam, s = LAMBDA_XENON, params.sigma_phi0
    room = jnp.maximum(lam + s - Xe * s, 1e-12)
    return jnp.clip(Xe * lam / room, 0.0, 1.0)


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
    """The control reactivity that leaves the core critical at these temperatures."""
    return -(
        params.alpha_fuel * (T_fuel - params.T_fuel_ref)
        + params.alpha_coolant * (T_avg - params.T_coolant_ref)
        - params.rho_Xe_full * (Xe_hat - 1.0)
    )


def burnt(state, params: ReactorParams):
    """The ppm of boron the fuel has burnt since reset or the last refuelling
    (``burn_t0``), at full power's rate."""
    return state.burn_rate * (state.physics_time - state.burn_t0) * params.delta_t


def rod_worth(rod_in, params: ReactorParams):
    """The rods' worth with a fraction ``rod_in`` of the stroke in: the integral
    worth of a cosine flux, ``y - sin(2 pi y) / (2 pi)``, negative going in."""
    y = rod_in
    return params.rho_ext_min * (y - jnp.sin(2.0 * jnp.pi * y) / (2.0 * jnp.pi))


def xenon_stiffness(params: ReactorParams):
    """``S``, sigma_x X0: full-power xenon's absorption at the core's peak, 1/cm."""
    p = params
    X0 = (p.gamma_I + p.gamma_x) * p.Sigma_f * p.Phi_full
    return p.sigma_x * X0 / (LAMBDA_XENON + p.sigma_phi0)


def rod_insertion_limit(n, params: ReactorParams):
    """How far in the rods may sit at power ``n``, a fraction of the stroke: a
    straight line through ``ril_full`` at full power and ``ril_half`` at half."""
    p = params
    slope = (p.ril_half - p.ril_full) / 0.5
    return jnp.clip(p.ril_full + slope * (1.0 - n), 0.0, 1.0)


def tavg_dnb_limit(params: ReactorParams):
    """Tavg's upper limit (K): the full-power program plus ``dnb_margin``, an
    absolute value as a COLR gives it."""
    return tavg_program(1.0, params) + params.dnb_margin


def flux_tilt(Xe_tilt, Xe_hat, n, eps, params: ReactorParams):
    """The axial offset the neutrons settle to: eqs 2.17 and 2.18 of Mastroianni
    (1979), with ``eps`` the absorption difference between the halves (1/cm) added
    by the same projection and the mean xenon and power free to move. Written so
    a zero tilt is no 0/0."""
    p = params
    S = xenon_stiffness(p)
    leakage = 45.0 * jnp.pi**3 * p.D_diffusion / (8.0 * p.H_core**2)
    feedback = 7.0 * n * p.Phi_full * p.alpha_power * p.nu_Sigma_f
    N = jnp.maximum(leakage - S * Xe_hat + feedback, 1e-4)
    b = S * Xe_tilt
    root = jnp.sqrt(jnp.maximum(N * N + 64.0 * b * (b + 1.25 * eps), 1e-10))
    return jnp.clip(-(8.0 * b + 10.0 * eps) / (N + root), -0.99, 0.99)


def rod_tilt(rod_in, params: ReactorParams):
    """``eps`` of a bank in from the top: half its absorption, weighted by
    sin(pi y)^3, nothing when it is out or all the way in."""
    return -0.5 * params.rho_ext_min * params.nu_Sigma_f * jnp.sin(jnp.pi * rod_in) ** 3


def moderator_tilt(T_hot, T_cold, params: ReactorParams):
    """``eps`` of the coolant's rise up the core, against the rise at full power:
    the node means sit a quarter of the rise either side of Tavg, and a smaller
    rise warms the top less, which moves power up the core (Krsko, section 2)."""
    p = params
    full_rise = p.P_thermal_ref / p.m_dot_cp
    return p.alpha_tilt * p.nu_Sigma_f * (full_rise - (T_hot - T_cold)) / 4.0


def tilt_rates(n, I_hat, Xe_hat, I_tilt, Xe_tilt, eps, p: ReactorParams):
    """Rates of ``[I_tilt, Xe_tilt]``: half the difference of the two halves."""
    A = flux_tilt(Xe_tilt, Xe_hat, n, eps, p)
    lam_sum = LAMBDA_XENON + p.sigma_phi0
    feeds = lam_sum * (p.gamma_ratio * n * A + I_tilt) / (1.0 + p.gamma_ratio)
    burn = p.sigma_phi0 * n * 8.0 / (3.0 * jnp.pi) * (A * Xe_hat + Xe_tilt)
    return (
        LAMBDA_IODINE * (n * A - I_tilt),
        feeds - LAMBDA_XENON * Xe_tilt - burn,
    )


def _reactivity(rho_ext, T_fuel, T_hot, T_cold, Xe_hat, p: ReactorParams):
    return rho_ext - rod_balance(T_fuel, 0.5 * (T_hot + T_cold), Xe_hat, p)


def _slow_rates(slow, n, load, rod_eps, p: ReactorParams):
    """Rates of ``[T_fuel, T_hot, T_cold, T_sg, I_hat, Xe_hat, I_tilt, Xe_tilt]``."""
    T_f, T_h, T_c, T_s, I_h, Xe, I_t, Xe_t = slow
    to_coolant = p.UA * (T_f - T_h)
    around_loop = p.m_dot_cp * (T_h - T_c)
    to_steam = p.UA_sg * (0.5 * (T_h + T_c) - T_s)
    lam_sum = LAMBDA_XENON + p.sigma_phi0
    feeds = lam_sum * (p.gamma_ratio * n + I_h) / (1.0 + p.gamma_ratio)
    eps = rod_eps + moderator_tilt(T_h, T_c, p)
    dI_t, dXe_t = tilt_rates(n, I_h, Xe, I_t, Xe_t, eps, p)
    A = flux_tilt(Xe_t, Xe, n, eps, p)
    return jnp.array(
        [
            (p.P_thermal_ref * n - to_coolant) / p.C_fuel,
            (to_coolant - around_loop) / p.C_coolant,
            (around_loop - to_steam) / p.C_cold,
            (to_steam - p.P_thermal_ref * load) / p.C_sg,
            LAMBDA_IODINE * (n - I_h),
            feeds - (LAMBDA_XENON + p.sigma_phi0 * n * (1.0 + A * Xe_t / Xe)) * Xe,
            dI_t,
            dXe_t,
        ]
    )


def compute_velocity(position, action, params: ReactorParams, load=1.0, rod_eps=0.0):
    """
    Right-hand side of the coupled ODE system.

    ``position = [n, C_1..6, T_fuel, T_hot, T_cold, T_sg, I_hat, Xe_hat, I_tilt,
    Xe_tilt]`` (shape (15,)). ``action`` is the control reactivity, rods and boron.
    """
    n = position[0]
    C = position[1 : 1 + N_GROUPS]
    slow = position[1 + N_GROUPS :]
    T_f, T_h, T_c = slow[:3]
    Xe = slow[5]

    rho = _reactivity(action, T_f, T_h, T_c, Xe, params)
    dn_dt = ((rho - BETA_TOT) / params.Lambda_gen) * n + jnp.sum(LAMBDA_I * C)
    dC_dt = (BETA_I / params.Lambda_gen) * n - LAMBDA_I * C

    dposition = jnp.concatenate(
        [jnp.array([dn_dt]), dC_dt, _slow_rates(slow, n, load, rod_eps, params)]
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
    positions, rho_ext, params: ReactorParams, n_substeps: int, load=1.0, rod_eps=0.0
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
        T_f, T_h, T_c, _T_s, _I, Xe, _It, _Xt = slow
        rho = _reactivity(rho_ext, T_f, T_h, T_c, Xe, params)
        n_new, C_new, n_g = _fast_block_trbdf2(n_, C_, rho, params, h)
        # RK4 on the non-stiff slow block; n sampled from the fast solve.
        k1 = _slow_rates(slow, n_, load, rod_eps, params)
        k2 = _slow_rates(slow + 0.5 * h * k1, n_g, load, rod_eps, params)
        k3 = _slow_rates(slow + 0.5 * h * k2, n_g, load, rod_eps, params)
        k4 = _slow_rates(slow + h * k3, n_new, load, rod_eps, params)
        slow_next = slow + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return (n_new, C_new, slow_next), None

    (n_f, C_f, slow_f), _ = jax.lax.scan(
        body, (n, C, slow0), xs=None, length=n_substeps
    )
    return jnp.concatenate([jnp.array([n_f]), C_f, slow_f]), None


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state(
    action: jnp.ndarray,
    state: ReactorState,
    params: ReactorParams,
    integration_method: str = "tr_bdf2_4",
):
    """
    ``action``: raw ``[rod demand, boron flow]`` in [-1, 1]. The rods move toward
    the demand at their one speed, and the core feels the boron flow at the far
    end of ``state.boron_queue`` (``Reactor.step_env`` feeds it).
    """
    p = params
    # Rods: +1 asks for all the way out, -1 all the way in.
    wanted = 0.5 * (1.0 - action[0])
    most = p.rod_speed * p.delta_t
    rod_in = jnp.clip(state.rod_in + jnp.clip(wanted - state.rod_in, -most, most), 0, 1)
    rods = rod_worth(rod_in, p)

    # Boron: feed and bleed of well-mixed coolant, exact over the sub-step.
    arriving = state.boron_queue[state.boron_delay - 1]
    flow = jnp.where(
        arriving < 0, -arriving * p.dilution_flow, arriving * p.boration_flow
    )
    feed = jnp.where(arriving < 0, 0.0, p.boric_acid_ppm)
    closes = 1.0 - jnp.exp(-flow / p.primary_mass * p.delta_t)
    boron = state.boron + (feed - state.boron) * closes

    rho_ext = rods - rod_worth(p.rod_park, p) + p.boron_worth * (boron - p.boron_ref)
    # Burnup since reset, as the boron it is worth (D9).
    rho_ext = rho_ext + p.boron_worth * burnt(state, p)
    rod_eps = rod_tilt(rod_in, p)

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
                    state.I_tilt,
                    state.Xe_tilt,
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
            positions, rho_ext, p, n_substeps, load, rod_eps
        )
    else:
        _compute_velocity = partial(
            compute_velocity, action=rho_ext, params=p, load=load, rod_eps=rod_eps
        )
        new_positions, metrics = integrate_dynamics(
            positions=positions,
            delta_t=p.delta_t,
            compute_velocity=_compute_velocity,
            method=integration_method,
        )

    new_n = new_positions[0]
    new_C = new_positions[1 : 1 + N_GROUPS]
    T_fuel, T_hot, T_cold, T_sg, I_hat, Xe_hat, I_tilt, Xe_tilt = new_positions[
        1 + N_GROUPS :
    ]
    eps = rod_eps + moderator_tilt(T_hot, T_cold, p)
    delta_I = new_n * flux_tilt(Xe_tilt, Xe_hat, new_n, eps, p)
    command = state.boron_queue[0]
    injected = jnp.where(
        command < 0, -command * p.dilution_flow, command * p.boration_flow
    )

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
    day = planned_load(plan_seconds(new_physics_time, state, p), state)
    new_load = day + wobble

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
            I_tilt=I_tilt,
            Xe_tilt=Xe_tilt,
            delta_I=delta_I,
            boron=boron,
            target_n=new_load,
            T_ref=tavg_program(new_load, p),
            load_wobble=wobble,
            rod_in=rod_in,
            rho_ext=rods,
            rho_ext_cmd=rod_worth(wanted, p),
            rod_moved=jnp.abs(rod_in - state.rod_in),
            effluent=injected,
            below_floor=jnp.where(
                0.5 * (T_hot + T_cold) < p.T_avg_min, state.below_floor + p.delta_t, 0.0
            ),
            ao_outside=state.ao_outside
            + p.delta_t * (jnp.abs(delta_I - new_n * p.ao_ref) > p.ao_band * new_n),
            above_dnb=jnp.where(
                0.5 * (T_hot + T_cold) > tavg_dnb_limit(p),
                state.above_dnb + p.delta_t,
                0.0,
            ),
            ril_outside=jnp.where(
                rod_in > rod_insertion_limit(new_n, p),
                state.ril_outside + p.delta_t,
                0.0,
            ),
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

    ``obs = [n, T_hot, rod position, load, T_avg, T_ref, delta_I, boron, to_down,
    to_up]``, the rod position in [-1, 1] with +1 fully out, as the demand is.
    ``to_down`` and ``to_up`` are the hours until the planned ramps start, 0 once
    they have: the dispatch an operator gets in advance (OAPS, section 1), so boron
    can lead its delay.

    Hidden: C_i, T_fuel, T_sg, iodine, xenon, and the boron on its way.
    """
    T_avg = 0.5 * (state.T_coolant + state.T_cold)
    now = plan_seconds(state.physics_time, state, params)
    return jnp.array(
        [
            state.n,
            state.T_coolant,
            1.0 - 2.0 * state.rod_in,
            state.target_n,
            T_avg,
            state.T_ref,
            state.delta_I,
            state.boron_meter[-1],
            jnp.maximum(state.ramp_down_at - now, 0.0) / 3600.0,
            jnp.maximum(state.ramp_up_at - now, 0.0) / 3600.0,
            state.xenon_power,
        ]
    )


def timed_out(state: ReactorState, params: ReactorParams):
    """A timed limit overrun: Tavg under its floor or above its DNB limit, or the rods
    past their insertion limit, for longer than allowed. The plant goes to hot standby, not a scram.
    """
    return (
        (state.below_floor >= params.floor_allowed)
        | (state.ril_outside >= params.ril_allowed)
        | (state.above_dnb >= params.dnb_allowed)
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
    # Tavg under the floor is a limit with a time to act, not a trip: it ends the
    # run only once it has lasted ``floor_allowed``.
    T_coolant_out = xp.logical_or(
        timed_out(state, params), state.T_coolant >= params.T_coolant_max
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
    # Tavg past its dead band on either side of its program: the band both cited
    # procedures keep (physics review, finding 1, 2026-11-10), priced as the cold
    # side's lost output.
    error = state.T_ref - 0.5 * (state.T_coolant + state.T_cold)
    tracking = (
        per_K_s
        * p.delta_t
        * p.e_floor
        * R.tracking_cost(error, p.e_floor, p.e_tol, p.tracking_exponent, xp)
    )
    # Past the offset's allowed hour, the derate's lost output.
    per_GW_s = p.P_electric_GW * 1000.0 * p.derate_price_per_MWh / 3600.0
    derated = xp.where(
        state.ao_outside >= p.ao_allowed, xp.maximum(state.n - p.derate_to, 0.0), 0.0
    )
    tracking = tracking + per_GW_s * derated * p.delta_t
    rod_wear = p.rod_step_price * p.rod_stroke_steps * state.rod_moved
    treatment = p.effluent_price * state.effluent * p.delta_t
    if tripped is None:
        tripped, _ = check_is_terminal(state, p, xp)
    terms = {
        "tracking": tracking,
        "running": rod_wear + treatment,
    }
    restart = xp.where(timed_out(state, p), p.standby_steps, p.restart_steps)
    return R.with_trip(terms, tripped, restart * p.failure_cost, xp)


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
