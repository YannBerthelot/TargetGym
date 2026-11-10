"""The F/A-18 Hornet airframe the patrol and formation tasks fly (owner,
2026-11-10: a fighter's formation on a fighter, the A320 keeps the transport
tasks). The 3D plant is the A320's (``plane3d.dynamics``); this module gives it
the Hornet's numbers. plane3d/PHYSICS.md, section "The F/A-18", is the contract.

Sources:

- [HARV] Chakraborty, Seiler and Balas, "Applications of linear and nonlinear
  robustness analysis techniques to the F/A-18 flight control laws", AIAA GNC
  2009: Table 1 (geometry, weight, inertias), Table 2 (actuators), Tables 7 and 8
  (the F/A-18 HARV aerodynamic model), section III.A (F404-GE-400, 16 100 lbf).
- [AFF] Hanson, Ryan, Allen and Jacobson, "An overview of flight test results
  for a formation flight autopilot", NASA/TM-2002-210729: two F/A-18s at M 0.56
  and 25 000 ft, a Class B envelope of 250 KCAS and 20 000 to 32 000 ft, a
  10 ft one-sigma station-keeping requirement, 56 ft nose to tail at least.
- [F404] Secondary: dry thrust 10 700 lbf and dry SFC 0.81 lb/(lbf h) (Wikipedia,
  General Electric F404); idle to full thrust in about 3.25 s (a GE figure as
  quoted on forum.dcs.world).
- [NATOPS] Secondary: +7.5 g, as forum posts quote the F/A-18 NATOPS manual.

Everything else is ours, labelled where it is set.
"""

import math

FT = 0.3048
LB = 0.45359237
LBF = 4.448222
SLUG_FT2 = 1.355818

# ---- Geometry, weight and inertia [HARV Table 1] ----
MASS = 33_310.0 * LB  # 15 109 kg
WING_AREA = 400.0 * FT**2  # 37.16 m^2
SPAN = 37.42 * FT  # 11.41 m
MAC = 11.52 * FT  # 3.51 m
ASPECT_RATIO = SPAN**2 / WING_AREA  # 3.50
PITCH_GYRATION = math.sqrt(151_293.0 * SLUG_FT2 / MASS)  # 3.69 m
ROLL_GYRATION = math.sqrt(23_000.0 * SLUG_FT2 / MASS)  # 1.44 m

# ---- Lift [HARV Table 8], alpha in rad: CL = 1.1645 a^3 - 5.4246 a^2
# + 5.6770 a - 0.0204. Its secant from 0 to 10 deg, the formation's range:
# CL(10 deg) = 0.811, so (0.811 + 0.020) / 10 = 0.0831 per deg. Its peak, 1.70
# at 38 deg, is LEX vortex lift; the plant's linear curve reaches it at 21 deg,
# where the stall blend begins (a known deviation, outside the task's range).
CL_ALPHA = 0.0831  # per deg
CL0 = -0.0204
CL_MAX = 1.70
AOA_STALL = (CL_MAX - CL0) / CL_ALPHA  # 20.7 deg
#: Half-chord sweep, rad (ours: 13 deg, from 26.7 deg at the leading edge
#: and a taper of 0.35 as the planform is drawn).
SWEEP_HALF_CHORD = math.radians(13.0)
#: Zero-lift drag and induced drag factor (ours: a clean fighter's 0.02, and
#: 1 / (pi A e) with e = 0.8).
CD0 = 0.020
K_INDUCED = 1.0 / (math.pi * ASPECT_RATIO * 0.8)
#: Drag-divergence Mach (ours: the plant is subsonic; 0.88 keeps it so).
M_CRIT = 0.88

# ---- Pitch: the stabilator, calibrated to [HARV Table 7] at 4 deg of incidence
# (0.070 rad): Cm_delta = 0.6401 x 0.070 - 1.1055 = -1.061 per rad; Cm_q =
# -14.30 x 0.070 - 2.00 = -3.00 per q c / 2V; Cm_alpha = -0.9931 per rad.
# An all-moving stabilator turns its whole incidence (effectiveness 1). For a
# tail at l behind the centre of gravity, Cm_q / Cm_delta = 2 l / c, so
# l = c x 3.00 / (2 x 1.061) = 4.96 m; then eta S_t a_t = 1.061 S c / l.
CM_DELTA, CM_Q, CM_ALPHA = -1.061, -3.00, -0.9931
TAIL_ARM = MAC * CM_Q / (2.0 * CM_DELTA)  # 4.96 m
TAIL_EFFICIENCY = 0.9  # ours, as the A320's
#: The stabilator's planform (ours: aspect ratio 2.4, 35 deg of half-chord
#: sweep), its slope DATCOM's as the wing's: 2 pi A / (2 + sqrt(A^2 (1 +
#: tan^2 L) + 4)).
TAIL_ASPECT_RATIO = 2.4
TAIL_SWEEP_HALF_CHORD = math.radians(35.0)
_TAIL_SLOPE = (
    2.0
    * math.pi
    * TAIL_ASPECT_RATIO
    / (
        2.0
        + math.sqrt(
            TAIL_ASPECT_RATIO**2 * (1.0 + math.tan(TAIL_SWEEP_HALF_CHORD) ** 2) + 4.0
        )
    )
)  # per rad
TAIL_CL_ALPHA = math.radians(_TAIL_SLOPE)  # per deg
TAIL_AREA = -CM_DELTA * WING_AREA * MAC / (TAIL_ARM * TAIL_EFFICIENCY * _TAIL_SLOPE)
#: Downwash at the tail, as a share of lifting-line's far field (ours: brings
#: de/da to 0.4, a fighter's; the far field, 2 a / (pi A) = 0.86 on this short
#: wing with its LEX lift, overstates it at a low-set stabilator).
DOWNWASH_RATIO = 0.4 / (2.0 * math.degrees(CL_ALPHA) / (math.pi * ASPECT_RATIO))
#: The wing-body's aerodynamic centre ahead of the centre of gravity, m, from
#: Cm_alpha = (arm / c) a_w - eta (S_t / S)(l / c) a_t (1 - de/da): negative, the
#: wing-body itself sits behind the centre of gravity.
WING_ARM = MAC * (CM_ALPHA + (-CM_DELTA) * (1.0 - 0.4)) / math.degrees(CL_ALPHA)
#: The stabilator's trim setting at the AFF point, rad (calibrated: zero
#: deflection there; ``scripts/calibrate_hornet.py``).
STABILIZER_SETTING = -0.09309

# ---- Roll [HARV Table 7] at 4 deg: Cl_p = -0.337; Cl_delta_a = -0.1553 x
# 0.070 + 0.1542 = 0.143 per rad. The plant's aileron moment is q S_a a tau d l,
# so S_a tau l = 0.143 S b / a (a per rad).
C_LP = -0.337
AILERON_EFFECTIVENESS = 0.4  # ours, as the A320's
AILERON_ARM = 4.0  # m, ours: 0.7 of the semi-span
AILERON_AREA = (
    0.143
    * WING_AREA
    * SPAN
    / (math.degrees(CL_ALPHA) * AILERON_EFFECTIVENESS * AILERON_ARM)
)

# ---- Engines: two F404-GE-400 at military power, the afterburner not modelled
# (ours: station-keeping is flown below it) [F404, HARV].
THRUST_SEA_LEVEL = 2.0 * 10_700.0 * LBF  # 95.2 kN
#: Dry SFC 0.81 lb/(lbf h) = 22.9 g/(kN s), in the plant's kg/(kN s) / 1000.
SFC = 0.81 * 28.325 / 1000.0
#: Idle to full in 3.25 s at sea level: a first-order lag reaches 95 % in
#: ln 19 of its time constant, 1.10 s. The plant scales it in corrected time.
POWER_RESPONSE_RATE = math.log(19.0) / 3.25
#: Fuel aboard at the patrol, kg (ours: about 60 % of the C's 4 926 kg internal).
FUEL = 3_000.0

# ---- Envelope ----
LOAD_FACTOR_MAX = 7.5  # [NATOPS], secondary
LOAD_FACTOR_MIN = -3.0  # ours: a fighter's usual negative limit
#: No autopilot manoeuvre limit: the formation is flown to the airframe's.
AP_LOAD_INCREMENT = LOAD_FACTOR_MAX
#: Bank: the inner loop's 45 deg demand limit, and no protection below it
#: (ours: the Hornet's flight controls do not limit bank in formation).
PROTECTED_BANK = math.radians(67.0)
MMO = 0.85  # ours: the plant's subsonic validity
VMO_EAS = 700.0 * 1852.0 / 3600.0  # ours: a fighter's, not reached here

#: The AFF design point and Class B band [AFF].
DESIGN_ALTITUDE = 25_000.0 * FT
DESIGN_MACH = 0.56
ALTITUDE_BAND = (20_000.0 * FT, 32_000.0 * FT)
CLASS_B_CAS = 250.0 * 1852.0 / 3600.0
#: The AFF point's true airspeed, m/s, and air density, kg/m^3: M 0.56 in the
#: ISA troposphere at 25 000 ft.
DESIGN_SPEED = DESIGN_MACH * math.sqrt(
    1.4 * 287.05 * (288.15 - 0.0065 * DESIGN_ALTITUDE)
)
DESIGN_DENSITY = 1.225 * (1.0 - 0.0065 * DESIGN_ALTITUDE / 288.15) ** 4.2559
#: The pitch a banked turn adds per g of extra load factor at the AFF point,
#: rad: the angle of attack that g takes, W / (q S a) = 0.101.
PITCH_PER_G = (
    MASS
    * 9.80665
    / (0.5 * DESIGN_DENSITY * DESIGN_SPEED**2 * WING_AREA * math.degrees(CL_ALPHA))
)

# ---- The roll damper [HARV IV.A.2]: roll rate fed back to the ailerons, 0.8 rad
# per rad/s at 250 ft/s and 25 000 ft, "reduced at high dynamic pressure" to
# 0.08. At the AFF point, the same altitude, it is taken as 1 / q (ours: the
# paper gives the range, not the schedule): 0.8 x (76.2 / 173.4)^2 = 0.154,
# inside the range. It caps the inner loop's roll D (scripts/tune_inner_loop.py).
ROLL_RATE_GAIN = 0.8 * (250.0 * FT / DESIGN_SPEED) ** 2


def airframe():
    """The Hornet as ``PlaneParams3D`` field values."""
    return dict(
        initial_mass=MASS,
        initial_fuel_quantity=FUEL,
        wings_surface=WING_AREA,
        wingspan=SPAN,
        aspect_ratio=ASPECT_RATIO,
        cl_alpha=CL_ALPHA,
        cl0=CL0,
        CL_max=CL_MAX,
        aoa_stall=AOA_STALL,
        sweep_half_chord=SWEEP_HALF_CHORD,
        cd0=CD0,
        k=K_INDUCED,
        M_crit=M_CRIT,
        pitch_gyration_radius=PITCH_GYRATION,
        roll_gyration_radius=ROLL_GYRATION,
        moment_arm_wings=WING_ARM,
        moment_arm_stabilizer=TAIL_ARM,
        stabilizer_surface=TAIL_AREA,
        tail_cl_alpha=TAIL_CL_ALPHA,
        tail_aspect_ratio=TAIL_ASPECT_RATIO,
        tail_sweep_half_chord=TAIL_SWEEP_HALF_CHORD,
        tail_k=1.0 / (math.pi * TAIL_ASPECT_RATIO * 0.8),
        tail_efficiency=TAIL_EFFICIENCY,
        elevator_effectiveness=1.0,
        downwash_ratio=DOWNWASH_RATIO,
        stabilizer_setting=STABILIZER_SETTING,
        C_lp=C_LP,
        aileron_surface=AILERON_AREA,
        aileron_effectiveness=AILERON_EFFECTIVENESS,
        moment_arm_aileron=AILERON_ARM,
        thrust_output_at_sea_level=THRUST_SEA_LEVEL,
        specific_fuel_consumption=SFC,
        power_response_rate=POWER_RESPONSE_RATE,
        load_factor_max=LOAD_FACTOR_MAX,
        load_factor_min=LOAD_FACTOR_MIN,
        ap_load_increment=AP_LOAD_INCREMENT,
        protected_bank=PROTECTED_BANK,
        mmo=MMO,
        vmo_eas=VMO_EAS,
    )
