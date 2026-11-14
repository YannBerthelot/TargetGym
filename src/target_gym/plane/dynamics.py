# from jax.tree_util import Partial as partial
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import EnvParams

from target_gym.base import FieldReading
from target_gym.utils import compute_norm_from_coordinates

_FT = 0.3048  # m
_KT = 1852.0 / 3600.0  # m/s
_DEG = float(jnp.pi) / 180.0  # rad

#: The instrument behind each state field of an aircraft, by the sections on
#: readings in PHYSICS.md. ``gamma`` and ``psi`` have none: an IRS works them
#: out from its velocities (``read_angles``).
INSTRUMENTS = {
    "x": dict(sigma=0.27),
    "y": dict(sigma=0.27),
    "z": dict(sigma=0.2, step=_FT),
    "x_dot": dict(sigma=0.03, step=0.125 * _KT),
    "y_dot": dict(sigma=0.03, step=0.125 * _KT),
    "z_dot": dict(sigma=0.03, step=_FT / 60.0),
    "theta": dict(sigma=0.0, step=0.010986 * _DEG),
    "phi": dict(sigma=0.0, step=0.010986 * _DEG),
    "theta_dot": dict(sigma=4.3e-6, step=0.015625 * _DEG),
    "phi_dot": dict(sigma=4.3e-6, step=0.015625 * _DEG),
    "power": dict(sigma=0.0, step=0.003),
    "stick": dict(sigma=0.0, step=1.6e-3),
    "aileron": dict(sigma=0.0, step=2.6e-3),
}
#: What a 2D aircraft reads, what a 3D one adds, and its GNSS fix.
LONGITUDINAL = ("z", "x_dot", "z_dot", "theta", "theta_dot", "power", "stick")
LATERAL = ("y_dot", "phi", "phi_dot", "aileron")
POSITION = ("x", "y")


def instruments(fields, prefix=""):
    """One reading per name in ``fields``, on the state field ``prefix + name``
    (``"lead."`` for an aircraft nested in the state)."""
    return tuple(FieldReading(prefix + f, **INSTRUMENTS[f]) for f in fields)


def read_angles(state, params, **angles):
    """``state`` with ``angles`` as worked out from its measured velocities, so
    they inherit that noise instead of taking their own. At
    ``sensor_noise_scale = 0`` nothing is measured, so the stored values stay."""
    quiet = params.sensor_noise_scale == 0
    return state.replace(
        **{k: jnp.where(quiet, getattr(state, k), v) for k, v in angles.items()}
    )


def advance_gust(
    gust, theta: float, sigma: float, dt: float, key, impulse_prob: float = 0.0
):
    """Advance a turbulence gust one step.

    OU mode (``impulse_prob == 0``, default): ``gust`` mean-reverts to zero,
    ``g' = g - theta*dt*g + sigma*sqrt(dt)*N``, an autocorrelated gust.

    Impulse mode (``impulse_prob > 0``): a rare memoryless kick. Each step, with
    per-component probability ``impulse_prob``, a kick of magnitude ``~sigma*N``
    is added; between kicks the gust decays as ``g' = (1 - theta*dt)*g``. Arrivals
    carry no autocorrelation, but the kick has temporal extent (its decay), so a
    lookahead sensor anticipates the onset while memory can only track the decay.

    With ``key is None`` or ``sigma == 0`` the gust is unchanged. Shared by the
    2D and 3D engines so turbulence is a physics-engine property.
    """
    if key is None:
        return gust
    k_ou, k_arr, k_mag = jax.random.split(key, 3)
    noise = jax.random.normal(k_ou, jnp.shape(gust))
    ou = gust - theta * dt * gust + sigma * jnp.sqrt(dt) * noise
    arrive = (jax.random.uniform(k_arr, jnp.shape(gust)) < impulse_prob).astype(
        gust.dtype
    )
    kick = arrive * sigma * jax.random.normal(k_mag, jnp.shape(gust))
    impulse = (1.0 - theta * dt) * gust + kick
    return jnp.where(impulse_prob > 0.0, impulse, ou)


def advance_dryden_gust(gust, air_speed, std, scale_length, dt, key):
    """Advance a turbulence gust one step along the flight path.

    A first-order Gauss-Markov gust, the Dryden form of the longitudinal
    component, whose correlation distance is the scale length: the gust
    decorrelates as ``exp(-V dt / L)`` over the ``V dt`` metres flown in the
    step, and its stationary standard deviation is ``std`` exactly. MIL-F-8785C
    sets L = 1 750 ft above 2 000 ft [MADDEN]. ``std = 0`` lets the gust decay
    to calm, which is how a turbulence zone ends. ``key is None`` freezes it.
    """
    if key is None:
        return gust
    phi = jnp.exp(-jnp.maximum(air_speed, 1.0) * dt / scale_length)
    noise = jax.random.normal(key, jnp.shape(gust))
    return phi * gust + std * jnp.sqrt(1.0 - phi**2) * noise


def in_zone(time, zone_start, zone_end):
    """Whether the step at ``time`` (s) is inside the episode's turbulence zone."""
    return (time >= zone_start) & (time < zone_end)


def step_key(key, time):
    """Per-step PRNG key derived from the episode key *and* the step index.

    ``step_env`` receives whatever key the caller supplies, and a caller may
    supply the *same* key at every step (``save_video`` does, and the runners
    did until they moved onto ``benchmark.run_policy``). Drawing noise
    straight from it would then redraw one identical innovation forever, collapsing a zero-mean disturbance into a deterministic
    ramp toward ``innovation / (1 - rho)``.

    Folding the step index in makes each step's draw distinct regardless of
    whether the caller splits keys, while staying deterministic given
    ``(key, time)`` -- so episodes remain exactly reproducible.
    """
    if key is None:
        return None
    return jax.random.fold_in(key, time)


def total_wind_2d(z, gust_x, gust_z, params):
    """Total world-frame wind = steady mean + linear altitude shear + gust.

    Single source of truth used by both the transition and the (optional)
    wind observation, so they never disagree.
    """
    shear = params.wind_shear_x * (z - params.shear_ref_alt)
    return params.wind_x + shear + gust_x, params.wind_z + gust_z


def total_wind_3d(z, gust_x, gust_y, gust_z, params):
    """Total world-frame 3D wind = mean + linear altitude shear + gust."""
    dz = z - params.shear_ref_alt
    return (
        params.wind_x + params.wind_shear_x * dz + gust_x,
        params.wind_y + params.wind_shear_y * dz + gust_y,
        params.wind_z + gust_z,
    )


def compute_drag(S: float, C: float, V: float, rho: float) -> float:
    """
    Compute the drag.

    Args:
        S (float): The surface (m^2) relative to the direction of interest.
        C (float): The drag coefficient (no units) relative to the direction of interest.
        V (float): The relative (w.r.t to the wind) speed (m.s^-1) on the axis of the direction of interest.
        rho (float): The air density (kg.m^-3) at the current altitude.

    Returns:
        float: The drag (in Newtons).
    """
    return 0.5 * rho * S * C * (V**2)


def compute_weight(mass: float, g: float) -> float:
    """Compute the weight of the plane given its mass and g"""
    return mass * g


def newton_second_law(
    thrust: float,
    lift: float,
    drag: float,
    P: float,
    gamma: float,  # flight path angle [rad]
    theta: float,  # pitch angle [rad]
) -> tuple[float, float]:
    """
    Newton's second law (vectorized form). Computes net aerodynamic, thrust, and weight forces.
    Returns (F_x, F_z) in world coordinates.
    """
    # velocity direction from gamma
    v_hat = jnp.array([jnp.cos(gamma), jnp.sin(gamma)])  # unit vector along velocity

    # drag: always opposite velocity
    F_drag = -drag * v_hat

    # lift: perpendicular to velocity (90° CCW rotation)
    perp_v = jnp.array([-v_hat[1], v_hat[0]])
    F_lift = lift * perp_v

    # thrust: along body axis (theta is pitch angle)
    t_hat = jnp.array([jnp.cos(theta), jnp.sin(theta)])
    F_thrust = thrust * t_hat

    # weight: acts downward
    F_weight = jnp.array([0.0, -P])

    # total force
    F_total = F_drag + F_lift + F_thrust + F_weight
    return F_total[0], F_total[1]


def check_power(power):
    assert 0.0 <= power <= 1.0, f"Power should be between 0 and 1, got {power}"


EPS = 1e-8


# First-order actuator lags. They are parameters so that a slower engine or a
# stiffer control system is expressible, and so that they appear in the parameter
# table rather than sitting unsourced in the code.
POWER_RESPONSE_RATE = 1.0 / 1.7
IDLE_POWER = 0.05
STICK_RESPONSE_RATE = 1.0 / 0.06


def compute_next_power(
    lever,
    current_power,
    delta_t,
    rate: float = POWER_RESPONSE_RATE,
    idle: float = IDLE_POWER,
):
    """The engine after ``delta_t``: a first-order lag, solved exactly, toward
    the power the lever in [0, 1] asks for, its range mapped onto idle to full."""
    requested_power = idle + (1.0 - idle) * jnp.clip(lever, 0.0, 1.0)
    return requested_power + (current_power - requested_power) * jnp.exp(
        -rate * delta_t
    )


def engine_response_rate(z, rate_sea_level):
    """The engine's first-order rate at height ``z``, 1/s.

    ``rate_sea_level`` holds the certification's sea-level response (14 CFR
    33.73(b)). A gas generator's transients scale in corrected time,
    ``t delta / sqrt(theta)``: the torque that accelerates the spools goes with the
    inlet pressure ``delta`` and the speed it must reach with ``sqrt(theta)`` (the
    corrected parameters every engine map and FADEC schedule is drawn in). So the
    same change of corrected thrust takes ``sqrt(theta) / delta`` times as long:
    5 s at FL290, 6 s at FL350, against 1.7 s at sea level. Derived, from the
    standard atmosphere.
    """
    temperature, pressure = standard_atmosphere(z)
    theta, delta = temperature / 288.15, pressure / 101325.0
    return rate_sea_level * delta / jnp.sqrt(theta)


def compute_next_surface(requested, current, delta_t, rate, rate_limit):
    """A control surface after ``delta_t``: a first-order lag, solved exactly,
    toward ``requested``, moving no faster than ``rate_limit``."""
    move = (requested - current) * (1.0 - jnp.exp(-rate * delta_t))
    most = rate_limit * delta_t
    return current + jnp.clip(move, -most, most)


# ---------------------------------------------------------------------------
# The fixed inner loop (PHYSICS.md section on the inner loop)
# ---------------------------------------------------------------------------

#: Control sub-steps per step: the inner loop closes at 4 Hz, around a short
#: period of about 2.5 s that a 1 s step could not resolve.
INNER_STEPS = 4
#: What a full action demands: the pitch TECS is limited to and the bank
#: ArduPilot's FBWA gives at full stick, which is the model's validity too.
PITCH_DEMAND_MAX = 15.0 * _DEG
BANK_DEMAND_MAX = 45.0 * _DEG
#: The surfaces' travel, which the loop's output in [-1, 1] spans.
STICK_TRAVEL = 15.0 * _DEG
AILERON_TRAVEL = 25.0 * _DEG


@struct.dataclass
class InnerLoop:
    """The attitude loop's gains, per radian of error and rad/s of rate, one
    tune for every task (``scripts/tune_inner_loop.py``, by the manual procedure
    of ``docs/proposals/expert-protocols-aircraft.md``), and ArduPilot's airspeed
    scaler about the tuning test's EAS, 230 m/s at 5500 m."""

    Kp_theta: float = 1.5279
    Ki_theta: float = 0.0637
    Kd_theta: float = 0.1273
    Kp_roll: float = 1.2732
    Ki_roll: float = 0.0318
    Kd_roll: float = 0.1273
    scaling_speed: float = 173.5  # m/s, SCALING_SPEED
    scale_min: float = 0.3  # min(0.5, 0.5 ARSPD_FBW_MIN / SCALING_SPEED)
    scale_max: float = 2.0  # max(2, 1.5 VMO / SCALING_SPEED)


def airspeed_scaler(eas, gains: InnerLoop):
    """ArduPilot's surface scaler: a surface's moment goes with EAS^2."""
    return jnp.clip(
        gains.scaling_speed / jnp.maximum(eas, 1.0), gains.scale_min, gains.scale_max
    )


def attitude_axis(kp, ki, kd, error, rate, integral, scaler, dt):
    """One axis of the attitude loop: ``(surface in [-1, 1], new integral)``.

    ``AP_RollController``'s law reduced to P on the angle and D on the rate
    (the proposal's section 1), with the scaler on the angle path and squared
    on the rate path. The integral is in surface units, so a scaler change does
    not move its trim, and it freezes while the surface is saturated.
    """
    output = kp * scaler * error + integral - kd * scaler**2 * rate
    surface = jnp.clip(output, -1.0, 1.0)
    grown = integral + ki * scaler**2 * error * dt
    return surface, jnp.where(output == surface, grown, integral)


def compute_thrust_output(
    power: float,  # throttle setting (0–1)
    thrust_output_at_sea_level: float,  # max thrust at sea level, N
    M: float,  # Mach number
    rho: float,  # air density at current altitude, kg/m³
    M_crit: float = 0.85,  # critical Mach number for thrust drop
    k1: float = 0.5,  # ram drag factor
    k2: float = 10.0,  # shock-induced thrust drop factor
) -> float:
    """
    Computes jet engine thrust with Mach and altitude effects.
    """
    # --- altitude factor (simple density scaling) ---
    sigma = rho / 1.225  # density ratio
    # altitude_factor = 0.8 * sigma + 0.2  # tunable
    altitude_factor = sigma
    # --- Mach effects ---
    # Ram drag effect (gradual quadratic decrease)
    mach_loss = 1 / (1 + k1 * M**2)

    # Shock-induced thrust drop beyond critical Mach
    shock_drop = jnp.exp(-k2 * jnp.maximum(M - M_crit, 0) ** 2)

    # --- final thrust ---
    thrust = (
        power * thrust_output_at_sea_level * altitude_factor * mach_loss * shock_drop
    )
    return thrust


def standard_atmosphere(z, xp=jnp):
    """Temperature (K) and pressure (Pa) of the International Standard Atmosphere
    at ``z`` metres: a lapse of 6.5 K/km to the tropopause at 11 000 m, then
    216.65 K with the pressure falling as ``exp(-(z - 11 000) / 6 341.6)``."""
    temperature = xp.maximum(288.15 - 0.0065 * z, 216.65)
    troposphere = (xp.minimum(temperature, 288.15) / 288.15) ** 5.2559
    above = 0.22336 * xp.exp(-(z - 11_000.0) / 6341.6)
    return temperature, 101325.0 * xp.where(z <= 11_000.0, troposphere, above)


def compute_air_density_from_altitude(altitude: float) -> float:
    """The standard atmosphere's air density (kg/m3) at ``altitude`` metres."""
    # Clipped so a divergent gradient-tuning rollout past the terminal state
    # stays finite; termination bounds are well inside.
    altitude = jnp.clip(altitude, -50_000.0, 40_000.0)
    temperature, pressure = standard_atmosphere(altitude)
    return pressure / (287.05 * temperature)


def compressibility(mach, aspect_ratio, sweep_half_chord):
    """How much a wing's lift slope has grown at ``mach``, over its value at 0.

    DATCOM's subsonic lift slope of a swept wing (Helmbold's, as extended by
    Diederich and Polhamus), ``2 pi A / (2 + sqrt(A^2 (beta^2 + tan^2 L) + 4))``
    with ``L`` the half-chord sweep and the section's slope taken as 2 pi. The
    ``2 pi A`` cancels in the ratio. For an unswept wing of infinite span this is
    Prandtl-Glauert's 1/beta; sweep and a finite span hold it well below that,
    1.33 at M 0.78 for the A320's wing against 1.60.
    """
    tan2 = jnp.tan(sweep_half_chord) ** 2
    beta2 = jnp.maximum(1e-6, 1.0 - mach**2)

    def inverse_slope(b2):
        return 2.0 + jnp.sqrt(aspect_ratio**2 * (b2 + tan2) + 4.0)

    return inverse_slope(1.0) / inverse_slope(beta2)


def aero_coefficients(aoa_deg, mach, params):
    """
    Realistic lift (CL) and drag (CD) coefficients for an A320.
    AoA in degrees. Mach effects included.

    Stall model: lift rises linearly at ``cl_alpha`` and *peaks at*
    ``aoa_stall``. The separation sigmoid is therefore centred
    ``aoa_stall_width`` degrees *beyond* the stall angle -- centring it on
    ``aoa_stall`` itself (as it was originally) halves the lift exactly where
    it should be maximal, which capped attainable CL at ~0.70 and put the
    clean stall speed at 228 kt. See ``PHYSICS.md`` §4.
    """

    # Wrap into [-180, 180]. ``theta`` is not wrapped by the integrator, so a
    # departed aircraft can arrive here with an incidence of several thousand
    # degrees; without this the model would be evaluated at an angle it has no
    # meaning for.
    aoa_deg = ((aoa_deg + 180.0) % 360.0) - 180.0

    # --- Lift coefficient ---
    CL_linear = params.cl0 + params.cl_alpha * aoa_deg
    stall_centre = params.aoa_stall + params.aoa_stall_width
    # How abruptly attached flow gives way to separated flow. This is not a free
    # constant: the wing has to reach CL_max essentially *attached* at
    # ``aoa_stall``, because the separated branch develops far less lift and any
    # appreciable blend at the peak would eat CL_max and move the validated
    # stall speed. Requiring the separated fraction to be 1 % there, given that
    # the sigmoid is centred ``aoa_stall_width`` beyond the stall angle, fixes
    # it at ln(99)/aoa_stall_width = 1.53 -- which is the 1.5 that used to be
    # written here, rounded.
    #
    # It is bounded from the other side too: much sharper and the wing would
    # separate over well under a degree, which no wing does. The 10-90 % band
    # this gives is 2.9 deg, in the 2-5 deg a clean transport wing shows.
    stall_sharpness = jnp.log(99.0) / params.aoa_stall_width
    # ``jax.nn.sigmoid`` rather than ``1 / (1 + exp(u))``, which is the same
    # function written so it survives float32. A departed aircraft reaches
    # incidences of 80 deg and beyond, and there ``exp(u)`` overflows: at 90 deg
    # the exponent is 110, ``exp`` returns inf, and while the forward value is
    # still a perfectly good 0, the reverse-mode derivative of ``x / (1 + inf)``
    # is NaN. That poisoned every gradient in any rollout long enough to depart,
    # which is what made ``tune_plane_pid`` return NaN gains and pinned two of
    # the 3D aircraft tuners as expected failures.
    CL = CL_linear * jax.nn.sigmoid(-(aoa_deg - stall_centre) * stall_sharpness)
    # Positive and negative stall limits. A cambered transport wing stalls
    # asymmetrically: CL_max ~ +1.5 near +15 deg, CL_min ~ -1.0 near -10 deg.
    # Only the positive limit existed before; with the corrected (steeper)
    # lift slope the negative branch runs away without this clamp.
    CL = jnp.clip(CL, params.CL_min, params.CL_max)

    # --- Drag coefficient ---
    CD = params.cd0 + params.k * CL**2

    # --- Mach corrections ---
    # The swept wing's own growth of lift slope with Mach (``compressibility``),
    # not Prandtl-Glauert's 1/beta, which is a 2D section's and overstated the
    # A320's slope at M 0.78 by a fifth.
    growth = compressibility(mach, params.aspect_ratio, params.sweep_half_chord)
    CL = CL * growth

    # Shock stall (PHYSICS.md D3). Compressibility raises the lift *slope* with
    # Mach, which is right, but on its own it also raises the attainable PEAK
    # lift -- the opposite of what happens. Past the critical Mach number,
    # shock-induced separation makes CL_max collapse; that is lift divergence,
    # and it is why transport aircraft have an overspeed limit at all.
    #
    # The cap is therefore the compressibility-scaled stall limit up to
    # ``M_crit``, and falls beyond it. The two branches agree at ``M_crit``, so
    # this is a no-op everywhere the model was validated (cruise sits near
    # M 0.7) and only bites in the regime the model previously got backwards.
    growth_crit = compressibility(
        params.M_crit, params.aspect_ratio, params.sweep_half_chord
    )
    shock = jnp.clip(
        1.0 - params.k_shock_stall * jnp.maximum(mach - params.M_crit, 0.0),
        params.shock_stall_floor,
        1.0,
    )
    cl_cap = jnp.where(
        mach <= params.M_crit,
        params.CL_max * growth,
        params.CL_max * growth_crit * shock,
    )
    cl_floor = jnp.where(
        mach <= params.M_crit,
        params.CL_min * growth,
        params.CL_min * growth_crit * shock,
    )
    CL = jnp.clip(CL, cl_floor, cl_cap)

    drag_rise = jnp.where(
        mach > params.M_crit, params.k_drag * (mach - params.M_crit) ** 2, 0.0
    )
    CD = CD + drag_rise

    # --- Post-stall: blend to a flat plate -------------------------------
    # Below stall the wing is an aerofoil; well past it, it is a barn door.
    # The attached-flow model above cannot represent that, and its failure is
    # not merely a loss of accuracy: ``CD = cd0 + k*CL**2`` ties drag to lift,
    # so the stall sigmoid that collapses CL collapses CD with it. A fully
    # separated wing came out at CD = cd0 = 0.02 -- less drag than in cruise --
    # which left a departed aircraft with no aerodynamic forces at all. It fell
    # at 300 m/s (the implied terminal velocity is 767 m/s) and tumbled without
    # damping, because nothing opposed the rotation either.
    #
    # A flat plate at incidence gives CL = sin 2a and CD = 2 sin^2 a, so CD is
    # about 2 at 90 deg rather than 0.02, and the implied terminal velocity is
    # ~108 m/s. The blend uses the same centre and steepness as the stall
    # sigmoid above, so lift is handed over to the plate exactly as separation
    # takes it away, and it is driven by |aoa| so both branches stall.
    #
    # Applied after the Mach corrections on purpose: separated flow is not a
    # compressibility effect, and the Mach factor has no business scaling it.
    # Same sharpness as the lift break above: one separation process, so lift is
    # handed to the plate exactly as it is taken from the aerofoil.
    separated = 1.0 / (
        1.0 + jnp.exp(-(jnp.abs(aoa_deg) - stall_centre) * stall_sharpness)
    )
    aoa_rad = jnp.deg2rad(aoa_deg)
    # Peak normal-force coefficient of the separated wing. A 2D flat plate
    # reaches 2; a finite wing reaches less, because flow relieves around the
    # tips. Viterna & Corrigan (1981) give CD_max = 1.11 + 0.018 AR for the
    # fully separated case, which for this wing's AR of 9.48 is 1.28 -- so this
    # is the wing's own geometry rather than an assumed value.
    cn_max = 1.11 + 0.018 * params.aspect_ratio
    CL_plate = 0.5 * cn_max * jnp.sin(2.0 * aoa_rad)
    CD_plate = params.cd0 + cn_max * jnp.sin(aoa_rad) ** 2
    CL = (1.0 - separated) * CL + separated * CL_plate
    CD = (1.0 - separated) * CD + separated * CD_plate

    CL = jnp.clip(CL, -2.0, 2.0)  # typical A320: max lift ~1.5-1.7
    # Upper bound is now the flat plate's ~2.0, not 1.0: the old cap predates
    # there being any post-stall drag to accommodate.
    CD = jnp.clip(CD, 0.0, 2.1)  # headroom over the separated wing's ~1.3

    return CL, CD


def buffet_line(mach, params):
    """The most lift the wing holds at ``mach`` before buffet (PHYSICS.md):
    ``CL_max`` at low speed, falling linearly. Compressibility raises the lift
    slope with Mach but not this. The wing's only, since buffet is a wing's."""
    return params.CL_max - params.buffet_slope * jnp.maximum(
        mach - params.buffet_mach, 0.0
    )


def scheduled_airspeed(z, cas, mach, xp=jnp):
    """The true airspeed of a schedule of ``cas`` (m/s) below the crossover and
    ``mach`` above it, by the standard atmosphere and the compressible pitot
    relation between CAS, impact pressure and Mach."""
    temperature, pressure = standard_atmosphere(z, xp)
    sound = xp.sqrt(1.4 * 287.0 * temperature)
    impact = 101325.0 * ((1.0 + 0.2 * (cas / 340.29) ** 2) ** 3.5 - 1.0)
    mach_of_cas = xp.sqrt(5.0 * ((impact / pressure + 1.0) ** (2.0 / 7.0) - 1.0))
    return xp.minimum(mach_of_cas, mach) * sound


def pitch_demand_bounds(gamma_air, air_speed, z, mass, params, load_reference=1.0):
    """The pitch demand the flight controls let through (REALITY.md, limits).

    Two limits, the tighter binding. Normal law holds the load factor inside the
    limit loads and keeps the angle of attack under its maximum: "the airplane
    cannot be stalled in normal law" [NTSB 1.6.3, BEA447]. And the autopilot
    shapes its own manoeuvres to ``ap_load_increment`` either side of the load
    the flight path needs, ``load_reference`` (1 level, ``1 / cos(bank)`` in a
    turn): an altitude capture uses "an acceptable acceleration limit with
    consideration for occupant comfort" [AMC 25.1329, altitude capture], and a load factor
    below zero is a Hazardous effect [AMC 25.1329, A4.2]. Autopilots in the
    record bound it at 0.1 to 0.2 g [KFC 500, TM 1-1510-225-10]; the policy, the
    expert and the oracle all fly through the same limit.

    Here that is one clip on the demanded attitude: no more incidence than the
    wing needs for the upper load or to reach the buffet line, the model's own
    stall boundary, and no less than it needs for the lower load. Attitude is
    flight path plus incidence in both the 2D and the 3D model
    (``compute_alpha``, ``compute_alpha_3d``). The wing's linear lift curve and
    its compressibility factor invert exactly; the tail's share of lift is left
    out (PHYSICS.md, protections).
    """
    rho = compute_air_density_from_altitude(z)
    q = 0.5 * rho * jnp.maximum(air_speed, 1.0) ** 2
    mach = air_speed / compute_speed_of_sound_from_altitude(z)
    growth = compressibility(mach, params.aspect_ratio, params.sweep_half_chord)
    per_g = mass * params.gravity / (q * params.wings_surface)
    n_hi = jnp.minimum(
        params.load_factor_max, load_reference + params.ap_load_increment
    )
    n_lo = jnp.maximum(
        params.load_factor_min, load_reference - params.ap_load_increment
    )
    cl_hi = jnp.minimum(n_hi * per_g, buffet_line(mach, params))
    cl_lo = n_lo * per_g

    def incidence(cl):
        deg = (cl / growth - params.cl0) / params.cl_alpha
        return jnp.deg2rad(jnp.clip(deg, -params.aoa_stall, params.aoa_stall))

    return gamma_air + incidence(cl_lo), gamma_air + incidence(cl_hi)


def economy_cost(power, z, tas, target_tas, params, xp=jnp):
    """What the operator pays per second inside the bands, in reference-cruise
    seconds: fuel plus time, by the cost index, per air distance flown.

    The FMGS's ECON speed minimises ``C_F fuel + C_T time``, the cost index
    being ``C_T / C_F`` in kg/min [Airbus, cost index]. Over a fixed window the
    time is spent whatever the speed, so what a controller changes is the cost
    of the distance it covers: ``(fuel flow + CI / 60) / TAS``, here per second
    at the assigned speed. Air distance, not ground: the wind is not the
    controller's. Fuel flow is the engine's at the step's end (``power``).
    """
    rho = compute_air_density_from_altitude(z)
    mach = tas / compute_speed_of_sound_from_altitude(z)
    thrust = compute_thrust_output(power, params.thrust_output_at_sea_level, mach, rho)
    flow = params.specific_fuel_consumption * 1e-3 * thrust
    per_second = (flow + params.cost_index / 60.0) * target_tas / xp.maximum(tas, 1.0)
    return params.economy_weight * per_second / params.economy_scale


def protected_pitch_demand(
    pitch_demand, gamma_air, air_speed, z, mass, params, load_reference=1.0
):
    """``pitch_demand`` clipped to ``pitch_demand_bounds``: what the flight
    controls let through."""
    lo, hi = pitch_demand_bounds(gamma_air, air_speed, z, mass, params, load_reference)
    return jnp.clip(pitch_demand, lo, hi)


def calibrated_airspeed(z, tas, xp=jnp):
    """The CAS of true airspeed ``tas`` at ``z``, by the standard atmosphere and
    the compressible pitot relation (the inverse of ``scheduled_airspeed``)."""
    temperature, pressure = standard_atmosphere(z, xp)
    mach = tas / xp.sqrt(1.4 * 287.0 * temperature)
    impact = pressure * ((1.0 + 0.2 * mach**2) ** 3.5 - 1.0)
    return 340.29 * xp.sqrt(5.0 * ((impact / 101325.0 + 1.0) ** (2.0 / 7.0) - 1.0))


def speed_band_excess(z, tas, params, xp=jnp, cas=None):
    """How far the airspeed sits past the assigned speed's band, in band widths.

    Pilots complying with a speed adjustment "are expected to maintain a speed
    within plus or minus 10 knots or 0.02 Mach number of the specified speed"
    [AIM 4-4-12]. The schedule assigns CAS below its crossover and Mach above
    it, so the band is read in the unit the schedule assigns at ``z``.
    ``cas`` is the assigned CAS, ``params.target_cas`` by default.
    """
    temperature, _ = standard_atmosphere(z, xp)
    sound = xp.sqrt(1.4 * 287.0 * temperature)
    cas = params.target_cas if cas is None else cas
    target = scheduled_airspeed(z, cas, params.target_mach, xp)
    on_mach = target / sound >= params.target_mach - 1e-6
    cas_excess = (
        xp.abs(calibrated_airspeed(z, tas, xp) - cas) - params.speed_band
    ) / params.speed_band
    mach_excess = (
        xp.abs(tas / sound - params.target_mach) - params.mach_band
    ) / params.mach_band
    return xp.maximum(xp.where(on_mach, mach_excess, cas_excess), 0.0)


def overspeed(air_speed, z, params):
    """Whether the aircraft is past VMO or MMO: the TCDS's "VMO/MMO warning".
    A warning, reported in ``info``, never a trip [TCDS item 10]."""
    rho = compute_air_density_from_altitude(z)
    eas = air_speed * jnp.sqrt(rho / params.air_density_at_sea_level)
    return (eas > params.vmo_eas) | (
        air_speed / compute_speed_of_sound_from_altitude(z) > params.mmo
    )


def outside_envelope(load_factor, air_speed, z, params):
    """Whether a flight condition is past the limit load factors, VMO or MMO."""
    rho = compute_air_density_from_altitude(z)
    eas = air_speed * jnp.sqrt(rho / params.air_density_at_sea_level)
    return (
        (load_factor > params.load_factor_max)
        | (load_factor < params.load_factor_min)
        | (eas > params.vmo_eas)
        | (air_speed / compute_speed_of_sound_from_altitude(z) > params.mmo)
    )


def compute_gamma(x_dot: float, z_dot: float) -> float:
    """Flight path angle from velocity vector."""
    return jnp.arctan2(z_dot, x_dot)  # handles negative x_dot safely


def compute_alpha(theta: float, x_dot: float, z_dot: float) -> float:
    """Angle of attack = pitch - flight path angle."""
    gamma = compute_gamma(x_dot, z_dot)
    alpha = theta - gamma
    # wrap into [-π, π] to avoid angle spirals
    return jnp.arctan2(jnp.sin(alpha), jnp.cos(alpha)), gamma


def compute_speed_of_sound_from_altitude(z):
    temperature, _ = standard_atmosphere(z)
    return jnp.sqrt(1.4 * 287.0 * temperature)


def compute_Mach_from_velocity_and_speed_of_sound(velocity, speed_of_sound):
    return velocity / speed_of_sound


def compute_velocity_from_horizontal_and_vertical_speed(x_dot, z_dot):
    return compute_norm_from_coordinates(jnp.array((x_dot, z_dot)))


def pitch_inertia(mass, params):
    """The pitch moment of inertia, ``m k_y^2``: the radius of gyration is the
    airframe's, so the inertia follows the mass as it does the weight."""
    return mass * params.pitch_gyration_radius**2


def tail_section(params):
    """The horizontal tail as a section for ``aero_coefficients``: symmetric, on
    its own lift slope, span and sweep, with its own stall."""
    return params.replace(
        cl0=0.0,
        cl_alpha=params.tail_cl_alpha,
        aspect_ratio=params.tail_aspect_ratio,
        sweep_half_chord=params.tail_sweep_half_chord,
        k=params.tail_k,
        CL_max=params.tail_CL_max,
        CL_min=-params.tail_CL_max,
        aoa_stall=params.tail_aoa_stall,
    )


def downwash(cl_wing, params):
    """The wing's downwash at the tail, rad: lifting-line theory's far-field
    value, ``2 CL / (pi A)``, twice the wing's own induced angle. It takes
    ``2 a / (pi A)`` of every change of incidence away from the tail, 0.33 at
    low speed and 0.43 at M 0.78 on this wing. ``downwash_ratio`` scales it
    where the far field overstates the downwash at the tail (1 on the A320)."""
    return params.downwash_ratio * 2.0 * cl_wing / (jnp.pi * params.aspect_ratio)


def longitudinal_airframe(alpha, speed, mach, rho, pitch_rate, stick, params):
    """The wing-body and the horizontal tail: ``(lift, drag, pitching moment,
    (tail drag coefficient, tail lift coefficient))``, nose-up positive.

    The wing-body's force acts at its aerodynamic centre, ``moment_arm_wings``
    ahead of the centre of gravity. The tail sits ``moment_arm_stabilizer``
    behind it and meets the air at the body's incidence less the wing's
    downwash, plus its fixed setting (the trimmable stabiliser at its cruise
    trim), less the elevator's deflection times its flap effectiveness, plus
    the incidence a pitch rate sweeps it through. Its dynamic pressure is
    ``tail_efficiency`` of the free stream's. Lift and drag are both defined
    against the local wind, so each surface's force normal to the body is
    ``L cos a + D sin a``: past the stall drag exceeds lift, and the omitted
    term would be the larger one.
    """
    cz_w, cx_w = aero_coefficients(jnp.rad2deg(alpha), mach, params)
    cz_w = jnp.minimum(cz_w, buffet_line(mach, params))
    lift_w = compute_drag(S=params.wings_surface, C=cz_w, V=speed, rho=rho)
    drag_w = compute_drag(S=params.wings_surface, C=cx_w, V=speed, rho=rho)
    normal_w = lift_w * jnp.cos(alpha) + drag_w * jnp.sin(alpha)
    moment = normal_w * params.moment_arm_wings

    # A nose-up rate q sweeps the tail down at q l: arctan rather than q l / V,
    # bounded as V falls, where the moment vanishes with the dynamic pressure.
    swept = jnp.arctan2(
        pitch_rate * params.moment_arm_stabilizer, jnp.maximum(speed, 1e-3)
    )
    tail = (
        alpha
        - downwash(cz_w, params)
        + params.stabilizer_setting
        - params.elevator_effectiveness * stick
        + swept
    )
    cz_t, cx_t = aero_coefficients(jnp.rad2deg(tail), mach, tail_section(params))
    area = params.tail_efficiency * params.stabilizer_surface
    lift_t = compute_drag(S=area, C=cz_t, V=speed, rho=rho)
    drag_t = compute_drag(S=area, C=cx_t, V=speed, rho=rho)
    normal_t = lift_t * jnp.cos(tail) + drag_t * jnp.sin(tail)
    moment = moment - normal_t * params.moment_arm_stabilizer
    return lift_w + lift_t, drag_w + drag_t, moment, (cx_t, cz_t)


@partial(
    jax.jit, static_argnames=["clip", "min_clip_boundaries", "max_clip_boundaries"]
)
def compute_acceleration(
    velocities: jnp.ndarray,
    positions: jnp.ndarray,
    action: tuple,
    params: EnvParams,
    clip: bool = False,
    min_clip_boundaries: Optional[tuple] = None,
    max_clip_boundaries: Optional[tuple] = None,
) -> tuple[float]:
    """
    Compute linear and angular accelerations for the aircraft.
    Returns: (a_x, a_z, alpha_y, metrics)

    Aerodynamics act on the **air-relative** velocity ``V_ground - wind`` (wind
    read from ``params.wind_x``/``params.wind_z`` — a physics-engine property,
    so any plane-based env that uses these params gets it): the angle of attack,
    flight-path angle, airspeed, Mach and dynamic pressure all derive from it,
    so a headwind/tailwind/crosswind changes the forces correctly.  The
    resulting accelerations still act on the ground velocity (Newton's law is in
    the inertial frame), and position integrates the ground velocity — so with
    ``wind = 0`` (the default) the behaviour is unchanged.
    """
    xp = jnp
    thrust, stick = action
    x_dot, z_dot, theta_dot = velocities

    _, z, theta = positions
    # Air-relative velocity drives all aerodynamics.
    air_x = x_dot - params.wind_x
    air_z = z_dot - params.wind_z
    alpha, gamma = compute_alpha(theta, air_x, air_z)
    # The all-up mass of this step: the caller threads what the tanks hold
    # through ``initial_mass`` (``compute_next_state``).
    m = params.initial_mass
    rho = compute_air_density_from_altitude(z)
    M = compute_Mach_from_velocity_and_speed_of_sound(
        velocity=compute_velocity_from_horizontal_and_vertical_speed(air_x, air_z),
        speed_of_sound=compute_speed_of_sound_from_altitude(z),
    )
    # --- Weight & airspeed ---
    P = compute_weight(m, params.gravity)
    V = compute_norm_from_coordinates(xp.array([air_x, air_z]))

    lift_total, drag_total, M_y, (C_x_e, C_z_e) = longitudinal_airframe(
        alpha, V, M, rho, theta_dot, stick, params
    )

    F_x, F_z = newton_second_law(
        thrust=thrust, lift=lift_total, drag=drag_total, P=P, gamma=gamma, theta=theta
    )

    metrics = (drag_total, lift_total, C_x_e, C_z_e, F_x, F_z)
    accelerations = xp.array([F_x / m, F_z / m, M_y / pitch_inertia(m, params)])

    if clip:
        assert (
            min_clip_boundaries is not None
        ), "Clipped without providing min_clip_boundaries"
        assert (
            max_clip_boundaries is not None
        ), "Clipped without providing max_clip_boundaries"
        accelerations = jnp.clip(
            accelerations,
            jnp.array(min_clip_boundaries),
            jnp.array(max_clip_boundaries),
        )
    return accelerations, metrics


def trim_stick(speed, z, theta, params):
    """The elevator that balances the pitching moment at this speed, height and
    level attitude, clipped to its travel, so a reset aircraft engages in trim.
    The moment is close to linear in the deflection, so one secant does."""

    def moment(stick):
        velocities = jnp.array([speed, 0.0, 0.0])
        positions = jnp.array([0.0, z, theta])
        return compute_acceleration(velocities, positions, (0.0, stick), params)[0][2]

    probe = -0.1
    at_zero, at_probe = moment(0.0), moment(probe)
    stick = probe * at_zero / (at_zero - at_probe)
    return jnp.clip(stick, -STICK_TRAVEL, STICK_TRAVEL)


def trim_level(air_speed, z, mass, params, iterations: int = 12):
    """``(pitch, stick, power)`` that hold level flight at ``air_speed`` and
    ``z``: no net force and no pitching moment, solved by Newton's method on the
    model's own accelerations (12 iterations, under rule 7's 100). Wings level,
    in still mean air, at the given all-up mass."""
    level = params.replace(initial_mass=mass, wind_x=0.0, wind_z=0.0)
    rho = compute_air_density_from_altitude(z)
    mach = air_speed / compute_speed_of_sound_from_altitude(z)

    def residual(u):
        theta, stick, power = u
        thrust = compute_thrust_output(
            power=power,
            thrust_output_at_sea_level=params.thrust_output_at_sea_level,
            rho=rho,
            M=mach,
        )
        acc, _ = compute_acceleration(
            jnp.array([air_speed, 0.0, 0.0]),
            jnp.array([0.0, z, theta]),
            (thrust, stick),
            level,
        )
        # The moment in m/s^2 at the tail, so the three rows share a scale.
        return acc * jnp.array([1.0, 1.0, params.moment_arm_stabilizer])

    def newton(u, _):
        step = jnp.linalg.solve(jax.jacfwd(residual)(u), residual(u))
        return u - step, None

    guess = jnp.array([0.03, -0.05, 0.4])
    u, _ = jax.lax.scan(newton, guess, None, length=iterations)
    return u[0], u[1], u[2]


def clamp_altitude(z, z_dot):
    """Clamp altitude to ground and zero vertical velocity if descending."""
    z_clamped = jnp.maximum(z, 0.0)
    z_dot_clamped = jnp.where((z <= 0.0) & (z_dot < 0.0), 0.0, z_dot)
    return z_clamped, z_dot_clamped
