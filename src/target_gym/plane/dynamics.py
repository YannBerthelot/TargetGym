# from jax.tree_util import Partial as partial
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from gymnax.environments import EnvParams

from target_gym.utils import compute_norm_from_coordinates


def advance_gust(gust, theta: float, sigma: float, dt: float, key):
    """Advance an Ornstein-Uhlenbeck turbulence gust one step.

    ``gust`` is a wind-deviation vector (m/s) that mean-reverts to zero:
    ``g' = g - theta*dt*g + sigma*sqrt(dt)*N``.  With ``key is None`` (e.g. a
    caller that does not model turbulence) or ``sigma == 0`` the gust is
    unchanged, so the total wind stays the steady ``params.wind``.  Shared by
    the 2D and 3D engines so turbulence is a physics-engine property.
    """
    if key is None:
        return gust
    noise = jax.random.normal(key, jnp.shape(gust))
    return gust - theta * dt * gust + sigma * jnp.sqrt(dt) * noise


def step_key(key, time):
    """Per-step PRNG key derived from the episode key *and* the step index.

    ``step_env`` receives whatever key the caller supplies, and every rollout
    helper in this repository -- ``run_episode_headless``, ``save_video``, the
    ``lax.scan`` bodies in the runners -- supplies the *same* key at every
    step. Drawing noise straight from it therefore redraws one identical
    innovation forever, collapsing a zero-mean disturbance into a deterministic
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


def compute_next_power(requested_power, current_power, delta_t):
    requested_power = jnp.clip(requested_power, 0.0 + EPS, 1.0)
    power_diff = requested_power - current_power
    current_power += (
        0.05 * delta_t * power_diff
    )  # TODO : parametrize how fast we reach the desired value
    # jax.debug.callback(check_power, current_power)
    return current_power


def compute_next_stick(requested_stick, current_stick, delta_t):
    stick_diff = requested_stick - current_stick
    current_stick += (
        0.9 * delta_t * stick_diff
    )  # TODO : parametrize how fast we reach the desired value
    return current_stick


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


def compute_air_density_from_altitude(altitude: float) -> float:
    """Compute the air density given the air density value (in kg.m-3) at sea level and a multiplicative factor (no unit) depending on altitude."""
    # ISA up to 11 km, altitude is assumed to be in meters

    T0 = 288.15  # K
    P0 = 101325.0  # Pa
    L = 0.0065  # K/m
    g = 9.80665  # m/s^2
    R = 287.05  # J/(kg·K)

    # Clip to keep T > 0 so (T/T0)**5.26 stays finite. Termination bounds are
    # well within [-50_000, 40_000]; clipping only matters for divergent
    # gradient-tuning rollouts past the terminal state, where it prevents NaN
    # from poisoning the backward pass via jnp.where.
    altitude = jnp.clip(altitude, -50_000.0, 40_000.0)
    T = T0 - L * altitude
    P = P0 * (T / T0) ** (g / (R * L))
    rho = P / (R * T)
    return rho


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

    # --- Lift coefficient ---
    CL_linear = params.cl0 + params.cl_alpha * aoa_deg
    stall_centre = params.aoa_stall + params.aoa_stall_width
    CL = CL_linear / (1 + jnp.exp((aoa_deg - stall_centre) * 1.5))
    # Positive and negative stall limits. A cambered transport wing stalls
    # asymmetrically: CL_max ~ +1.5 near +15 deg, CL_min ~ -1.0 near -10 deg.
    # Only the positive limit existed before; with the corrected (steeper)
    # lift slope the negative branch runs away without this clamp.
    CL = jnp.clip(CL, params.CL_min, params.CL_max)

    # --- Drag coefficient ---
    CD = params.cd0 + params.k * CL**2

    # --- Mach corrections ---
    beta = jnp.sqrt(jnp.maximum(1e-6, 1 - mach**2))

    CL = CL / beta

    # Shock stall (PHYSICS.md D3). Prandtl-Glauert raises the lift *slope* with
    # Mach, which is right, but on its own it also raises the attainable PEAK
    # lift -- the opposite of what happens. Past the critical Mach number,
    # shock-induced separation makes CL_max collapse; that is lift divergence,
    # and it is why transport aircraft have an overspeed limit at all.
    #
    # The cap is therefore the Prandtl-Glauert-scaled stall limit up to
    # ``M_crit``, and falls beyond it. The two branches agree at ``M_crit``, so
    # this is a no-op everywhere the model was validated (cruise sits near
    # M 0.7) and only bites in the regime the model previously got backwards.
    beta_crit = jnp.sqrt(jnp.maximum(1e-6, 1 - params.M_crit**2))
    shock = jnp.clip(
        1.0 - params.k_shock_stall * jnp.maximum(mach - params.M_crit, 0.0),
        params.shock_stall_floor,
        1.0,
    )
    cl_cap = jnp.where(
        mach <= params.M_crit,
        params.CL_max / beta,
        params.CL_max / beta_crit * shock,
    )
    cl_floor = jnp.where(
        mach <= params.M_crit,
        params.CL_min / beta,
        params.CL_min / beta_crit * shock,
    )
    CL = jnp.clip(CL, cl_floor, cl_cap)

    drag_rise = jnp.where(
        mach > params.M_crit, params.k_drag * (mach - params.M_crit) ** 2, 0.0
    )
    CD = CD + drag_rise

    CL = jnp.clip(CL, -2.0, 2.0)  # typical A320: max lift ~1.5-1.7
    CD = jnp.clip(CD, 0.0, 1.0)  # drag can’t be negative or huge

    return CL, CD


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
    h = z
    gamma_air = 1.4
    R = 287.0
    T0 = 288.15
    L = 0.0065
    T11 = 216.65
    T = jnp.where(h <= 11000, T0 - L * h, T11)
    return jnp.sqrt(gamma_air * R * T)


def compute_Mach_from_velocity_and_speed_of_sound(velocity, speed_of_sound):
    return velocity / speed_of_sound


def compute_velocity_from_horizontal_and_vertical_speed(x_dot, z_dot):
    return compute_norm_from_coordinates(jnp.array((x_dot, z_dot)))


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
    x_dot, z_dot, _ = velocities

    _, z, theta = positions
    # Air-relative velocity drives all aerodynamics.
    air_x = x_dot - params.wind_x
    air_z = z_dot - params.wind_z
    alpha, gamma = compute_alpha(theta, air_x, air_z)
    # jax.debug.print(
    #     "{x} {y} {z}", x=jnp.rad2deg(alpha), y=jnp.rad2deg(gamma), z=jnp.rad2deg(theta)
    # )
    m = (
        params.initial_mass
    )  # TODO : make it the actual mass when we start considering fuel consumption
    rho = compute_air_density_from_altitude(z)
    M = compute_Mach_from_velocity_and_speed_of_sound(
        velocity=compute_velocity_from_horizontal_and_vertical_speed(air_x, air_z),
        speed_of_sound=compute_speed_of_sound_from_altitude(z),
    )
    # --- Weight & airspeed ---
    P = compute_weight(m, params.gravity)
    V = compute_norm_from_coordinates(xp.array([air_x, air_z]))

    # ====================================================
    # WINGS
    # ====================================================
    C_z_w, C_x_w = aero_coefficients(xp.rad2deg(alpha), M, params=params)
    lift_wings = compute_drag(S=params.wings_surface, C=C_z_w, V=V, rho=rho)
    drag_wings = compute_drag(S=params.wings_surface, C=C_x_w, V=V, rho=rho)
    M_wings = lift_wings * params.moment_arm_wings

    # ====================================================
    # STABILIZER
    # ====================================================
    C_z_s, C_x_s = aero_coefficients(xp.rad2deg(alpha) - 3.0, M, params=params)
    lift_stab = compute_drag(S=params.stabilizer_surface, C=C_z_s, V=V, rho=rho)
    drag_stab = compute_drag(S=params.stabilizer_surface, C=C_x_s, V=V, rho=rho)
    F_stab = lift_stab - drag_stab
    M_stabilizer = -F_stab * params.moment_arm_stabilizer

    # ====================================================
    # ELEVATOR
    # ====================================================
    C_z_e, C_x_e = aero_coefficients(
        xp.rad2deg(alpha) - xp.rad2deg(stick) - 3.0, M, params=params
    )
    lift_elev = compute_drag(S=params.elevator_surface, C=C_z_e, V=V, rho=rho)
    drag_elev = compute_drag(S=params.elevator_surface, C=C_x_e, V=V, rho=rho)
    F_elev = lift_elev * xp.cos(stick) - drag_elev * xp.sin(stick)
    M_elevator = -F_elev * params.moment_arm_stabilizer

    # ====================================================
    # TOTAL MOMENT & FORCES
    # ====================================================
    M_y = M_wings + M_stabilizer + M_elevator
    drag_total = drag_wings + drag_stab + drag_elev
    lift_total = lift_wings + lift_stab + lift_elev

    F_x, F_z = newton_second_law(
        thrust=thrust, lift=lift_total, drag=drag_total, P=P, gamma=gamma, theta=theta
    )

    metrics = (drag_total, lift_total, C_x_e, C_z_e, F_x, F_z)
    accelerations = xp.array([F_x / m, F_z / m, M_y / params.I])

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


def clamp_altitude(z, z_dot):
    """Clamp altitude to ground and zero vertical velocity if descending."""
    z_clamped = jnp.maximum(z, 0.0)
    z_dot_clamped = jnp.where((z <= 0.0) & (z_dot < 0.0), 0.0, z_dot)
    return z_clamped, z_dot_clamped


if __name__ == "__main__":
    # experiment with power
    power = [
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.5,
        0.3,
        0.3,
        0.3,
        0.3,
    ]
    current_power = 0
    vals = []
    max_output = 1000
    for i in range(len(power)):
        current_power = compute_next_power(power[i], current_power, 1.0)
        vals.append(current_power)

    plt.plot(vals)
    plt.show()
