# from jax.tree_util import Partial as partial
from functools import partial
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from gymnax.environments import EnvParams

from plane_env.utils import compute_norm_from_coordinates


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


def speed_of_sound(h):
    # convert meters to km for clarity
    km = h / 1000.0

    # Troposphere 0-11 km: T = 288.15 - 6.5*h
    T0 = 288.15 - 6.5 * km
    # Stratosphere 11-20 km: T = 216.65
    T1 = 216.65 + 0.0 * (km - 11)
    # Stratosphere 20-32 km: T = 216.65 + (km-20)*1.0*10
    T2 = 216.65 + (km - 20) * 1.0 * 10  # simplified linear increase
    # Stratosphere 32-47 km: T = 228.65 + (km-32)*2.8
    T3 = 228.65 + (km - 32) * 2.8

    # Smooth weighting using jnp.clip
    w0 = jnp.clip((11 - km) / 11, 0, 1)
    w1 = jnp.clip((km - 11) / 9, 0, 1) * jnp.clip((20 - km) / 9, 0, 1)
    w2 = jnp.clip((km - 20) / 12, 0, 1) * jnp.clip((32 - km) / 12, 0, 1)
    w3 = jnp.clip((km - 32) / 15, 0, 1) * jnp.clip((47 - km) / 15, 0, 1)

    T = w0 * T0 + w1 * T1 + w2 * T2 + w3 * T3

    gamma = 1.4
    R = 287.05

    return jnp.sqrt(gamma * R * T)


def compute_weight(mass: float, g: float) -> float:
    """Compute the weight of the plane given its mass and g"""
    return mass * g


def compute_initial_z_drag_coefficient(
    alpha: float,
    C_z_max: float,
    min_alpha: float = -5.0,
    threshold_alpha: float = 15.0,
    stall_alpha: float = 20.0,
) -> float:
    """
    Compute an *approximated* version of the drag coefficient of the plane on the z-axis.

    Args:
        alpha (float): The angle of attack (in degrees).
        C_z_max (float): The maximum possible value for the drag coefficient (no units) along the z-axis.
        min_alpha (float, optional): The angle of attack (in degrees) under which the wings create no lift (it stalls). Defaults to -5.
        threshold_alpha (float, optional): The angle of attack (in degrees) where lift starts to decrease. Defaults to 15.
        stall_alpha (float, optional): The angle of attack (in degrees) above which where the airplanes stalls (creating no lift). Defaults to 20.

    Returns:
        float: The value of the lift coefficient/drag coefficient along the z-axis
    """
    return jax.lax.select(
        jnp.logical_or(
            jnp.greater_equal(jnp.abs(alpha), stall_alpha),
            jnp.greater(min_alpha, alpha),
        ),
        0.0,
        jax.lax.select(
            jnp.greater_equal(threshold_alpha, jnp.abs(alpha)),
            jnp.abs((alpha + 5.0) / threshold_alpha) * C_z_max,
            1.0 - jnp.abs((alpha - threshold_alpha) / threshold_alpha) * C_z_max,
        ),
    )


def compute_initial_x_drag_coefficient(
    alpha: float, C_x_min: float, x_drag_coef: float = 0.02
) -> float:
    """
    Compute an *approximated* version of the drag coefficient of the plane on the x-axis.

    Args:
        alpha (float): The angle of attack (in degrees).
        C_x_min (float): The minimal value for the drag coefficient (no units).
        x_drag_coef (float, optional): Hyperparameter representing how much drag increases with the angle. Defaults to 0.02.

    Returns:
        float: The value of the drag coefficient along the x-axis
    """
    return C_x_min + (x_drag_coef * alpha) ** 2


def compute_mach_impact_on_x_drag_coefficient(
    C_x: float, M: float, M_critic: float
) -> float:
    """
    Compute the impact of the Mach number on the drag coefficient.

    Args:
        C_x (float): The drag coefficient (no unit) without Mach number impact.
        M (float): The Mach number (no unit).
        M_critic (float): The critic Mach number (no unit).

    Returns:
        float: The drag coefficient (no unit).
    """
    return jax.lax.select(
        jnp.greater_equal(M_critic, M),
        C_x / (jnp.sqrt(1 - jnp.square(M))),
        7 * C_x * (M - M_critic) + C_x / (jnp.sqrt(1 - jnp.square(M))),
    )


def compute_mach_impact_on_z_drag_coefficient(
    C_z: float, M: float, M_critic: float
) -> float:
    """
    Compute the impact of the Mach number on the lift coefficient.

    Args:
        C_x (float): The drag coefficient (no unit) without Mach number impact.
        M (float): The Mach number (no unit).
        M_critic (float): The critic Mach number (no unit).

    Returns:
        float: The drag coefficient (no unit).
    """
    M_d = M_critic + (1 - M_critic) / 4
    return jax.lax.select(
        jnp.greater_equal(M_critic, M),
        C_z,
        jax.lax.select(
            jnp.greater(M, M_d),
            C_z + 0.1 * (M_d - M_critic) - 0.8 * (M - M_d),
            C_z + 0.1 * (M - M_critic),
        ),
    )


def compute_x_drag_coefficient(
    alpha: float, M: float, C_x_min: float, M_critic: float, x_drag_coef: float = 0.02
) -> float:
    """
    Compute the drag coefficient on the x-axis. Includes impact of the angle of attack and the Mach number.

    Args:
        alpha (float): The angle of attack (in degrees).
        M (float): The Mach number (no unit).
        C_x_min (float): Minimal value of the drag coefficient (no unit).
        M_critic (float): The critical Mach number (no unit).
        x_drag_coef (float, optional): The approximated drag coefficient. Defaults to 0.02.

    Returns:
        float: The drag coefficient (no unit) for the x-axis.
    """
    C_x = compute_initial_x_drag_coefficient(alpha, C_x_min, x_drag_coef=x_drag_coef)
    return compute_mach_impact_on_x_drag_coefficient(C_x, M, M_critic)


def compute_z_drag_coefficient(
    alpha: float,
    M: float,
    C_z_max: float,
    M_critic: float,
    min_alpha: float = -5.0,
    threshold_alpha: float = 15.0,
    stall_alpha: float = 20.0,
) -> float:
    """
    Compute the drag coefficient on the z-axis. Includes impact of the angle of attack and the Mach number.

    Args:
        alpha (float): The angle of attack (in degrees).
        M (float): The Mach number (no unit).
        C_x_min (float): Minimal value of the drag coefficient (no unit).
        M_critic (float): The critical Mach number (no unit).
        x_drag_coef (float, optional): The approximated drag coefficient. Defaults to 0.02.

    Returns:
        float: The drag coefficient (no unit) for the x-axis.
    """
    C_z = compute_initial_z_drag_coefficient(
        alpha=alpha,
        C_z_max=C_z_max,
        min_alpha=min_alpha,
        threshold_alpha=threshold_alpha,
        stall_alpha=stall_alpha,
    )
    return compute_mach_impact_on_z_drag_coefficient(C_z, M, M_critic)


def newton_second_law(thrust, lift, drag, P, gamma, theta):
    # v_hat = [cos(gamma), sin(gamma)]
    vhat_x = jnp.cos(gamma)
    vhat_z = jnp.sin(gamma)
    # drag opposite velocity
    F_drag_x = -drag * vhat_x
    F_drag_z = -drag * vhat_z

    # perp_v = [-sin(gamma), cos(gamma)]
    F_lift_x = -lift * jnp.sin(gamma)
    F_lift_z = lift * jnp.cos(gamma)

    # thrust along body axis
    t_hat_x = jnp.cos(theta)
    t_hat_z = jnp.sin(theta)
    F_thrust_x = thrust * t_hat_x
    F_thrust_z = thrust * t_hat_z

    # weight
    F_weight_x = 0.0
    F_weight_z = -P

    F_total_x = F_drag_x + F_lift_x + F_thrust_x + F_weight_x
    F_total_z = F_drag_z + F_lift_z + F_thrust_z + F_weight_z
    return F_total_x, F_total_z


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
    altitude_factor = 0.7 * sigma + 0.3  # tunable

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

    T = T0 - L * altitude
    P = P0 * (T / T0) ** (g / (R * L))
    rho = P / (R * T)
    return rho


def compute_exposed_surfaces(
    S_front: float, S_wings: float, alpha: float
) -> tuple[float, float]:
    """
    Compute the exposed surface (w.r.t. the relative wind) relative to x and z-axis depending on the angle of attack

    Args:
        S_front (float): Front surface (in m^2) of the plane.
        S_wings (float): Wings surface (in m^2) of the plane.
        alpha (float): angle of attack (in degrees).

    Returns:
        tuple[float,float]: The exposed surface on the x and z-axis
    """

    S_z = S_front * jnp.sin(alpha) + S_wings * jnp.cos(alpha)
    S_x = S_front * jnp.cos(alpha) + S_wings * jnp.sin(alpha)
    return S_x, S_z


def aero_coefficients(aoa_deg, mach=0.0):
    """
    Realistic lift (CL) and drag (CD) coefficients for an A320.
    AoA in degrees. Mach effects included.
    """
    # --- A320 parameters ---
    cl_alpha = 0.05  # per deg
    cl0 = 0.2  # zero-lift AoA
    cd0 = 0.02  # zero-lift drag
    k = 0.045  # induced drag factor
    aoa_stall = 15.0  # deg
    CL_max = 1.5

    # --- Lift coefficient ---
    CL_linear = cl0 + cl_alpha * aoa_deg
    CL = CL_linear / (1 + jnp.exp((aoa_deg - aoa_stall) * 1.5))
    CL = jnp.minimum(CL, CL_max)

    # --- Drag coefficient ---
    CD = cd0 + k * CL**2

    # --- Mach corrections ---
    beta = jnp.sqrt(jnp.maximum(1e-6, 1 - mach**2))

    CL = CL / beta

    M_crit = 0.82
    k_drag = 5.0
    drag_rise = jnp.where(mach > M_crit, k_drag * (mach - M_crit) ** 2, 0.0)
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


@partial(jax.jit)
def compute_acceleration(
    velocities: jnp.ndarray, positions: jnp.ndarray, action: tuple, params: EnvParams
) -> tuple[float]:
    """
    Compute linear and angular accelerations for the aircraft.
    Returns: (a_x, a_z, alpha_y, metrics)
    """
    xp = jnp
    thrust, stick = action
    x_dot, z_dot, _ = velocities

    _, z, theta = positions
    alpha, gamma = compute_alpha(theta, x_dot, z_dot)
    m = (
        params.initial_mass
    )  # TODO : make it the actual mass when we start considering fuel consumption
    rho = compute_air_density_from_altitude(z)
    M = compute_Mach_from_velocity_and_speed_of_sound(
        velocity=compute_velocity_from_horizontal_and_vertical_speed(x_dot, z_dot),
        speed_of_sound=compute_speed_of_sound_from_altitude(z),
    )
    # --- Weight & velocity ---
    P = compute_weight(m, params.gravity)
    V = compute_norm_from_coordinates(xp.array([x_dot, z_dot]))

    # ====================================================
    # WINGS
    # ====================================================
    C_z_w, C_x_w = aero_coefficients(xp.rad2deg(alpha), M)
    lift_wings = compute_drag(S=params.wings_surface, C=C_z_w, V=V, rho=rho)
    drag_wings = compute_drag(S=params.wings_surface, C=C_x_w, V=V, rho=rho)
    moment_arm_wings = 1.5
    M_wings = lift_wings * moment_arm_wings

    # ====================================================
    # STABILIZER
    # ====================================================
    stabilizer_surface = 27
    C_z_s, C_x_s = aero_coefficients(xp.rad2deg(alpha) - 3.0, M)
    lift_stab = compute_drag(S=stabilizer_surface, C=C_z_s, V=V, rho=rho)
    drag_stab = compute_drag(S=stabilizer_surface, C=C_x_s, V=V, rho=rho)
    F_stab = lift_stab - drag_stab
    moment_arm_stabilizer = 15.0
    M_stabilizer = -F_stab * moment_arm_stabilizer

    # ====================================================
    # ELEVATOR
    # ====================================================
    elevator_surface = 10
    C_z_e, C_x_e = aero_coefficients(xp.rad2deg(alpha) - xp.rad2deg(stick) - 3.0, M)
    lift_elev = compute_drag(S=elevator_surface, C=C_z_e, V=V, rho=rho)
    drag_elev = compute_drag(S=elevator_surface, C=C_x_e, V=V, rho=rho)
    F_elev = lift_elev * xp.cos(stick) - drag_elev * xp.sin(stick)
    M_elevator = -F_elev * moment_arm_stabilizer

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
        current_power = compute_next_power(power[i], current_power)
        vals.append(current_power)

    plt.plot(vals)
    plt.show()
