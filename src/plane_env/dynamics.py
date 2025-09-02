import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

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


def newton_second_law(
    thrust: float, lift: float, drag: float, P: float, gamma: float, theta: float
) -> tuple[float, float]:
    """
    Compute Netwon's second law on the x and z axis to get the resulting forces.

    Args:
        thrust (float): The thrust (in Newtons) magnitude.
        lift (float): The lift (in Newtons) magnitude.
        drag (float): The drag (in Newtons) magnitude.
        P (float): The weight (in Newtons) magnitude.
        gamma (float): The slope (in degrees) angle.
        theta (float): The pitch (in degrees) angle.

    Returns:
        tuple[float, float]: The resulting forces (in Newtons) on the x-axis and z-axis respectively
    """
    # Z-Axis
    # Project onto Z-axis
    lift_z = jnp.cos(theta) * lift
    drag_z = -jnp.sin(gamma) * drag
    thrust_z = jnp.sin(theta) * thrust
    # Compute the sum
    F_z = lift_z + drag_z + thrust_z - P

    # X-Axis
    # Project on X-axis
    lift_x = -jnp.sin(theta) * lift
    drag_x = -jnp.abs(jnp.cos(gamma) * drag)
    thrust_x = jnp.cos(theta) * thrust
    # Compute the sum
    F_x = lift_x + drag_x + thrust_x

    return F_x, F_z


def check_power(power):
    assert 0.0 <= power <= 1.0, f"Power should be between 0 and 1, got {power}"


def compute_next_power(requested_power, current_power, delta_t):
    power_diff = requested_power - current_power
    current_power += (
        0.05 * delta_t * power_diff
    )  # TODO : parametrize how fast we reach the desired value
    jax.debug.callback(check_power, current_power)
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
    Compute lift (CL) and drag (CD) coefficients for a given angle of attack (in degrees),
    including basic Mach number effects (Prandtl-Glauert + drag rise).
    """

    # --- parameters from JS model ---
    aoa1 = -5  # deg
    aoa2 = 18  # deg
    cl1 = 0.2
    cl2 = 5.16
    cd1 = 0.02
    cd2 = 0.026
    clpOffset = 0.1
    q = 25  # slope for sigmoid blending

    # Mach correction parameters
    M_crit = 0.8  # critical Mach for drag rise
    k_drag = 5.0  # quadratic drag rise factor

    def CL(aoa):
        return cl1 + cl2 * aoa

    def CD(aoa):
        return cd1 + cd2 * (CL(aoa) ** 2)

    def CLP(aoa):
        return jnp.sin(2 * aoa) + clpOffset

    def CDP(aoa):
        return 0.02 + (1 - jnp.cos(2 * aoa)) * 0.02

    def Up(x, pos):
        return 1 / (1 + jnp.exp(-q * (x - pos)))

    def Down(x, pos):
        return 1 / (1 + jnp.exp(q * (x - pos)))

    def Win(x, pos1, pos2):
        return Up(x, pos1) * Down(x, pos2)

    # --- normalize aoa ---
    aoa = jnp.deg2rad(aoa_deg)

    aoa1_rad = jnp.deg2rad(aoa1)
    aoa2_rad = jnp.deg2rad(aoa2)

    # --- blended lift ---
    CLT = (
        Down(aoa, aoa1_rad) * CLP(aoa)
        + Win(aoa, aoa1_rad, aoa2_rad) * CL(aoa)
        + Up(aoa, aoa2_rad) * CLP(aoa)
    )

    # --- blended drag ---
    CDT = (
        Down(aoa, aoa1_rad) * CDP(aoa)
        + Win(aoa, aoa1_rad, aoa2_rad) * CD(aoa)
        + Up(aoa, aoa2_rad) * CDP(aoa)
    )

    # --- Mach corrections ---

    # Prandtl–Glauert correction for CL
    beta = jnp.sqrt(jnp.maximum(1e-6, 1 - mach**2))
    CLT = CLT / beta

    # Drag divergence (wave drag) beyond M_crit
    drag_rise = jnp.where(mach > M_crit, k_drag * (mach - M_crit) ** 2, 0.0)
    CDT = CDT + drag_rise

    return CLT, CDT


def compute_acceleration(
    thrust: float,
    stick: float,
    m: float,
    gravity: float,
    x_dot: float,
    z_dot: float,
    frontal_surface: float,
    wings_surface: float,
    alpha: float,
    M: float,
    M_crit: float,
    C_x0: float,
    C_z0: float,
    gamma: float,
    theta: float,
    rho: float,
    I: float = 9_000_000,
) -> tuple[float]:
    P = compute_weight(m, gravity)
    V = compute_norm_from_coordinates(jnp.array([x_dot, z_dot]))
    S_x, S_z = compute_exposed_surfaces(
        S_front=frontal_surface, S_wings=wings_surface, alpha=alpha
    )
    C_x = compute_x_drag_coefficient(alpha=alpha, M=M, C_x_min=C_x0, M_critic=M_crit)
    C_z = compute_z_drag_coefficient(alpha=alpha, C_z_max=C_z0, M=M, M_critic=M_crit)

    C_z = 0.6
    C_x = C_z / 16
    C_z, C_x = aero_coefficients(jnp.rad2deg(alpha), M)
    # jax.debug.print('C_x: {C_x}, C_z: {C_z}, rho: {rho}, V: {V} m/s, thrust:{thrust} N, M:{M}, AoA {aoa} deg', C_x=C_x, C_z=C_z, rho=rho, V=V, thrust=thrust, M=M, aoa=jnp.rad2deg(alpha))

    wings_surface = 100
    drag_wings = compute_drag(S=wings_surface, C=C_x, V=V, rho=rho)
    lift_wings = compute_drag(S=wings_surface, C=C_z, V=V, rho=rho)

    moment_arm_wings = (
        1.5  # distance from the center of gravity to the wings (in meters)
    )
    M_wings = lift_wings * moment_arm_wings  # moment arm for wings (in meters)

    stabilizer_surface = 27
    C_z, C_x = aero_coefficients(jnp.rad2deg(alpha) - 3, M)
    lift_stabilizer = compute_drag(S=stabilizer_surface, C=C_z, V=V, rho=rho)
    drag_stabilizer = compute_drag(S=stabilizer_surface, C=C_x, V=V, rho=rho)
    F_m = lift_stabilizer - drag_stabilizer

    moment_arm_stabilizer = (
        15  # distance from the center of gravity to the stabilizer (in meters)
    )
    M_stabilizer = -F_m * moment_arm_stabilizer

    elevator_surface = 10
    C_z, C_x = aero_coefficients(
        jnp.rad2deg(alpha) - jnp.rad2deg(stick) - 3, M
    )  # TODO : is it minus stick or + stick?
    lift_elevator = compute_drag(S=elevator_surface, C=C_z, V=V, rho=rho)
    drag_elevator = compute_drag(S=elevator_surface, C=C_x, V=V, rho=rho)
    F_elevator = lift_elevator * jnp.cos(stick) - drag_elevator * jnp.sin(stick)
    M_elevator = moment_arm_stabilizer * F_elevator

    M_y = M_stabilizer + M_wings + M_elevator

    drag = (
        drag_wings + drag_stabilizer + drag_elevator
    )  # total drag including stabilizer
    lift = (
        lift_wings + lift_stabilizer + lift_elevator
    )  # total lift including stabilizer

    F_x, F_z = newton_second_law(
        thrust=thrust, lift=lift, drag=drag, P=P, gamma=gamma, theta=theta
    )
    metrics = (drag, lift, S_x, S_z, C_x, C_z, F_x, F_z)
    # jax.debug.print("{M_y} Nm, M_stabilizer: {M_stabilizer} Nm, M_wings: {M_wings} Nm", M_y=M_y, M_stabilizer=M_stabilizer, M_wings=M_wings)
    return F_x / m, F_z / m, M_y / I, metrics


def compute_speed_and_pos_from_acceleration(
    V_x, V_z, theta_dot, x, z, theta, a_x, a_z, alpha_y, delta_t
):
    # Update speed
    V_x += a_x * delta_t
    V_z += a_z * delta_t
    new_theta_dot = alpha_y * delta_t + theta_dot
    # Update position
    x += V_x * delta_t
    z += V_z * delta_t
    new_theta = new_theta_dot * delta_t + theta

    return V_x, V_z, new_theta_dot, x, z, new_theta


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
