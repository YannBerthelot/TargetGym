from typing import Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit

from plane.env import EnvParams, EnvState
from plane.utils import compute_norm_from_coordinates


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


def compute_initial_z_drag_coefficient(
    alpha: float,
    C_z_max: float,
    min_alpha: float = -5,
    threshold_alpha: float = 15,
    stall_alpha: float = 20,
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
    return jax.lax.cond(
        jnp.logical_or(
            jnp.greater_equal(jnp.abs(alpha), stall_alpha),
            jnp.greater(min_alpha, alpha),
        ),
        lambda: 0.0,
        lambda: jax.lax.cond(
            jnp.greater_equal(threshold_alpha, jnp.abs(alpha)),
            lambda: jnp.abs((alpha + 5) / threshold_alpha) * C_z_max,
            lambda: 1 - jnp.abs((alpha - threshold_alpha) / threshold_alpha) * C_z_max,
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
    return jax.lax.cond(
        jnp.greater_equal(M_critic, M),
        lambda: C_x / (jnp.sqrt(1 - jnp.square(M))),
        lambda: 7 * C_x * (M - M_critic) + C_x / (jnp.sqrt(1 - jnp.square(M))),
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
    return jax.lax.cond(
        jnp.greater_equal(M_critic, M),
        lambda: C_z,
        lambda: jax.lax.cond(
            jnp.greater(M, M_d),
            lambda: C_z + 0.1 * (M_d - M_critic) - 0.8 * (M - M_d),
            lambda: C_z + 0.1 * (M - M_critic),
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
    min_alpha: float = -5,
    threshold_alpha: float = 15,
    stall_alpha: float = 20,
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
    assert 0.0 <= power <= 1.0


def compute_next_power(requested_power, current_power):
    target_power = requested_power - current_power
    current_power += jnp.tanh(1) * target_power
    jax.debug.callback(check_power, current_power)
    return current_power


def compute_thrust_output(
    power: float,
    altitude_factor: float,
    thrust_output_at_sea_level: float,
) -> float:
    # TODO : think about composition
    max_output = thrust_output_at_sea_level * altitude_factor
    return power * max_output


def compute_air_density_from_altitude(
    initial_rho: float, altitude_factor: float
) -> float:
    """Compute the air density given the air density value (in kg.m-3) at sea level and a multiplicative factor (no unit) depending on altitude."""
    return initial_rho * altitude_factor


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

    S_x = S_front * jnp.sin(alpha) + S_wings * jnp.cos(alpha)
    S_z = S_front * jnp.cos(alpha) + S_wings * jnp.sin(alpha)
    return S_x, S_z


def compute_acceleration(
    thrust: float, state: EnvState, params: EnvParams
) -> tuple[float]:
    P = compute_weight(state.m, params.gravity)
    V = compute_norm_from_coordinates([state.x_dot, state.z_dot])
    rho = compute_air_density_from_altitude(
        initial_rho=params.air_density_at_sea_level,
        altitude_factor=state.atltitude_factor,
    )
    S_x, S_z = compute_exposed_surfaces(
        S_front=params.frontal_surface, S_wings=params.wings_surface, alpha=state.alpha
    )
    C_x = compute_x_drag_coefficient(
        alpha=state.alpha, M=state.M, C_x_min=params.C_x0, M_critic=params.M_crit
    )
    C_z = compute_z_drag_coefficient(
        alpha=state.alpha, C_z_max=params.C_z0, M=state.M, M_critic=params.M_crit
    )
    drag = compute_drag(S=S_x, C=C_x, V=V, rho=rho)  # TODO : check if it's V or V_x
    lift = compute_drag(S=S_z, C=C_z, V=V, rho=rho)

    F_x, F_z = newton_second_law(
        thrust=thrust, lift=lift, drag=drag, P=P, gamma=state.gamma, theta=state.theta
    )

    return F_x / state.m, F_z / state.m


def compute_speed_and_pos_from_acceleration(V_x, V_z, x, z, a_x, a_z, delta_t):
    V_x += a_x * delta_t
    V_z += a_z * delta_t
    x += V_x * delta_t
    z += V_z * delta_t
    return V_x, V_z, x, z


def check_mass_does_not_increase(old_mass, new_mass):
    assert old_mass >= new_mass


def compute_next_state(
    power_requested: float, state: EnvState, params: EnvParams
) -> EnvState:
    # power
    power = compute_next_power(power_requested, state.power)
    thrust = compute_thrust_output(
        power=power,
        altitude_factor=state.atltitude_factor,
        thrust_output_at_sea_level=params.thrust_output_at_sea_level,
    )
    # acceleration, speed and position
    a_x, a_z = compute_acceleration(thrust, state, params)
    (
        x_dot,
        z_dot,
        x,
        z,
    ) = compute_speed_and_pos_from_acceleration(
        state.x_dot, state.z_dot, state.x, state.z, a_x, a_z, params.delta_t
    )

    # time
    t = state.t + 1

    # angles
    gamma = jnp.arcsin(z_dot / compute_norm_from_coordinates([x_dot, z_dot + 1e-6]))
    alpha = state.theta - gamma

    # mass
    m = params.initial_mass + state.fuel
    jax.debug.callback(check_mass_does_not_increase, state.m, m)

    new_state = EnvState(
        x=x,
        x_dot=x_dot,
        z=z,
        z_dot=z_dot,
        theta=state.theta,  # no change atm
        alpha=alpha,
        gamma=gamma,
        m=m,  # no change atm
        power=power,
        fuel=state.fuel,  # no change atm
        rho=compute_air_density_from_altitude(
            params.air_density_at_sea_level, altitude_factor=state.atltitude_factor
        ),
        t=t,
    )
    return new_state


if __name__ == "__main__":
    # experiment with power
    power = [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3]
    current_power = 0
    vals = []
    max_output = 1000
    for i in range(len(power)):
        current_power = compute_next_power(power[i], current_power)
        print(current_power)
        vals.append(current_power)

    plt.plot(vals)
    plt.show()
