from math import cos, sin

import numpy as np
import pytest

from plane.dynamics import (
    compute_air_density_from_altitude,
    compute_drag,
    compute_exposed_surfaces,
    compute_initial_x_drag_coefficient,
    compute_initial_z_drag_coefficient,
    compute_mach_impact_on_x_drag_coefficient,
    compute_mach_impact_on_z_drag_coefficient,
    compute_norm_from_coordinates,
    compute_weight,
    newton_second_law,
)
from plane.env import EnvParams, EnvState, compute_next_state


def test_compute_drag():
    S = 2
    C = 10
    V = 5
    rho = 0.01
    expected_drag = 0.5 * rho * S * C * V**2
    assert compute_drag(S, C, V, rho) == pytest.approx(expected_drag)


def test_compute_weight():
    m = 10
    g = 9.81
    expected_weight = m * g
    assert compute_weight(m, g) == pytest.approx(expected_weight)


def test_compute_initial_z_drag_coefficient():
    C_z_max = 0.9
    threshold_alpha = 15
    stall_alpha = 20
    min_alpha = 5
    assert (
        compute_initial_z_drag_coefficient(
            alpha=stall_alpha + 0.1,
            C_z_max=C_z_max,
            threshold_alpha=threshold_alpha,
            stall_alpha=stall_alpha,
            min_alpha=min_alpha,
        )
        == 0
    )
    assert (
        compute_initial_z_drag_coefficient(
            alpha=min_alpha - 0.01,
            C_z_max=C_z_max,
            threshold_alpha=threshold_alpha,
            stall_alpha=stall_alpha,
            min_alpha=min_alpha,
        )
        == 0
    )
    assert (
        compute_initial_z_drag_coefficient(
            alpha=threshold_alpha - min_alpha,
            C_z_max=C_z_max,
            threshold_alpha=threshold_alpha,
            stall_alpha=stall_alpha,
            min_alpha=min_alpha,
        )
        == C_z_max
    )


def test_newton_second_law():
    thrust = 10000
    lift = 20000
    drag = 5000
    gamma = 5
    theta = 5
    P = 15000
    f_x, f_z = newton_second_law(
        thrust=thrust, lift=lift, drag=drag, P=P, gamma=gamma, theta=theta
    )

    lift_z = cos(theta) * lift
    drag_z = -sin(gamma) * drag
    thrust_z = sin(theta) * thrust
    # Compute the sum
    expected_f_z = lift_z + drag_z + thrust_z - P

    assert f_z == expected_f_z

    lift_x = -sin(theta) * lift
    drag_x = -abs(cos(gamma) * drag)
    thrust_x = cos(theta) * thrust
    # Compute the sum
    expected_f_x = lift_x + drag_x + thrust_x

    assert f_x == expected_f_x


def test_compute_norm_from_coordinates():
    vec = [2, 3]
    norm = compute_norm_from_coordinates(vec)
    expected_norm = np.sqrt(np.sum(np.array(vec) ** 2))
    assert norm == expected_norm


def test_compute_exposed_surfaces():
    S_front = 4
    S_wings = 2
    alpha = 5
    expected_S_x = S_front * sin(alpha) + S_wings * cos(alpha)
    expected_S_z = S_front * cos(alpha) + S_wings * sin(alpha)
    S_x, S_z = compute_exposed_surfaces(S_front, S_wings, alpha)
    assert expected_S_x == S_x
    assert expected_S_z == S_z


def test_compute_mach_impact_on_z_drag_coefficient():
    C_z = 5.0
    M_critic = 0.8
    modified_C_z = compute_mach_impact_on_z_drag_coefficient(
        C_z, M=0.7, M_critic=M_critic
    )
    assert modified_C_z == C_z

    modified_C_z = compute_mach_impact_on_z_drag_coefficient(
        C_z, M=0.81, M_critic=M_critic
    )
    assert modified_C_z > C_z

    modified_C_z = compute_mach_impact_on_z_drag_coefficient(
        C_z, M=0.9, M_critic=M_critic
    )
    assert modified_C_z < C_z


def test_compute_mach_impact_on_x_drag_coefficient():
    C_x = 5.0
    M_critic = 0.8
    modified_C_z = compute_mach_impact_on_x_drag_coefficient(
        C_x, M=0.7, M_critic=M_critic
    )
    assert (
        compute_mach_impact_on_x_drag_coefficient(C_x, M=0.8, M_critic=M_critic)
        > modified_C_z
    )


def test_compute_initial_x_drag_coefficient():
    C_x_min = 2
    assert compute_initial_x_drag_coefficient(alpha=0, C_x_min=C_x_min) == C_x_min
    assert compute_initial_x_drag_coefficient(alpha=5, C_x_min=C_x_min) > C_x_min


def test_compute_air_density_from_altitude():
    initial_rho = 2
    altitude_factor = 3
    expected_air_density = initial_rho * altitude_factor
    assert (
        compute_air_density_from_altitude(initial_rho, altitude_factor)
        == expected_air_density
    )


def test_compute_next_state():
    params = EnvParams()
    state = EnvState(
        x=0,
        x_dot=0,
        z=0,
        z_dot=0,
        theta=0,
        alpha=0,
        gamma=0,
        m=params.initial_mass + params.initial_fuel_quantity,
        power=0,
        fuel=params.initial_fuel_quantity,
        rho=params.air_density_at_sea_level,
        t=0,
    )
    new_state = compute_next_state(1, state, params)
    assert new_state.x > state.x
    assert new_state.x_dot > state.x_dot
    assert new_state.z < state.z
    assert new_state.z_dot < state.z_dot
    assert new_state.power > state.power
    assert new_state.t == state.t + 1
    assert new_state.theta == state.theta
    assert new_state.alpha > state.alpha
    assert new_state.gamma < state.gamma
