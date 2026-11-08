"""The turbine's plant, against the published figures.

Contract, from docs/proposals/industrial-fixes.md (PR 6) and these sources:

* Jonkman et al. (2009), NREL/TP-500-38060, Table 7-1: the pitch that holds
  5.296610 MW at 12.1 rpm, and its sensitivity. That sensitivity holds the wake
  frozen, so a steady Cp surface reads it shallower.
* Table 7-2: generator torque moves at most 15,000 N m/s, pitch 8 deg/s.
"""

import jax
import numpy as np
import pytest
from scipy.optimize import brentq

from target_gym.energy.wind_turbine.env import aerodynamic_torque, omega_rated
from target_gym.registry import REGISTRY

SPEC = REGISTRY["wind_turbine"]
ENV, P = SPEC.make_env(), SPEC.make_test_params()
W = float(omega_rated(P))
KEY = jax.random.PRNGKey(0)


def _aero(v, theta):
    return aerodynamic_torque(W, v, theta, P)[1]


def _reset(seed=0):
    return ENV.reset_env(jax.random.PRNGKey(seed), P)[1]


def _raw(pitch, torque):
    return np.array([2 * pitch / P.pitch_max - 1, 2 * torque / P.torque_max - 1])


@pytest.mark.parametrize(
    "v, pitch, frozen",  # Table 7-1: m/s, deg, MW/rad
    [(13, 6.60, 51.66), (15, 10.45, 64.44), (20, 17.47, 94.71), (25, 23.47, 125.30)],
)
def test_the_surface_reproduces_the_published_region_3(v, pitch, frozen):
    theta = brentq(lambda t: float(_aero(float(v), t)) - 5.296610e6, 0.0, 30.0)
    assert theta == pytest.approx(pitch, abs=0.7)
    slope = float(jax.grad(_aero, 1)(float(v), float(theta))) * 180 / np.pi / 1e6
    assert 0.55 * frozen <= -slope <= frozen


@pytest.mark.parametrize("seed", range(4))
def test_the_reset_balances_the_rotor(seed):
    """One 0.1 deg of pitch at -60 MW/rad is about 0.1 MW of 4 MW."""
    s = _reset(seed)
    tau = aerodynamic_torque(s.omega, s.v_wind, s.pitch, P)[0]
    assert float(tau) == pytest.approx(float(P.N_gear * s.torque), rel=0.04)


@pytest.mark.parametrize("command, reached", [(47_400.0, 23_750.0), (0.0, 16_250.0)])
def test_the_torque_moves_at_most_15_kN_m_per_second(command, reached):
    """From 20,000 N m, 15,000 N m/s over a 0.25 s step is 3,750 N m."""
    s = _reset().replace(torque=20_000.0)
    new = ENV.step_env(KEY, s, _raw(float(s.pitch), command), P)[1]
    assert float(new.torque) == pytest.approx(reached, rel=1e-4)


def test_the_applied_action_is_where_the_actuators_went():
    s = _reset()
    np.testing.assert_allclose(s.actuator_held, _raw(s.pitch, s.torque), rtol=1e-5)
    new, info = (lambda o: (o[1], o[4]))(ENV.step_env(KEY, s, _raw(30.0, 0.0), P))
    np.testing.assert_allclose(
        info["applied_action"], _raw(new.pitch, new.torque), rtol=1e-5
    )
