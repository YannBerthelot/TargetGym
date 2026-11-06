"""The turbine's plant, against the published figures.

Contract, from docs/proposals/industrial-fixes.md (PR 6) and these sources:

* Jonkman et al. (2009), NREL/TP-500-38060, Table 7-1: the pitch that holds
  5.296610 MW at 12.1 rpm, and its sensitivity. That sensitivity holds the wake
  frozen, so a steady Cp surface reads it shallower.
* Table 7-2: generator torque moves at most 15,000 N m/s, pitch 8 deg/s.
* Ours: the dispatch moves every 25 s, each step above 1.10 x rated pays 200
  floor-widths per unit of speed ratio, and fatigue charges pitch travel.
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
#: What one floor-width of tracking error costs: $ per W per step times e_floor.
FLOOR = 100.0 / 1e6 * 0.25 / 3600.0 * 1680.0


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
    """From 20,000 N m, 15,000 N m/s over a 0.25 s step is 3,750 N m. The lag
    alone would close 92 % of the gap."""
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


def test_fatigue_charges_the_steps_pitch_travel():
    """A far command moves 8 deg/s x 0.25 s = 2 deg, 0.05 of ``pitch_max``, and
    the lag behind the command, about 18 deg here, is free."""
    s = _reset()
    new = ENV.step_env(KEY, s, _raw(float(s.pitch) + 20.0, float(s.torque)), P)[1]
    charged = FLOOR * (0.05 - P.c_hold) / P.c_hold
    assert float(ENV.reward_terms(new, P)["running"]) == pytest.approx(
        charged, rel=1e-3
    )


@pytest.mark.parametrize(
    "ratio, floors", [(0.9, 0.0), (1.0, 0.0), (1.1, 0.0), (1.15, 10.0), (1.2, 20.0)]
)
def test_overspeed_is_charged_past_110_percent_of_rated(ratio, floors):
    s = _reset().replace(omega=ratio * W, pitch_travel=0.0)
    charged = float(ENV.reward_terms(s, P)["running"])
    assert charged == pytest.approx(floors * FLOOR, abs=0.02 * FLOOR)


@pytest.mark.parametrize(
    "clock, level",  # each step moves the clock on by one
    [(0, 0), (98, 0), (99, 1), (199, 2), (299, 3), (1000, 3)],
)
def test_the_dispatch_holds_each_level_for_25_s(clock, level):
    s = _reset().replace(block_clock=clock)
    hold = _raw(float(s.pitch), float(s.torque))
    obs, new = ENV.step_env(KEY, s, hold, P.replace(sensor_noise_scale=0.0))[:2]
    assert float(new.target_power) == float(s.target_levels[level])
    assert float(obs[4]) == pytest.approx(float(s.target_levels[level]) / 1e6)


def test_each_episode_draws_its_own_dispatch_inside_its_range():
    levels = np.asarray(_reset(100).target_levels)
    assert np.all((levels >= 3.5e6) & (levels <= 5.0e6)) and len(set(levels)) == 4
