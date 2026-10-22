"""The wind turbine's oracle: torque by a Newton solve, pitch by a slew-capped PI.

Its claims, each checked below. With the turbulence off, the Newton solve on
the torque lands the next step's power on the target to float32 rounding, so
what is left with turbulence on is the one-step wind innovation. The pitch
command moves by at most 0.13 deg per step near the setpoint, the cap opens
linearly to the actuator's rate toward the edges of the 0.97-1.13 band, and
outside it the PI is not capped. Version-1 params keep the gradient planner
their recorded baseline was taken with.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import registry
from target_gym.energy.wind_turbine.env import electrical_power, omega_rated
from target_gym.energy.wind_turbine.experts import (
    WindTurbineNewtonPI,
    make_wind_turbine_gradient_mpc,
)
from target_gym.experts.mpc import GradientMPC, plan_params

SPEC = registry.get("wind_turbine")


def _episode(params, seed, steps=None):
    """Rotor speed over rated, |target - power| and trips over one episode."""
    env = SPEC.make_env()
    oracle = SPEC.make_mpc(env, plan_params(SPEC, params))
    oracle.reset()
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset_env(key, params)
    step = jax.jit(env.step_env)
    w_rated = float(omega_rated(params))
    ratio, error, trips = [], [], 0
    for _ in range(steps or int(params.max_steps_in_episode)):
        action = oracle.step(obs, state)
        obs, state, _, _, info = step(key, state, action, params)
        trips += int(bool(info["tripped"]))
        ratio.append(float(state.omega) / w_rated)
        power = electrical_power(state.omega, state.torque, params)
        error.append(abs(float(state.target_power) - float(power)))
    return np.array(ratio), np.array(error), trips


def test_the_registry_builds_the_feedback_law():
    env = SPEC.make_env()
    oracle = SPEC.make_mpc(env, SPEC.make_test_params())
    assert isinstance(oracle, WindTurbineNewtonPI)
    assert oracle.solver_report() == {}


def test_version_1_keeps_the_gradient_planner():
    """The version-1 baseline was recorded with the gradient planner, which
    stays importable under its own name."""
    env = SPEC.make_env()
    params = SPEC.make_test_params().replace(reward_version=1)
    assert isinstance(SPEC.make_mpc(env, params), GradientMPC)
    assert isinstance(make_wind_turbine_gradient_mpc(env, params), GradientMPC)


def test_without_turbulence_the_torque_lands_the_target():
    """With the wind on its mean path the model is the plant, so the power
    after every step equals the target to float32 rounding (about 0.5 W at
    4-5 MW). A torque law that missed would leave kilowatts."""
    params = SPEC.make_test_params().replace(turbulence_std=0.0)
    for seed in range(3):
        ratio, error, trips = _episode(params, seed, steps=100)
        assert trips == 0
        assert ratio.min() > 0.9  # the low-speed protection never engages
        assert error.max() <= 2.0


@pytest.mark.parametrize(
    "ratio, cap",
    [
        (1.05, 0.13),  # inside 1.01-1.09: the slew cap
        (1.08, 0.13),
        (1.11, 0.13 + (2.0 - 0.13) * 0.5),  # halfway up the outer ramp
        (1.12, 0.13 + (2.0 - 0.13) * 0.75),
        (0.99, 0.13 + (2.0 - 0.13) * 0.5),  # the same on the low side
    ],
)
def test_the_pitch_command_moves_at_most_the_cap(ratio, cap):
    """A speed jump of 5% of rated asks the PI for about 7 deg in one step;
    the command moves by the cap at that speed instead. The actuator's own
    rate is 8 deg/s, 2 deg per 0.25 s step."""
    env = SPEC.make_env()
    params = SPEC.make_test_params()
    oracle = WindTurbineNewtonPI(env, params)
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    w_rated = float(omega_rated(params))
    sign = 1.0 if ratio > 1.0 else -1.0
    before = state.replace(
        omega=jnp.float32((ratio - sign * 0.05) * w_rated), pitch_cmd=jnp.float32(10.0)
    )
    oracle.reset()
    oracle.step(None, before)
    action = oracle.step(None, before.replace(omega=jnp.float32(ratio * w_rated)))
    cmd = (float(action[0]) + 1.0) / 2.0 * float(params.pitch_max)
    assert cmd - 10.0 == pytest.approx(sign * cap, abs=1e-4)


def test_outside_the_band_the_pi_is_not_capped():
    env = SPEC.make_env()
    params = SPEC.make_test_params()
    oracle = WindTurbineNewtonPI(env, params)
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    w_rated = float(omega_rated(params))
    before = state.replace(
        omega=jnp.float32(1.15 * w_rated), pitch_cmd=jnp.float32(10.0)
    )
    oracle.reset()
    oracle.step(None, before)
    action = oracle.step(None, before.replace(omega=jnp.float32(1.20 * w_rated)))
    cmd = (float(action[0]) + 1.0) / 2.0 * float(params.pitch_max)
    assert cmd - 10.0 > 2.0  # more than the actuator can follow in one step


@pytest.mark.slow
def test_the_screened_extremes_stay_inside_the_envelope():
    """The oracle audit (2026-10) ran seeds 3-199 of the 400-step episode: no
    trip, rotor speed 0.836-1.178 of rated, against trips at 0.40 and 1.25.
    These are its three extremes: seed 149 peaks at 1.178 (start-up, step 45),
    seed 56 at 1.175 inside the scored window, and seed 142, whose lull has
    less power in the wind than its target, falls to 0.836."""
    params = SPEC.make_test_params()
    for seed in (149, 56, 142):
        ratio, _, trips = _episode(params, seed)
        assert trips == 0, seed
        assert ratio.max() < 1.18, (seed, ratio.max())
        assert ratio.min() > 0.83, (seed, ratio.min())


@pytest.mark.slow
def test_protocol_seed_0_holds_without_sagging():
    """Over measure_hold's 700 steps a hard 0.90-1.15 band (no ramp) let the
    rotor sag after a gust on protocol seed 0, holding 27.2 kW; the ramp holds
    1454 W there, the best seed that sets ``rho_floor``
    (``hold_measurements.json``)."""
    params = SPEC.make_test_params().replace(max_steps_in_episode=700)
    _, error, trips = _episode(params, 0)
    assert trips == 0
    assert error[300:].mean() == pytest.approx(1453.8, rel=0.02)
