"""Validation for the first-order system.

Enforces ``src/target_gym/pc_gym/first_order/PHYSICS.md``. There is no physics
to validate here -- this is a lag, not a plant -- so everything is checked
against the closed-form solution ``x(t) = K u (1 - exp(-t/tau))``. The
environment exists as a conformance fixture, and these tests keep it honest:
correct dynamics, resolvable time constant, reachable setpoints.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.pc_gym.first_order.env import FirstOrderParams
from target_gym.registry import REGISTRY


@pytest.fixture(scope="module")
def params():
    return FirstOrderParams()


def _step_response(u, steps, params):
    """Open-loop response to a constant input, from x = 0."""
    spec = REGISTRY["first_order"]
    env = spec.make_env()
    p = params.replace(max_steps_in_episode=steps + 5)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    state = state.replace(x=jnp.zeros_like(state.x))
    raw = 2.0 * (u - p.u_min) / (p.u_max - p.u_min) - 1.0
    step = jax.jit(env.step_env)
    xs = []
    for _ in range(steps):
        _, state, _, terminated, _ = step(key, state, jnp.array([raw]), p)
        xs.append(float(state.x))
        if bool(terminated):
            break
    return np.array(xs)


def test_time_constant_is_resolved_by_the_timestep(params):
    """At least five steps per time constant, or the lag is not a lag."""
    assert params.tau / params.delta_t >= 5.0


def test_episode_covers_settling(params):
    """The horizon must be several time constants."""
    assert params.max_steps_in_episode * params.delta_t >= 4.0 * params.tau


def test_step_response_reaches_63_percent_at_one_time_constant(params):
    """The definition of a time constant."""
    u = 1.0
    n = int(round(params.tau / params.delta_t))
    xs = _step_response(u, 4 * n, params)
    assert xs[n - 1] == pytest.approx(0.632 * params.K * u, rel=0.05)


def test_step_response_settles_at_the_dc_gain(params):
    u = 1.0
    xs = _step_response(u, int(8 * params.tau / params.delta_t), params)
    assert xs[-1] == pytest.approx(params.K * u, rel=0.01)


def test_step_response_is_monotone_and_does_not_overshoot(params):
    """A first-order lag cannot overshoot; if it does, the integrator is wrong."""
    xs = _step_response(1.0, int(8 * params.tau / params.delta_t), params)
    assert np.all(np.diff(xs) >= -1e-6)
    assert xs.max() <= params.K * 1.0 * 1.001


def test_response_scales_linearly_with_input(params):
    """It is a linear system; two inputs must give proportional outputs."""
    n = int(6 * params.tau / params.delta_t)
    a = _step_response(0.5, n, params)[-1]
    b = _step_response(1.5, n, params)[-1]
    assert b / a == pytest.approx(3.0, rel=0.02)


def test_every_target_is_reachable_with_margin(params):
    """The same reachability check the four-tank environment failed.

    Costs nothing here, and its absence there made every episode unwinnable.
    """
    for target in params.target_x_range:
        u_needed = target / params.K
        assert params.u_min < u_needed < params.u_max
        assert abs(u_needed) < 0.85 * params.u_max, "no input margin at the setpoint"


def test_targets_stay_inside_the_state_bounds(params):
    for target in params.target_x_range:
        assert params.x_min < target < params.x_max
