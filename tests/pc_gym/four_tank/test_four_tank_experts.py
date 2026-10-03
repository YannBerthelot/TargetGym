"""The four-tank's oracle: a shrinking-horizon NLP on the environment's own
step.

The NLP is only the plant's optimum if its map is the plant, so the fast
tests hold ``four_tank_step_map`` equal to ``step_env``, under every
integration method the environment accepts. The slow test is the closed loop
on the protocol (``eval.evaluate_controller``, seeds 0-2), against the oracle
audit's measurement (2026-10): 3.1e-8 per step, against 0.0074 for the oracle
it replaced.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import eval as E
from target_gym.experts.mpc import ShrinkingHorizonNLP, plan_params
from target_gym.pc_gym.four_tank.env_jax import FourTank
from target_gym.pc_gym.four_tank.experts import (
    _LEVEL_MARGIN,
    four_tank_step_map,
    make_four_tank_mpc,
)
from target_gym.registry import REGISTRY

SPEC = REGISTRY["four_tank"]
FIELDS = ("h1", "h2", "h3", "h4")
METHODS = ["rk4_1", "rk4_3", "euler_2", "rk2_1", "rk3_2"]


@pytest.mark.parametrize("method", METHODS)
def test_the_map_is_the_envs_step(method):
    """On random levels across the envelope and random pump commands, the
    map is ``step_env`` to float32 rounding (measured within 1.4e-7
    relative), whatever integration method the environment was built with."""
    env = FourTank(integration_method=method)
    p = SPEC.make_test_params()
    F = four_tank_step_map(env, p)
    _, state = env.reset_env(jax.random.PRNGKey(0), p)
    step = jax.jit(env.step_env)
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = rng.uniform(0.06, 1.4, size=4)
        u = rng.uniform(-1.0, 1.0, size=2)
        s = state.replace(**{f: jnp.float32(v) for f, v in zip(FIELDS, x)})
        _, s2, *_ = step(jax.random.PRNGKey(1), s, jnp.asarray(u, jnp.float32), p)
        x32 = np.array([float(getattr(s, f)) for f in FIELDS])
        got = np.array(F(x32, u)).ravel()
        want = np.array([float(getattr(s2, f)) for f in FIELDS])
        np.testing.assert_allclose(got, want, rtol=5e-6)


def test_an_empty_tank_drains_nothing():
    """The env's ``_safe_sqrt`` is 0 at and below an empty tank, and so is
    the map's, with a finite derivative there."""
    import casadi

    p = SPEC.make_test_params()
    F = four_tank_step_map(SPEC.make_env(), p)
    x = np.array([0.0, 0.2, 0.0, 0.1])
    nxt = np.array(F(x, [-1.0, -1.0])).ravel()
    assert nxt[0] == pytest.approx(0.0, abs=1e-12)  # nothing in, nothing out
    xs, us = casadi.MX.sym("x", 4), casadi.MX.sym("u", 2)
    J = casadi.Function("J", [xs, us], [casadi.jacobian(F(xs, us), xs)])
    assert np.all(np.isfinite(np.array(J(x, [0.0, 0.0]))))


def test_the_oracle_plans_on_the_protocols_window():
    """The factory reads the window from the protocol's own burn-in, and
    keeps every planned level 5 mm inside the trip limits."""
    p = plan_params(SPEC, SPEC.make_test_params())
    mpc = make_four_tank_mpc(SPEC.make_env(), p)
    assert isinstance(mpc, ShrinkingHorizonNLP)
    assert mpc.window_start == E.scored_burn_in("four_tank", p) == 250
    assert (mpc.resolve_every, mpc.pre_weight, mpc.horizon) == (25, 1e-3, 500)
    np.testing.assert_allclose(mpc.x_lb, p.h_min + _LEVEL_MARGIN)
    np.testing.assert_allclose(mpc.x_ub, p.h_max - _LEVEL_MARGIN)


def test_a_short_closed_loop_solves_once_per_period():
    env = SPEC.make_env()
    p = SPEC.make_test_params(max_steps_in_episode=30)
    mpc = make_four_tank_mpc(env, p, resolve_every=25)
    _, state = env.reset_env(jax.random.PRNGKey(0), p)
    step = jax.jit(env.step_env)
    for _ in range(30):
        u = np.asarray(mpc.step(None, state))
        assert u.shape == (2,) and np.all(np.abs(u) <= 1.0)
        _, state, *_ = step(jax.random.PRNGKey(0), state, jnp.asarray(u), p)
    report = mpc.solver_report()
    assert report["solver_calls"] == 2 and report["solver_failures"] == 0


@pytest.mark.slow
def test_the_protocol_gain_is_at_the_plants_optimum():
    """Measured 3.08e-8 per step with zero trips, 28 s for three seeds; the
    oracle it replaced gave 0.0074."""
    m = E.evaluate_controller("four_tank", "mpc", seeds=3)
    assert m["failure_rate"] == 0.0
    assert m["gain"] < 1e-6
