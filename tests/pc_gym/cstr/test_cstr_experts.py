"""The CSTR's oracle: a shrinking-horizon NLP on the environment's own step.

The NLP is only the plant's optimum if its map is the plant, so the fast
tests hold ``cstr_step_map`` equal to ``step_env``, under every integration
method the environment accepts. The slow test is the closed loop on the
protocol (``eval.evaluate_controller``, seeds 0-2), against the oracle audit's
measurement (2026-10): 9.4e-9 per step, against 0.317 for the oracle it
replaced.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import eval as E
from target_gym.experts.mpc import ShrinkingHorizonNLP, plan_params
from target_gym.pc_gym.cstr.env_jax import CSTR
from target_gym.pc_gym.cstr.experts import cstr_step_map, make_cstr_mpc
from target_gym.registry import REGISTRY

SPEC = REGISTRY["cstr"]
METHODS = ["rk4_1", "rk4_3", "euler_2", "rk2_1", "rk3_2"]


def _pairs(method, n=50, seed=0):
    """``(map's x_next, env's x_next)`` on random states and actions in the box."""
    env = CSTR(integration_method=method)
    p = SPEC.make_test_params()
    F = cstr_step_map(env, p)
    _, state = env.reset_env(jax.random.PRNGKey(0), p)
    step = jax.jit(env.step_env)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x = rng.uniform([0.75, 310.0], [0.95, 345.0])
        u = rng.uniform(-1.0, 1.0, size=1)
        s = state.replace(C_a=jnp.float32(x[0]), T=jnp.float32(x[1]))
        _, s2, *_ = step(jax.random.PRNGKey(1), s, jnp.asarray(u, jnp.float32), p)
        got = np.array(F(np.array([float(s.C_a), float(s.T)]), u)).ravel()
        out.append((got, np.array([float(s2.C_a), float(s2.T)])))
    return out


@pytest.mark.parametrize("method", METHODS)
def test_the_map_is_the_envs_step(method):
    """On random states and actions, the map is ``step_env`` to float32
    rounding (measured within 9e-7 relative), whatever integration method
    the environment was built with."""
    for got, want in _pairs(method):
        np.testing.assert_allclose(got, want, rtol=5e-6)


def test_the_map_follows_the_envs_integration_method():
    """The comparison has teeth: Euler at the registered step is off by far
    more than the tolerance above, so the map really is the env's RK4."""
    env = CSTR(integration_method="euler_1")
    p = SPEC.make_test_params()
    F = cstr_step_map(CSTR(integration_method="rk4_1"), p)
    _, s = env.reset_env(jax.random.PRNGKey(0), p)
    _, s2, *_ = env.step_env(jax.random.PRNGKey(1), s, jnp.asarray([0.3]), p)
    got = np.array(F(np.array([float(s.C_a), float(s.T)]), [0.3])).ravel()
    want = np.array([float(s2.C_a), float(s2.T)])
    assert np.max(np.abs(got - want) / np.abs(want)) > 1e-4


def test_the_oracle_plans_on_the_protocols_window():
    """The factory reads the window from the protocol's own burn-in."""
    p = plan_params(SPEC, SPEC.make_test_params())
    mpc = make_cstr_mpc(SPEC.make_env(), p)
    assert isinstance(mpc, ShrinkingHorizonNLP)
    assert mpc.window_start == E.scored_burn_in("cstr", p) == 12
    assert (mpc.resolve_every, mpc.pre_weight, mpc.horizon) == (1, 1e-3, 100)


def test_a_short_closed_loop_solves_every_step_and_resets_cold():
    env = SPEC.make_env()
    p = SPEC.make_test_params(max_steps_in_episode=8)
    mpc = make_cstr_mpc(env, p)
    _, state = env.reset_env(jax.random.PRNGKey(0), p)
    step = jax.jit(env.step_env)
    for _ in range(8):
        u = mpc.step(None, state)
        assert -1.0 <= u <= 1.0
        _, state, *_ = step(jax.random.PRNGKey(0), state, jnp.asarray([u]), p)
    report = mpc.solver_report()
    assert report["solver_calls"] == 8 and report["solver_failures"] == 0
    mpc.reset()
    assert mpc._plan is None


@pytest.mark.slow
def test_the_protocol_gain_is_at_the_plants_optimum():
    """Measured 9.42e-9 per step with zero trips, 15 s for three seeds; the
    oracle it replaced gave 0.317."""
    m = E.evaluate_controller("cstr", "mpc", seeds=3)
    assert m["failure_rate"] == 0.0
    assert m["gain"] < 1e-6
