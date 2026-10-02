"""``ShrinkingHorizonNLP`` and ``casadi_step_map``, the shared machinery of
the CSTR and four-tank oracles, on a toy plant whose optimum is known.

The toy is an integrator, ``x' = u / 2`` over one-unit steps, started at 0
with a target of 1: two full-scale steps reach the target exactly, so the
scored window costs nothing once it opens at step 3.
"""

from types import SimpleNamespace

import casadi
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.experts.mpc import ShrinkingHorizonNLP, casadi_step_map
from target_gym.integration import integrate_dynamics

METHODS = ["euler_1", "euler_3", "rk2_1", "rk3_2", "rk4_1", "rk4_4"]


@pytest.mark.parametrize("method", METHODS)
def test_the_step_map_is_integrate_dynamics(method):
    """Same method string, same stages: a nonlinear right-hand side gives
    the same step in CasADi as through ``integrate_dynamics``."""

    def v_jax(p, u):
        return jnp.array([-p[0] * p[1] + u, jnp.sin(p[0]) - 0.3 * p[1]]), None

    def v_ca(p, u):
        return casadi.vertcat(-p[0] * p[1] + u[0], casadi.sin(p[0]) - 0.3 * p[1])

    F = casadi_step_map(v_ca, 2, 1, 0.4, method)
    for x, u in (([0.5, -1.0], 0.2), ([1.3, 0.7], -0.9)):
        want, _ = integrate_dynamics(
            positions=jnp.asarray(x),
            delta_t=0.4,
            compute_velocity=lambda p, u=u: v_jax(p, u),
            method=method,
        )
        np.testing.assert_allclose(np.array(F(x, [u])).ravel(), want, rtol=1e-5)


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError, match="Unknown integration method"):
        casadi_step_map(lambda p, u: p, 1, 1, 1.0, "midpoint_1")


def _toy(steps=6, window_start=3, **kw):
    x, u = casadi.MX.sym("x", 1), casadi.MX.sym("u", 1)
    F = casadi.Function("F", [x, u], [x + 0.5 * u])
    params = SimpleNamespace(
        max_steps_in_episode=steps, e_floor=0.01, e_tol=0.0, tracking_exponent=2.0
    )
    nlp = ShrinkingHorizonNLP(
        None,
        params,
        step_map=F,
        state_fields=("x",),
        target_fields=("target",),
        tracked=(0,),
        window_start=window_start,
        x_scale=(1.0,),
        **kw,
    )
    return nlp, F


def _run(nlp, F, steps=6):
    state = SimpleNamespace(time=0, x=0.0, target=1.0)
    xs = []
    for t in range(steps):
        u = nlp.step(None, state)
        assert -1.0 <= u <= 1.0
        state = SimpleNamespace(time=t + 1, x=float(F(state.x, u)), target=state.target)
        xs.append(state.x)
    return np.array(xs)


def test_it_reaches_the_optimum_and_counts_its_solves():
    nlp, F = _toy()
    xs = _run(nlp, F)
    np.testing.assert_allclose(xs[2:], 1.0, atol=1e-6)  # the scored window
    report = nlp.solver_report()
    assert report["solver_calls"] == 6 and report["solver_failures"] == 0
    nlp.reset()
    assert nlp._plan is None


def test_it_resolves_every_period_and_follows_the_plan_between():
    nlp, F = _toy(resolve_every=4)
    xs = _run(nlp, F)
    np.testing.assert_allclose(xs[2:], 1.0, atol=1e-6)
    assert nlp.solver_report()["solver_calls"] == 2  # steps 0 and 4


def test_a_new_episode_without_a_reset_starts_cold():
    """When time goes back to 0 the old plan is dropped before the first
    solve, so that solve starts from zeros. Reaching the target alone would
    not show it: in this toy the old plan ends at 0, so a warm guess would be
    zeros too."""
    nlp, F = _toy()
    _run(nlp, F)
    cold = []
    guess = nlp._guess

    def spy(t, n):
        cold.append(nlp._plan is None)
        return guess(t, n)

    nlp._guess = spy
    xs = _run(nlp, F)  # time goes back to 0
    np.testing.assert_allclose(xs[2:], 1.0, atol=1e-6)
    assert cold[0], "the first solve of the new episode saw the old plan"


def test_a_failed_solve_keeps_the_previous_plan():
    """A solve that fails for a reason other than the cap (here an
    infeasible state bound) is counted and leaves the last plan in place."""
    nlp, F = _toy()
    state = SimpleNamespace(time=0, x=0.0, target=1.0)
    first = nlp.step(None, state)
    plan = nlp._plan
    nlp.x_lb, nlp.x_ub = np.array([5.0]), np.array([6.0])  # unreachable
    state = SimpleNamespace(time=1, x=float(F(0.0, first)), target=1.0)
    u = nlp.step(None, state)
    report = nlp.solver_report()
    assert report["solver_failures"] == 1 and report["solver_capped"] == 0
    assert report["solver_last_status"]
    assert nlp._plan is plan and u == pytest.approx(plan[0, 1])


def test_a_capped_solve_is_applied():
    """Every solve hits the cap, and each capped plan replaces the last, so
    the plan in use starts at the last step solved (5). Were capped solves
    discarded like other failures, only the first would be used and the
    plan would still start at 0."""
    nlp, F = _toy(max_iter=1)
    _run(nlp, F)
    report = nlp.solver_report()
    assert report["solver_capped"] == report["solver_failures"] > 0
    assert report["solver_last_status"] == "Maximum_Iterations_Exceeded"
    assert nlp._t0 == 5


@pytest.mark.parametrize("field,value", [("e_tol", 0.1), ("tracking_exponent", 1.0)])
def test_it_refuses_a_cost_its_objective_is_not(field, value):
    """The objective is the reward's tracking term only when that term is
    quadratic with no tolerance band."""
    x, u = casadi.MX.sym("x", 1), casadi.MX.sym("u", 1)
    params = SimpleNamespace(
        max_steps_in_episode=4, e_floor=0.01, e_tol=0.0, tracking_exponent=2.0
    )
    setattr(params, field, value)
    with pytest.raises(ValueError, match="quadratic"):
        ShrinkingHorizonNLP(
            None,
            params,
            step_map=casadi.Function("F", [x, u], [x + u]),
            state_fields=("x",),
            target_fields=("target",),
            tracked=(0,),
            window_start=0,
            x_scale=(1.0,),
        )
