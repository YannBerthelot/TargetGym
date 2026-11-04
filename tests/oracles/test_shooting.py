"""The shared multiple-shooting builder, ``oracles.shooting``.

Contract (its docstring and docs/proposals/oracle-solver.md): ``Shooting``
plans ``n_stages`` moves, each held for ``block`` task steps, that minimise
the sum of the step costs over every visited state plus the epigraph terms
(T2) and the elastic limits (T3) on each visited state's outputs, within the
move bounds, to IPOPT's tolerance; it certifies each solve; a failed solve
returns the plan it started from, the previous plan shifted one stage.
``stage_derivatives`` gives the stage Lagrangian's exact Hessian.

Oracles: a finite-horizon LQR by a NumPy Riccati recursion on the lifted
(held-move) system, a tracking problem whose optimum is worked out by hand,
and ``jax.hessian`` for the forward-over-forward Hessian.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.oracles.shooting import (
    Hinge,
    Limit,
    Shooting,
    StageModel,
    stage_derivatives,
)

try:
    from jax import enable_x64
except ImportError:  # JAX 0.6; 0.10 removed the experimental name
    from jax.experimental import enable_x64

A = np.array([[1.0, 0.1], [0.0, 0.9]])
B = np.array([[0.0], [0.1]])
Q, R = np.diag([1.0, 0.5]), np.array([[0.1]])


def _lqr_moves(x0, S):
    """Finite-horizon LQR on the held-move system, written from the plant:
    over a stage of two steps with u held, s1 = A x + B u and
    s2 = A^2 x + (A + I) B u, costing s1'Q s1 + s2'Q s2 + 2 u'R u. Nothing
    is charged past the last stage (the builder's stage past the horizon is
    uncosted), so P_S = 0."""
    M1, M2 = np.hstack([A, B]), np.hstack([A @ A, (A + np.eye(2)) @ B])
    M = M1.T @ Q @ M1 + M2.T @ Q @ M2 + np.diag([0.0, 0.0, 2.0 * R[0, 0]])
    Ab, Bb = M2[:, :2], M2[:, 2:]
    P, gains = np.zeros((2, 2)), []
    for _ in range(S):
        Qxx, Qux = M[:2, :2] + Ab.T @ P @ Ab, M[2:, :2] + Bb.T @ P @ Ab
        Kk = -np.linalg.solve(M[2:, 2:] + Bb.T @ P @ Bb, Qux)
        P, gains = Qxx + Qux.T @ Kk, [Kk] + gains
    x, us = np.asarray(x0, float), []
    for Kk in gains:
        us.append(Kk @ x)
        x = Ab @ x + Bb @ us[-1]
    return np.array(us)


def _linear(data, x, u, t):
    x1 = jnp.asarray(A) @ x + jnp.asarray(B) @ u
    return x1, x1 @ jnp.asarray(Q) @ x1 + u @ jnp.asarray(R) @ u, jnp.zeros(0)


@pytest.mark.medium
def test_plans_the_finite_horizon_lqr_of_the_held_move_system():
    """Bounds of 100 sit far from moves of order 1, so they are inactive."""
    model = StageModel(step=_linear, block=2, lo=(-100.0,), hi=(100.0,))
    x0 = np.array([1.0, -0.5])
    with enable_x64():
        planner = Shooting(model, n_stages=6, nx=2)
        plan = planner.solve(None, jnp.asarray(x0), jnp.zeros(1))
        moves = np.asarray(planner.moves(plan))
    cert = plan.certificate
    assert not bool(cert.failed) and not bool(cert.capped)
    assert 0.0 <= float(cert.error) <= 1e-8
    # The barrier on bounds 100 away biases a move by about mu / 100, and
    # the scaled KKT error stops at 1e-8: 1e-6 is two decades above both.
    np.testing.assert_allclose(moves, _lqr_moves(x0, 6), atol=1e-6)


def _walk(data, x, u, t):
    """x moves by u per step; the only cost is |x - 2.5| through a hinge."""
    x1 = x + u
    return x1, jnp.zeros((), x.dtype), x1 - 2.5


@pytest.mark.medium
def test_tracks_through_the_epigraph_rows_on_every_visited_state():
    """min sum |x_t - 2.5| over 8 steps, x_0 = 0, x_{t+1} = x_t + u_k, each
    move held 2 steps, |u| <= 1. By hand: stage 0 costs (2.5 - a) +
    (2.5 - 2a) for a move a <= 1; from 2a the next stage's best is
    |d - u| + |d - 2u| at u = d / 2, d = 2.5 - 2a, worth d / 2, and every
    later stage holds 2.5 at no cost. The total, 6.25 - 4a, is least at
    a = 1: moves (1, 0.25, 0, 0), cost 1.5 + 0.5 + 0.25 = 2.25. Only the
    visited states' terms make it 2.25: the stage ends alone cost 0.5."""
    model = StageModel(
        step=_walk, block=2, lo=(-1.0,), hi=(1.0,), hinges=(Hinge(absolute=True),)
    )
    with enable_x64():
        planner = Shooting(model, n_stages=4, nx=1)
        plan = planner.solve(None, jnp.zeros(1), jnp.zeros(1))
        moves = np.asarray(planner.moves(plan))[:, 0]
    assert not bool(plan.certificate.failed)
    assert 0.0 <= float(plan.certificate.error) <= 1e-8
    # An LP solved to a scaled error of 1e-8: each of the 8 terms is off by
    # about the final barrier parameter, far below 1e-6.
    np.testing.assert_allclose(moves, [1.0, 0.25, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(float(plan.cost), 2.25, atol=1e-6)


def _walk_limited(data, x, u, t):
    x1 = x + u
    return x1, jnp.zeros((), x.dtype), jnp.concatenate([x1 - 2.5, x1])


@pytest.mark.medium
def test_a_limit_priced_above_any_gain_holds_on_every_visited_state():
    """The walk above with x <= 2 on every visited state, elastic at 100 per
    unit, more than the 6 a unit of x could save over the 3 later stages'
    6 visited states. By hand: stage 0 at a = 1 visits 1 and 2; from 2 every
    later move must be 0, as 2 + u and 2 + 2u stay <= 2. Cost 1.5 + 0.5 +
    6 x 0.5 = 5, no slack used."""
    model = StageModel(
        step=_walk_limited, block=2, lo=(-1.0,), hi=(1.0,),
        hinges=(Hinge(absolute=True),), limits=(Limit(hi=2.0, price=100.0),),
    )  # fmt: skip
    with enable_x64():
        planner = Shooting(model, n_stages=4, nx=1)
        plan = planner.solve(None, jnp.zeros(1), jnp.zeros(1))
        moves = np.asarray(planner.moves(plan))[:, 0]
    assert not bool(plan.certificate.failed)
    np.testing.assert_allclose(moves, [1.0, 0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(float(plan.cost), 5.0, atol=1e-6)


def _pendulum(v):
    """Five Euler steps of a forced pendulum from (v0, v1) under v2 held,
    and the steps' cost: smooth, through a scan as the integrators are."""

    def one(c, _):
        x, acc = c
        x = x + 0.1 * jnp.stack([x[1], -jnp.sin(x[0]) + v[2]])
        return (x, acc + x[0] ** 2 + jnp.cos(x[1]) * v[2] ** 2), None

    (x, cost), _ = jax.lax.scan(one, (v[:2], 0.0), None, length=5)
    return x, cost


def test_the_forward_over_forward_hessian_is_the_exact_hessian():
    with enable_x64():
        v, lam = jnp.array([0.4, -0.9, 0.7]), jnp.array([0.3, -1.2])

        def lagrangian(v):
            x, cost = _pendulum(v)
            return 0.7 * cost + lam @ x

        H = stage_derivatives(_pendulum, v, lam, 0.7)[4]
        expected = jax.hessian(lagrangian)(v)
    # Two AD orders of one float64 function: they differ by rounding only.
    np.testing.assert_allclose(H, expected, rtol=1e-12, atol=1e-13)


def _brittle(data, x, u, t):
    """x moves by u at cost (x - 3)^2. With ``data`` = (1, moves), any move
    other than ``moves[t]`` takes x to NaN."""
    on, planned = data
    x1 = x + u + jnp.where((on > 0) & (u[0] != planned[t]), jnp.nan, 0.0)
    return x1, jnp.sum((x1 - 3.0) ** 2), jnp.zeros(0)


@pytest.mark.medium
def test_a_failed_solve_keeps_the_previous_plan_shifted():
    """The warm solve starts 0.05 off the plan, so every step it tries moves
    the moves, which the brittle model answers with NaN: its line searches
    and restorations fail away from its start, at a re-simulated point."""
    model = StageModel(step=_brittle, block=1, lo=(-1.0,), hi=(1.0,))
    with enable_x64():
        planner = Shooting(model, n_stages=5, nx=1)
        first = planner.solve((0.0, jnp.zeros(6)), jnp.zeros(1), jnp.zeros(1))
        shifted = planner.shift(first)[0]
        data = (1.0, planner.unpack(shifted)[1][:, 0])
        x0 = planner.states(first)[0] + 0.05
        failed = planner.solve(data, x0, jnp.zeros(1), first)
    assert not bool(first.certificate.failed)
    assert bool(failed.certificate.failed)
    np.testing.assert_array_equal(failed.z, shifted)
    # The move applied next is the one the previous plan had planned next.
    assert float(planner.moves(failed)[0, 0]) == float(planner.moves(first)[1, 0])


def _runaway(data, x, u, t):
    """x+ = 100 x + u, open-loop unstable, tracking ``data`` + 0.1 data t:
    0 when data = 0, the ramp 0.1 (t + 1) when data = 1."""
    x1 = 100.0 * x + u
    return x1, jnp.sum((x1 - 0.1 * data * (t + 1)) ** 2), jnp.zeros(0)


@pytest.mark.medium
def test_a_cold_solve_plans_an_unstable_plant_whose_held_move_overflows():
    """From x0 = 1 the move held at 0 takes x to 1e320, past float64, within
    160 stages, so a cold start cannot rest on that rollout (oracle-solver.md,
    G8). By hand: the first move -100 brings x to 0, where 0 holds it at no
    cost."""
    model = StageModel(step=_runaway, block=1, lo=(-200.0,), hi=(200.0,))
    with enable_x64():
        planner = Shooting(model, n_stages=160, nx=1)
        plan = planner.solve(0.0, jnp.ones(1), jnp.zeros(1))
    assert not bool(plan.certificate.failed)
    np.testing.assert_allclose(planner.moves(plan)[0], [-100.0], atol=1e-6)
    np.testing.assert_allclose(planner.states(plan), 0.0, atol=1e-6)


@pytest.mark.medium
def test_stages_past_the_active_horizon_do_not_run_the_plant_away():
    """Five live stages of 160, as at an episode's end (H_k = N_ep - k): the
    live moves track the ramp 0.1 k, by hand u_k = 0.1 (k + 1) - 100 (0.1 k),
    and the plan stays finite past them, where the plant held at the last
    move would grow a hundredfold a stage (G9)."""
    model = StageModel(step=_runaway, block=1, lo=(-1e3,), hi=(1e3,))
    with enable_x64():
        planner = Shooting(model, n_stages=160, nx=1)
        plan = planner.solve(1.0, jnp.zeros(1), jnp.zeros(1), active=5)
    assert not bool(plan.certificate.failed)
    k = np.arange(5)
    np.testing.assert_allclose(
        planner.moves(plan)[:5, 0], 0.1 * (k + 1) - 10.0 * k, atol=1e-6
    )
    assert np.all(np.isfinite(planner.states(plan)))
