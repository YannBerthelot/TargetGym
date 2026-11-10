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


def _ramp(data, x, u, t):
    """A position moving 0.1 u a step, charged for its distance from 1.5."""
    x1 = x + 0.1 * u
    return x1, jnp.sum((x1 - 1.5) ** 2), jnp.zeros(0)


@pytest.mark.medium
def test_a_moves_rows_do_not_bind_past_the_plan():
    """x_{t+1} = x_t + 0.1 u from 0.6, |u| <= 1, x + 0.1 u <= 1 from each stage.
    The cost falls with x up to 1.5, past the row, so by hand four stages at u = 1
    reach 1.0 with the row active on the last: every planned move is 1. The stage
    past the plan starts at 1.0, where the same row would ask for u <= 0, and its
    move is held to the last planned one: a row left on it would drag that move
    off 1, for a stage nothing is costed on."""
    model = StageModel(
        step=_ramp, block=1, lo=(-1.0,), hi=(1.0,),
        move_rows=lambda x, u: 1.0 - x - 0.1 * u,
    )  # fmt: skip
    with enable_x64():
        planner = Shooting(model, n_stages=4, nx=1)
        plan = planner.solve(None, jnp.full(1, 0.6), jnp.zeros(1))
        moves = np.asarray(planner.moves(plan))[:, 0]
    assert not bool(plan.certificate.failed)
    # Rows 1e-8 relaxed, and a barrier at a scaled error of 1e-8: 1e-5 is far
    # over both and far under the 0.5 the row dragged the move by, unmasked.
    np.testing.assert_allclose(moves, [1.0, 1.0, 1.0, 1.0], atol=1e-5)


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


def _hidden(data, x, u, t):
    """x[0] multiplies by 100 a step, out of any move's reach and free of any
    cost; x[1] moves by u and pays (x[1] - 3)^2."""
    x1 = jnp.stack([100.0 * x[0], x[1] + u[0]])
    return x1, (x1[1] - 3.0) ** 2, jnp.zeros(0)


@pytest.mark.medium
def test_multipliers_that_overflow_do_not_end_a_cold_solve():
    """From x = 0 the first state stays at 0, unpriced, so the solve is plain.
    The cold start's least-squares multipliers solve the same rows with K = I,
    which does price it: their Riccati term grows 1e4 a stage, past float64 in
    80 stages, so they come out NaN with every derivative finite. IPOPT zeroes
    multipliers past its cap, and a NaN has to count as past it. By hand: the
    first move takes x[1] to 3, and 0 holds it."""
    model = StageModel(step=_hidden, block=1, lo=(-100.0,), hi=(100.0,))
    with enable_x64():
        planner = Shooting(model, n_stages=80, nx=2)
        plan = planner.solve(0.0, jnp.zeros(2), jnp.zeros(1))
    assert not bool(plan.certificate.failed)
    np.testing.assert_allclose(planner.moves(plan)[0], [3.0], atol=1e-6)


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


def _actuator(data, x, u, t):
    """A valve that reaches any command within its travel, tracking 1.05."""
    return u, jnp.sum((u - 1.05) ** 2), jnp.zeros(0)


def _reach(x, u):
    """One step's travel, |u - x| <= 0.1, as rows >= 0."""
    return jnp.concatenate([0.1 - (u - x), 0.1 + (u - x)])


@pytest.mark.medium
def test_move_rows_bound_each_move_from_its_stages_state():
    """The valve from 0.5 to 1.05 at most 0.1 a step: every command within
    one step's travel of the valve, stage 0's from the measured 0.5. By hand
    the plan ramps at full travel, 0.6, 0.7, ..., 1.0, then reaches 1.05 and
    holds it; without the rows it would command 1.05 at once. (The target
    is off the ramp's grid so that no row binds with a zero multiplier.)"""
    model = StageModel(step=_actuator, block=1, lo=(-2.0,), hi=(2.0,), move_rows=_reach)
    with enable_x64():
        planner = Shooting(model, n_stages=8, nx=1)
        plan = planner.solve(0.0, jnp.full(1, 0.5), jnp.full(1, 0.5))
    assert not bool(plan.certificate.failed)
    np.testing.assert_allclose(
        planner.moves(plan)[:8, 0],
        [0.6, 0.7, 0.8, 0.9, 1.0, 1.05, 1.05, 1.05],
        atol=1e-6,
    )


@pytest.mark.medium
def test_a_limit_holds_its_back_off_inside_the_bound():
    """The limited walk above with a back-off of 0.25: every visited state at
    most 2 - 0.25 = 1.75. By hand: stage 0 moves 0.875 (visits 0.875, 1.75),
    the rest 0; cost 1.625 + 0.75 + 6 x 0.75 = 6.875."""
    model = StageModel(
        step=_walk_limited, block=2, lo=(-1.0,), hi=(1.0,),
        hinges=(Hinge(absolute=True),),
        limits=(Limit(hi=2.0, price=100.0, backoff=0.25),),
    )  # fmt: skip
    with enable_x64():
        planner = Shooting(model, n_stages=4, nx=1)
        plan = planner.solve(None, jnp.zeros(1), jnp.zeros(1))
    assert not bool(plan.certificate.failed)
    np.testing.assert_allclose(
        np.asarray(planner.moves(plan))[:4, 0], [0.875, 0.0, 0.0, 0.0], atol=1e-6
    )
    np.testing.assert_allclose(float(plan.cost), 6.875, atol=1e-6)


@pytest.mark.medium
def test_a_limit_with_no_inside_is_not_backed_off():
    """lo = hi = 2, an equality kept elastic as a terminal line-out is: a
    bound with no inside has nowhere to back off to, so a back-off of 0.25
    leaves the plan and its cost as they are without one. (Backed off, the
    bounds would cross and force a slack of 0.25 a stage at 100 a unit.)"""

    def plan(backoff):
        model = StageModel(
            step=_walk_limited, block=2, lo=(-1.0,), hi=(1.0,),
            hinges=(Hinge(absolute=True),),
            limits=(Limit(lo=2.0, hi=2.0, price=100.0, backoff=backoff),),
        )  # fmt: skip
        with enable_x64():
            return Shooting(model, n_stages=4, nx=1).solve(
                None, jnp.zeros(1), jnp.zeros(1)
            )

    plain, backed = plan(0.0), plan(0.25)
    assert not bool(plain.certificate.failed) and not bool(backed.certificate.failed)
    np.testing.assert_allclose(float(backed.cost), float(plain.cost), atol=1e-6)


def _split(data, x, u, t):
    """Two moves, one output, as a fat plant: x+ = u_1 + u_2, tracking 1.
    Every split with u_1 + u_2 = 1 tracks exactly, so the optimum is a line."""
    x1 = (u[0] + u[1])[None]
    return x1, jnp.sum((x1 - 1.0) ** 2), jnp.zeros(0)


@pytest.mark.medium
def test_a_tie_break_picks_the_least_travel_among_equal_plans():
    """Warm from a plan at (2, -1), which tracks exactly, with the actuators
    measured at (1, -1). The plan is the point of the line nearest the
    actuators: minimise (a - 1)^2 + (b + 1)^2 s.t. a + b = 1, by hand
    (1.5, -0.5). Without a tie-break nothing picks a point of the line."""
    model = StageModel(step=_split, block=1, lo=(-3.0,) * 2, hi=(3.0,) * 2, tie=1e-4)
    with enable_x64():
        planner = Shooting(model, n_stages=4, nx=1)
        start = planner.solve(0.0, jnp.ones(1), jnp.array([2.0, -1.0]))
        plan = planner.solve(0.0, jnp.ones(1), jnp.array([1.0, -1.0]), start)
    assert not bool(plan.certificate.failed) and not bool(plan.certificate.capped)
    np.testing.assert_allclose(planner.moves(plan)[0], [1.5, -0.5], atol=1e-3)
