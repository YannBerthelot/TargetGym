"""The shared oracle solver, ``oracles.ipm``.

Contract (its docstring; Waechter and Biegler 2006): ``interior_point``
returns a point that meets the KKT conditions of min f s.t. c = 0,
G z + g0 >= 0 to ``tol``, with the monotone barrier and with the adaptive
one. ``stage_kkt_solve`` solves the Newton system of the multiple-shooting
layout with ``ne`` slacks per stage (none to several, G5) and reports
whether every stage's reduced Hessian is positive definite.

Oracles: a closed form (the Euclidean projection onto the probability
simplex), a published optimum (Hock and Schittkowski's problem 71), NumPy's
dense solve of the same system, and the inertia theorem. The first two are
the compressor's port's tests, run against this copy.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.oracles.ipm import (
    NLP,
    StageHessian,
    StageJacobian,
    StageRows,
    interior_point,
    stage_barrier,
    stage_kkt_factor,
    stage_kkt_solve,
    stage_kkt_solve_factored,
    stage_multipliers,
    stage_shift,
)

try:
    from jax import enable_x64
except ImportError:  # JAX 0.6; 0.10 removed the experimental name
    from jax.experimental import enable_x64


def _dense_kkt(K, J, rz, rc):
    n, m = K.shape[0], J.shape[0]
    M = jnp.block([[K, -J.T], [J, jnp.zeros((m, m))]])
    sol = jnp.linalg.solve(M, jnp.concatenate([rz, rc]))
    Z = jnp.linalg.svd(J)[2][m:].T
    return sol[:n], sol[n:], jnp.all(jnp.linalg.eigvalsh(Z.T @ K @ Z) > 0)


def _nlp(f, c, G, g0):
    def derivs(z, lam):
        lagrangian = lambda v: f(v) - lam @ c(v)  # noqa: E731
        return jax.grad(f)(z), jax.hessian(lagrangian)(z), jax.jacfwd(c)(z), c(z)

    return NLP(f=f, c=c, derivs=derivs, G=G, g0=g0, kkt=_dense_kkt)


@pytest.mark.parametrize("adaptive", [False, True])
def test_projects_onto_the_simplex(adaptive):
    """min 1/2 |z - a|^2 s.t. sum z = 1, z >= 0, a = (0.8, 0.6, -0.2, 0.1).
    By hand: the threshold (0.8 + 0.6 - 1) / 2 = 0.2 keeps the first two,
    so z = max(a - 0.2, 0) = (0.6, 0.4, 0, 0)."""
    with enable_x64():
        a = jnp.array([0.8, 0.6, -0.2, 0.1], jnp.float64)
        nlp = _nlp(
            lambda z: 0.5 * jnp.sum((z - a) ** 2),
            lambda z: jnp.atleast_1d(jnp.sum(z) - 1.0),
            jnp.eye(4),
            jnp.zeros(4),
        )
        z0 = jnp.full(4, 0.25)
        r = interior_point(
            nlp, z0, z0, jnp.zeros(1), jnp.ones(4), mu0=0.1, max_iter=100,
            adaptive=adaptive,
        )  # fmt: skip
        np.testing.assert_allclose(r.z, [0.6, 0.4, 0.0, 0.0], atol=1e-6)
        assert 0.0 <= float(r.error) <= 1e-8


@pytest.mark.parametrize("adaptive", [False, True])
def test_solves_hock_schittkowski_71(adaptive):
    """HS071: min x1 x4 (x1 + x2 + x3) + x3 s.t. x1 x2 x3 x4 >= 25,
    sum x^2 = 40, 1 <= x <= 5, from (1, 5, 5, 1); the product bound through
    a variable t >= 25. Optimum (Hock and Schittkowski, 1981):
    x* = (1, 4.74299963, 3.82114998, 1.37940829), f* = 17.0140173."""
    with enable_x64():

        def f(z):
            x = z[:4]
            return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

        def c(z):
            return jnp.stack([jnp.sum(z[:4] ** 2) - 40.0, jnp.prod(z[:4]) - z[4]])

        eye = jnp.eye(5)[:4]
        G = jnp.concatenate([eye, -eye, jnp.eye(5)[4:]])
        g0 = jnp.concatenate([-jnp.ones(4), 5.0 * jnp.ones(4), jnp.array([-25.0])])
        z0 = jnp.array([1.0, 5.0, 5.0, 1.0, 25.0])
        r = interior_point(
            _nlp(f, c, G, g0), z0, jnp.maximum(G @ z0 + g0, 1e-2), jnp.zeros(2),
            jnp.ones(9), mu0=0.1, max_iter=100, adaptive=adaptive,
        )  # fmt: skip
        np.testing.assert_allclose(
            r.z[:4], [1.0, 4.74299963, 3.82114998, 1.37940829], atol=1e-6
        )
        np.testing.assert_allclose(f(r.z), 17.0140173, rtol=1e-7)
        assert 0.0 <= float(r.error) <= 1e-8


# Roundoff near an optimum, made deterministic. IPOPT 3.14's line search
# (IpBacktrackingLineSearch.cpp, IpFilterLSAcceptor.cpp, IpUtils.cpp) has three
# answers to it, each tested on min f(z) s.t. z1 - z2 = 0, z >= lower, whose
# reported gradient or value the test perturbs: a restoration called at an
# acceptable point ends the solve there (ACCEPTABLE_POINT_REACHED); its accept
# tests allow 10 eps |reference| (Compare_le); and a step below 10 eps of the
# variables is taken whole, the line search skipped (DetectTinyStep).
EPS = float(np.finfo(np.float64).eps)


def _perturbed(f, grad, lower, restore=None, residual=0.0):
    """``residual`` stays in the row as the line search evaluates it (c), not
    as the derivatives report it, as roundoff leaves in a shooting row: no
    step, second-order correction included, can remove it."""

    def derivs(z, lam):
        return grad(z), jnp.eye(2), jnp.array([[1.0, -1.0]]), reported(z)

    def reported(z):
        return jnp.atleast_1d(z[0] - z[1])

    def c(z):
        return jnp.abs(reported(z)) + residual if residual else reported(z)

    return NLP(
        f=f, c=c, derivs=derivs, G=jnp.eye(2), g0=-lower * jnp.ones(2),
        kkt=_dense_kkt, restore=restore,
    )  # fmt: skip


def _solve_from(nlp, z0, lower, mu0):
    """From z0 on the constraint, barrier mu0, each w at its centre mu0 / s."""
    s = z0 - lower
    return interior_point(
        nlp, z0, s, jnp.zeros(1), mu0 / s, mu0=mu0, max_iter=30
    )  # fmt: skip


def test_a_line_search_that_fails_at_an_acceptable_point_ends_the_solve_there():
    """f = |z|^2 / 2 with its gradient reported 1e-7 high, from z = 0 with
    the bound at -1, and a 1e-16 residual in the row, as cstr's warm solves
    met: every step (-1e-7, -1e-7) raises the barrier objective phi from
    exactly 0 (f = 0, log s = 0) and leaves the violation at 1e-16, so no
    step length down to alpha_min (2.5e-12 here) passes, not even with the
    accept tests' slack 10 eps |phi| = 0. The error is 1e-7, inside
    acceptable_tol 1e-6, so IPOPT stops at z = 0, before the restoration
    (which here would move the point by 1)."""
    with enable_x64():
        nlp = _perturbed(
            lambda z: 0.5 * jnp.sum(z**2), lambda z: z + 1e-7, -1.0,
            restore=lambda z: z + 1.0, residual=1e-16,
        )  # fmt: skip
        r = _solve_from(nlp, jnp.zeros(2), -1.0, 1e-9)
        assert int(r.iterations) < 15
        assert 1e-8 < float(r.error) <= 1e-6
        np.testing.assert_array_equal(r.z, [0.0, 0.0])


def test_a_shift_that_failed_does_not_set_the_next_shift():
    """IPOPT's inertia correction (Waechter and Biegler 2006, Algorithm IC)
    remembers the last shift that worked. f = sum (z^2 - 1)^2 / 4 on z1 = z2
    has its minima at z = +-1. The Hessian is NaN at the start, z = 3, so no
    shift works there and a restoration moves to z = 0.2, where f'' = -0.88
    needs a shift near 1. Remembering the failed 1e40 would start that search
    at 3e39, whose step is zero, and a third of it at each of the 80 iterations
    it takes to come down. From 0.2 the gradient z^3 - z is negative, so the
    solve descends to z = 1."""
    with enable_x64():

        def f(z):
            return jnp.sum((z**2 - 1.0) ** 2) / 4.0

        def derivs(z, lam):
            H = jnp.diag(3.0 * z**2 - 1.0)
            poison = jnp.where(z[0] > 2.0, jnp.nan, 1.0)
            return (
                jax.grad(f)(z), poison * H, jnp.array([[1.0, -1.0]]),
                jnp.atleast_1d(z[0] - z[1]),
            )  # fmt: skip

        nlp = NLP(
            f=f, c=lambda z: jnp.atleast_1d(z[0] - z[1]), derivs=derivs,
            G=jnp.eye(2), g0=10.0 * jnp.ones(2), kkt=_dense_kkt,
            restore=lambda z: 0.2 * jnp.ones(2),
        )  # fmt: skip
        z0 = 3.0 * jnp.ones(2)
        r = interior_point(
            nlp, z0, z0 + 10.0, jnp.zeros(1), 0.01 / (z0 + 10.0), mu0=0.01, max_iter=60
        )
        assert 0.0 <= float(r.error) <= 1e-8
        np.testing.assert_allclose(r.z, [1.0, 1.0], atol=1e-6)


@pytest.mark.parametrize("adaptive", [False, True])
def test_every_direction_comes_from_the_accepted_shift(adaptive):
    """f = sum (z^2 - 1)^2 / 4 on z1 = z2 has its minima at z = +-1 and a
    maximum at 0. At the start, z = 0.2, f'' = 3 z^2 - 1 = -0.88: K + delta I
    is the wrong inertia until delta is near 1, and the unshifted Newton step
    heads for the maximum. The gradient z^3 - z is negative there, so a step
    from the accepted shift descends to z = 1; the step, the adaptive mode's
    affine and centring directions and the second-order corrections must all
    come from that shift's factorisation."""
    with enable_x64():
        nlp = _nlp(
            lambda z: jnp.sum((z**2 - 1.0) ** 2) / 4.0,
            lambda z: jnp.atleast_1d(z[0] - z[1]),
            jnp.eye(2), 10.0 * jnp.ones(2),
        )  # fmt: skip
        z0 = 0.2 * jnp.ones(2)
        r = interior_point(
            nlp, z0, z0 + 10.0, jnp.zeros(1), 0.01 / (z0 + 10.0), mu0=0.01,
            max_iter=60, adaptive=adaptive,
        )  # fmt: skip
    assert float(r.shift) > 0.0
    assert 0.0 <= float(r.error) <= 1e-8
    np.testing.assert_allclose(r.z, [1.0, 1.0], atol=1e-6)


def test_the_accept_tests_allow_ten_eps_of_the_reference():
    """f = 1 + |z - 1|^2 / 2, plus 4 eps on any point past the start, from
    z = 1 - 2e-8: the Newton step to z = 1 lowers the smooth part by 4e-16
    and raises f by 8.9e-16, a net rise of 2 ulp of f, inside the 10 eps
    |phi| = 2.2e-15 that Compare_le allows. Taken, the solve converges to
    tol; refused, no step passes and it ends at the start's error, 2e-8."""
    with enable_x64():
        start = 1.0 - 2e-8

        def f(z):
            return (
                1.0
                + 0.5 * jnp.sum((z - 1.0) ** 2)
                + jnp.where(z[0] > start, 4.0 * EPS, 0.0)
            )

        nlp = _perturbed(f, lambda z: z - 1.0, -10.0)
        r = _solve_from(nlp, jnp.full(2, start), -10.0, 1e-12)
        assert 0.0 <= float(r.error) <= 1e-8
        np.testing.assert_allclose(r.z, [1.0, 1.0], rtol=0, atol=1e-12)


def test_a_step_below_ten_eps_of_the_variables_is_taken_whole():
    """f = |z - 1e8|^2 / 2, plus 1 on any point past the start, from 7 ulp
    (1.04e-7) below 1e8: the step is 1e-15 of the variables, under
    tiny_step_tol = 10 eps, with no violation, so IPOPT takes it whole
    although f rises by 1. Then the gradient is 0 and the solve converges."""
    with enable_x64():
        Z = 1e8
        start = Z - 7.0 * np.spacing(Z)

        def f(z):
            return 0.5 * jnp.sum((z - Z) ** 2) + jnp.where(z[0] > start, 1.0, 0.0)

        nlp = _perturbed(f, lambda z: z - Z, 0.0)
        r = _solve_from(nlp, jnp.full(2, start), 0.0, 1e-12)
        assert 0.0 <= float(r.error) <= 1e-8
        np.testing.assert_array_equal(r.z, [Z, Z])


def _blocks(seed, ne, S=4, nx=3, nu=2, m=5):
    """Random stage blocks, well posed, and the same matrices written dense
    from the layout z = (x_1..x_S, u_0..u_{S-1}, e_0..e_{S-1}): K's block k
    on (x_k, u_k, e_k), symmetric, coupling u_{k-1} with (u_k, e_k), shifted
    to be positive definite; J's row block k on x_{k+1} (an invertible E_k),
    x_k and u_k; G's row block k on x_k, u_k and e_k. Stage 0 has no x_0."""
    rng = np.random.default_rng(seed)
    nl = nx + nu + ne
    xi = np.arange(nx * S).reshape(S, nx)
    ui = nx * S + np.arange(nu * S).reshape(S, nu)
    ei = (nx + nu) * S + np.arange(ne * S).reshape(S, ne)
    n = nl * S
    D = rng.normal(size=(S, nl, nl))
    D = D + np.swapaxes(D, 1, 2)
    D[0, :nx], D[0, :, :nx] = 0.0, 0.0
    P = rng.normal(size=(S, nu, nu + ne))
    P[0] = 0.0
    end = rng.normal(size=(nx, nx))
    end = end + end.T
    K = np.zeros((n, n))
    for k in range(S):
        own = np.concatenate([xi[k - 1] if k else [], ui[k], ei[k]]).astype(int)
        K[np.ix_(own, own)] += D[k][-len(own) :, -len(own) :]
        if k:
            v = np.concatenate([ui[k], ei[k]])
            K[np.ix_(ui[k - 1], v)] += P[k]
            K[np.ix_(v, ui[k - 1])] += P[k].T
    K[np.ix_(xi[-1], xi[-1])] += end
    shift = abs(np.linalg.eigvalsh(K).min()) + 1.0
    K += shift * np.eye(n)
    D, end = D + shift * np.eye(nl), end + shift * np.eye(nx)
    E = np.eye(nx) + 0.3 * rng.normal(size=(S, nx, nx))
    A, B = rng.normal(size=(S, nx, nx)), rng.normal(size=(S, nx, nu))
    A[0] = 0.0
    J = np.zeros((nx * S, n))
    Gb = rng.normal(size=(S, m, nl))
    Gb[0, :, :nx] = 0.0
    G = np.zeros((m * S, n))
    for k in range(S):
        rows = slice(nx * k, nx * (k + 1))
        J[rows, xi[k]], J[rows, ui[k]] = E[k], B[k]
        cols = np.concatenate([ui[k], ei[k]])
        G[m * k : m * (k + 1), cols] = Gb[k, :, nx:]
        if k:
            J[rows, xi[k - 1]] = A[k]
            G[m * k : m * (k + 1), xi[k - 1]] = Gb[k, :, :nx]
    blocks = (
        StageHessian(*map(jnp.asarray, (D, P, end))),
        StageJacobian(*map(jnp.asarray, (E, A, B)), ne),
        StageRows(
            *map(jnp.asarray, (Gb[..., :nx], Gb[..., nx : nx + nu], Gb[..., nx + nu :]))
        ),
    )
    vecs = rng.normal(size=n), rng.normal(size=nx * S), rng.normal(size=m * S)
    return blocks, (K, J, G), vecs, ui


def _inertia_ok(K, J):
    """[[K, J'], [J, 0]] has exactly as many negative eigenvalues as J has
    rows iff K is positive definite on J's null space."""
    m = J.shape[0]
    M = np.block([[K, J.T], [J, np.zeros((m, m))]])
    return int(np.sum(np.linalg.eigvalsh(M) < 0)) == m


# float64 on systems of condition about 1e3: rounding is near 1e-13.
TIGHT = dict(rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("ne", [0, 1, 3])
@pytest.mark.parametrize("seed", [0, 1])
def test_stage_kkt_solve_is_the_dense_solve(seed, ne):
    with enable_x64():
        (Kb, Jb, _), (K, J, _), (rz, rc, _), _ = _blocks(seed, ne)
        dz, dlam, ok = stage_kkt_solve(Kb, Jb, jnp.asarray(rz), jnp.asarray(rc))
    n, m = K.shape[0], J.shape[0]
    M = np.block([[K, -J.T], [J, np.zeros((m, m))]])
    expected = np.linalg.solve(M, np.concatenate([rz, rc]))
    np.testing.assert_allclose(dz, expected[:n], **TIGHT)
    np.testing.assert_allclose(dlam, expected[n:], **TIGHT)
    assert bool(ok) and _inertia_ok(K, J)


@pytest.mark.parametrize("ne", [0, 3])
def test_one_factorisation_solves_every_right_hand_side(ne):
    """An iteration factorises K + delta I once and takes every direction
    (the step, the affine and centring ones, the second-order corrections)
    from it: each right-hand side is the dense solve of the same system."""
    with enable_x64():
        (Kb, Jb, _), (K, J, _), _, _ = _blocks(4, ne)
        fac, ok = stage_kkt_factor(Kb, Jb)
        rng = np.random.default_rng(7)
        rhs = [
            (rng.normal(size=K.shape[0]), rng.normal(size=J.shape[0])) for _ in range(3)
        ]
        got = [
            stage_kkt_solve_factored(fac, jnp.asarray(rz), jnp.asarray(rc))
            for rz, rc in rhs
        ]
    n, m = K.shape[0], J.shape[0]
    M = np.block([[K, -J.T], [J, np.zeros((m, m))]])
    assert bool(ok)
    for (rz, rc), (dz, dlam, _) in zip(rhs, got):
        expected = np.linalg.solve(M, np.concatenate([rz, rc]))
        np.testing.assert_allclose(dz, expected[:n], **TIGHT)
        np.testing.assert_allclose(dlam, expected[n:], **TIGHT)


@pytest.mark.parametrize("ne", [0, 3])
def test_blockwise_products_are_the_dense_products(ne):
    """Every product the iteration takes: G z, G'w, J'lam, K = H + G'diag(s)G
    and K + delta I (checked through the solve, which reads K only), and the
    least-squares multipliers (J J')^-1 J r."""
    with enable_x64():
        (Hb, Jb, Gb), (H, J, G), (z, lam, w), _ = _blocks(2, ne)
        sig, r = np.linspace(0.5, 2.0, len(w)), np.cos(np.arange(len(z)))
        Kb = stage_shift(stage_barrier(Hb, Gb, jnp.asarray(sig)), 0.7)
        dz = stage_kkt_solve(Kb, Jb, jnp.asarray(z), jnp.asarray(lam))[0]
        out = [Gb @ jnp.asarray(z), Gb.T @ jnp.asarray(w), Jb.T @ jnp.asarray(lam)]
        mult = stage_multipliers(Jb, jnp.asarray(r))
    K = H + G.T @ (sig[:, None] * G) + 0.7 * np.eye(len(z))
    m = J.shape[0]
    M = np.block([[K, -J.T], [J, np.zeros((m, m))]])
    expected = np.linalg.solve(M, np.concatenate([z, lam]))[: len(z)]
    for got, want in zip(out + [dz], [G @ z, G.T @ w, J.T @ lam, expected]):
        np.testing.assert_allclose(got, want, **TIGHT)
    np.testing.assert_allclose(mult, np.linalg.solve(J @ J.T, J @ r), **TIGHT)


def test_stage_kkt_solve_reports_the_wrong_inertia():
    """Negative curvature on one stage's move, larger than anything the
    dynamics can couple in, makes the reduced Hessian indefinite."""
    with enable_x64():
        (Kb, Jb, _), (K, J, _), (rz, rc, _), ui = _blocks(3, ne=3)
        K[ui[2][0], ui[2][0]] = -1e3
        Kb = Kb._replace(D=Kb.D.at[2, 3, 3].set(-1e3))  # u_2's first entry, nx = 3
        assert not _inertia_ok(K, J)
        assert not bool(stage_kkt_solve(Kb, Jb, jnp.asarray(rz), jnp.asarray(rc))[2])
