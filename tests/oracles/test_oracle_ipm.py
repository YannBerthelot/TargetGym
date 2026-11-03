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

from target_gym.oracles.ipm import NLP, interior_point, stage_kkt_solve

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


def _stage_system(seed, ne, S=4, nx=3, nu=2):
    """A random, well-posed Newton system in ``stage_kkt_solve``'s layout:
    K symmetric on the allowed pattern (x_k, u_k and the ne slacks e_k
    together; u_k with u_{k-1}) and shifted to be positive definite; J's row
    block k on x_{k+1} (an invertible E_k), x_k and u_k."""
    rng = np.random.default_rng(seed)
    xi = np.arange(nx * S).reshape(S, nx)
    ui = nx * S + np.arange(nu * S).reshape(S, nu)
    ei = (nx + nu) * S + np.arange(ne * S).reshape(S, ne)
    n, m = (nx + nu + ne) * S, nx * S
    allowed = np.zeros((n, n), bool)
    for k in range(S):
        own = np.concatenate([xi[k - 1] if k else [], ui[k], ei[k]]).astype(int)
        allowed[np.ix_(own, own)] = True
        if k:
            allowed[np.ix_(ui[k - 1], ui[k])] = allowed[np.ix_(ui[k], ui[k - 1])] = True
    R = rng.normal(size=(n, n)) * allowed
    K = R + R.T + (abs(np.linalg.eigvalsh(R + R.T).min()) + 1.0) * np.eye(n)
    J = np.zeros((m, n))
    for k in range(S):
        rows = slice(nx * k, nx * (k + 1))
        J[rows, xi[k]] = np.eye(nx) + 0.3 * rng.normal(size=(nx, nx))
        if k:
            J[rows, xi[k - 1]] = rng.normal(size=(nx, nx))
        J[rows, ui[k]] = rng.normal(size=(nx, nu))
    return K, J, rng.normal(size=n), rng.normal(size=m), ui


def _inertia_ok(K, J):
    """[[K, J'], [J, 0]] has exactly as many negative eigenvalues as J has
    rows iff K is positive definite on J's null space."""
    m = J.shape[0]
    M = np.block([[K, J.T], [J, np.zeros((m, m))]])
    return int(np.sum(np.linalg.eigvalsh(M) < 0)) == m


def _riccati(K, J, rz, rc, ne):
    with enable_x64():
        return stage_kkt_solve(*map(jnp.asarray, (K, J, rz, rc)), 4, 3, 2, ne)


@pytest.mark.parametrize("ne", [0, 1, 3])
@pytest.mark.parametrize("seed", [0, 1])
def test_stage_kkt_solve_is_the_dense_solve(seed, ne):
    K, J, rz, rc, _ = _stage_system(seed, ne)
    n, m = K.shape[0], J.shape[0]
    M = np.block([[K, -J.T], [J, np.zeros((m, m))]])
    expected = np.linalg.solve(M, np.concatenate([rz, rc]))
    dz, dlam, ok = _riccati(K, J, rz, rc, ne)
    # float64 on systems of condition about 1e3: rounding is near 1e-13.
    np.testing.assert_allclose(dz, expected[:n], rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(dlam, expected[n:], rtol=1e-9, atol=1e-10)
    assert bool(ok) and _inertia_ok(K, J)


def test_stage_kkt_solve_reports_the_wrong_inertia():
    """Negative curvature on one stage's move, larger than anything the
    dynamics can couple in, makes the reduced Hessian indefinite."""
    K, J, rz, rc, ui = _stage_system(3, ne=3)
    K[ui[2][0], ui[2][0]] = -1e3
    assert not _inertia_ok(K, J)
    assert not bool(_riccati(K, J, rz, rc, 3)[2])
