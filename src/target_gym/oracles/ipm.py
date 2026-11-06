"""The shared oracle solver: IPOPT's interior-point method in JAX.

The compressor's port (``compressor_surge.experts_ipm`` until that oracle
moved here). The algorithm is the same; what differs is how its matrices
are held. H, J and G may be dense arrays or, for
a multiple-shooting NLP, stage blocks (``StageHessian``, ``StageJacobian``,
``StageRows``), so no n x n matrix is ever formed: every product the
iteration takes (``G @ z``, ``G.T @ w``, ``J.T @ lam``) works on either, and
``NLP.barrier`` and ``NLP.shift`` form K and K + delta I. ``stage_kkt_solve``
reads the blocks and carries ``ne`` slacks per stage rather than one, and
``Result.shift`` reports the largest inertia shift a solve needed
(docs/proposals/oracle-solver.md, sections 2, 6 and 7).

IPOPT's algorithm in JAX, as CasADi ran it for the compressor until 0.7:
Waechter and Biegler, "On the implementation of an interior-point filter
line-search algorithm for large-scale nonlinear programming", Math. Program.
106 (2006), Algorithm A, with IPOPT 3.14's default constants, and IPOPT's
adaptive barrier update (Nocedal, Waechter and Waltz, SIAM J. Optim. 19,
2009) for warm starts. The problem is

    min f(z)  subject to  c(z) = 0  and  G z + g0 >= 0,

with slacks s (G z + g0 - s = 0, s > 0), multipliers ``lam`` for c and
``w > 0`` for s. The slack's equality multiplier equals ``w`` at every
iterate and is eliminated, so each Newton step solves

    [K + delta I   -J'] [dz  ]   [rz]
    [J              0 ] [dlam] = [rc],   K = H + G' diag(w / s) G,

H the Hessian of the Lagrangian f - lam'c and J the Jacobian of c, through
``kkt``. ``stage_kkt_solve`` solves it by a Riccati recursion for a
multiple-shooting layout.

What is IPOPT's, with its defaults: the monotone (Fiacco-McCormick) barrier,
or the adaptive one (``adaptive=True``: the quality-function oracle in free
mode, and the filter of (objective, violation) pairs that falls back to the
monotone mode and back); the filter line search with second-order
corrections, its accept tests with 10 eps of slack, and its full step when
the step is tiny; the inertia correction; the kappa_sigma reset of ``w``;
the step of ``lam`` along the primal step; the error measure and its
scaling; termination at ``tol``, or at ``acceptable_tol`` held
``acceptable_iter`` iterations, or at once when the line search fails at an
acceptable point. What is not: IPOPT's golden-section search over the
barrier centring is a scan of a log grid here; its soft restoration and its
stop after repeated tiny steps are not ported; and its restoration phase is
the caller's ``restore``, a point that meets c(z) = 0, taken at most
``max_restore`` times per solve. Without ``restore`` a failed line search
ends the solve as failed. The line-search details are IPOPT 3.14's source
(IpBacktrackingLineSearch.cpp, IpFilterLSAcceptor.cpp, IpUtils.cpp).

Every quantity is in whatever units the caller scales the problem to; the
caller applies IPOPT's scaling (``oracles.shooting``).
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

# IPOPT's defaults (option names in parentheses).
KAPPA_EPS = 10.0  # barrier_tol_factor
KAPPA_MU = 0.2  # mu_linear_decrease_factor
THETA_MU = 1.5  # mu_superlinear_decrease_power
TAU_MIN = 0.99  # tau_min
KAPPA_SIGMA = 1e10  # kappa_sigma
GAMMA_THETA = 1e-5  # gamma_theta
GAMMA_PHI = 1e-8  # gamma_phi
DELTA = 1.0  # delta
S_THETA = 1.1  # s_theta
S_PHI = 2.3  # s_phi
ETA_PHI = 1e-8  # eta_phi
GAMMA_ALPHA = 0.05  # alpha_min_frac
KAPPA_SOC = 0.99  # kappa_soc
MAX_SOC = 4  # max_soc
S_MAX = 100.0  # s_max
MU_MIN = 1e-11  # mu_min
MU_MAX = 1e5  # mu_max
MU_MAX_FACT = 1e3  # mu_max_fact
SIGMA_MIN = 1e-6  # sigma_min
SIGMA_MAX = 1e2  # sigma_max
MONOTONE_INIT_FACTOR = 0.8  # adaptive_mu_monotone_init_factor
FILTER_MARGIN_FACT = 1e-5  # filter_margin_fact
FILTER_MAX_MARGIN = 1.0  # filter_max_margin
FIRST_HESSIAN_SHIFT = 1e-4  # first_hessian_perturbation
TINY_STEP_EPS = 10.0  # tiny_step_tol, in machine epsilons
#: The violation above which a small step is not tiny (IPOPT's DetectTinyStep,
#: a constant in its source).
TINY_STEP_VIOLATION = 1e-4
#: The slack of IPOPT's accept tests, in machine epsilons of the reference
#: value (Compare_le).
COMPARE_EPS = 10.0
#: Points of the barrier-centring grid (ours), log-spaced over IPOPT's
#: [sigma_min, sigma_max]; IPOPT searches the same range by golden section.
N_SIGMA = 41
#: Entries each filter keeps (ours). A solve that adds more overwrites the
#: last, which only makes the filter more permissive.
FILTER_SIZE = 512


class NLP(NamedTuple):
    """One problem, as ``interior_point`` reads it.

    ``derivs(z, lam)`` returns (grad f, H, J, c) with H the Hessian of
    f - lam'c; ``kkt(K, J, rz, rc)`` returns (dz, dlam, ok), ``ok`` false
    when the reduced Hessian is not positive definite; ``restore(z)``, if
    given, returns a point with c = 0. ``barrier(H, G, sig)`` returns
    K = H + G' diag(sig) G and ``shift(K, delta)`` K + delta I: dense by
    default, ``stage_barrier`` and ``stage_shift`` for stage blocks.
    """

    f: Callable
    c: Callable
    derivs: Callable
    G: jnp.ndarray
    g0: jnp.ndarray
    kkt: Callable
    restore: Callable | None = None
    barrier: Callable = lambda H, G, sig: H + G.T @ (sig[:, None] * G)  # noqa: E731
    shift: Callable = lambda K, delta: K + delta * jnp.eye(K.shape[0])  # noqa: E731


class Result(NamedTuple):
    z: jnp.ndarray
    s: jnp.ndarray
    lam: jnp.ndarray
    w: jnp.ndarray
    iterations: jnp.ndarray
    #: The final error (IPOPT's, with mu = 0); -1 for a failed solve.
    error: jnp.ndarray
    #: The largest inertia shift delta any iteration added; 0 if none.
    shift: jnp.ndarray


def _to_boundary(v, dv, tau):
    """The largest step in (0, 1] that keeps ``v + a dv >= (1 - tau) v``."""
    neg = dv < 0
    ratio = jnp.where(neg, -tau * v / jnp.where(neg, dv, -1.0), jnp.inf)
    return jnp.minimum(1.0, jnp.min(ratio))


def interior_point(
    nlp: NLP,
    z,
    s,
    lam,
    w,
    *,
    mu0: float,
    max_iter: int,
    tol: float = 1e-8,
    adaptive: bool = False,
    acceptable_tol: float = 1e-6,
    acceptable_iter: int = 15,
    max_restore: int = 3,
) -> Result:
    """Solve ``nlp`` from (z, s, lam, w), s and w strictly positive."""
    f, c_fn, derivs, G, g0, kkt, restore, barrier, shift = nlp
    n, me, mi = z.shape[0], lam.shape[0], s.shape[0]
    mu_floor = tol / (KAPPA_EPS + 1.0)
    eps = jnp.finfo(z.dtype).eps
    tiny_tol, cmp_eps = TINY_STEP_EPS * eps, COMPARE_EPS * eps

    def residuals(gf, J, c, z, s, lam, w):
        rd = gf - J.T @ lam - G.T @ w
        ri = G @ z + g0 - s
        sd = jnp.maximum(
            S_MAX, (jnp.sum(jnp.abs(lam)) + jnp.sum(jnp.abs(w))) / (me + mi)
        )
        sc = jnp.maximum(S_MAX, jnp.sum(jnp.abs(w)) / mi)
        primal = jnp.maximum(jnp.max(jnp.abs(c)), jnp.max(jnp.abs(ri)))
        return rd, ri, jnp.max(jnp.abs(rd)) / (sd / S_MAX), primal, sc / S_MAX

    def error(gf, J, c, z, s, lam, w, mu):
        _, _, dual, primal, sc = residuals(gf, J, c, z, s, lam, w)
        return jnp.maximum(jnp.maximum(dual, primal), jnp.max(jnp.abs(s * w - mu)) / sc)

    def theta(z, s):
        return jnp.sum(jnp.abs(c_fn(z))) + jnp.sum(jnp.abs(G @ z + g0 - s))

    def phi(z, s, mu):
        return f(z) - mu * jnp.sum(jnp.log(s))

    def body(carry):
        (k, z, s, lam, w, mu, d_last, F, nF, gf, H, J, c, err, n_acc, th_lim,
         free, GF, nG, mu_max, n_rest, d_max) = carry  # fmt: skip
        th_min = th_lim[0]
        rd, ri, dual, primal, sc = residuals(gf, J, c, z, s, lam, w)

        # A-3: lower mu while the barrier problem is solved to kappa_eps mu.
        def mu_cond(st):
            m, i = st
            e = jnp.maximum(jnp.maximum(dual, primal), jnp.max(jnp.abs(s * w - m)) / sc)
            return (e <= KAPPA_EPS * m) & (m > mu_floor) & (i < 20)

        def mu_next(st):
            m, i = st
            return jnp.maximum(mu_floor, jnp.minimum(KAPPA_MU * m, m**THETA_MU)), i + 1

        mu_mono, _ = jax.lax.while_loop(mu_cond, mu_next, (mu, 0))

        sig = w / s
        K = barrier(H, G, sig)
        mu_avg = s @ w / mi

        def direction(delta, cvec, target):
            comp = w - target / s
            rz = -rd - G.T @ (sig * ri + comp)
            dz, dl, ok = kkt(shift(K, delta), J, rz, -cvec)
            ds = G @ dz + ri
            return dz, dl, ds, -sig * ds - comp, ok & jnp.all(jnp.isfinite(dz))

        # Inertia correction: no shift, then 1e-4 (or a third of the last
        # shift), growing 100x (8x once a shift has been needed) to 1e40.
        def in_cond(st):
            return ~st[2] & (st[1] < 1e40)

        def in_next(st):
            i, delta = st[0], st[1]
            first = jnp.where(
                d_last == 0, FIRST_HESSIAN_SHIFT, jnp.maximum(1e-20, d_last / 3.0)
            )
            grow = jnp.where(d_last == 0, 100.0, 8.0)
            delta = jnp.where(i == 0, 0.0, jnp.where(i == 1, first, delta * grow))
            return i + 1, delta, direction(delta, c, mu)[-1]

        _, delta, usable = jax.lax.while_loop(in_cond, in_next, (0, 0.0, False))
        d_last = jnp.where((delta > 0) & usable, delta, d_last)
        d_max = jnp.maximum(d_max, delta)

        if adaptive:
            # Globalisation: this point must be acceptable to the filter of
            # (objective, violation) pairs; if it is, it joins the filter.
            f_k = f(z)
            th_k = jnp.sum(jnp.abs(c)) + jnp.sum(jnp.abs(ri))
            live = jnp.arange(FILTER_SIZE) < nG
            progress = ~jnp.any(live & (f_k >= GF[:, 0]) & (th_k >= GF[:, 1]))
            margin = FILTER_MARGIN_FACT * jnp.minimum(FILTER_MAX_MARGIN, err)
            entry = jnp.array([f_k - margin, th_k - margin])
            GF = jnp.where(
                progress, GF.at[jnp.minimum(nG, FILTER_SIZE - 1)].set(entry), GF
            )
            nG = jnp.where(progress, jnp.minimum(nG + 1, FILTER_SIZE), nG)
            # Free mode: the centring that minimises the quality function, the
            # step's predicted residuals; the step is linear in the target.
            aff = direction(delta, c, 0.0)
            cen = [b - a for a, b in zip(aff[:4], direction(delta, c, mu_avg)[:4])]
            lo = jnp.maximum(SIGMA_MIN, MU_MIN / mu_avg)
            hi = jnp.maximum(lo, jnp.minimum(SIGMA_MAX, mu_max / mu_avg))
            sigmas = jnp.exp(jnp.linspace(jnp.log(lo), jnp.log(hi), N_SIGMA))
            primal2 = (jnp.sum(c**2) + jnp.sum(ri**2)) / (me + mi)
            dual2 = jnp.sum(rd**2) / n

            def quality(sigma):
                ds_, dw_ = aff[2] + sigma * cen[2], aff[3] + sigma * cen[3]
                tau_ = jnp.maximum(TAU_MIN, 1.0 - sigma * mu_avg)
                ap, ad = _to_boundary(s, ds_, tau_), _to_boundary(w, dw_, tau_)
                compl = (s + ap * ds_) * (w + ad * dw_)
                return (
                    (1 - ad) ** 2 * dual2
                    + (1 - ap) ** 2 * primal2
                    + jnp.sum(compl**2) / mi
                )

            mu_free = jnp.clip(
                sigmas[jnp.argmin(jax.vmap(quality)(sigmas))] * mu_avg, MU_MIN, mu_max
            )
            mu_fixed = jnp.clip(MONOTONE_INIT_FACTOR * mu_avg, MU_MIN, mu_max)
            # Free on progress; otherwise fixed, entered at 0.8 times the
            # average complementarity, and monotone from there.
            mu_new = jnp.where(progress, mu_free, jnp.where(free, mu_fixed, mu_mono))
            free = progress
        else:
            mu_new = mu_mono
        # A new barrier problem starts with an empty line-search filter.
        reset = mu_new != mu
        mu = mu_new
        F = jnp.where(reset, F.at[1:].set(jnp.inf), F)
        nF = jnp.where(reset, 1, nF)
        tau = jnp.maximum(TAU_MIN, 1.0 - mu)
        dz, dl, ds, dw, _ = direction(delta, c, mu)

        a_max = _to_boundary(s, ds, tau)
        th0, ph0 = theta(z, s), phi(z, s, mu)
        m = gf @ dz - mu * jnp.sum(ds / s)  # the barrier objective's slope

        def verdict(zt, st, a):
            """(accepted, f-type, theta) of a trial point; ``a`` is the step
            the switching and Armijo conditions are written for."""
            th, ph = theta(zt, st), phi(zt, st, mu)
            finite = jnp.isfinite(th) & jnp.isfinite(ph) & jnp.all(st > 0)
            live = jnp.arange(FILTER_SIZE) < nF
            in_filter = jnp.any(live & (th >= F[:, 0]) & (ph >= F[:, 1]))
            switching = (m < 0) & (a * (-m) ** S_PHI > DELTA * th0**S_THETA)
            ftype = switching & (th0 <= th_min)
            ph_slack, th_slack = cmp_eps * jnp.abs(ph0), cmp_eps * jnp.abs(th0)
            armijo = ph - ph0 - ETA_PHI * a * m <= ph_slack
            decrease = (th - (1.0 - GAMMA_THETA) * th0 <= th_slack) | (
                ph - ph0 + GAMMA_PHI * th0 <= ph_slack
            )
            return finite & ~in_filter & jnp.where(ftype, armijo, decrease), ftype, th

        ok0, ft0, th_t = verdict(z + a_max * dz, s + a_max * ds, a_max)
        # A step this small against the variables is roundoff: taken whole,
        # with no line search, unless the point is far from feasible.
        rel = jnp.maximum(
            jnp.max(jnp.abs(dz) / (1.0 + jnp.abs(z))),
            jnp.max(jnp.abs(ds) / (1.0 + jnp.abs(s))),
        )
        ok0 = ok0 | ((rel <= tiny_tol) & (th0 <= TINY_STEP_VIOLATION))

        # Second-order corrections, on a rejected full step that raised theta.
        def soc_cond(st):
            return (st[0] < MAX_SOC) & ~st[1] & st[2]

        def soc_next(st):
            p, c_soc, th_old = st[0], st[3], st[4]
            sz, sl, ss, sw, _ = direction(delta, c_soc, mu)
            a_s = _to_boundary(s, ss, tau)
            zt = z + a_s * sz
            accepted, ft, th = verdict(zt, s + a_s * ss, a_max)
            c_next = a_s * c_soc + c_fn(zt)
            return (
                p + 1,
                accepted,
                th <= KAPPA_SOC * th_old,
                c_next,
                th,
                (sz, sl, ss, sw, a_s, ft),
            )

        soc0 = (0, False, ~ok0 & (th_t >= th0), a_max * c + c_fn(z + a_max * dz), th_t,
                (dz, dl, ds, dw, a_max, ft0))  # fmt: skip
        _, soc_ok, _, _, _, soc_step = jax.lax.while_loop(soc_cond, soc_next, soc0)

        # Backtracking on the plain direction, down to IPOPT's alpha_min.
        a_min = GAMMA_ALPHA * jnp.where(
            m < 0,
            jnp.where(
                th0 <= th_min,
                jnp.minimum(
                    GAMMA_THETA,
                    jnp.minimum(
                        GAMMA_PHI * th0 / -m, DELTA * th0**S_THETA / (-m) ** S_PHI
                    ),
                ),
                jnp.minimum(GAMMA_THETA, GAMMA_PHI * th0 / -m),
            ),
            GAMMA_THETA,
        )

        def bt_cond(st):
            # The next step tried must exceed alpha_min, as IPOPT's loop
            # condition: alpha_min is 0 when theta is (W&B eq. 23), and the
            # halving then stops when the step underflows to 0.
            return ~st[1] & (0.5 * st[0] > a_min)

        def bt_next(st):
            a = 0.5 * st[0]
            accepted, ft, _ = verdict(z + a * dz, s + a * ds, a)
            return a, accepted, ft

        a_bt, bt_ok, bt_ft = jax.lax.while_loop(
            bt_cond, bt_next, (a_max, ok0 | soc_ok, ft0)
        )
        sz, sl, ss, sw, a_s, ft_s = soc_step
        use_soc = ~ok0 & soc_ok
        a = jnp.where(use_soc, a_s, jnp.where(ok0, a_max, a_bt))
        Dz, Dl, Ds, Dw = (
            jnp.where(use_soc, u, v)
            for u, v in ((sz, dz), (sl, dl), (ss, ds), (sw, dw))
        )
        ftype = jnp.where(use_soc, ft_s, jnp.where(ok0, ft0, bt_ft))
        accepted = ok0 | soc_ok | bt_ok
        a_z = _to_boundary(w, Dw, tau)

        # A failed line search: in free mode IPOPT's fallback, the monotone
        # mode from this point; otherwise a restoration step.
        fallback = (~accepted & free) if adaptive else jnp.array(False)
        # IPOPT does not restore at a point already acceptable: the solve ends
        # there (ACCEPTABLE_POINT_REACHED), by this port's acceptable test.
        at_acc = ~accepted & ~fallback & (err <= acceptable_tol)
        resto = ~accepted & ~fallback & ~at_acc
        if restore is not None:
            z_r = restore(z)
            s_r = jnp.maximum(G @ z_r + g0, mu)
        else:
            z_r, s_r = z, s
        z1 = jnp.where(accepted, z + a * Dz, jnp.where(resto, z_r, z))
        s1 = jnp.where(accepted, s + a * Ds, jnp.where(resto, s_r, s))
        lam1 = jnp.where(accepted, lam + a * Dl, lam)
        w1 = jnp.where(accepted, w + a_z * Dw, w)
        w1 = jnp.where(
            at_acc, w, jnp.clip(w1, mu / (KAPPA_SIGMA * s1), KAPPA_SIGMA * mu / s1)
        )
        n_rest = n_rest + resto
        if adaptive:
            mu = jnp.where(
                fallback, jnp.clip(MONOTONE_INIT_FACTOR * mu_avg, MU_MIN, mu_max), mu
            )
            free = free & ~fallback
        # A-7: a step that was not f-type joins the filter; a fallback or a
        # restoration empties it.
        entry = jnp.array([(1.0 - GAMMA_THETA) * th0, ph0 - GAMMA_PHI * th0])
        joins = accepted & ~ftype
        F = jnp.where(joins, F.at[jnp.minimum(nF, FILTER_SIZE - 1)].set(entry), F)
        nF = jnp.where(joins, jnp.minimum(nF + 1, FILTER_SIZE), nF)
        F = jnp.where(accepted, F, F.at[1:].set(jnp.inf))
        nF = jnp.where(accepted, nF, 1)

        gf1, H1, J1, c1 = derivs(z1, lam1)
        err1 = error(gf1, J1, c1, z1, s1, lam1, w1, 0.0)
        recovered = fallback | (resto & (restore is not None) & (n_rest <= max_restore))
        finite = (
            jnp.isfinite(err1) & jnp.all(jnp.isfinite(z1)) & jnp.all(jnp.isfinite(lam1))
        )
        good = ((accepted | recovered) & finite) | at_acc

        def pick(u, v):
            return jax.tree_util.tree_map(lambda a, b: jnp.where(good, a, b), u, v)

        n_acc = jnp.where(err1 <= acceptable_tol, n_acc + 1, 0)
        n_acc = jnp.where(at_acc, acceptable_iter, n_acc)
        return (
            jnp.where(good, k + 1, max_iter), pick(z1, z), pick(s1, s), pick(lam1, lam),
            pick(w1, w), mu, d_last, F, nF, pick(gf1, gf), pick(H1, H), pick(J1, J),
            pick(c1, c), jnp.where(good, err1, -1.0), n_acc, th_lim, free, GF, nG,
            mu_max, n_rest, d_max,
        )  # fmt: skip

    def cond(carry):
        k, err, n_acc = carry[0], carry[13], carry[14]
        return (k < max_iter) & (err > tol) & (n_acc < acceptable_iter)

    gf, H, J, c = derivs(z, lam)
    th_init = jnp.sum(jnp.abs(c)) + jnp.sum(jnp.abs(G @ z + g0 - s))
    th_lim = jnp.array([1e-4, 1e4]) * jnp.maximum(1.0, th_init)
    F = jnp.full((FILTER_SIZE, 2), jnp.inf).at[0].set(jnp.array([th_lim[1], -jnp.inf]))
    GF = jnp.full((FILTER_SIZE, 2), jnp.inf)
    mu_max = jnp.minimum(MU_MAX, MU_MAX_FACT * (s @ w) / mi)
    err0 = error(gf, J, c, z, s, lam, w, 0.0)
    carry = (0, z, s, lam, w, mu0, 0.0, F, 1, gf, H, J, c, err0, 0, th_lim, True, GF, 0,
             mu_max, 0, 0.0)  # fmt: skip
    k, z, s, lam, w, *rest = jax.lax.while_loop(cond, body, carry)
    return Result(z, s, lam, w, k, rest[8], rest[-1])


def _parts(z, N, nx, nu):
    """(x_1..x_N, u_0..u_{N-1}, e_0..e_{N-1}) of a vector in stage layout."""
    a, b = N * nx, N * (nx + nu)
    return z[:a].reshape(N, nx), z[a:b].reshape(N, nu), z[b:].reshape(N, -1)


def _starts(xs):
    """x_k for each stage k, x_0 (measured, no variable) as zeros."""
    return jnp.concatenate([jnp.zeros_like(xs[:1]), xs[:-1]])


def _from_starts(gx):
    """The adjoint of ``_starts``: stage k's x_k entries onto x_1..x_N."""
    return jnp.concatenate([gx[1:], jnp.zeros_like(gx[:1])])


class _Adjoint(NamedTuple):
    matvec: Callable

    def __matmul__(self, v):
        return self.matvec(v)


class StageRows(NamedTuple):
    """G in stage blocks: stage k's rows read only x_k (``Gx``, stage 0's
    unused), u_k (``Gu``) and e_k (``Ge``), each (N, rows, width)."""

    Gx: jnp.ndarray
    Gu: jnp.ndarray
    Ge: jnp.ndarray

    def __matmul__(self, z):
        N, nx = self.Gx.shape[0], self.Gx.shape[2]
        xs, us, es = _parts(z, N, nx, self.Gu.shape[2])
        out = jnp.einsum("kri,ki->kr", self.Gx, _starts(xs))
        out = out + jnp.einsum("kri,ki->kr", self.Gu, us)
        return (out + jnp.einsum("kri,ki->kr", self.Ge, es)).ravel()

    @property
    def T(self):
        def matvec(w):
            W = w.reshape(self.Gx.shape[:2])
            gx, gu, ge = (jnp.einsum("kri,kr->ki", g, W) for g in self)
            return jnp.concatenate([_from_starts(gx).ravel(), gu.ravel(), ge.ravel()])

        return _Adjoint(matvec)


@jax.tree_util.register_pytree_node_class
class StageJacobian:
    """J of the shooting rows in stage blocks: row block k is E_k dx_{k+1} +
    A_k dx_k + B_k du_k (A_0 unused); no row reads the ``ne`` slacks."""

    def __init__(self, E, A, B, ne: int):
        self.E, self.A, self.B, self.ne = E, A, B, ne

    def tree_flatten(self):
        return (self.E, self.A, self.B), self.ne

    @classmethod
    def tree_unflatten(cls, ne, children):
        return cls(*children, ne)

    @property
    def T(self):
        def matvec(lam):
            L = lam.reshape(self.E.shape[:2])
            gx = jnp.einsum("kij,ki->kj", self.E, L)
            gx = gx + _from_starts(jnp.einsum("kij,ki->kj", self.A, L))
            gu = jnp.einsum("kij,ki->kj", self.B, L)
            e = jnp.zeros(L.shape[0] * self.ne, L.dtype)
            return jnp.concatenate([gx.ravel(), gu.ravel(), e])

        return _Adjoint(matvec)


class StageHessian(NamedTuple):
    """A symmetric matrix in stage blocks: ``D[k]`` on v_k = (x_k, u_k, e_k)
    (stage 0's x_0 part unused), ``P[k]`` coupling u_{k-1} with (u_k, e_k),
    ``end`` on x_N. The pattern ``stage_kkt_solve`` takes."""

    D: jnp.ndarray
    P: jnp.ndarray
    end: jnp.ndarray


def stage_barrier(H: StageHessian, G: StageRows, sig) -> StageHessian:
    """H + G' diag(sig) G, stage by stage."""
    Gk = jnp.concatenate(list(G), axis=-1)
    sk = sig.reshape(Gk.shape[:2])
    return H._replace(D=H.D + jnp.einsum("kri,kr,krj->kij", Gk, sk, Gk))


def stage_shift(K: StageHessian, delta) -> StageHessian:
    """K + delta I."""
    return K._replace(
        D=K.D + delta * jnp.eye(K.D.shape[-1]), end=K.end + delta * jnp.eye(len(K.end))
    )


def stage_multipliers(J: StageJacobian, r):
    """The least-squares multipliers (J J')^-1 J r, as IPOPT computes them:
    the Newton system with K = I, by the Riccati recursion."""
    N, nx = J.E.shape[:2]
    nu, ne = J.B.shape[2], J.ne
    eye = StageHessian(
        jnp.broadcast_to(jnp.eye(nx + nu + ne), (N, nx + nu + ne, nx + nu + ne)),
        jnp.zeros((N, nu, nu + ne)),
        jnp.eye(nx),
    )
    return stage_kkt_solve(eye, J, -r, jnp.zeros(N * nx, r.dtype))[1]


def stage_kkt_solve(K: StageHessian, J: StageJacobian, rz, rc):
    """The Newton system of a multiple-shooting NLP, by a Riccati recursion.

    Solves [K -J'; J 0] [dz; dlam] = [rz; rc] for the layout
    z = (x_1..x_S, u_0..u_{S-1}, e_0..e_{S-1}): states (nx each; x_0 is
    measured, not a variable), inputs (nu each) and ``ne`` slacks per stage
    (none to several). K, in stage blocks, couples only x_k, u_k and e_k with
    each other and u_k with u_{k-1}; row block k of J is E_k dx_{k+1} +
    A_k dx_k + B_k du_k with E_k invertible. That is the equality-constrained
    LQ problem min 1/2 dz'K dz - rz'dz s.t. J dz = rc in the augmented state
    y_k = (dx_k, du_{k-1}) and control v_k = (du_k, de_k).

    Returns (dz, dlam, ok), ``ok`` false unless every stage's reduced
    Hessian Q_vv is positive definite, which is the inertia the step needs.
    """
    S, nx, nu = K.D.shape[0], K.end.shape[0], K.P.shape[1]
    ne = K.P.shape[2] - nu
    ny, nv = nx + nu, nu + ne
    xi = jnp.arange(nx * S).reshape(S, nx)  # x_1..x_S; x_k at row k - 1
    ui = nx * S + jnp.arange(nu * S).reshape(S, nu)
    ei = (nx + nu) * S + jnp.arange(ne * S).reshape(S, ne)
    xk = jnp.concatenate([xi[:1], xi[:-1]])  # x_k; stage 0's is masked out
    vk = jnp.concatenate([ui, ei], axis=1)
    mask = (jnp.arange(S) > 0)[:, None].astype(K.D.dtype)
    Hxx = K.D[:, :nx, :nx] * mask[:, :, None] * mask[:, None, :]
    Hxv = K.D[:, :nx, nx:] * mask[:, :, None]
    Hvv = K.D[:, nx:, nx:]
    Hpv = K.P * mask[:, :, None]
    gx = -rz[xk] * mask
    gv = -rz[vk]
    E = J.E
    Ei = jnp.linalg.inv(E)
    A = -Ei @ (J.A * mask[:, None, :])
    B = -Ei @ J.B
    b = (Ei @ rc.reshape(S, nx)[:, :, None])[:, :, 0]
    P_end = jnp.zeros((ny, ny), K.D.dtype).at[:nx, :nx].set(K.end)
    p_end = jnp.zeros(ny, K.D.dtype).at[:nx].set(-rz[xi[-1]])

    def backward(carry, k):
        P, p, ok = carry
        Ab = jnp.zeros((ny, ny), K.D.dtype).at[:nx, :nx].set(A[k])
        Bb = (
            jnp.zeros((ny, nv), K.D.dtype)
            .at[:nx, :nu]
            .set(B[k])
            .at[nx:, :nu]
            .set(jnp.eye(nu))
        )
        bb = jnp.zeros(ny, K.D.dtype).at[:nx].set(b[k])
        Myy = jnp.zeros((ny, ny), K.D.dtype).at[:nx, :nx].set(Hxx[k])
        Myv = jnp.zeros((ny, nv), K.D.dtype).at[:nx].set(Hxv[k]).at[nx:].set(Hpv[k])
        Qyy = Myy + Ab.T @ P @ Ab
        Qyv = Myv + Ab.T @ P @ Bb
        Qvv = Hvv[k] + Bb.T @ P @ Bb
        Qvv = 0.5 * (Qvv + Qvv.T)
        g = P @ bb + p
        qy = jnp.zeros(ny, K.D.dtype).at[:nx].set(gx[k]) + Ab.T @ g
        qv = gv[k] + Bb.T @ g
        L = jnp.linalg.cholesky(Qvv)
        ok = ok & jnp.all(jnp.isfinite(L))
        Kk = -jax.scipy.linalg.cho_solve((L, True), Qyv.T)
        kk = -jax.scipy.linalg.cho_solve((L, True), qv)
        P = Qyy + Qyv @ Kk
        P, p = 0.5 * (P + P.T), qy + Qyv @ kk
        return (P, p, ok), (Kk, kk, P, p)

    (_, _, ok), (Ks, ks, Ps, ps) = jax.lax.scan(
        backward, (P_end, p_end, True), jnp.arange(S)[::-1]
    )
    Ks, ks, Ps, ps = Ks[::-1], ks[::-1], Ps[::-1], ps[::-1]
    P_next = jnp.concatenate([Ps[1:], P_end[None]])
    p_next = jnp.concatenate([ps[1:], p_end[None]])

    def forward(y, k):
        v = Ks[k] @ y + ks[k]
        dx = A[k] @ y[:nx] + B[k] @ v[:nu] + b[k]
        y1 = jnp.concatenate([dx, v[:nu]])
        # The multiplier is the value function's slope at x_{k+1}.
        dlam = jnp.linalg.solve(E[k].T, (P_next[k] @ y1 + p_next[k])[:nx])
        return y1, (v, dx, dlam)

    _, (V, DX, DL) = jax.lax.scan(forward, jnp.zeros(ny, K.D.dtype), jnp.arange(S))
    dz = jnp.zeros(rz.shape[0], K.D.dtype)
    dz = dz.at[xi.ravel()].set(DX.ravel()).at[ui.ravel()].set(V[:, :nu].ravel())
    dz = dz.at[ei.ravel()].set(V[:, nu:].ravel())
    return dz, DL.ravel(), ok
