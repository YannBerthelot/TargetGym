"""Multiple shooting on Cooper's T_c grid, solved by the shared IPM.

The problem builder the oracles are to share (docs/proposals/oracle-solver.md
and the horizon and smoothing protocols beside it). A task gives its planner
model one task step at a time (``StageModel``); ``Shooting`` holds each move
for ``block`` steps and plans ``n_stages`` moves to IPOPT's tolerance.

Stage k runs ``block`` steps from x_k with u_k held, visiting s_1..s_b, and
costs the sum of the step costs. Its next state is s_b augmented with the
outputs y(s_1)..y(s_b): algebraic components with no dynamics, one per
kinked term or limited quantity per visited state (G5). The rows on them are
linear, as the IPM needs:

- an epigraph term (T2), cost w t^p: t >= r - tol and t >= 0, plus
  t >= -r - tol when ``absolute``, one slack t per visited state;
- a hard limit (T3), lo <= q <= hi on every visited state, elastic: one
  slack per limit and stage, priced per unit at the trip cost.

The rows on stage k's outputs sit in stage k + 1 with its slacks, so every
row is stage-local (``stage_kkt_solve``). So, as in the compressor's NLP, one
stage past the horizon carries the last planned stage's rows. It costs
nothing, and its move is held to the last planned one by a cost that cannot
move the optimum, since that move reaches nothing costed. Stage 0's rows
would bind the measured state and are dropped.

A solve may plan fewer stages than it was built for (``active``), as the
horizon shrinks at the episode's end (H_k = min(P, N_ep - k)): the stages
past it cost nothing, carry no rows, hold their moves to the last planned
one in the same way and hold their state, since no plant runs past the
episode (G9), so one compiled problem serves every H_k.
"""

import functools
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from target_gym.oracles.ipm import (
    NLP,
    StageHessian,
    StageJacobian,
    StageRows,
    interior_point,
    stage_barrier,
    stage_kkt_solve,
    stage_multipliers,
    stage_shift,
)

#: IPOPT's iteration cap per solve, as the compressor's (G4: kept, capped
#: solves recorded); IPOPT's default tolerance (read: ``tol``).
MAX_ITER, TOL = 150, 1e-8
#: Barrier start and bound push: IPOPT's defaults cold, the compressor's
#: warm-start options warm (ours, inherited from its CasADi build).
COLD_MU, COLD_PUSH, WARM_MU, WARM_PUSH = 0.1, 1e-2, 1e-4, 1e-9
#: IPOPT's ``bound_relax_factor`` and ``nlp_scaling_max_gradient`` (read).
BOUND_RELAX, MAX_GRADIENT = 1e-8, 100.0


class Hinge(NamedTuple):
    """An epigraph term (T2): ``weight * max(r - tol, 0) ** power``, of
    ``abs(r)`` rather than r when ``absolute``."""

    weight: float = 1.0
    absolute: bool = False
    tol: float = 0.0
    power: int = 1


class Limit(NamedTuple):
    """A hard limit (T3) ``lo <= q <= hi``, elastic at ``price`` per unit."""

    lo: float = -np.inf
    hi: float = np.inf
    price: float = 1.0


class StageModel(NamedTuple):
    """A task's planner model. ``step(data, x, u, t)`` is one task step from
    state ``x`` under move ``u``, ``t`` steps into the plan, and returns
    ``(x_next, cost, y)``: the cost and outputs of the state reached, ``y``
    one argument r per hinge then one quantity q per limit. ``data`` is the
    solve's exogenous data, any pytree. ``lo`` and ``hi`` bound the move
    (T1: the model never clips it)."""

    step: Callable
    block: int
    lo: tuple
    hi: tuple
    hinges: tuple = ()
    limits: tuple = ()


class Certificate(NamedTuple):
    """One solve, as audit decision 6 stores it with each record row:
    IPOPT's final scaled KKT error (-1 when failed), the iterations, whether
    the cap stopped it above ``tol``, the largest row violation in the scaled
    problem, whether it failed, and the largest inertia shift it needed."""

    error: jnp.ndarray
    iterations: jnp.ndarray
    capped: jnp.ndarray
    violation: jnp.ndarray
    failed: jnp.ndarray
    shift: jnp.ndarray


class Plan(NamedTuple):
    """A plan in the task's units, z = (x_1..x_N, u_0..u_{N-1},
    e_0..e_{N-1}) with N = n_stages + 1, its multipliers, its cost."""

    z: jnp.ndarray
    lam: jnp.ndarray
    w: jnp.ndarray
    cost: jnp.ndarray
    certificate: Certificate


def stage_derivatives(stage, v, lam, weight):
    """``(F, l, dF, dl, H)`` of ``stage(v) = (F, l)``, H the exact Hessian
    of ``weight * l + lam'F`` (G1: Gauss-Newton did not converge on
    plane3d_circle), taken forward over forward (G2): ``jax.hessian``, which
    reverses through the integrators' scans, took 105 s and 10 GB to compile
    one copy there, this 14 s and 2.4 GB."""

    def lagrangian(v):
        F, l = stage(v)
        return weight * l + lam @ F

    F, l = stage(v)
    dF, dl = jax.jacfwd(stage)(v)
    return F, l, dF, dl, jax.jacfwd(jax.jacfwd(lagrangian))(v)


class Shooting:
    """The NLP of ``model`` over ``n_stages`` held moves, for ``nx`` states.

    ``solve(data, x0, u_hold, previous=None)`` plans from the measured state
    x0. Cold, it starts with every state at x0 and the moves ``u_hold`` that
    hold the actuators where they are, never from an expert (G8: multiple
    shooting needs no simulated start, which an unstable plant or a pump
    held off would run away from); warm, from ``previous`` and its
    multipliers shifted one stage. A failed solve returns the plan it
    started from, so its next move is the previous plan's next one. Jitted;
    run it under x64 for its tolerance.
    """

    def __init__(self, model: StageModel, n_stages: int, nx: int, max_iter=MAX_ITER):
        m, b = model, model.block
        self.model, self.S, self.N, self.nx = m, n_stages, n_stages + 1, nx
        self.max_iter = max_iter
        self.nu, self.nh, nl = len(m.lo), len(m.hinges), len(m.limits)
        self.ny, self.ne = self.nh + nl, b * self.nh + nl
        self.nxa = nx + b * self.ny
        N, nxa, nu, ne = self.N, self.nxa, self.nu, self.ne
        self.n = n = N * (nxa + nu + ne)
        self.xi = np.arange(N * nxa).reshape(N, nxa)
        self.ui = N * nxa + np.arange(N * nu).reshape(N, nu)
        self.ei = N * (nxa + nu) + np.arange(N * ne).reshape(N, ne)
        xk = np.concatenate([np.full((1, nxa), n), self.xi[:-1]])  # x_0: no variable
        self.dyn = np.concatenate([xk[:, :nx], self.ui], 1)  # what a stage reads
        self.loc = np.concatenate([xk, self.ui, self.ei], 1)  # what its rows read
        self.htol = np.array([h.tol for h in m.hinges], float)
        self.absolute = np.array([h.absolute for h in m.hinges], bool)
        lim = np.array([(q.lo, q.hi, q.price) for q in m.limits], float).reshape(nl, 3)
        self.lim_lo, self.lim_hi, self.price = lim.T
        self.solve = jax.jit(self._solve)
        #: A warm solve from ``previous`` as it stands, not shifted: the next
        #: rung of a cold solve's width continuation (K7).
        self.resolve = jax.jit(functools.partial(self._solve, shift=False))

    def unpack(self, z):
        N, a, b = self.N, self.N * self.nxa, self.N * (self.nxa + self.nu)
        return z[:a].reshape(N, -1), z[a:b].reshape(N, -1), z[b:].reshape(N, -1)

    def moves(self, plan: Plan):
        """The planned moves u_0..u_{S-1}; u_0 is applied for one T_c."""
        return self.unpack(plan.z)[1][: self.S]

    def states(self, plan: Plan):
        """The planned states x_1..x_S at the stage boundaries."""
        return self.unpack(plan.z)[0][: self.S, : self.nx]

    def shift(self, plan: Plan):
        """(z, lam, w) one stage on, the last stage repeated."""

        def sh(v):
            v = v.reshape(self.N, -1)
            return jnp.concatenate([v[1:], v[-1:]]).ravel()

        return (
            jnp.concatenate([sh(v) for v in self.unpack(plan.z)]),
            sh(plan.lam),
            sh(plan.w),
        )

    def _stage(self, data, x0, v, k):
        """(x_{k+1} with its outputs, stage cost) from v = (x_k, u_k)."""
        nx, b = self.nx, self.model.block

        def one(carry, j):
            x, acc = carry
            x1, c, y = self.model.step(data, x, v[nx:], k * b + j)
            return (x1, acc + c), jnp.asarray(y, x.dtype).reshape(self.ny)

        x = jnp.where(k == 0, x0, v[:nx])
        (x, cost), Y = jax.lax.scan(one, (x, jnp.zeros((), x.dtype)), jnp.arange(b))
        return jnp.concatenate([x, Y.ravel()]), cost

    def _split(self, xa, e):
        b, nh = self.model.block, self.nh
        Y = xa[self.nx :].reshape(b, self.ny)
        return Y[:, :nh], Y[:, nh:], e[: b * nh].reshape(b, nh), e[b * nh :]

    def _rows(self, xa, u, e, a):
        """A stage's rows, each >= 0 and affine in (x_k, u_k, e_k); ``a`` is
        0 on stage 0, whose rows then only bound its slacks."""
        R, Q, t, el = self._split(xa, e)
        lo, hi = (np.flatnonzero(np.isfinite(v)) for v in (self.lim_lo, self.lim_hi))
        return jnp.concatenate([
            u - jnp.asarray(self.model.lo), jnp.asarray(self.model.hi) - u,
            (t - a * (R - self.htol)).ravel(), t.ravel(),
            (t + a * (R + self.htol))[:, self.absolute].ravel(),
            (a * (Q[:, lo] - self.lim_lo[lo]) + el[lo]).ravel(),
            (a * (self.lim_hi[hi] - Q[:, hi]) + el[hi]).ravel(), el,
        ])  # fmt: skip

    def _need(self, xa, a):
        """The least slacks that meet the rows on ``xa``."""
        R, Q, _, _ = self._split(xa, jnp.zeros(self.ne))
        t = jnp.maximum(
            a * (R - self.htol), jnp.where(self.absolute, a * (-R - self.htol), 0.0)
        )
        lo, hi = (np.where(np.isfinite(v), v, 0.0) for v in (self.lim_lo, self.lim_hi))
        over = jnp.maximum(
            jnp.where(np.isfinite(self.lim_lo), lo - Q, 0.0),
            jnp.where(np.isfinite(self.lim_hi), Q - hi, 0.0),
        )
        el = a * jnp.max(over, axis=0, initial=0.0)
        return jnp.concatenate([jnp.maximum(t, 0.0).ravel(), el])

    def _e_cost(self, e):
        _, _, t, el = self._split(jnp.zeros(self.nxa), e)
        out = el @ self.price
        for i, h in enumerate(self.model.hinges):
            out = out + h.weight * jnp.sum(t[:, i] ** h.power)
        return out

    def _solve(self, data, x0, u_hold, previous=None, active=None, shift=True):
        N, nx, nxa, nu, n = self.N, self.nx, self.nxa, self.nu, self.n
        b, ui, ei, dyn, loc = self.model.block, self.ui, self.ei, self.dyn, self.loc
        cold = previous is None
        ks = jnp.arange(N)
        active = self.S if active is None else active
        wk = (ks < active).astype(x0.dtype)  # the costed stages
        act = ((ks > 0) & (ks <= active)).astype(x0.dtype)  # rows on their outputs
        pin = ((ks > 0) & (ks >= active)).astype(x0.dtype)  # moves reaching nothing
        x0a = jnp.concatenate([x0, jnp.zeros(nxa - nx, x0.dtype)])

        def stage(v, k):
            nxt, cost = self._stage(data, x0, v, k)
            return jnp.where(k >= active, nxt.at[:nx].set(v[:nx]), nxt), cost

        def V_of(z):
            xs, us, _ = self.unpack(z)
            return jnp.concatenate(
                [jnp.concatenate([x0a[None], xs[:-1]])[:, :nx], us], 1
            )

        def slacked(xs, us, es):
            """z of the states, moves and slacks, each slack raised to what
            its rows need."""
            need = jax.vmap(self._need)(jnp.concatenate([x0a[None], xs[:-1]]), act)
            return jnp.concatenate(
                [xs.ravel(), us.ravel(), jnp.maximum(es, need).ravel()]
            )

        def feasible(us, es):
            """The states re-simulated from the moves (the restoration)."""

            def advance(x, k):
                nxt = stage(jnp.concatenate([x, us[k]]), k)[0]
                return nxt[:nx], nxt

            return slacked(jax.lax.scan(advance, x0, ks)[1], us, es)

        if cold:
            # Every state at x0, each with the outputs of one stage from it.
            held = jnp.concatenate([x0, u_hold])
            xs = jax.vmap(lambda k: stage(held, k)[0])(ks).at[:, :nx].set(x0)
            z0 = slacked(xs, jnp.tile(u_hold, (N, 1)), jnp.zeros((N, self.ne)))
            lam0 = jnp.zeros(N * nxa)
        else:
            z0, lam0, w0 = self.shift(previous) if shift else previous[:3]

        def f(z):
            l = jax.vmap(stage)(V_of(z), ks)[1]
            return wk @ l + jnp.sum(jax.vmap(self._e_cost)(self.unpack(z)[2]))

        def c_fn(z):
            return (self.unpack(z)[0] - jax.vmap(stage)(V_of(z), ks)[0]).ravel()

        ne, nloc = self.ne, loc.shape[1]
        own = np.concatenate([np.arange(nx), nxa + np.arange(nu)])  # in v_k
        eye = jnp.broadcast_to(jnp.eye(nxa), (N, nxa, nxa))

        def derivs(z, lam, hess=True):
            """(grad f, H, J, c) in the task's units and stage blocks, H the
            Lagrangian's."""
            es, V = self.unpack(z)[2], V_of(z)
            if hess:

                def one(v, k, l, w):
                    return stage_derivatives(lambda v: stage(v, k), v, l, w)

                F, _, dF, dl, Hk = jax.vmap(one)(V, ks, lam.reshape(N, nxa), wk)
            else:
                F = jax.vmap(stage)(V, ks)[0]
                dF, dl = jax.vmap(jax.jacfwd(stage))(V, ks)
            gf = jnp.zeros(n + 1).at[dyn].add(wk[:, None] * dl)[:n]
            gf = gf.at[ei].add(jax.vmap(jax.grad(self._e_cost))(es))
            A = jnp.zeros((N, nxa, nxa)).at[:, :, :nx].set(-dF[:, :, :nx])
            J = StageJacobian(eye, A, -dF[:, :, nx:], ne)
            c = (self.unpack(z)[0] - F).ravel()
            if not hess:
                return gf, None, J, c
            Dk = jnp.zeros((N, nloc, nloc)).at[:, own[:, None], own[None, :]].set(Hk)
            e = nxa + nu + np.arange(ne)
            Dk = Dk.at[:, e[:, None], e[None, :]].set(
                jax.vmap(jax.hessian(self._e_cost))(es)
            )
            H = StageHessian(Dk, jnp.zeros((N, nu, nu + ne)), jnp.zeros((nxa, nxa)))
            return gf, H, J, c

        # IPOPT-style scaling. Variables by typical magnitudes: the measured
        # state, the start's outputs, the move range, and each limit slack in
        # cost units, so a trip-sized price does not set the objective's
        # scale. Shooting rows in their state's scale; the objective by its
        # largest gradient at the start; each inequality row by its largest
        # scaled gradient, so a bound is a bound on its scaled variable.
        dy = jnp.maximum(
            jnp.max(jnp.abs(self.unpack(z0)[0][:, nx:].reshape(N * b, -1)), 0), 1.0
        )
        dx = jnp.concatenate([jnp.maximum(jnp.abs(x0), 1.0), jnp.tile(dy, b)])
        du = jnp.maximum(
            jnp.abs(jnp.asarray(self.model.lo)), jnp.abs(jnp.asarray(self.model.hi))
        )
        de = jnp.concatenate([jnp.ones(b * self.nh), 1.0 / jnp.asarray(self.price)])
        D = jnp.concatenate([jnp.tile(dx, N), jnp.tile(du, N), jnp.tile(de, N)])
        Rs, Dl = D[: N * nxa], jnp.concatenate([D, jnp.ones(1)])[loc]  # (N, nloc)
        Rk = Rs.reshape(N, nxa)
        zeros = [jnp.zeros((N, nxa)), jnp.zeros((N, nu)), jnp.zeros((N, ne))]
        Gk = jax.vmap(jax.jacfwd(self._rows, argnums=(0, 1, 2)))(*zeros, act)
        Gk = jnp.concatenate(Gk, axis=-1)  # (N, rows, nloc)
        Gh = Gk * Dl[:, None, :]
        big = jnp.max(jnp.abs(Gh), axis=-1)
        rs = jnp.where(
            jnp.sum(Gk != 0, -1) == 1, 1.0 / big, jnp.minimum(1.0, MAX_GRADIENT / big)
        )
        Gh, mrow = Gh * rs[:, :, None], Gk.shape[1]
        G = StageRows(Gh[..., :nxa], Gh[..., nxa : nxa + nu], Gh[..., nxa + nu :])
        rs, g0k = rs.ravel(), (jax.vmap(self._rows)(*zeros, act) * rs).ravel()
        g0 = g0k + BOUND_RELAX * jnp.maximum(1.0, jnp.abs(g0k))

        def held(zh):
            """The moves past the plan, each held to the one before."""
            dU = zh[ui[1:]] - zh[ui[:-1]]
            return dU * pin[1:, None]

        osc = jnp.minimum(
            1.0, MAX_GRADIENT / jnp.max(jnp.abs(D * derivs(z0, None, False)[0]))
        )

        def scaled(zh, lam, hess=True):
            gf, H, J, c = derivs(D * zh, lam / (osc * Rs), hess)
            d = 2.0 * held(zh)
            gf = (osc * D * gf).at[ui[1:]].add(d).at[ui[:-1]].add(-d)
            J = StageJacobian(
                J.E / Rk[:, :, None] * Rk[:, None, :],
                J.A / Rk[:, :, None] * Dl[:, None, :nxa],
                J.B / Rk[:, :, None] * Dl[:, None, nxa : nxa + nu], ne,
            )  # fmt: skip
            if hess:
                u = nxa + np.arange(nu)
                Dk = osc * Dl[:, :, None] * H.D * Dl[:, None, :]
                both = 2.0 * (pin + jnp.concatenate([pin[1:], jnp.zeros(1)]))
                Dk = Dk.at[:, u, u].add(both[:, None])
                P = H.P.at[:, np.arange(nu), np.arange(nu)].add(-2.0 * pin[:, None])
                H = StageHessian(Dk, P, osc * H.end)
            return gf, H, J, c / Rs

        def restore(zh):
            _, us, es = self.unpack(D * zh)
            return feasible(us, es) / D

        nlp = NLP(
            f=lambda zh: osc * f(D * zh) + jnp.sum(held(zh) ** 2),
            c=lambda zh: c_fn(D * zh) / Rs, derivs=scaled, G=G, g0=g0, restore=restore,
            kkt=stage_kkt_solve, barrier=stage_barrier, shift=stage_shift,
        )  # fmt: skip
        zh = z0 / D
        if cold:
            zh = zh.at[ei].add(COLD_PUSH)
            w = jnp.ones(N * mrow)
            gf, _, J, _ = scaled(zh, jnp.zeros(N * nxa), False)
            lam = stage_multipliers(J, gf - G.T @ w)
            lam = jnp.where(jnp.max(jnp.abs(lam)) > 1e3, 0.0, lam)  # IPOPT's cap
            w0 = jnp.zeros(N * mrow)
        else:
            zh = zh.at[ei].set(jnp.maximum(zh[ei], WARM_PUSH))
            w, lam = jnp.maximum(w0 * osc / rs, WARM_PUSH), lam0 * osc * Rs
        s = jnp.maximum(G @ zh + g0, COLD_PUSH if cold else WARM_PUSH)
        r = interior_point(
            nlp, zh, s, lam, w, mu0=COLD_MU if cold else WARM_MU,
            max_iter=self.max_iter, tol=TOL, adaptive=not cold,
        )  # fmt: skip

        failed = ~(r.error >= 0.0) | ~jnp.all(jnp.isfinite(r.z))
        cert = Certificate(
            error=jnp.where(failed, -1.0, r.error), iterations=r.iterations,
            capped=~failed & (r.iterations >= self.max_iter) & (r.error > TOL),
            violation=jnp.maximum(jnp.max(jnp.abs(nlp.c(r.z))),
                                  jnp.max(-(G @ r.z + g0k), initial=0.0)),
            failed=failed, shift=r.shift,
        )  # fmt: skip
        z = jnp.where(failed, z0, D * r.z)
        lam = jnp.where(failed, lam0, r.lam / (osc * Rs))
        return Plan(z, lam, jnp.where(failed, w0, r.w * rs / osc), f(z), cert)
