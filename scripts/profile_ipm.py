"""Where an interior-point iteration of the reactor's ceiling solve spends its time.

Builds the whole-day NLP of ``reactor_ipopt.py`` over a short horizon and times,
after compiling each, the pieces one iteration calls: the stage simulation (c and
f, the line search's trials), the stage Jacobians, the exact stage Hessians
(forward over forward, as ``stage_derivatives``, and forward over reverse), the
Riccati KKT solve, and whole solves of a few iterations. The wobble path is zero
here; it changes values, not cost.

    uv run python scripts/profile_ipm.py --steps 24 96 --iters 3 8
"""

import argparse
import json
import pathlib
import resource
import sys
import time

import jax
import jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from reactor_ipopt import WIDTH, build  # noqa: E402

from target_gym.oracles.controller import enable_x64, float64  # noqa: E402
from target_gym.oracles.ipm import (  # noqa: E402
    StageHessian,
    StageJacobian,
    stage_kkt_solve,
)
from target_gym.oracles.shooting import stage_derivatives  # noqa: E402
from target_gym.reactor import Reactor, ReactorParams  # noqa: E402


def timed(fn, *args, reps=3):
    """(compile + first run, median of ``reps`` later runs) in seconds."""
    t0 = time.time()
    jax.block_until_ready(fn(*args))
    first = time.time() - t0
    runs = []
    for _ in range(reps):
        t0 = time.time()
        jax.block_until_ready(fn(*args))
        runs.append(time.time() - t0)
    return round(first, 2), round(sorted(runs)[len(runs) // 2], 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=[24])
    ap.add_argument("--iters", type=int, nargs="+", default=[3, 8])
    ap.add_argument("--skip-solve", action="store_true")
    ap.add_argument("--skip-ff", action="store_true")
    ap.add_argument(
        "--warm", action="store_true", help="time warm (adaptive) re-solves instead"
    )
    args = ap.parse_args()
    env, p = Reactor(), ReactorParams()
    _, state = env.reset_env(jax.random.PRNGKey(102), p)
    with enable_x64():
        q, st = float64(p), float64(state)
        for steps in args.steps:
            model, sh = build(env, q, steps, 1, max(args.iters))
            data = {
                "task": model.data(st), "end": steps,
                "path": jnp.zeros(steps + 2), "widths": jnp.asarray([WIDTH]),
                "boron_width": jnp.asarray(0.01),
            }  # fmt: skip
            x0, hold = model.state_x(st), model.hold(st)
            N, nx, nxa, nu = sh.N, sh.nx, sh.nxa, sh.nu
            out = {"steps": steps, "N": N, "n": sh.n, "nxa": nxa, "ne": sh.ne,
                   "nloc": sh.loc.shape[1]}  # fmt: skip
            ks = jnp.arange(N)
            V = jnp.tile(jnp.concatenate([x0, hold]), (N, 1))
            lam = jnp.ones((N, nxa)) * 1e-3
            wk = jnp.ones(N)

            def stage(v, k):
                return sh._stage(data, x0, v, k)

            sim = jax.jit(jax.vmap(stage))
            jac = jax.jit(jax.vmap(jax.jacfwd(stage)))

            def ff(v, k, l, w):
                return stage_derivatives(lambda v: stage(v, k), v, l, w)[-1]

            def fr(v, k, l, w):
                def lag(v):
                    F, c = stage(v, k)
                    return w * c + l @ F

                return jax.jacfwd(jax.grad(lag))(v)

            def rf(v, k, l, w):
                def lag(v):
                    F, c = stage(v, k)
                    return w * c + l @ F

                def row(e):
                    return jax.grad(lambda v: jax.jvp(lag, (v,), (e,))[1])(v)

                return jax.vmap(row)(jnp.eye(v.shape[0], dtype=v.dtype))

            out["sim"] = timed(sim, V, ks)
            out["hess_rf"] = timed(jax.jit(jax.vmap(rf)), V, ks, lam, wk)
            out["jac"] = timed(jac, V, ks)
            if not args.skip_ff:
                out["hess_ff"] = timed(jax.jit(jax.vmap(ff)), V, ks, lam, wk)
            out["hess_fr"] = timed(jax.jit(jax.vmap(fr)), V, ks, lam, wk)
            if not args.skip_ff:
                a = jax.jit(jax.vmap(ff))(V, ks, lam, wk)
                b = jax.jit(jax.vmap(fr))(V, ks, lam, wk)
                out["hess_rel_diff"] = float(
                    jnp.max(jnp.abs(a - b)) / jnp.maximum(jnp.max(jnp.abs(a)), 1e-300)
                )
            nloc = sh.loc.shape[1]
            K = StageHessian(
                jnp.broadcast_to(jnp.eye(nloc), (N, nloc, nloc)) * 2.0,
                jnp.zeros((N, nu, nu + sh.ne)), jnp.eye(nxa),
            )  # fmt: skip
            dF = jac(V, ks)[0]
            A = jnp.zeros((N, nxa, nxa)).at[:, :, :nx].set(-dF[:, :, :nx])
            J = StageJacobian(
                jnp.broadcast_to(jnp.eye(nxa), (N, nxa, nxa)), A, -dF[:, :, nx:], sh.ne
            )
            rz, rc = jnp.ones(sh.n), jnp.ones(N * nxa)
            out["kkt"] = timed(jax.jit(stage_kkt_solve), K, J, rz, rc)
            if not args.skip_solve:
                for it in args.iters:
                    _, s2 = build(env, q, steps, 1, it)
                    if args.warm:  # a warm, adaptive re-solve from a 5-iteration plan
                        _, s5 = build(env, q, steps, 1, 5)
                        start = s5.solve(data, x0, hold)

                        def run(s2=s2, start=start):
                            return s2.resolve(data, x0, hold, start)

                    else:

                        def run(s2=s2):
                            return s2.solve(data, x0, hold)

                    t0 = time.time()
                    plan = run()
                    jax.block_until_ready(plan.z)
                    first = time.time() - t0
                    t0 = time.time()
                    plan = run()
                    jax.block_until_ready(plan.z)
                    c = plan.certificate
                    out[f"solve_{it}"] = {
                        "first": round(first, 1), "run": round(time.time() - t0, 2),
                        "iterations": int(c.iterations), "error": float(c.error),
                    }  # fmt: skip
            out["maxrss_gb"] = round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2
            )
            print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
