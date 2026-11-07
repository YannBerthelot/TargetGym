"""Judging step 3's ceiling: a whole-episode plan on the rebuilt furnace.

It knows the true model, the true state and the whole future, because it rolls
``step_env`` itself on the episode's own keys, so the pull, the gas and the moves are
the ones that will happen. The move is the valve's path at knots; a penalty keeps it
inside the slew, and the reported cost is a clean rollout of the command, in which
the actuator enforces the slew exactly. Convergence is shown by refining the knots.
"""

import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from target_gym.glass_furnace import rebuilt as B  # noqa: E402

env = B.RebuiltFurnace()
SKIP = 240


def rollout(seed, u, p):
    key = jax.random.PRNGKey(seed)
    ob, st = env.reset_env(key, p)

    def f(s, tu):
        t, a = tu
        _, s, r, _, info = env.step_env(jax.random.fold_in(key, t), s, a[None], p)
        return s, (-r, s.T_crown - s.target_T_crown, s.T_crown_meas - s.target_T_crown, info["tripped"])

    return jax.lax.scan(f, st, (jnp.arange(u.shape[0]), u))[1]


def rate_step(p):
    return p.actuator_rate * 2.0 * p.delta_t


def plan(seed, u0, p, every, iters):
    """L-BFGS on the valve at knots ``every`` steps apart, from ``u0``, all steps.
    The objective is the whole episode's cost, the score's window included."""
    n = u0.shape[0]
    grid = jnp.arange(n, dtype=jnp.float64)
    knots = jnp.arange(0, n + every, every, dtype=jnp.float64)
    limit = rate_step(p)

    def expand(v):
        return jnp.interp(grid, knots, v)

    def objective(v):
        u = expand(v)
        cost = rollout(seed, u, p)[0][SKIP:].mean()
        d = jnp.diff(u)
        over = jnp.maximum(jnp.abs(d) - limit, 0.0) / limit
        return cost + 1e2 * jnp.mean(over**2)

    vg = jax.jit(jax.value_and_grad(objective))
    v0 = np.interp(np.asarray(knots), np.arange(n), np.asarray(u0))

    def fun(v):
        val, g = vg(jnp.asarray(v))
        return float(val), np.asarray(g, dtype=np.float64)

    res = minimize(
        fun, v0, jac=True, method="L-BFGS-B", bounds=[(-1.0, 1.0)] * v0.size,
        options=dict(maxiter=iters, maxcor=30, ftol=1e-10, gtol=1e-7),
    )  # fmt: skip
    return np.asarray(expand(jnp.asarray(res.x))), res


def true_cost(seed, u, p):
    y = jax.jit(lambda a: rollout(seed, a, p))(jnp.asarray(u))
    c = np.asarray(y[0])[SKIP:].mean()
    e = np.asarray(y[1])[SKIP:]
    return float(c), float(np.sqrt((e**2).mean())), int(np.asarray(y[3]).sum())


if __name__ == "__main__":
    npz = np.load(sys.argv[1])
    u_all = npz["lead_u"]
    seed_index = int(sys.argv[2])
    seed = 100 + seed_index
    out = sys.argv[3]
    p = B.RebuiltParams()
    u = np.asarray(u_all[seed_index, :, 0] if u_all.ndim == 3 else u_all[seed_index], dtype=np.float64)
    print("start", true_cost(seed, u, p), flush=True)
    for every, iters in ((20, 300), (5, 300), (2, 300), (1, 300)):
        t0 = time.time()
        u, res = plan(seed, u, p, every, iters)
        print(
            f"knots every {every * 30} s: {true_cost(seed, u, p)} objective {res.fun:.4f}"
            f" iters {res.nit} {res.message} {time.time() - t0:.0f} s",
            flush=True,
        )
        np.save(out, u)
