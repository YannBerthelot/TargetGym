"""Which stage Hessian is right: forward over forward, forward over reverse, or
central differences of the reverse gradient (x64)."""

import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from reactor_ipopt import WIDTH, build  # noqa: E402

from target_gym.oracles.controller import enable_x64, float64  # noqa: E402
from target_gym.reactor import Reactor, ReactorParams  # noqa: E402

env, p = Reactor(), ReactorParams()
_, state = env.reset_env(jax.random.PRNGKey(102), p)
with enable_x64():
    q, st = float64(p), float64(state)
    model, sh = build(env, q, 8, 1, 3)
    data = {
        "task": model.data(st),
        "end": 8,
        "path": jnp.zeros(10),
        "widths": jnp.asarray([WIDTH]),
        "boron_width": jnp.asarray(0.01),
    }
    x0, hold = model.state_x(st), model.hold(st)
    v = jnp.concatenate([x0, hold + jnp.array([0.01, 0.02])])
    lam = jax.random.normal(jax.random.PRNGKey(0), (sh.nxa,))
    k = 3

    def lag(v):
        F, c = sh._stage(data, x0, v, k)
        return c + lam @ F

    Hff = np.asarray(jax.jit(jax.jacfwd(jax.jacfwd(lag)))(v))
    Hfr = np.asarray(jax.jit(jax.jacfwd(jax.grad(lag)))(v))
    Hrr = np.asarray(jax.jit(jax.jacrev(jax.jacfwd(lag)))(v))
    Hrf = np.asarray(
        jax.jit(
            lambda v: jax.vmap(
                lambda e: jax.grad(lambda v: jax.jvp(lag, (v,), (e,))[1])(v)
            )(jnp.eye(len(v)))
        )(v)
    )
    g = jax.jit(jax.grad(lag))
    scale = np.maximum(np.abs(np.asarray(v)), 1.0)
    Hfd = np.zeros_like(Hff)
    for i in range(len(v)):
        h = 1e-5 * scale[i]
        e = jnp.zeros_like(v).at[i].set(h)
        Hfd[:, i] = (np.asarray(g(v + e)) - np.asarray(g(v - e))) / (2 * h)
    ref = np.max(np.abs(Hfd))
    for name, H in [
        ("ff", Hff),
        ("fr", Hfr),
        ("rev(fwd)", Hrr),
        ("vmap grad jvp", Hrf),
    ]:
        d = np.abs(H - Hfd)
        i, j = np.unravel_index(np.argmax(d), d.shape)
        print(
            name,
            "max|H-FD|/max|FD| =",
            d.max() / ref,
            "at",
            (i, j),
            "asym =",
            np.max(np.abs(H - H.T)) / ref,
        )
