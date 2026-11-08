"""Attempt 3's oracle: the fuel a controller with foresight would fire.

    python glass_slow_oracle.py SETTLED.npz SCENARIO SEED [iterations] [knot_hours]

It knows the true plant, its state and every walk and event of the seed's episode, and
chooses the valve as a piecewise-constant path (one value per ``knot_hours``) by Adam on
the gradient through the whole episode. Lost glass is a step in cost, so the gradient
uses a smooth step of it; every cost reported is the true one, from the environment's own
``reward_terms``. Converged is shown, not assumed: two starts, the design valve and the
middle of the box, must end within 0.5 % of each other, with the last tenth of the
iterations flat.
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from target_gym.glass_furnace import physical as P
from target_gym.glass_furnace import physical_env as E
from target_gym.glass_furnace.physical_scenarios import suite

d = np.load(sys.argv[1])
F0, Q0 = float(d["fuel"]), float(d["cooling"])
S0 = P.Fields(**{k: jnp.asarray(d[k]) for k in d if k not in ("fuel", "cooling")})
SCENARIO = {s.name: s for s in suite()}[sys.argv[2]]
SEED = int(sys.argv[3])
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 400
KNOT_H = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
EP = E.EnvParams().replace(**SCENARIO.overrides)
N = EP.plant.max_steps_in_episode
PER = int(round(KNOT_H * 3600.0 / EP.plant.delta_t))
KNOTS = -(-N // PER)
LO, HI = EP.fuel_min, EP.fuel_max


def valve(z):
    return LO + (HI - LO) * jax.nn.sigmoid(z)


def costs(z, soft):
    """Episode cost, EUR: smooth (for the gradient) or true."""
    k0, k1 = jax.random.split(jax.random.split(jax.random.PRNGKey(0), 64)[SEED])
    _, st = E.reset(k0, S0, EP, fuel=F0, cooling=Q0)
    u = jnp.repeat(valve(z), PER)[:N]
    step = jax.checkpoint(lambda st, k, a: E.step(k, st, a, EP))

    def f(st, x):
        k, a = x
        _, st, _, c = step(st, k, a)
        if not soft:
            return st, c["fuel"] + c["lost_glass"]
        r = P.sensors(st.fields, EP.grid, EP.plant)
        glass = st.pull * EP.plant.delta_t / 1000.0 * EP.lost_glass_price
        ok = jax.nn.sigmoid((r["fining"] - EP.fining_pass) / 0.005) * jax.nn.sigmoid(
            (r["canal"] - EP.canal_target) / 0.5
        )
        return st, c["fuel"] + glass * (1.0 - ok)

    return jnp.sum(jax.lax.scan(f, st, (jax.random.split(k1, N), u))[1])


def solve(start):
    z = jnp.full((KNOTS,), jnp.log((start - LO) / (HI - start)))
    opt = optax.adam(optax.exponential_decay(0.1, ITERS, 0.1))
    state = opt.init(z)
    grad = jax.jit(jax.value_and_grad(lambda z: costs(z, True)))
    true = jax.jit(lambda z: costs(z, False))
    trace = []
    for i in range(ITERS):
        v, g = grad(z)
        upd, state = opt.update(g, state)
        z = optax.apply_updates(z, upd)
        if i % 10 == 0 or i == ITERS - 1:
            trace.append(float(true(z)))
    return z, trace


def main():
    days = N * EP.plant.delta_t / 86400.0
    t0 = time.time()
    out = {}
    for name, start in (("design", F0), ("box mid", 0.5 * (LO + HI))):
        z, trace = solve(start)
        out[name] = (z, trace)
        tail = np.array(trace[-max(len(trace) // 10, 2) :])
        print(
            f"{SCENARIO.name} seed {SEED} start {name}: EUR/day {trace[-1] / days:.0f},"
            f" last tenth spread {100 * np.ptp(tail) / tail.mean():.2f} %,"
            f" mean valve {float(valve(z).mean()):.4f} kg/s, {time.time() - t0:.0f} s",
            flush=True,
        )
    a, b = (out[k][1][-1] for k in out)
    best = min(a, b)
    print(
        f"{SCENARIO.name} seed {SEED}: oracle EUR/day {best / days:.0f}; starts agree to"
        f" {100 * abs(a - b) / best:.2f} % ({'converged' if abs(a - b) / best < 0.005 else 'NOT converged'})",
        flush=True,
    )
    z = out["design"][0] if a <= b else out["box mid"][0]
    np.save(
        sys.argv[1].replace(".npz", f"_oracle_{SCENARIO.name}_{SEED}.npy"),
        np.asarray(valve(z)),
    )


if __name__ == "__main__":
    main()
