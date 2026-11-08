"""Attempt 3's oracle: the fuel a controller with foresight would fire.

    python glass_slow_oracle.py SETTLED.npz SCENARIO SEED [iterations] [knot_hours]

It knows the true plant, its state and every walk and event of the seed's episode. First
it finds the best constant valve in hindsight on a grid; then it shapes the valve as a
piecewise-constant path (one value per ``knot_hours``) by Adam on the gradient through the
whole episode, from that constant and from the design valve. Lost glass is a step in
cost, so the gradient sees a smooth step whose width shrinks over the run; every cost
reported is the true one from the environment's ``reward_terms``, and each start keeps
its best true path, so the oracle is never worse than the best constant. A step whose
gradient is not finite is skipped and counted. Converged is shown, not assumed: the two
starts must end within 0.5 % of each other.
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


def unvalve(u):
    u = jnp.clip(u, LO + 1e-4, HI - 1e-4)
    return jnp.log((u - LO) / (HI - u))


def costs(z, width):
    """Episode cost, EUR: with ``width`` 0 the true one, else a smooth step of that
    width on the fining share (and 100 times it, in K, on the canal)."""
    k0, k1 = jax.random.split(jax.random.split(jax.random.PRNGKey(0), 64)[SEED])
    _, st = E.reset(k0, S0, EP, fuel=F0, cooling=Q0)
    u = jnp.repeat(valve(z), PER)[:N]
    step = jax.checkpoint(lambda st, k, a: E.step(k, st, a, EP))

    def f(st, x):
        k, a = x
        _, st, _, c = step(st, k, a)
        r = P.sensors(st.fields, EP.grid, EP.plant)
        glass = st.pull * EP.plant.delta_t / 1000.0 * EP.lost_glass_price
        w = jnp.maximum(width, 1e-9)
        ok = jax.nn.sigmoid((r["fining"] - EP.fining_pass) / w) * jax.nn.sigmoid(
            (r["canal"] - EP.canal_target) / (100.0 * w)
        )
        smooth = c["fuel"] + glass * (1.0 - ok)
        return st, jnp.where(width > 0, smooth, c["fuel"] + c["lost_glass"])

    return jnp.sum(jax.lax.scan(f, st, (jax.random.split(k1, N), u))[1])


TRUE = jax.jit(lambda z: costs(z, 0.0))
GRAD = jax.jit(jax.value_and_grad(costs))


def best_constant():
    grid = jnp.linspace(0.40, 0.60, 41)
    c = np.asarray(
        jax.jit(jax.vmap(lambda u: costs(jnp.full((KNOTS,), unvalve(u)), 0.0)))(grid)
    )
    return float(grid[int(np.argmin(c))]), float(c.min())


def solve(start):
    z = jnp.full((KNOTS,), unvalve(jnp.asarray(start)))
    opt = optax.adam(optax.exponential_decay(0.05, ITERS, 0.1))
    state = opt.init(z)
    best_z, best = z, float(TRUE(z))
    skipped = 0
    for i in range(ITERS):
        width = 0.02 * (0.1 ** (i / max(ITERS - 1, 1)))  # 0.02 down to 0.002
        _, g = GRAD(z, width)
        if not bool(jnp.all(jnp.isfinite(g))):
            skipped += 1
            continue
        upd, state = opt.update(g, state)
        z = optax.apply_updates(z, upd)
        if i % 10 == 9 or i == ITERS - 1:
            v = float(TRUE(z))
            if v < best:
                best_z, best = z, v
    return best_z, best, skipped


def main():
    days = N * EP.plant.delta_t / 86400.0
    t0 = time.time()
    c_best, c_cost = best_constant()
    print(
        f"{SCENARIO.name} seed {SEED}: best constant {c_best:.3f} kg/s, EUR/day {c_cost / days:.0f}",
        flush=True,
    )
    out = {}
    for name, start in (("best constant", c_best), ("design", F0)):
        z, best, skipped = solve(start)
        out[name] = (z, best)
        print(
            f"{SCENARIO.name} seed {SEED} start {name}: EUR/day {best / days:.0f},"
            f" mean valve {float(valve(z).mean()):.4f} kg/s, skipped {skipped},"
            f" {time.time() - t0:.0f} s",
            flush=True,
        )
    a, b = (out[k][1] for k in out)
    best = min(a, b)
    print(
        f"{SCENARIO.name} seed {SEED}: oracle EUR/day {best / days:.0f}; best constant"
        f" {c_cost / days:.0f}; starts agree to {100 * abs(a - b) / best:.2f} %"
        f" ({'converged' if abs(a - b) / best < 0.005 else 'NOT converged'})",
        flush=True,
    )
    z = out["best constant"][0] if a <= b else out["design"][0]
    np.save(
        sys.argv[1].replace(".npz", f"_oracle_{SCENARIO.name}_{SEED}.npy"),
        np.asarray(valve(z)),
    )


if __name__ == "__main__":
    main()
