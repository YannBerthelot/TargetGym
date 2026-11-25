"""Glass room study, probe 1: the design state's fields, the step's speed, and the
steady map from the valve to the fining share, the crown and the canal.

Run: PYTHONPATH=src python scripts/glass_room/probe1.py <out.json>
"""

import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from target_gym.glass_furnace import env_jax as E
from target_gym.glass_furnace import physical as P

np.set_printoptions(precision=3, suppress=True, linewidth=220)
env = E.GlassFurnace()
p = env.default_params
quiet = p.replace(walk_sd=(0.0,) * 5, sensor_noise_scale=0.0)
a = P.layout(p.grid)
out = {}

# ---- the design state
fuel0, cooling0, fields = E.design_state()
s0 = P.Fields(**{k: jnp.asarray(v) for k, v in fields.items()})
print("design fuel", fuel0, "cooling", cooling0)
x = np.cumsum(np.asarray(a.dx)) - 0.5 * np.asarray(a.dx)
z = np.cumsum(np.asarray(a.dz)) - 0.5 * np.asarray(a.dz)
print("x centres", x)
print("z centres", z, "dz", np.asarray(a.dz))
print("T melt, C, layers down, columns along:")
print(np.asarray(s0.T) - 273.15)
print("fining share:")
print(np.asarray(s0.fining))
print("crown hot face C", np.asarray(s0.T_roof[0]) - 273.15)
print("blanket m", np.asarray(s0.m), "cover", np.asarray(P.cover(s0.m)))
print("foam", np.asarray(s0.foam))
new, books = P.plant_step(
    s0, fuel0, p.pull, p.cullet, p.thickness, cooling0, p.grid, p.plant
)
u = np.asarray(books["u"])  # (layers, interior faces) m/s
print("u m/h at faces:")
print(u * 3600.0)
F = u * np.asarray(a.dz)[:, None] * np.asarray(a.wf[1:-1])  # m3/s per layer
fwd = np.where(F > 0, F, 0).sum(0) * p.plant.rho_glass
print("forward mass flow over pull at each face", fwd / p.pull)
read = P.sensors(s0, p.grid, p.plant)
print({k: float(v) for k, v in read.items()})
print(
    "books in/out MW",
    float(books["in"]) / 1e6,
    float(books["out"]) / 1e6,
    "stack C",
    float(books["T_stack"]),
    "air C",
    float(books["T_air"]),
)
out["design"] = dict(
    fuel=fuel0,
    cooling=cooling0,
    T=np.asarray(s0.T).tolist(),
    X=np.asarray(s0.fining).tolist(),
    u=u.tolist(),
    crown=np.asarray(s0.T_roof[0]).tolist(),
)

# ---- timing and the steady map: the env's own step, walks off, valve held
key = jax.random.PRNGKey(0)
_, st0 = env.reset_env(key, quiet)


def held(fuel, n, params):
    def f(s, k):
        _, s1, r, _, _ = env.step_env(k, s, E.to_action(fuel, params), params)
        read = P.sensors(s1.plant, params.grid, params.plant)
        return s1, (-r, read["fining"], read["hot"], read["canal"], s1.cooling, s1.fuel)

    keys = jax.random.split(jax.random.PRNGKey(1), n)
    return jax.lax.scan(f, st0, keys)


run = jax.jit(lambda fuel: held(fuel, 5040, quiet))
t = time.time()
r = run(jnp.asarray(fuel0))
jax.block_until_ready(r)
print("compile + 5040 steps", time.time() - t)
t = time.time()
r = run(jnp.asarray(fuel0 + 0.001))
jax.block_until_ready(r)
dt = time.time() - t
print("5040 steps", dt, "s;", 5040 / dt, "steps/s")
out["steps_per_s"] = 5040 / dt

fuels = jnp.asarray(np.round(np.linspace(0.44, 0.56, 13), 4))
n = 30 * 720
long = jax.jit(jax.vmap(lambda fuel: held(fuel, n, quiet)[1]))
t = time.time()
cost, fin, hot, canal, cool, got = (np.asarray(v) for v in long(fuels))
print("steady map", time.time() - t, "s")
rows = []
for i, f in enumerate(np.asarray(fuels)):
    row = dict(
        asked=float(f),
        got=float(got[i, -1]),
        fining=float(fin[i, -1]),
        fining_10d=float(fin[i, 7200]),
        hot_C=float(hot[i, -1] - 273.15),
        canal_C=float(canal[i, -1] - 273.15),
        cooling_MW=float(cool[i, -1] / 1e6),
        cost=float(cost[i, -720:].mean()),
    )
    rows.append(row)
    print(row)
out["steady"] = rows
if len(sys.argv) > 1:
    json.dump(out, open(sys.argv[1], "w"))
