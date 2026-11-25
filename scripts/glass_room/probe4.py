"""Glass room study, probe 4: the unfined share of the glass leaving when fining is
faster than the tank's transit, as real seeds need (many e-folds), and how steeply it
then answers fuel, pull and sulfate. Valve held, walks off, 30 days.

Run: PYTHONPATH=src python scripts/glass_room/probe4.py <out.json>
"""

import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import lib as L  # noqa: E402

quiet = L.scenario_params("normal", walk_sd=(0.0,) * 5, sensor_noise_scale=0.0)
H, N = L.HOUR, 40 * 24 * L.HOUR
a = L.P.layout(quiet.grid)
keys = jax.random.split(jax.random.PRNGKey(3), N)
_, st0 = L.ENV.reset_env(jax.random.PRNGKey(0), quiet)


def settle(fuel, tau, pull, sulfate):
    p = quiet.replace(
        pull=pull, sulfate=sulfate, plant=quiet.plant.replace(fining_time=tau)
    )

    def f(s, k):
        _, s1, _, _, _ = L.ENV.step_env(k, s, L.E.to_action(fuel, p), p)
        return s1, None

    s, _ = jax.lax.scan(f, st0, keys)
    X = s.plant.fining
    U = 1.0 - X
    out_U = jnp.sum(U[:, -1] * a.dz) / jnp.sum(a.dz)
    read = L.P.sensors(s.plant, p.grid, p.plant)
    return out_U, U[:, -1], U[:, 12], read["hot"], s.fuel, read["canal"]


run = jax.jit(jax.vmap(settle, in_axes=(0, 0, 0, 0)))
f0, p0, s0 = 0.4772, float(quiet.pull), float(quiet.sulfate)
cases = []
for tau in (3600.0, 1200.0, 600.0, 300.0, 150.0, 60.0):
    for fuel in (0.468, 0.4725, 0.4772, 0.482, 0.4868):
        cases.append((f"tau {tau:.0f} fuel {fuel}", fuel, tau, p0, s0))
    cases.append((f"tau {tau:.0f} pull +5%", f0, tau, p0 * 1.05, s0))
    cases.append((f"tau {tau:.0f} pull -5%", f0, tau, p0 * 0.95, s0))
    cases.append((f"tau {tau:.0f} sulfate +17%", f0, tau, p0, s0 * 1.17))
    cases.append((f"tau {tau:.0f} sulfate -17%", f0, tau, p0, s0 * 0.83))
arr = [jnp.asarray([c[i] for c in cases], jnp.float32) for i in (1, 2, 3, 4)]
out_U, U_exit, U_mid, hot, fuel, canal = (np.asarray(v) for v in run(*arr))
res = {}
for i, c in enumerate(cases):
    res[c[0]] = dict(
        U=float(out_U[i]),
        lnU=float(np.log(max(out_U[i], 1e-12))),
        hot_C=float(hot[i] - 273.15),
        fuel=float(fuel[i]),
        canal_C=float(canal[i] - 273.15),
        U_exit_layers=[float(x) for x in U_exit[i]],
    )
    print(
        c[0],
        "U %.3e lnU %.2f hot %.1f fuel %.4f"
        % (out_U[i], np.log(max(out_U[i], 1e-12)), hot[i] - 273.15, fuel[i]),
        "exit layers",
        np.array2string(U_exit[i], precision=2),
        flush=True,
    )
json.dump(res, open(sys.argv[1], "w"), indent=1)
