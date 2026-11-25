"""Glass room study, probe 3: with the crown held and no quality feedback, how far each
hidden disturbance moves the fining share, and how fast.

Run: PYTHONPATH=src python scripts/glass_room/probe3.py <out.json>
"""

import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import lib as L  # noqa: E402

row = L.ladder_row()
quiet = L.scenario_params("normal", walk_sd=(0.0,) * 5, sensor_noise_scale=0.0)
H = L.HOUR
res = {}

# ---- settle under the full cascade at its placed setpoint, to find the crown reading
# it holds; then freeze the crown setpoint there (no quality feedback).
full = L.cascade(bias=row.balance)
keys = jax.random.split(jax.random.PRNGKey(7), 20 * 24 * H)
carry, tr = jax.jit(lambda: L.roll(full, quiet, L.start(full, quiet, 0), keys))()
crown_sp = float(np.asarray(tr["crown"])[-24 * H :].mean())
x0 = float(np.asarray(tr["X"])[-1])
print(
    "settled: crown couple",
    crown_sp,
    "X",
    x0,
    "valve",
    float(np.asarray(tr["valve"])[-1]),
    "hot C",
    float(np.asarray(tr["hot"])[-1]) - 273.15,
    flush=True,
)
res["settled"] = dict(crown_sp=crown_sp, X=x0)

# the crown-only reference: the inner loop on a fixed setpoint. The cascade adds the
# outer output to the rest, so the rest is the setpoint itself with the outer off.
held = L.cascade(outer=False, bias=row.balance, crown_rest=crown_sp)
held_noff = L.cascade(
    outer=False,
    bias=lambda o, xp: row.balance(o, xp) * 0
    + L.E.to_action(jnp.asarray(0.4739), quiet),
    crown_rest=crown_sp,
)
# its own settled start
keys2 = jax.random.split(jax.random.PRNGKey(8), 10 * 24 * H)
c0 = (carry[0], jax.tree.map(jnp.asarray, held.init_state), carry[2])
c_held, tr = jax.jit(lambda: L.roll(held, quiet, c0, keys2))()
print(
    "crown held, settled X",
    float(np.asarray(tr["X"])[-1]),
    "crown",
    float(np.asarray(tr["crown"])[-1]),
    flush=True,
)

# ---- steps at held crown
N = 7 * 24 * H
keys3 = jax.random.split(jax.random.PRNGKey(9), N)
pp = quiet.plant
steps = {
    "sulfate +50%": dict(sulfate=quiet.sulfate * 1.5),
    "sulfate -50%": dict(sulfate=quiet.sulfate * 0.5),
    "sulfate +17%": dict(sulfate=quiet.sulfate * 1.17),
    "blanket +20%": dict(thickness=quiet.thickness * 1.2),
    "blanket -20%": dict(thickness=quiet.thickness * 0.8),
    "cullet +10 pts": dict(cullet=quiet.cullet + 0.10),
    "cullet -10 pts": dict(cullet=quiet.cullet - 0.10),
    "pull +5%": dict(pull=quiet.pull * 1.05),
    "pull -5%": dict(pull=quiet.pull * 0.95),
    "pull +10%": dict(pull=quiet.pull * 1.10),
    "band +10 K": dict(
        plant=pp.replace(T_fine_low=pp.T_fine_low + 10, T_fine_high=pp.T_fine_high + 10)
    ),
    "band -10 K": dict(
        plant=pp.replace(T_fine_low=pp.T_fine_low - 10, T_fine_high=pp.T_fine_high - 10)
    ),
    "fining time +20%": dict(plant=pp.replace(fining_time=pp.fining_time * 1.2)),
    "emissivity -0.05": dict(plant=pp.replace(eps_surface=0.75)),
}


def marks(x, base):
    d = np.asarray(x) - base
    out = {f"{h}h": float(d[h * H - 1]) for h in (2, 6, 12, 24, 48, 96, 168)}
    big = np.abs(d) > 0.002
    out["first_move_h"] = float(np.argmax(big) / H) if big.any() else None
    return out


for label, change in steps.items():
    p2 = quiet.replace(**change)
    _, tr = jax.jit(lambda p2=p2: L.roll(held, p2, c_held, keys3))()
    m = marks(
        tr["X"],
        float(
            np.asarray(c_held[0].plant.fining[:, -1] @ L.P.layout(quiet.grid).dz) / 1.2
        ),
    )
    m["hot_dK_24h"] = float(
        np.asarray(tr["hot"])[24 * H] - np.asarray(c_held[0].plant.T_roof[0]).max()
    )
    m["valve_24h"] = float(np.asarray(tr["valve"])[24 * H])
    m["valve_168h"] = float(np.asarray(tr["valve"])[-1])
    m["floor_dK_24h"] = float(
        np.asarray(tr["floor"])[24 * H] - np.asarray(tr["floor"])[0]
    )
    res[label] = m
    print(
        label,
        json.dumps({k: (None if v is None else round(v, 4)) for k, v in m.items()}),
        flush=True,
    )

# gas swing at held crown: the event param, at once
p2 = quiet.replace(gas_swing=0.06, pull_change_window_h=(0.0, 0.0))
c_g = (
    c_held[0].replace(events=jnp.zeros(2), event_size=L.E.event_sizes(p2), time=0),
    c_held[1],
    c_held[2],
)
_, tr = jax.jit(lambda: L.roll(held, p2, c_g, keys3))()
m = marks(tr["X"], float(np.asarray(tr["X"])[0]))
res["gas +6% over 6 h"] = m
print(
    "gas +6% over 6 h",
    json.dumps({k: (None if v is None else round(v, 4)) for k, v in m.items()}),
    flush=True,
)

# ---- the walks at held crown, the scored normal day and the events
for name in ("normal", "pull_up", "pull_down", "gas_rich", "gas_lean"):
    p = L.scenario_params(name)
    f = jax.jit(
        lambda seed, p=p: L.roll(
            held, p, L.start(held, p, seed), L.episode_keys(seed, 5040)[1]
        )[1]
    )
    rows = [L.summary(f(seed)) for seed in (100, 101, 102, 103, 104)]
    agg = {
        k: float(np.mean([r[k] for r in rows]))
        for k in rows[0]
        if rows[0][k] is not None
    }
    agg["X_min_worst"] = float(min(r["X_min"] for r in rows))
    res[f"held {name}"] = agg
    print(
        "held", name, json.dumps({k: round(v, 4) for k, v in agg.items()}), flush=True
    )
json.dump(res, open(sys.argv[1], "w"), indent=1)
