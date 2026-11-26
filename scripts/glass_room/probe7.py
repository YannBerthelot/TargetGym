"""Glass room study, probe 7: a suite built on job changes. A furnace alternating
between a low and a high pull, the change planned and shown ``notice`` hours ahead.

Every policy is the ladder's cascade with its crown setpoint following the pull charged
(the feedforward on the inner loop's setpoint), differing only in the fining setpoint:
``sp`` always, plus ``boost`` per unit of announced rise while a rise is announced. So
``boost = 0`` is a controller blind to the plan, and a long ``notice`` with each seed's
best ``(sp, boost)`` is a clairvoyant schedule, an upper bound on the ceiling.

Run: PYTHONPATH=src python scripts/glass_room/probe7.py <out.json> <low> <high>
"""

import itertools
import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import lib as L  # noqa: E402

from target_gym.experts.ladder import Cascade, Pi, Rung, ladder_controller  # noqa: E402

H = L.HOUR
SEEDS = tuple(range(100, 110))
LOW, HIGH = float(sys.argv[2]), float(sys.argv[3])
row = L.ladder_row()
base = L.scenario_params("normal")
G_TABLE = 185.0
CROWN_DESIGN, PULL_DESIGN = 1820.75, float(base.pull)  # K at X 0.5523, design pull


def controller(sp, boost, valve_ff=False, table=True):
    loops = (Pi(0, L.OUTER["kc"], L.OUTER["ti"], L._alpha(L.OUTER["tf"])),)
    inner = Pi(0, L.INNER["kc"], L.INNER["ti"], L._alpha(L.INNER["tf"]))
    if table:
        rest = lambda o, xp: CROWN_DESIGN + G_TABLE * (
            o[8] / PULL_DESIGN - 1.0
        )  # noqa: E731
    else:
        rest = lambda o, xp: CROWN_DESIGN  # noqa: E731
    cas = Cascade(lambda o, xp: o[0], rest, inner)
    rung = Rung("cascade", loops, np.ones((1, 1)), (cas,))
    read = lambda o, xp: xp.stack([o[4]])  # noqa: E731

    def target(o, xp):
        rise = xp.maximum(o[9] / o[8] - 1.0, 0.0)
        return xp.stack([sp + boost * rise])

    return ladder_controller(rung, read, target, row.balance, L.DT)


def episode(sp, boost, notice, pull0, change, seed, valve_ff=False, table=True):
    c = controller(sp, boost, table=table)
    p0 = base.replace(pull=PULL_DESIGN * pull0, pull_notice_h=notice)
    quiet = p0.replace(walk_sd=(0.0,) * 5, sensor_noise_scale=0.0)
    keys = jax.random.split(jax.random.PRNGKey(31), 25 * 24 * H)
    (s, m, _), _ = L.roll(c, quiet, L.start(c, quiet, 0), keys)
    p = p0.replace(pull_change=change)
    st, _, _ = L.start(c, p, seed)
    st = st.replace(
        plant=s.plant,
        cooling=s.cooling,
        integral=s.integral,
        fuel=s.fuel,
        limit=s.limit,
    )
    _, tr = L.roll(c, p, (st, m, L.ENV.get_obs(st, p)), L.episode_keys(seed, 5040)[1])
    return (
        jnp.mean(tr["cost"]),
        jnp.sum(tr["lost"]) / H,
        jnp.mean(tr["fuel_cost"]),
        jnp.min(tr["X"]),
    )


SCEN = {
    "normal_low": (LOW, 0.0),
    "normal_high": (HIGH, 0.0),
    "job_up": (LOW, HIGH / LOW - 1.0),
    "job_down": (HIGH, LOW / HIGH - 1.0),
}
SPS = (0.505, 0.51, 0.515, 0.52, 0.53, 0.54, 0.5523, 0.57, 0.59, 0.61, 0.63, 0.65, 0.67)
BOOSTS = (0.0, 0.1, 0.2, 0.3, 0.4)
NOTICES = (8.0, 24.0, 48.0, 96.0)
grid = list(itertools.product(SPS, BOOSTS))
sp_a = jnp.asarray([g[0] for g in grid], jnp.float32)
bo_a = jnp.asarray([g[1] for g in grid], jnp.float32)
seeds_a = jnp.asarray(SEEDS)
res = {
    "low": LOW,
    "high": HIGH,
    "sps": SPS,
    "boosts": BOOSTS,
    "notices": NOTICES,
    "runs": {},
}
for name, (pull0, change) in SCEN.items():
    for notice in NOTICES if change > 0 else (8.0,):
        f = jax.jit(
            jax.vmap(
                jax.vmap(
                    lambda sp, bo, seed: episode(sp, bo, notice, pull0, change, seed),
                    in_axes=(None, None, 0),
                ),
                in_axes=(0, 0, None),
            )
        )
        cost, lost, fuel, xmin = (np.asarray(v) for v in f(sp_a, bo_a, seeds_a))
        res["runs"][f"{name} notice {notice:.0f}"] = dict(
            cost=cost.tolist(),
            lost=lost.tolist(),
            fuel=fuel.tolist(),
            xmin=xmin.tolist(),
        )
        blind = [i for i, g in enumerate(grid) if g[1] == 0.0]
        mean = cost.mean(1)
        best_blind = min(blind, key=lambda i: mean[i])
        best_any = int(np.argmin(mean))
        clair = cost.min(0).mean()
        placed = grid.index((0.5523, 0.0))
        print(
            f"{name} notice {notice:.0f}: ladder setpoint 0.5523 -> {mean[placed]:.2f} "
            f"(lost {lost[placed].mean():.1f} h) | best fixed margin sp {grid[best_blind][0]} -> "
            f"{mean[best_blind]:.2f} (lost {lost[best_blind].mean():.1f} h) | best plan reader "
            f"sp {grid[best_any][0]} boost {grid[best_any][1]} -> {mean[best_any]:.2f} "
            f"(lost {lost[best_any].mean():.1f} h) | per-seed best {clair:.2f}",
            flush=True,
        )
    # the ladder as recorded: valve feedforward only, crown rest fixed
    if True:
        g = jax.jit(
            jax.vmap(
                lambda seed: episode(0.5523, 0.0, 8.0, pull0, change, seed, table=False)
            )
        )
        cost, lost, fuel, xmin = (np.asarray(v) for v in g(seeds_a))
        res["runs"][f"{name} recorded ladder"] = dict(
            cost=cost.tolist(), lost=lost.tolist()
        )
        print(
            f"{name}: recorded ladder (valve feedforward) -> {cost.mean():.2f} (lost {lost.mean():.1f} h)",
            flush=True,
        )
json.dump(res, open(sys.argv[1], "w"))
