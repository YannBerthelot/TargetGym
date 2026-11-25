"""Glass room study: where the foresight plans keep the glass the kept cascade loses.
Each seed's best plan and the cascade's own valve path, replayed on the same episode
(seeds 100 to 104, sensor noise 1) with both defect sources, with the blocks' alone and
with reboil alone, and what the blocks did (the cascade also on readings without
noise, to see what its own valve movement costs it): the share of their area at or over the
release temperature, and how far that area rose over the week.

Run: PYTHONPATH=src python scripts/glass_room/plan_mechanisms.py <ceil_dir> <out.json> [scenario ...]
"""

import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import lib as L  # noqa: E402
from plan_traces import SEEDS, best_knots  # noqa: E402

P, E = L.P, L.E
CASCADE = L.cascade()


def fields_run(valve, p, seed):
    """One episode of a valve path, kg/s per step, with what its blocks do."""
    a, rho = P.layout(p.grid), p.plant.rho_glass
    mass = rho * a.dz[:, None] * a.dx * a.wc  # kg of glass per cell
    # block area a cell touches: two side walls, and the floor under the bottom layer
    area = 2.0 * a.dz[:, None] * a.dx * jnp.ones_like(a.wc)
    area = area.at[-1].add(a.dx * a.wc)
    k_reset, keys = L.episode_keys(seed, valve.shape[0])
    _, st = L.ENV.reset_env(k_reset, p)

    def f(s, x):
        k, u = x
        _, s1, r, _, _ = L.ENV.step_env(k, s, E.to_action(u, p), p)
        hot = s1.plant.block >= p.plant.block_release_T
        rise = jnp.maximum(s1.plant.block - s.plant.block, 0.0)
        return s1, dict(
            cost=-r,
            glass=E.compute_reward_terms(s1, p)["tracking"],
            hot_area=jnp.sum(hot * area) / jnp.sum(area),
            rise_hot=jnp.sum(rise * hot * area) / jnp.sum(hot * area + 1e-9),
            bath=jnp.sum(s1.plant.T * mass) / jnp.sum(mass),
        )

    return jax.lax.scan(f, st, (keys, valve))[1]


def run(valve, p):
    f = jax.jit(jax.vmap(lambda v, s: fields_run(v, p, s)))
    return {k: np.asarray(v) for k, v in f(valve, np.asarray(SEEDS)).items()}


def asked(q):
    """The valve the kept cascade asks for on each seed, reading under ``q``."""
    f = jax.jit(jax.vmap(lambda s: L.run_controller(CASCADE, q, s)["asked"]))
    return f(np.asarray(SEEDS))


res = {}
for name in sys.argv[3:] or ["normal", "pull_up"]:
    p = L.scenario_params(name)
    n = int(p.max_steps_in_episode)
    knots = np.stack([best_knots(sys.argv[1], name, s) for s in SEEDS])
    plan = jnp.asarray(np.repeat(knots, -(-n // knots.shape[1]), axis=1)[:, :n])
    quiet = L.scenario_params(name, sensor_noise_scale=0.0)
    paths = {
        "plan": plan,
        "cascade": asked(p),
        "cascade reading without noise": asked(quiet),
    }
    only = {
        "blocks alone": p.replace(plant=p.plant.replace(reboil_plate=0.0)),
        "reboil alone": p.replace(plant=p.plant.replace(block_release=0.0)),
    }
    res[name] = {}
    for who, valve in paths.items():
        tr = run(valve, p)
        d = dict(
            cost=tr["cost"].mean(),
            glass=tr["glass"].mean(),
            hot_area=tr["hot_area"].mean(),
            rise_of_hot_blocks_K=tr["rise_hot"].sum(1).mean(),
            bath_mean_C=tr["bath"].mean() - 273.15,
            bath_sd=tr["bath"].std(1).mean(),
            bath_rise_K=np.maximum(np.diff(tr["bath"], axis=1), 0.0).sum(1).mean(),
        )
        for label, q in only.items():
            d[f"glass, {label}"] = run(valve, q)["glass"].mean()
        res[name][who] = {k: float(v) for k, v in d.items()}
        print(
            name, who, {k: round(v, 4) for k, v in res[name][who].items()}, flush=True
        )
json.dump(res, open(sys.argv[2], "w"), indent=1)
