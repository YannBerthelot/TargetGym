"""Glass room study: the room at other release rates, read by replay. The blocks'
release rate and the temperature they release from are ours (``ours.json``). Defects
feed nothing back into the furnace, so the kept cascade fires the same valve at any
rate, and the ceiling's and the library planner's valve paths, made at the registered
rate, keep their marks at any other. Replayed on a plant with another rate they are no
longer the best plans for it: they bound its ceiling and its planner from above, and
its room from below.

Run: PYTHONPATH=src python scripts/glass_room/rate_replay.py <ceil_dir> <lib_dir> <out.json>
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

from target_gym import scenarios  # noqa: E402

CASCADE = L.cascade()
#: (times the registered release rate, K over the registered release temperature)
VARIANTS = {
    "registered": (1.0, 0.0),
    "a third of the rate": (1.0 / 3.0, 0.0),
    "three times the rate": (3.0, 0.0),
    "release from 50 K lower": (1.0, -50.0),
    "release from 25 K lower": (1.0, -25.0),
    "release from 25 K higher": (1.0, 25.0),
    "release from 50 K higher": (1.0, 50.0),
}


def main():
    ceil_dir, lib_dir, out = sys.argv[1:4]
    weights, seeds = scenarios.weights(L.SPEC), np.asarray(SEEDS)
    res = {v: {} for v in VARIANTS}
    for name in weights:
        p = L.scenario_params(name)
        n = int(p.max_steps_in_episode)

        def replay(valve, rate, shift, p=p):
            plant = p.plant.replace(
                block_release=p.plant.block_release * rate,
                block_release_T=p.plant.block_release_T + shift,
            )
            q = p.replace(plant=plant)
            tr = jax.vmap(lambda v, s: L.run_valve(v, q, s))(valve, seeds)
            under = (tr["canal"] < q.canal_target) | (tr["X"] < q.fining_pass)
            return tr["cost"].mean(1), tr["fuel_cost"].mean(1), under.sum(1)

        replay = jax.jit(replay)
        asked = jax.jit(jax.vmap(lambda s, p=p: L.run_controller(CASCADE, p, s)["asked"]))  # fmt: skip
        knots = np.stack([best_knots(ceil_dir, name, s) for s in SEEDS])
        paths = {
            "reference": asked(seeds),
            "planner": jnp.asarray(np.stack(
                [np.load(f"{lib_dir}/{name}_{s}.npz")["valve"] for s in SEEDS])),
            "ceiling": jnp.asarray(np.repeat(knots, -(-n // knots.shape[1]), axis=1)[:, :n]),
        }  # fmt: skip
        for variant, (rate, shift) in VARIANTS.items():
            row = {}
            for who, valve in paths.items():
                cost, fuel, under = (np.asarray(v) for v in replay(valve, rate, shift))
                row[who] = dict(cost=cost.tolist(), fuel=fuel.tolist(),
                                glass=(cost - fuel).tolist(), under=under.tolist())  # fmt: skip
            res[variant][name] = row
            line = ", ".join(
                f"{who} {np.mean(r['cost']):.3f} (glass {np.mean(r['glass']):.3f},"
                f" under {int(np.sum(r['under']))})" for who, r in row.items()
            )  # fmt: skip
            print(f"{name}, {variant}: {line}", flush=True)
    for variant, rows in res.items():
        J = {who: sum(weights[n] * np.mean(rows[n][who][k]) for n in weights)
             for who in ("reference", "planner", "ceiling") for k in ("cost",)}  # fmt: skip
        G = {who: sum(weights[n] * np.mean(rows[n][who]["glass"]) for n in weights)
             for who in J}  # fmt: skip
        d = np.array([[np.array(rows[n]["reference"]["cost"]) - np.array(rows[n][who]["cost"])
                       for n in weights] for who in ("ceiling", "planner")])  # fmt: skip
        per_seed = np.einsum("s,wsk->wk", np.array([weights[n] for n in weights]), d)
        print(
            f"{variant}: J reference {J['reference']:.3f}, planner {J['planner']:.3f},"
            f" ceiling {J['ceiling']:.3f}; glass {G['reference']:.3f}, {G['planner']:.3f},"
            f" {G['ceiling']:.3f}; room at least {J['reference'] / J['ceiling']:.4f},"
            f" planner at least {J['reference'] / J['planner']:.4f}; per seed to the"
            f" ceiling {np.round(per_seed[0], 3).tolist()}, to the planner"
            f" {np.round(per_seed[1], 3).tolist()}",
            flush=True,
        )
    json.dump(res, open(out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
