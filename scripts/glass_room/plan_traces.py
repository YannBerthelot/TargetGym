"""Glass room study: what the foresight plans do that the kept cascade does not. Each
seed's best plan of ``scripts/glass_slow_oracle.py`` replayed beside the cascade on the
same episode (five scenarios, seeds 100 to 104, sensor noise 1): fuel, lost glass by
day, the fining share, the hot spot and the valve.

Run: PYTHONPATH=src python scripts/glass_room/plan_traces.py <ceil_dir> <out.json>
"""

import glob
import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import lib as L  # noqa: E402

from target_gym import scenarios  # noqa: E402

SEEDS = (100, 101, 102, 103, 104)
DAY = 24 * L.HOUR
CASCADE = L.cascade()
TAIL = 24.0 / 168.0  # the tail's steps per episode step, as the room step weighs it


def best_knots(out, name, seed):
    runs = [json.loads(pathlib.Path(f).read_text())
            for f in glob.glob(f"{out}/{name}_{seed}_4h*.json")]  # fmt: skip
    runs = [r for r in runs if r["replayed_lost"] == 0 and r["replayed_trips"] == 0]
    r = min(runs, key=lambda r: r["planned_cost"] + r["tail_running"] * TAIL)
    return np.asarray(r["knots"], np.float32)


def numbers(tr, p):
    glass = tr["cost"] - tr["fuel_cost"]
    days = glass[:, : 7 * DAY].reshape(len(glass), 7, DAY).mean(2)
    worth = tr["pull"] * L.DT / 1000.0 * float(p.lost_glass_price)
    return dict(
        cost=tr["cost"].mean(1),
        fuel=tr["fuel_cost"].mean(1),
        glass=glass.mean(1),
        lost_share=(glass / worth).mean(1),
        glass_by_day=days,
        fining_mean=tr["X"].mean(1),
        fining_min=tr["X"].min(1),
        hot_mean_C=tr["hot"].mean(1) - 273.15,
        hot_sd=tr["hot"].std(1),
        hot_max_C=tr["hot"].max(1) - 273.15,
        crown_max_C=tr["crown"].max(1) - 273.15,
        valve_mean=tr["valve"].mean(1),
        valve_move=np.abs(np.diff(tr["valve"], axis=1)).sum(1),
    )


def main():
    res = {}
    for name in scenarios.weights(L.SPEC):
        p = L.scenario_params(name)
        n = int(p.max_steps_in_episode)
        knots = np.stack([best_knots(sys.argv[1], name, s) for s in SEEDS])
        hold = -(-n // knots.shape[1])
        valve = jnp.asarray(np.repeat(knots, hold, axis=1)[:, :n])
        seeds = np.asarray(SEEDS)
        plan = jax.jit(jax.vmap(lambda v, s, p=p: L.run_valve(v, p, s)))(valve, seeds)
        ref = jax.jit(jax.vmap(lambda s, p=p: L.run_controller(CASCADE, p, s)))(seeds)
        res[name] = {
            who: {k: np.asarray(v).tolist() for k, v in
                  numbers({k: np.asarray(v) for k, v in tr.items()}, p).items()}
            for who, tr in (("plan", plan), ("cascade", ref))
        }  # fmt: skip
        for who in ("plan", "cascade"):
            d = res[name][who]
            line = ", ".join(
                f"{k} {np.mean(v):.4g}" for k, v in d.items() if k != "glass_by_day"
            )
            by_day = np.round(np.mean(d["glass_by_day"], 0), 2).tolist()
            print(f"{name} {who}: {line}\n   glass by day {by_day}", flush=True)
    json.dump(res, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
