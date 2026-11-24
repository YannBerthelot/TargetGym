"""Glass room study, probe 2: what the kept reference and the recorded ceiling plans
actually do, step by step, on the ladder's seeds.

Run: PYTHONPATH=src python scripts/glass_room/probe2.py <plans dir> <out dir>
"""

import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import lib as L  # noqa: E402

plans, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
SEEDS = (100, 101, 102, 103, 104)
NAMES = ("normal", "pull_up", "pull_down", "gas_rich", "gas_lean")
row = L.ladder_row()
ref = L.cascade(bias=row.balance)
res = {}
for name in NAMES:
    p = L.scenario_params(name)
    run = jax.jit(lambda seed, p=p: L.run_controller(ref, p, seed))
    replay = jax.jit(lambda valve, seed, p=p: L.run_valve(valve, p, seed))
    for seed in SEEDS:
        if not (out / f"ref_{name}_{seed}.npz").exists():
            tr = run(seed)
            np.savez(
                out / f"ref_{name}_{seed}.npz",
                **{k: np.asarray(v) for k, v in tr.items()},
            )
        s = L.summary(dict(np.load(out / f"ref_{name}_{seed}.npz")))
        res[f"ref {name} {seed}"] = s
        best = None
        for f in sorted(plans.glob(f"{name}_{seed}_4h*.json")):
            d = json.loads(f.read_text())
            if d.get("replayed_lost", 1) == 0 and (
                best is None or d["replayed_cost"] < best[0]
            ):
                best = (d["replayed_cost"], f)
        if best is None:
            continue
        valve = np.load(str(best[1]).replace(".json", "_plan.npy"))
        valve = np.repeat(valve, 4 * L.HOUR)[:5040]  # the file holds 4 h knots
        tr = replay(jnp.asarray(valve, jnp.float32), seed)
        s = L.summary(tr)
        s["recorded"] = best[0]
        res[f"oracle {name} {seed}"] = s
        print(
            "orc",
            name,
            seed,
            json.dumps({k: (None if v is None else round(v, 4)) for k, v in s.items()}),
            flush=True,
        )
        np.savez(
            out / f"orc_{name}_{seed}.npz", **{k: np.asarray(v) for k, v in tr.items()}
        )
json.dump(res, open(out / "probe2.json", "w"), indent=1)
