"""Glass room study: the kept cascade at other fining setpoints, on the plant with its
defects. The ladder places the setpoint two spreads of its PI rung over the pass mark
(0.5523, the PI's spread 0.026), not two of the cascade's own. Here, on the ladder's
suite and its ten seeds: the cascade's own spread of the fining share, and at each
setpoint its cost, its fuel, its lost glass and the steps and seeds under a mark.

Run: PYTHONPATH=src python scripts/glass_room/margin_scan.py <out.json> [setpoint ...]
"""

import json
import pathlib
import sys

import jax
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import lib as L  # noqa: E402

from target_gym import scenarios  # noqa: E402

SEEDS = np.arange(100, 110)
SETPOINTS = (L.PLACED, 0.54, 0.53, 0.52, 0.51)
SETTLED = 2 * 24 * L.HOUR  # steps: the week's first two days are the cooling from reset


def main():
    out = sys.argv[1]
    setpoints = [float(a) for a in sys.argv[2:]] or SETPOINTS
    weights = scenarios.weights(L.SPEC)
    res = {}
    for sp in setpoints:
        ctrl, rows = L.cascade(setpoint=sp), {}
        for name in weights:
            p = L.scenario_params(name)
            run = jax.jit(jax.vmap(lambda s, p=p, c=ctrl: L.run_controller(c, p, s)))
            tr = {k: np.asarray(v) for k, v in run(SEEDS).items()
                  if k in ("cost", "fuel_cost", "X", "canal")}  # fmt: skip
            under = (tr["canal"] < float(p.canal_target)) | (tr["X"] < float(p.fining_pass))  # fmt: skip
            late = tr["X"][:, SETTLED:]
            rows[name] = dict(
                cost=tr["cost"].mean(1).tolist(),
                fuel=tr["fuel_cost"].mean(1).tolist(),
                glass=(tr["cost"] - tr["fuel_cost"]).mean(1).tolist(),
                under=under.sum(1).tolist(),
                fining_mean=late.mean(1).tolist(),
                fining_low=tr["X"].min(1).tolist(),
                spread=late.std(1).tolist(),
                spread_pooled=float(late.std()),
            )
            r = rows[name]
            print(
                f"setpoint {sp:.4f}, {name}: cost {np.mean(r['cost']):.3f}, fuel"
                f" {np.mean(r['fuel']):.3f}, glass {np.mean(r['glass']):.3f}, steps under a"
                f" mark {int(np.sum(r['under']))} on {int(np.sum(np.array(r['under']) > 0))}"
                f" of {len(SEEDS)} seeds, fining {np.mean(r['fining_mean']):.4f}, lowest"
                f" {np.min(r['fining_low']):.4f}, spread {r['spread_pooled']:.4f}",
                flush=True,
            )
        res[f"{sp:.4f}"] = rows
        J = {k: sum(weights[n] * np.mean(rows[n][k]) for n in weights)
             for k in ("cost", "fuel", "glass")}  # fmt: skip
        J5 = sum(weights[n] * np.mean(rows[n]["cost"][:5]) for n in weights)
        lost = sum(int(np.sum(rows[n]["under"])) for n in weights)
        print(
            f"setpoint {sp:.4f}: J {J['cost']:.3f} (fuel {J['fuel']:.3f}, glass"
            f" {J['glass']:.3f}); on seeds 100 to 104 {J5:.3f}; steps under a mark {lost}",
            flush=True,
        )
        json.dump(res, open(out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
