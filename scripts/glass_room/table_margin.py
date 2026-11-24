"""Glass room study: what a reference without quality feedback pays for a family it
cannot tell apart. Plants read defects, which show nothing above the pass mark, so
their crown setpoints come from tables with a safety factor (REALITY.md section 10).
Here the cascade's crown loop alone, its setpoint held at a table value some kelvin
over the one at which the registered plant's fining sits at the cascade's setpoint, on
the registered plant and on members whose fining band the batch's redox has moved.
Normal scenario, the ladder's ten seeds, two weeks, the second read.

The band's moves (20 K either way) are ours, inside the 50 K the sources give for the
fining onset of oxidised melts (1430 to 1480 C, REALITY.md section 10).

Run: PYTHONPATH=src python scripts/glass_room/table_margin.py <out.json> [K over ...]
"""

import json
import pathlib
import sys

import jax
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import lib as L  # noqa: E402

SEEDS = np.arange(100, 110)
WEEK = 7 * 24 * L.HOUR
#: K: the crown couple under the kept cascade on the registered plant, second week
#: (``hidden_members.py``: 1544.7 C).
TABLE = 1817.85
MEMBERS = {
    "registered": {},
    "fining band 20 K higher": dict(T_fine_low=1713.15, T_fine_high=1793.15),
    "fining band 20 K lower": dict(T_fine_low=1673.15, T_fine_high=1753.15),
}
OVER = (0.0, 10.0, 20.0, 30.0)


def main():
    out = sys.argv[1]
    base, res = L.scenario_params("normal"), {}
    for over in [float(a) for a in sys.argv[2:]] or OVER:
        ctrl = L.cascade(outer=False, crown_rest=TABLE + over)
        for name, change in MEMBERS.items():
            p = base.replace(
                plant=base.plant.replace(**change), max_steps_in_episode=2 * WEEK
            )
            run = jax.jit(jax.vmap(lambda s, p=p, c=ctrl: L.run_controller(c, p, s)))
            tr = {k: np.asarray(v)[:, WEEK:] for k, v in run(SEEDS).items()
                  if k in ("cost", "fuel_cost", "X", "canal", "crown")}  # fmt: skip
            low = tr["X"] < float(p.fining_pass)
            cold = tr["canal"] < float(p.canal_target)
            row = dict(
                cost=tr["cost"].mean(1).tolist(), fuel=tr["fuel_cost"].mean(1).tolist(),
                glass=(tr["cost"] - tr["fuel_cost"]).mean(1).tolist(),
                under_fining=low.sum(1).tolist(), under_canal=cold.sum(1).tolist(),
                fining_mean=float(tr["X"].mean()), fining_low=float(tr["X"].min()),
                crown_C=float(tr["crown"].mean() - 273.15),
            )  # fmt: skip
            res[f"{over:g} K over, {name}"] = row
            print(
                f"table {over:g} K over, {name}: cost {np.mean(row['cost']):.3f}, fuel"
                f" {np.mean(row['fuel']):.3f}, glass {np.mean(row['glass']):.3f}, steps"
                f" under the fining mark {int(np.sum(row['under_fining']))} on"
                f" {int(np.sum(np.array(row['under_fining']) > 0))} of {len(SEEDS)} seeds,"
                f" under the canal's {int(np.sum(row['under_canal']))}, fining"
                f" {row['fining_mean']:.4f}, lowest {row['fining_low']:.4f}, crown"
                f" {row['crown_C']:.1f} C",
                flush=True,
            )
            json.dump(res, open(out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
