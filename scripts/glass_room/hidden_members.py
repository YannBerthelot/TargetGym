"""Glass room study: does a furnace with other numbers than the reference was tuned on
cost the reference anything? The kept cascade, as tuned on the registered plant, on
members of the family that differ in one number it is not told: fouled regenerators, a
duller or brighter surface, a fining band moved by the batch's redox, a slower fining,
a leakier crown. Normal scenario, the ladder's ten seeds, two weeks: the first lets the
cascade settle on the member (the reset state is the registered plant's), the second
is read.

Every size here is ours, picked to move the fuel or the fining by a few percent; only
the direction of each is sourced (REALITY.md; Beerkens 1992 for the regenerators).

Run: PYTHONPATH=src python scripts/glass_room/hidden_members.py <out.json>
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
MEMBERS = {
    "registered": {},
    "regenerators fouled (0.80 to 0.72 a node)": dict(eps_regen_node=0.72),
    "regenerators fouled (0.80 to 0.64 a node)": dict(eps_regen_node=0.64),
    "surface emissivity 0.7": dict(eps_surface=0.7),
    "surface emissivity 0.9": dict(eps_surface=0.9),
    "fining band 20 K higher": dict(T_fine_low=1713.15, T_fine_high=1793.15),
    "fining band 20 K lower": dict(T_fine_low=1673.15, T_fine_high=1753.15),
    "fining a quarter slower": dict(fining_time=4500.0),
    "crown loss up a quarter": dict(U_crown=2.125),
}


def main():
    out = sys.argv[1]
    ctrl, base = L.cascade(), L.scenario_params("normal")
    res = {}
    for name, change in MEMBERS.items():
        p = base.replace(
            plant=base.plant.replace(**change), max_steps_in_episode=2 * WEEK
        )
        run = jax.jit(jax.vmap(lambda s, p=p: L.run_controller(ctrl, p, s)))
        tr = {k: np.asarray(v)[:, WEEK:] for k, v in run(SEEDS).items()
              if k in ("cost", "fuel_cost", "X", "canal", "valve", "crown")}  # fmt: skip
        under = (tr["canal"] < float(p.canal_target)) | (tr["X"] < float(p.fining_pass))
        res[name] = dict(
            cost=tr["cost"].mean(1).tolist(),
            fuel=tr["fuel_cost"].mean(1).tolist(),
            glass=(tr["cost"] - tr["fuel_cost"]).mean(1).tolist(),
            under=under.sum(1).tolist(),
            fining_mean=float(tr["X"].mean()), fining_low=float(tr["X"].min()),
            spread=float(tr["X"].std()), valve=float(tr["valve"].mean()),
            crown_C=float(tr["crown"].mean() - 273.15),
        )  # fmt: skip
        r = res[name]
        print(
            f"{name}: cost {np.mean(r['cost']):.3f}, fuel {np.mean(r['fuel']):.3f}, glass"
            f" {np.mean(r['glass']):.3f}, steps under a mark {int(np.sum(r['under']))} on"
            f" {int(np.sum(np.array(r['under']) > 0))} of {len(SEEDS)} seeds, fining"
            f" {r['fining_mean']:.4f}, lowest {r['fining_low']:.4f}, spread"
            f" {r['spread']:.4f}, valve {r['valve']:.4f} kg/s, crown {r['crown_C']:.1f} C",
            flush=True,
        )
        json.dump(res, open(out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
