"""Glass room study: the furnace as a family run from a table. Prototype of what a
furnace that does not know itself would be as a task (the owner's go of 2026-11-24;
the registered plant is untouched).

The family: the fining band moved by the batch's redox, here by -20, -10, 0, 10 and
20 K, equal weights (ours, inside the 50 K the sources give for the fining onset). The
reference: the kept cascade's crown loop alone, its setpoint from a table, since a
plant's quality reading says nothing while the glass is good. The table's safety rule
(ours, for the owner to rule): the lowest setpoint, on a 5 K grid, at which no member
loses a step on the ladder's ten seeds.

Each member starts as its operators run it: a week on the walks' mean path under the
table, from the registered design state. The scored week starts there, on the
episode's own walks and keys. Beside the table: the cascade with the fining share read
(what the task's reference is today), and each member's own best table.

Run: PYTHONPATH=src python scripts/glass_room/family.py <out_dir> [K over ...]
"""

import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import lib as L  # noqa: E402

E = L.E
SEEDS = np.arange(100, 110)
WEEK = 7 * 24 * L.HOUR
#: K: the crown couple under the kept cascade on the registered plant
#: (``hidden_members.py``: 1544.7 C).
TABLE = 1817.85
BANDS = (-20.0, -10.0, 0.0, 10.0, 20.0)
OWN_TABLES = (-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0)
#: What a start keeps of the episode's own reset: its walks, events and sensor key.
FROM_RESET = ("time", "walk", "events", "event_until", "event_size", "event_t0",
              "sensor_key")  # fmt: skip


def member(band, base=None):
    """The scenario's params with the fining band moved by ``band`` K."""
    base = L.scenario_params("normal") if base is None else base
    plant = base.plant.replace(
        T_fine_low=base.plant.T_fine_low + band, T_fine_high=base.plant.T_fine_high + band
    )  # fmt: skip
    return base.replace(plant=plant)


def table(over):
    """The crown loop on a table setpoint ``over`` K above ``TABLE``."""
    return L.cascade(outer=False, crown_rest=TABLE + over)


def settled(band, over):
    """The member after a week under the table on the walks' mean path: the state
    and the controller's memory."""
    p = member(band).replace(walk_sd=(0.0,) * len(E.WALKS))
    ctrl = table(over)
    keys = L.episode_keys(0, WEEK)[1]
    end = jax.jit(lambda c: L.roll(ctrl, p, c, keys)[0])(L.start(ctrl, p, 0))
    return end[0], end[1]


def starts(state, p, seeds):
    """``state`` as each seed's episode meets it: the member's furnace, the episode's
    own walks, events and sensor key."""

    def one(seed):
        k_reset, keys = L.episode_keys(seed, WEEK)
        own = L.ENV.reset_env(k_reset, p)[1]
        st = state.replace(**{k: getattr(own, k) for k in FROM_RESET})
        return st, L.ENV.get_obs(st, p), keys

    return jax.vmap(one)(jnp.asarray(seeds))


def week(ctrl, p, state, mem, seeds=SEEDS):
    """The scored week of ``ctrl`` from ``state``: per seed cost, fuel, glass, steps
    under the fining mark and under the canal's, and the fining's mean and lowest."""

    def run(seeds):
        st, obs, keys = starts(state, p, seeds)
        roll = lambda s, o, k: L.roll(ctrl, p, (s, mem, o), k)[1]  # noqa: E731
        tr = jax.vmap(roll)(st, obs, keys)
        return tr["cost"], tr["fuel_cost"], tr["X"], tr["canal"]

    cost, fuel, X, canal = (np.asarray(v) for v in jax.jit(run)(jnp.asarray(seeds)))
    return dict(
        cost=cost.mean(1).tolist(), fuel=fuel.mean(1).tolist(),
        glass=(cost - fuel).mean(1).tolist(),
        under_fining=(X < float(p.fining_pass)).sum(1).tolist(),
        under_canal=(canal < float(p.canal_target)).sum(1).tolist(),
        fining_mean=float(X.mean()), fining_low=float(X.min()),
    )  # fmt: skip


def show(label, row):
    print(
        f"{label}: cost {np.mean(row['cost']):.3f}, fuel {np.mean(row['fuel']):.3f},"
        f" glass {np.mean(row['glass']):.3f}, steps under the fining mark"
        f" {int(np.sum(row['under_fining']))} on"
        f" {int(np.sum(np.array(row['under_fining']) > 0))} of {len(row['cost'])}"
        f" seeds, under the canal's {int(np.sum(row['under_canal']))}, fining"
        f" {row['fining_mean']:.4f}, lowest {row['fining_low']:.4f}",
        flush=True,
    )


def save(path, state, mem):
    leaves = jax.tree.leaves((state, mem))
    np.savez(path, **{f"leaf{i}": np.asarray(v) for i, v in enumerate(leaves)})


def load(path, band, over):
    """The saved start of a member: ``(state, memory)``, on the tree of a fresh one."""
    p = member(band)
    like = L.start(table(over), p, 0)[:2]
    z = np.load(path)
    leaves = [jnp.asarray(z[f"leaf{i}"]) for i in range(len(z.files))]
    return jax.tree.unflatten(jax.tree.structure(like), leaves)


def main():
    out = pathlib.Path(sys.argv[1])
    overs = [float(a) for a in sys.argv[2:]] or [15.0, 20.0, 25.0]
    res, safe = {}, None
    for over in overs:  # the safety rule: the lowest table no member loses a step on
        rows, at = {}, {}
        for band in BANDS:
            state, mem = at[band] = settled(band, over)
            rows[band] = week(table(over), member(band), state, mem)
            show(f"table {over:g} K over, band {band:+g} K", rows[band])
        res[f"table {over:g}"] = {f"{b:+g}": r for b, r in rows.items()}
        lost = sum(int(np.sum(r["under_fining"])) for r in rows.values())
        J = float(np.mean([np.mean(r["cost"]) for r in rows.values()]))
        print(f"table {over:g} K over: family cost {J:.3f}, steps under the fining mark {lost}", flush=True)  # fmt: skip
        if lost == 0 and safe is None:
            safe, start = over, at
            for band, (state, mem) in at.items():
                save(out / f"member_{band:+g}.npz", state, mem)
    json.dump(res | {"safe": safe}, open(out / "family.json", "w"), indent=1)
    if safe is None:
        sys.exit("no table tried keeps every member's mark")
    print(f"the family's table: {safe:g} K over", flush=True)
    for band in BANDS:  # from the same start: the cascade, and the member's own tables
        state, mem = start[band]
        p = member(band)
        rows = {"cascade": week(L.cascade(crown_rest=TABLE + safe), p, state, mem)}
        show(f"band {band:+g} K, the cascade reading the fining share", rows["cascade"])
        for over in OWN_TABLES:
            rows[f"table {over:g}"] = week(table(over), p, state, mem)
            show(f"band {band:+g} K, table {over:g} K over", rows[f"table {over:g}"])
        res[f"band {band:+g}"] = rows
        json.dump(res | {"safe": safe}, open(out / "family.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
