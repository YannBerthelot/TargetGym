"""The family sweep (charter B7): draw a task's ``ours`` numbers jointly across
their ranges and check the property still holds on every draw.

Run with ``uv run python scripts/family_sweep.py TASK``. The ranges are the task
package's ``ours.json``, at the member's operating state: ``ranges`` (each with
its source in ``why``), ``coupled`` groups that move together, ``derived``
numbers recomputed from the drawn ones on each draw, and ``no_range`` for the
ours numbers no source gives a range for. 8 Latin-hypercube draws on a fixed
seed; a table with no ranges is swept on its shipped member alone (owner,
2026-11-12). Ceilings are not run per draw (J6): the room is "not measured" here.

The reactor (expert reference): on each draw the expert is re-tuned by its
recipe's bump tests and run against the rod-park hold over the suite on seeds
100 to 109; a plant at rest must need nothing for a day.

Every other charter task (``ladder``): on each draw the task's registry entry
answers with the drawn numbers, so its ladder row, bump tests, rungs and nulls
all see the drawn member. The ladder is climbed afresh, and the property holds
when it keeps a rung (one that trips no seed) whose median beats every null's
in every scenario.
"""

import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

DRAWS = 8
SEED = 20261110


def table(task):
    """``task``'s ``ours.json``, None without one."""
    from target_gym.registry import REGISTRY

    module = type(REGISTRY[task].make_env()).__module__
    package = module.rsplit(".", 1)[0].split(".")
    path = ROOT / "src" / pathlib.Path(*package) / "ours.json"
    return json.loads(path.read_text()) if path.is_file() else None


def draws(table):
    """``DRAWS`` Latin-hypercube draws. Each group in ``table["coupled"]`` shares one
    coordinate, so its numbers move together, ``lo`` with ``lo`` (amendment 5).
    Without ranges, one draw that changes nothing. ``derived`` numbers are each an
    arithmetic expression over the drawn ones."""
    names = sorted(table["ranges"])
    if not names:
        return [{}]
    groups = [list(g) for g in table.get("coupled", [])]
    grouped = {n for g in groups for n in g}
    groups += [[n] for n in names if n not in grouped]
    unit = qmc.LatinHypercube(d=len(groups), seed=SEED).random(DRAWS)
    out = []
    for row in unit:
        d = {}
        for u, group in zip(row, groups):
            for n in group:
                r = table["ranges"][n]
                d[n] = float(r["lo"] + u * (r["hi"] - r["lo"]))
        if "restart_steps" in d:
            d["restart_steps"] = int(round(d["restart_steps"]))
        row = {n: d[n] for n in names}
        for n, expr in table.get("derived", {}).items():
            row[n] = float(eval(expr, {"__builtins__": {}}, dict(d)))
        out.append(row)
    return out


class Drawn:
    """A registry entry answering with a draw: ``make_test_params`` applies the
    draw over every other change (a scenario's, a ladder row's), an integer field
    rounded; everything else is the entry's own."""

    def __init__(self, spec, values):
        self._spec, self._values = spec, values

    def __getattr__(self, name):
        return getattr(self._spec, name)

    def make_test_params(self, **changes):
        default = self._spec.make_test_params()
        values = {
            k: int(round(v)) if isinstance(_get(default, k), int) else v
            for k, v in self._values.items()
        }
        return self._spec.make_test_params(**{**changes, **values})


def _get(params, name):
    return params[name] if isinstance(params, dict) else getattr(params, name)


def holds(reference, nulls) -> bool:
    """B7's property on one draw: a rung was kept, at least one null ran, and the
    kept rung's median beats every null's median in every scenario."""
    if reference is None or not nulls:
        return False
    return all(
        np.median(v["costs"]) < np.median(null["scenarios"][s]["costs"])
        for null in nulls.values()
        for s, v in reference["scenarios"].items()
    )


def _medians(run):
    return {s: float(np.median(v["costs"])) for s, v in run["scenarios"].items()}


def ladder(task, table):
    """The shared runner: the ladder climbed afresh on each draw, with the nulls."""
    import reference_ladder as L

    from target_gym.registry import REGISTRY

    spec, report = REGISTRY[task], []
    for i, d in enumerate(draws(table)):
        REGISTRY[task] = Drawn(spec, d)
        try:
            row = L.ROWS[task]()
            rungs = {
                k: L.run_rung(task, row, k, c) for k, c in L.build(task, row).items()
            }
            order = [k for k in L._ORDER if k in rungs]
            kept = L.best_rung({k: rungs[k] for k in order})
            nulls, broken = {}, {}
            for null in L.null_set(task):
                try:
                    nulls[null.name] = L.run_suite(task, row, null)
                except Exception as err:  # a stale null table, reported
                    broken[null.name] = f"{type(err).__name__}: {err}"
        finally:
            REGISTRY[task] = spec
        ref = rungs[kept] if kept else None
        out = {
            "draw": i,
            "values": d,
            "kept": kept,
            "rungs": {k: _medians(v) | {"trips": v["trips"]} for k, v in rungs.items()},
            "nulls": {k: _medians(v) for k, v in nulls.items()},
            "broken": broken,
            "holds": holds(ref, nulls),
        }
        print(json.dumps(out), flush=True)
        report.append(out)
    return report


def run(task, table):
    """The sweep of ``task`` on ``table``: the reactor's own runner, else the
    ladder's."""
    return (reactor if task == "reactor" else ladder)(table)


def at_rest_cost(env, p):
    """A day lined out at full load with nothing moving, rods parked, no boron."""
    q = p.replace(
        demand_sigma=0.0, initial_xenon_power_range=(1.0, 1.0),
        ramp_down_range=(1e6, 1e6), burnup_range=(0.0, 0.0),
        carried_days_range=(0.0, 0.0), sensor_noise_scale=0.0, initial_tilt_share=0.0,
    )  # fmt: skip
    _, s = env.reset_env(jax.random.PRNGKey(0), q)
    null = jnp.array([1.0 - 2.0 * q.rod_park, 0.0])
    keys = jax.random.split(jax.random.PRNGKey(1), q.max_steps_in_episode)

    def step(s, k):
        _, s1, r, _, info = env.step_env(k, s, null, q)
        return s1, (-r, info["tripped"])

    _, (cost, trip) = jax.jit(lambda s: jax.lax.scan(step, s, keys))(s)
    return float(np.sum(cost)), int(np.sum(trip))


def reactor(table):
    import reactor_ladder as L

    from target_gym import scenarios
    from target_gym.reactor import Reactor, ReactorParams
    from target_gym.registry import REGISTRY

    env = Reactor()
    suite = scenarios.suite(REGISTRY["reactor"])
    hold = lambda m, o: (jnp.array([L.PARK, 0.0]), m)  # noqa: E731
    report = []
    for i, d in enumerate(draws(table)):
        p = ReactorParams(**d)
        rest, rest_trips = at_rest_cost(env, p)
        row = {"draw": i, "values": d, "at_rest_cost": rest,
               "at_rest_trips": rest_trips, "scenarios": {}}  # fmt: skip
        try:
            tune, _ = L.bumps(env, p)
        except L.TrippedDuringTest as e:
            # The recipe cannot be run on this member: the property fails here.
            row.update(holds=False, tune_refused=str(e))
            print(json.dumps(row), flush=True)
            report.append(row)
            continue
        for sc in suite:
            q = ReactorParams(**{**sc.overrides, **d})
            act, memory = L.procedure(q, tune)
            ref = L.rows(env, q, act, memory)
            null = L.rows(env, q, hold, jnp.asarray(0.0))
            row["scenarios"][sc.name] = {
                "reference_median": float(np.median(ref["cost_by_seed"])),
                "null_median": float(np.median(null["cost_by_seed"])),
                "reference_tripped": ref["seeds_tripped"],
                "beats_null": float(np.median(ref["cost_by_seed"]))
                < float(np.median(null["cost_by_seed"])),
            }
        row["holds"] = rest < 1.0 and all(
            v["beats_null"] for v in row["scenarios"].values()
        )
        print(json.dumps(row), flush=True)
        report.append(row)
    return report


def main():
    task = sys.argv[1]
    t = table(task)
    if t is None:
        raise SystemExit(f"{task}: no ours.json")
    report = run(task, t)
    held = sum(r["holds"] for r in report)
    print(f"SWEEP {task}: the property holds on {held} of {len(report)} draws")


if __name__ == "__main__":
    main()
