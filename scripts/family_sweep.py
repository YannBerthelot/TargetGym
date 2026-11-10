"""The family sweep (charter B7): draw a task's ``ours`` numbers jointly across
their ranges and check the property still holds on every draw.

Run with ``uv run python scripts/family_sweep.py reactor``. The ranges are the
package's ``ours.json``, at the member's operating state. 8 Latin-hypercube
draws on a fixed seed; on each, the reference is re-tuned by its recipe's bump
tests and run against the null over the scenario suite on seeds 100 to 109.
Checked per draw: a plant at rest needs nothing for a day; the reference beats
the null on the median over seeds in every scenario; the reference's trips are
reported. Ceilings are not run per draw (J6): the room is "not measured" here.
Only the reactor has a table so far.
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


def draws(table):
    names = sorted(table["ranges"])
    lo = np.array([table["ranges"][n]["lo"] for n in names], float)
    hi = np.array([table["ranges"][n]["hi"] for n in names], float)
    unit = qmc.LatinHypercube(d=len(names), seed=SEED).random(DRAWS)
    values = qmc.scale(unit, np.minimum(lo, hi), np.maximum(lo, hi))
    out = []
    for row in values:
        d = dict(zip(names, row.tolist()))
        if "restart_steps" in d:
            d["restart_steps"] = int(round(d["restart_steps"]))
        out.append(d)
    return out


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
        tune, _ = L.bumps(env, p)
        row = {"draw": i, "values": d, "at_rest_cost": rest,
               "at_rest_trips": rest_trips, "scenarios": {}}  # fmt: skip
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
    table = json.loads((ROOT / "src" / "target_gym" / task / "ours.json").read_text())
    runners = {"reactor": reactor}
    if task not in runners:
        raise SystemExit(f"{task}: no sweep runner yet")
    report = runners[task](table)
    held = sum(r["holds"] for r in report)
    print(f"SWEEP {task}: the property holds on {held} of {len(report)} draws")


if __name__ == "__main__":
    main()
