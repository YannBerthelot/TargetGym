"""The glass furnace's foresight oracle on the ladder's seeds, and its room (charter J3).

    python glass_slow_oracle.py solve SCENARIO SEED KNOT_HOURS START OUT_DIR
    python glass_slow_oracle.py room OUT_DIR LADDER_JSON KNOT_HOURS [SEEDS]

``solve`` plans one episode with ``glass_furnace.oracle.solve`` (the env's own step and
reward, the seed's walks and events foreseen) from the design valve (``START`` "-"), a
constant ``START`` kg/s, or ("restart") the seed's last plan, then replays the plan through ``benchmark.run_policy`` on the seed at
sensor noise 1, as the reference ladder scores its rungs, and writes the solver's
certificate and the replayed cost. ``room`` reads those files and the ladder's
reference JSON: room per scenario is the kept rung's mean cost per step over the
oracle's replayed one on the same seeds, room on J the same on the scenario-weighted
costs, and writes ``data/oracle_ceilings/glass_furnace.json``.
"""

import json
import pathlib
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from target_gym import scenarios  # noqa: E402
from target_gym.benchmark import run_policy, task_info  # noqa: E402
from target_gym.experts.mpc import plan_params  # noqa: E402
from target_gym.glass_furnace import env_jax as E  # noqa: E402
from target_gym.glass_furnace import oracle as O  # noqa: E402
from target_gym.registry import REGISTRY  # noqa: E402

SPEC = REGISTRY["glass_furnace"]
SEEDS = tuple(range(100, 110))
NOISE = 1.0  # the ladder's sensor noise scale
ITER_CAP = 300  # SLSQP iterations per episode solve (ours: about 50 min on bob)


def scenario_params(name):
    s = {x.name: x for x in scenarios.suite(SPEC)}[name]
    return scenarios.params_for(SPEC, s, sensor_noise_scale=NOISE)


class Replay:
    """A plan as a ``benchmark.Policy``."""

    oracle = True

    def __init__(self, valve, params):
        to_action = jax.vmap(E.to_action, in_axes=(0, None))
        self.actions = np.asarray(to_action(jnp.asarray(valve), params)).reshape(-1)

    def reset(self, key, obs, task):
        return None

    def act(self, carry, obs, t, state=None):
        return self.actions[t].reshape(1, 1), carry


def solve(name, seed, knot_hours, start, out):
    env, params = SPEC.make_env(), scenario_params(name)
    n = int(params.max_steps_in_episode)
    reset, keys = O.episode_keys(seed, n)
    _, state = env.reset_env(reset, params)
    knot = int(round(knot_hours * 3600.0 / float(params.delta_t)))
    if start is None:
        z0, tag = None, f"{name}_{seed}_{knot_hours:g}h"
    elif start == "restart":  # from this seed's last plan, SLSQP's memory reset
        z0 = np.load(pathlib.Path(out, f"{name}_{seed}_{knot_hours:g}h_plan.npy"))
        tag = f"{name}_{seed}_{knot_hours:g}h_restart"
    else:
        z0, tag = (
            np.full(-(-n // knot), float(start)),
            f"{name}_{seed}_{knot_hours:g}h_s{start}",
        )
    t0 = time.time()
    plan = O.solve(
        env, plan_params(SPEC, params), state, keys, knot, start=z0, max_iter=ITER_CAP,
        log=lambda i, z, f, r: (
            print(f"it {i} {f:.6f} {r:.2e} {time.time() - t0:.0f}s", flush=True),
            np.save(pathlib.Path(out, f"{tag}_knots.npy"), z),
        ),
    )  # fmt: skip
    seconds = time.time() - t0
    np.save(pathlib.Path(out, f"{tag}_plan.npy"), plan.knots)
    rollout = run_policy(
        Replay(plan.valve, params), env, params, task_info(SPEC.name, env, params, 1),
        seeds=[seed],
    )  # fmt: skip
    terms = rollout["terms"]
    row = {
        "scenario": name, "seed": seed, "knot_hours": knot_hours, "start": start,
        "seconds": seconds, "status": plan.status, "message": plan.message,
        "iterations": plan.iterations, "capped": plan.capped, "kkt": plan.kkt,
        "violation": plan.violation, "active": plan.active, "how": plan.how,
        "planned_cost": plan.cost, "planned_lost": plan.lost_steps,
        "replayed_cost": float(-rollout["reward"][0].mean()),
        "replayed_lost": int((terms["tracking"][0] > 0).sum()),
        "replayed_trips": int(rollout["tripped"][0].sum()),
        "knots": [float(v) for v in plan.knots],
    }  # fmt: skip
    pathlib.Path(out, f"{tag}.json").write_text(json.dumps(row, indent=1))
    print(json.dumps({k: v for k, v in row.items() if k != "knots"}), flush=True)


def room(out, ladder, knot_hours=O.KNOT_HOURS, seeds=SEEDS):
    """Each seed's ceiling is its cheapest replayed solve that lost no glass and
    tripped nothing, over every start; it is converged only if that solve is."""
    ref = json.loads(pathlib.Path(ladder).read_text())
    kept = ref["rungs"][ref["kept"]]
    weights = scenarios.weights(SPEC)
    idx = [SEEDS.index(s) for s in seeds]
    rows, per, solves = {}, {}, {}
    for name in weights:
        got = []
        for seed in seeds:
            runs = [
                json.loads(p.read_text())
                for p in pathlib.Path(out).glob(f"{name}_{seed}_{knot_hours:g}h*.json")
            ]
            runs = [r for r in runs
                    if r["replayed_lost"] == 0 and r["replayed_trips"] == 0]  # fmt: skip
            r = min(runs, key=lambda r: r["replayed_cost"])
            solves[f"{name} {seed}"] = {
                k: r[k] for k in ("status", "message", "iterations", "capped", "kkt",
                                  "active", "how", "planned_cost", "replayed_cost")
            }  # fmt: skip
            solves[f"{name} {seed}"]["starts"] = len(runs)
            got.append(r["replayed_cost"])
        ceiling = np.array(got)
        reference = np.array(kept["scenarios"][name]["costs"])[idx]
        per[name] = ceiling
        rows[name] = {
            "weight": weights[name],
            "ceiling_per_seed": [round(float(c), 4) for c in ceiling],
            "ceiling": round(float(ceiling.mean()), 4),
            "reference": round(float(reference.mean()), 4),
            "room": round(float(reference.mean() / ceiling.mean()), 4),
        }
    j = sum(weights[n] * per[n] for n in weights)
    j_ref = np.array(kept["costs"])[idx]
    converged = [k for k, v in solves.items()
                 if v["status"] == 0 and v["kkt"] < O.KKT_TOL and not v["capped"]]  # fmt: skip
    result = {
        "task": "glass_furnace",
        "reference_rung": ref["kept"],
        "seeds": list(seeds),
        "knot_hours": knot_hours,
        "units": "EUR per step, mean over each episode, then over seeds",
        "scenarios": rows,
        "J": {
            "ceiling": round(float(j.mean()), 4),
            "reference": round(float(j_ref.mean()), 4),
            "room": round(float(j_ref.mean() / j.mean()), 4),
        },
        "converged": f"{len(converged)} of {len(solves)} solves",
        "solves": solves,
    }
    path = ROOT / "src/target_gym/data/oracle_ceilings/glass_furnace.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "solve":
        a = sys.argv[2:]
        start = None if len(a) < 5 or a[3] == "-" else a[3]
        solve(a[0], int(a[1]), float(a[2]), start, a[-1])
    else:
        seeds = (
            tuple(int(x) for x in sys.argv[5].split(","))
            if len(sys.argv) > 5
            else SEEDS
        )
        room(sys.argv[2], sys.argv[3], float(sys.argv[4]), seeds)
