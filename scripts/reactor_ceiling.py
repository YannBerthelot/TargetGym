"""The reactor's ceiling for its charter record: the day planner with the load's
wobble foreseen (``make_reactor_mpc(env, params, seed=...)``), on the ladder's
seeds, written to ``data/oracle_ceilings/reactor.json`` through
``charter.write_ceiling``.

A day takes the planner 5 to 10 minutes and 3 to 4 GB, so the 60 days are run in
pieces and merged:

    python scripts/reactor_ceiling.py --scenario edf_daily --seeds 100 101 --out a.json
    python scripts/reactor_ceiling.py --merge a.json b.json ...

or all in one process, some eight hours:

    python scripts/reactor_ceiling.py

It is a candidate, the best plan known with the wobble foreseen: a plan bounds the
best cost from above, and nothing here bounds it from below.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

TASK = "reactor"
SEEDS = tuple(range(100, 110))  # the ladder's (reference_ladder.SEEDS)
NOISE = 1.0  # the sensors' noise as scored (reference_ladder.NOISE)
WHAT = (
    "each seed's cost under the day planner with the load's wobble foreseen "
    "(reactor.oracle.make_reactor_mpc with the episode's seed): the best plan "
    "known, not a lower bound"
)


def measure(scenario: str, seeds) -> dict:
    """One scenario's piece: each seed's mean cost a step, its trips, and whether
    every one of its solves converged (none failed, none stopped at its cap)."""
    from target_gym import scenarios
    from target_gym.benchmark import (
        CallablePolicy,
        episodes_of,
        run_policy,
        task_info,
    )
    from target_gym.reactor.oracle import make_reactor_mpc
    from target_gym.registry import REGISTRY

    spec = REGISTRY[TASK]
    env = spec.make_env()
    (sc,) = [s for s in scenarios.suite(spec) if s.name == scenario]
    params = scenarios.params_for(spec, sc, sensor_noise_scale=NOISE)
    oracle = make_reactor_mpc(env, params, seed=0)

    def make(_task):
        oracle.reset()
        return oracle.step

    policy = CallablePolicy(make, name="ceiling", oracle=True)
    out = {"seeds": [], "costs": [], "trips": [], "converged": []}
    for seed in seeds:
        oracle.certificates.clear()
        task = task_info(TASK, env, params, 1)
        (ep,) = episodes_of(run_policy(policy, env, params, task, seeds=[seed]))
        report = oracle.solver_report()
        out["seeds"].append(int(seed))
        out["costs"].append(float(ep.cost.mean()))
        out["trips"].append(int(np.any(ep.failed)))
        out["converged"].append(
            report["solver_failures"] == 0 and report["solver_capped"] == 0
        )
        print(scenario, seed, out["costs"][-1], report, flush=True)
    return {scenario: out}


def assemble(pieces, weights: dict, seeds=SEEDS) -> dict:
    """The ceilings file from the pieces: every scenario on every seed, once, or
    no file. In ``charter.ceiling_line``'s shape."""
    rows: dict = {s: {} for s in weights}
    for piece in pieces:
        for s, v in piece.items():
            if s not in rows:
                raise SystemExit(f"{s} is not one of the reactor's scenarios")
            for i, seed in enumerate(v["seeds"]):
                if seed in rows[s]:
                    raise SystemExit(f"{s} seed {seed} was measured twice")
                rows[s][seed] = (v["costs"][i], v["trips"][i], v["converged"][i])
    missing = {s: [x for x in seeds if x not in r] for s, r in rows.items()}
    if any(missing.values()):
        raise SystemExit(
            f"not measured yet: { {s: m for s, m in missing.items() if m} }"
        )
    scenarios = {}
    for s, w in weights.items():
        costs, trips, ok = zip(*(rows[s][seed] for seed in seeds))
        scenarios[s] = {
            "weight": w,
            "ceiling_costs": list(costs),
            "ceiling_trips": int(sum(trips)),
            "solver": {"converged": int(sum(ok))},
        }
    return {
        "task": TASK,
        "seeds": list(seeds),
        "units": "dollars per step, mean over each episode",
        "ceiling": WHAT,
        "scenarios": scenarios,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--out", help="write this piece here instead of the record")
    ap.add_argument("--merge", nargs="+", help="pieces to merge into the record")
    args = ap.parse_args(argv)

    from target_gym import charter, scenarios
    from target_gym.provenance import environment_fingerprint
    from target_gym.registry import REGISTRY

    spec = REGISTRY[TASK]
    weights = scenarios.weights(spec)
    if args.merge:
        pieces = [json.loads(pathlib.Path(p).read_text()) for p in args.merge]
    else:
        names = [args.scenario] if args.scenario else list(weights)
        pieces = [measure(s, args.seeds) for s in names]
    if args.out:
        merged: dict = {}
        for piece in pieces:
            merged.update(piece)
        pathlib.Path(args.out).write_text(json.dumps(merged))
        return 0
    path = charter.write_ceiling(
        TASK, assemble(pieces, weights), environment_fingerprint(spec)
    )
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
