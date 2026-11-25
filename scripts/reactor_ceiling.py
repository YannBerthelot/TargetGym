"""The reactor's ceiling for its charter record: the day planner with the load's
wobble foreseen, on the protocol's runs (``target_gym.scoring_runs``: the stream
pinned to each kept scenario, 7 days from the reset), written to
``data/oracle_ceilings/reactor.json`` through ``charter.write_ceiling``.

A 7-day run takes the planner 10 to 25 minutes and 4 GB, so the 40 runs are run in
pieces and merged:

    python scripts/reactor_ceiling.py --scenario edf_daily --seeds 100 101 --out a.json
    python scripts/reactor_ceiling.py --merge a.json b.json ... [--extended baseload]

``--extended`` names the scenarios the record's stop rule gave extra seeds to.
``--kind mpc`` runs the record's planner line instead, the planner on the load's
mean path, in the same pieces; they are merged by ``record_charter.py
--mpc-pieces``.

It is a candidate, the best plan known with the wobble foreseen: a plan bounds the
best cost from above, and nothing here bounds it from below.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

TASK = "reactor"
SEEDS = tuple(range(100, 110))  # the protocol's first seeds (scoring_runs.seeds)
WHAT = (
    "each seed's mean cost per step over a 7-day run of the stream pinned to the "
    "scenario, under the day planner with the load's wobble foreseen "
    "(reactor.oracle.make_reactor_mpc with seed 0): the best plan known, not a "
    "lower bound"
)


def planner(scenario: str, kind: str = "ceiling"):
    """The day planner's params and seed for a run: the ceiling sees the run's
    world (its seed), the planner line plans on the mean path (``plan_params``);
    both plan to the run's end."""
    from target_gym import scoring_runs as S
    from target_gym.experts.mpc import plan_params
    from target_gym.registry import REGISTRY

    p = S.run_params(TASK, scenario)
    if kind == "ceiling":
        return p, 0
    if kind == "mpc":
        return plan_params(REGISTRY[TASK], p), None
    raise ValueError(f"unknown planner kind {kind!r}")


def measure(scenario: str, seeds, kind: str = "ceiling") -> dict:
    """One scenario's piece: each seed's mean cost a step, its trips, and whether
    every one of its solves converged (none failed, none stopped at its cap)."""
    from target_gym import scoring_runs as S
    from target_gym.benchmark import CallablePolicy
    from target_gym.reactor.oracle import make_reactor_mpc
    from target_gym.registry import REGISTRY

    env = REGISTRY[TASK].make_env()
    params, seed0 = planner(scenario, kind)
    oracle = make_reactor_mpc(env, params, seed=seed0)

    def make(_task):
        oracle.reset()
        return lambda obs, state: oracle.step(obs, state.env)

    policy = CallablePolicy(make, name=kind, oracle=True)
    out = {"seeds": [], "costs": [], "trips": [], "converged": []}
    for seed in seeds:
        oracle.certificates.clear()
        run = S.run(TASK, scenario, policy, [seed])
        report = oracle.solver_report()
        out["seeds"].append(int(seed))
        out["costs"].append(run["costs"][0])
        out["trips"].append(run["trips"][0])
        out["converged"].append(
            report["solver_failures"] == 0 and report["solver_capped"] == 0
        )
        print(scenario, seed, out["costs"][-1], report, flush=True)
    return {scenario: out}


def assemble(pieces, weights: dict, seeds=SEEDS) -> dict:
    """The ceilings file from the pieces: every scenario on every seed, once, or
    no file. In ``charter.ceiling_line``'s shape. ``seeds`` is one list, or
    ``{scenario: seeds}`` where the stop rule gave a scenario more."""
    want = {s: list(seeds.get(s, SEEDS) if isinstance(seeds, dict) else seeds)
            for s in weights}  # fmt: skip
    rows: dict = {s: {} for s in weights}
    for piece in pieces:
        for s, v in piece.items():
            if s not in rows:
                raise SystemExit(f"{s} is not one of the reactor's scenarios")
            for i, seed in enumerate(v["seeds"]):
                if seed in rows[s]:
                    raise SystemExit(f"{s} seed {seed} was measured twice")
                rows[s][seed] = (v["costs"][i], v["trips"][i], v["converged"][i])
    missing = {s: [x for x in want[s] if x not in r] for s, r in rows.items()}
    if any(missing.values()):
        raise SystemExit(
            f"not measured yet: { {s: m for s, m in missing.items() if m} }"
        )
    scenarios = {}
    for s, w in weights.items():
        costs, trips, ok = zip(*(rows[s][seed] for seed in want[s]))
        scenarios[s] = {
            "weight": w,
            "ceiling_costs": list(costs),
            "ceiling_trips": int(sum(trips)),
            "solver": {"converged": int(sum(ok))},
        }
    return {
        "task": TASK,
        "seeds": want,
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
    ap.add_argument("--extended", nargs="*", default=[], help="scenarios given more")
    ap.add_argument("--kind", default="ceiling", choices=("ceiling", "mpc"))
    args = ap.parse_args(argv)

    from target_gym import charter
    from target_gym import scoring_runs as S
    from target_gym.provenance import environment_fingerprint
    from target_gym.registry import REGISTRY

    spec = REGISTRY[TASK]
    weights = S.weights(TASK)
    seeds = {s: S.seeds(TASK, extended=s in args.extended) for s in weights}
    if args.merge:
        pieces = [json.loads(pathlib.Path(p).read_text()) for p in args.merge]
    else:
        names = [args.scenario] if args.scenario else list(weights)
        pieces = [measure(s, args.seeds, args.kind) for s in names]
    if args.out or args.kind == "mpc":
        if not args.out:
            raise SystemExit("a planner line is written as pieces (--out)")
        merged: dict = {}
        for piece in pieces:
            merged.update(piece)
        pathlib.Path(args.out).write_text(json.dumps(merged))
        return 0
    path = charter.write_ceiling(
        TASK, assemble(pieces, weights, seeds), environment_fingerprint(spec)
    )
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
