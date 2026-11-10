"""Measure each task's slow mode ``t_s`` (docs/proposals/long-strata.md, R1).

    python scripts/slow_modes.py --tasks cstr first_order --out slow_modes.json

For each task: ten coupled pairs under the shipped reference (``pid``), on seeds
100 to 109, run B in manual mode for half a window, then ``--windows`` windows of
the reference on both. Prints and writes ``t_s``, the slowest element, the
elements that never forgot (to name in ``IGNORE`` if they are bookkeeping) and the
start-up excess cost. Heavy on the slow plants: run it on bob.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp

import numpy as np

#: Bookkeeping per task, never plant state: name -> reason. Filled from the first
#: run's ``never_forgot`` lists, each entry checked against the task's env code.
IGNORE: dict[str, dict[str, str]] = {}

SEEDS = range(100, 110)


def measure(name: str, windows: float, offset_share: float, pre: float) -> dict:
    from target_gym.registry import REGISTRY
    from target_gym.runners.runners import baseline_policy
    from target_gym.slow_mode import coupled_pair, slow_mode

    spec = REGISTRY[name]
    env, params = spec.make_env(), spec.make_test_params()
    space = env.action_space(params)
    low, high = np.asarray(space.low, float), np.asarray(space.high, float)
    n = int(params.max_steps_in_episode)
    pairs = [
        coupled_pair(
            env,
            params,
            lambda p: baseline_policy(spec, "pid", p),
            seed=k,
            pre=int(pre * n),
            length=int(windows * n),
            offset=offset_share * (high - low),
            low=low,
            high=high,
        )
        for k in SEEDS
    ]
    result = slow_mode(pairs, ignore=tuple(IGNORE.get(name, {})))
    return {
        "task": name,
        "N": n,
        "t_s": result.t_s,
        "slowest": result.slowest,
        "never_forgot": list(result.never_forgot),
        "start_excess": result.start_excess,
        "tripped_pairs": result.tripped_pairs,
        "pre": pre,
        "offset": offset_share,
        "ignored": IGNORE.get(name, {}),
        "windows": windows,
        "seeds": list(SEEDS),
    }


def main() -> None:
    from target_gym.registry import REGISTRY

    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=list(REGISTRY))
    ap.add_argument("--windows", type=float, default=3.0)
    ap.add_argument("--offset", type=float, default=0.1, help="share of the span")
    ap.add_argument("--pre", type=float, default=0.5, help="share of a window")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="slow_modes.json")
    args = ap.parse_args()
    jobs = [(t, args.windows, args.offset, args.pre) for t in args.tasks]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = []
        for row in pool.starmap(measure, jobs):
            print(json.dumps(row), flush=True)
            rows.append(row)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
