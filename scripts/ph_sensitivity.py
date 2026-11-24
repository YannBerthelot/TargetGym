"""pH's sensitivity run (owner, 2026-11-24): the charter record's checks on members
of the plant that scale one group of unsourced numbers at a time.

Run with ``uv run python scripts/ph_sensitivity.py [--out FILE]``. On each member the
ladder is climbed afresh (placement, rungs, the kept one), the nulls run against the
kept rung on the ladder's seeds (J2's sanity), and the reference and the ceiling
(``oracle.perfect_hold``) run on the scoring seeds, for the room with a 95 % interval
on paired seeds. The groups are the buffer flow, the two tanks' volumes (their
residence time) and the two upset sizes, each at 0.5 and 1.5 times the shipped value.

The spans are stated, not sourced: none of these numbers has a documented family
range (``ours.json``), so this is a sensitivity and not the charter's family sweep
(B7), and the page says so. Nothing is gated on it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import jax
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

TASK = "ph_neutralization"
#: Each group's unsourced numbers, scaled together.
GROUPS = {
    "buffer": ("q2_nominal", "q2_noise_std"),
    "residence": ("V_s1", "V"),
    "upsets": ("acid_upset_size", "flow_swing"),
}
SPANS = (0.5, 1.5)
SCORING_SEEDS = tuple(range(200, 210))


def members() -> list:
    """``(name, {field: scale})``: the shipped plant, then each group at each span."""
    out = [("shipped", {})]
    for group, fields in GROUPS.items():
        for k in SPANS:
            out.append((f"{group} x{k:g}", {f: k for f in fields}))
    return out


class Scaled:
    """A registry entry whose parameters come out with ``scales`` applied after every
    change (a scenario's, a ladder row's), so a number a scenario sets to 0 stays 0;
    everything else is the entry's own."""

    def __init__(self, spec, scales):
        self._spec, self._scales = spec, dict(scales)

    def __getattr__(self, name):
        return getattr(self._spec, name)

    def make_test_params(self, **changes):
        p = self._spec.make_test_params(**changes)
        if not self._scales:
            return p
        return p.replace(**{f: getattr(p, f) * k for f, k in self._scales.items()})


def _interval(a, b, n=5000):
    rng = np.random.default_rng(0)
    i = rng.integers(0, len(a), (n, len(a)))
    return np.percentile(a[i].mean(1) / b[i].mean(1), [2.5, 97.5]).tolist()


def room(L, row, policy, spec) -> dict:
    """The reference and the ceiling on the scoring seeds, by scenario and on J."""
    from target_gym import scenarios
    from target_gym.benchmark import episode_keys
    from target_gym.pc_gym.ph_neutralization.oracle import perfect_hold

    env, weights = spec.make_env(), scenarios.weights(spec)
    ref, ceiling, out = 0.0, 0.0, {}
    episodes = L._suite_episodes(TASK, row, policy, spec, SCORING_SEEDS)
    for s, (name, eps) in zip(scenarios.suite(spec), episodes, strict=True):
        start = row.starts.get(policy.name, {})
        params = scenarios.params_for(
            spec, s, **{"sensor_noise_scale": L.NOISE, **row.score, **start}
        )
        T = int(params.max_steps_in_episode)
        costs = np.array([float(e.cost.mean()) for e in eps])
        holds = []
        for seed in SCORING_SEEDS:
            resets, steps = episode_keys(jax.random.PRNGKey(seed), 1, T)
            holds.append(perfect_hold(env, params, resets[0], steps[:, 0]))
        bound = np.array([h["cost"] for h in holds])
        ref, ceiling = ref + weights[name] * costs, ceiling + weights[name] * bound
        out[name] = {
            "reference": float(costs.mean()),
            "ceiling": float(bound.mean()),
            "trips": int(sum(bool(np.any(e.failed)) for e in eps)),
            "hold_feasible": all(h["feasible"] for h in holds),
        }
    return {
        "scenarios": out,
        "J_reference": float(ref.mean()),
        "J_ceiling": float(ceiling.mean()),
        "room": float(ref.mean() / ceiling.mean()),
        "room_interval": _interval(ref, ceiling),
    }


def run_member(L, spec, scales) -> dict:
    from target_gym.registry import REGISTRY

    REGISTRY[TASK] = Scaled(spec, scales)
    try:
        row = L.ROWS[TASK]()
        rungs = L.build(TASK, row)
        results = {k: L.run_rung(TASK, row, k, c) for k, c in rungs.items()}
        kept = L.keep_rung(row, results)
        nulls = L.check_nulls(TASK, row, results[kept]) if kept else None
        measured = room(L, row, L.Reference(kept, rungs[kept]), REGISTRY[TASK])
    finally:
        REGISTRY[TASK] = spec
    return {
        "kept": kept,
        "setpoints": {r: {k: v["setpoint"] for k, v in p.items()}
                      for r, p in row.placements.items()},
        "reference_trips": results[kept]["trips"] if kept else None,
        "null_wins": nulls["wins"] if nulls else None,
        "null_seeds": nulls["seeds"] if nulls else None,
        "null_medians": {k: float(np.median(v["costs"]))
                         for k, v in (nulls or {}).get("runs", {}).items()},
        **measured,
    }  # fmt: skip


def main() -> None:
    import reference_ladder as L

    from target_gym.registry import REGISTRY

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument("--members", nargs="*", default=None)
    args = parser.parse_args()
    spec, report = REGISTRY[TASK], {}
    for name, scales in members():
        if args.members and name not in args.members:
            continue
        print(f"== {name} {scales}", flush=True)
        report[name] = {"scales": scales, **run_member(L, spec, scales)}
        r = report[name]
        print(
            f"SENSITIVITY {name}: kept {r['kept']} at {r['setpoints'].get(r['kept'])}, "
            f"reference {r['J_reference']:.5f}, ceiling {r['J_ceiling']:.5f}, room "
            f"{r['room']:.3f} [{r['room_interval'][0]:.3f}, "
            f"{r['room_interval'][1]:.3f}], null wins {r['null_wins']}",
            flush=True,
        )
        if args.out:
            args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
