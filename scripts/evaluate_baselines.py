"""Score the shipped controllers with the reach-and-hold protocol.

Run with ``uv run python scripts/evaluate_baselines.py [--envs ...] [--seeds N]``.
For every plant with a PID (and an MPC where one exists) it runs
``target_gym.eval.evaluate_controller`` on the test episode and writes
``src/target_gym/data/protocol_results/<task>.json``: the long-run cost after burn-in
split into tracking and running cost, the reach cost per target change, the
failure rate, with each plant's fixed settle and its ceiling
(``target_gym.scoring``). ``--table`` prints the room map from the file without
re-running anything.

Unlike the recorded baselines (``record_baselines.py``, one return per
episode), this is the number the benchmark is about: what a controller pays
per step once it is holding, and what a target change costs it.

Each row carries the same fingerprint as a recorded baseline
(``provenance.baseline_fingerprint``: the environment's modules, the shared
controller code, the gains, the parameter values and the burn-in the protocol
scores from, the hold row's capped at half the episode), so a test can read the
MPC's recorded cost instead of re-running the MPC, and refuse the row once the
code it describes has moved. ``tests/test_reward_contract.py`` does that for the
floor check.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from target_gym import records  # noqa: E402
from target_gym.eval import evaluate_controller, fixed_settle  # noqa: E402
from target_gym.provenance import baseline_fingerprint  # noqa: E402
from target_gym.registry import REGISTRY  # noqa: E402
from target_gym.scoring import ceiling, headroom, score  # noqa: E402


def _fmt(x):
    if x is None or x != x:
        return "—"
    return f"{x:.3g}"


def table(rows: dict) -> str:
    lines = [
        "| plant | ceiling | PID gain (track / run) | MPC gain (track / run) | room | MPC score | PID hold | MPC hold | PID reach B (transient) | MPC reach B (transient) | fail |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    unsettled = []
    for name, r in rows.items():
        pid, mpc = r.get("pid"), r.get("mpc")
        c = r.get("ceiling", float("nan"))

        def cell(m):
            if not m:
                return "—"
            return f"{_fmt(m['gain'])} ({_fmt(m.get('tracking'))} / {_fmt(m.get('running'))})"

        def reach(m, who):
            if not m:
                return "—"
            b = m["reach_cost"]
            mark = ""
            if b is not None and b < 0:
                mark = "†"
                unsettled.append(f"{who} on `{name}`")
            t = m.get("transient_cost")
            return (
                f"{_fmt(b)}{mark} ({_fmt(t)})" if t is not None else f"{_fmt(b)}{mark}"
            )

        b = pid["hold_tracking"] if pid else float("nan")
        room = headroom(b, c) if pid else float("nan")
        s = score(mpc["hold_tracking"], b, c) if (pid and mpc) else float("nan")
        fail = max((m["failure_rate"] for m in (pid, mpc) if m), default=float("nan"))
        lines.append(
            f"| `{name}` | {_fmt(c)} | {cell(pid)} | {cell(mpc)} | {_fmt(room)} | {_fmt(s)} | "
            f"{_fmt(pid['hold']) if pid else '—'} | {_fmt(mpc['hold']) if mpc else '—'} | "
            f"{reach(pid, 'PID')} | {reach(mpc, 'MPC')} | {_fmt(fail)} |"
        )
    lines.append("")
    lines.append(
        "Reach B is each controller's transient cost above its *own* hold level "
        "(Theorem 4's bias), so it is not comparable between two controllers "
        "whose holds differ: a controller holding far off shows a small B "
        "because its level swallows its transient. The number in parentheses "
        "is the transient's summed cost, not relative to anything, and is the "
        "one to compare across controllers."
    )
    if unsettled:
        lines.append(
            "† cost still rising at the end of the window (no hold reached, "
            'so the transient is cheaper than the "hold" level and B is '
            "negative): " + ", ".join(unsettled) + "."
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--envs", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument(
        "--table", action="store_true", help="print the room map from the file"
    )
    args = ap.parse_args()
    rows = records.load("protocol_results")
    if args.table:
        print(table(rows))
        return 0
    names = args.envs or [n for n, s in REGISTRY.items() if s.has_pid]
    for name in names:
        spec = REGISTRY[name]
        p = spec.make_test_params()
        row = {
            "fingerprint": baseline_fingerprint(spec),
            "rho_floor": float(getattr(p, "rho_floor", float("nan"))),
            "ceiling": ceiling(name, p),
            "settle": fixed_settle(name),
            "seeds": args.seeds,
        }
        for kind in ("pid", "mpc"):
            if kind == "mpc" and spec.make_mpc is None:
                continue
            t0 = time.time()
            m = evaluate_controller(name, kind, seeds=args.seeds)
            m = {k: (float(v) if v == v else None) for k, v in m.items()}
            m["seconds"] = round(time.time() - t0, 1)
            row[kind] = m
            print(
                f"  {name:20s} {kind:3s} gain={m['gain']:.4g} hold={m['hold']:.4g} "
                f"track={m.get('tracking', float('nan')):.4g} "
                f"run={m.get('running', float('nan')) or 0:.4g} reach={m['reach_cost']:.4g} "
                f"fail={m['failure_rate']:.2f} [{m['seconds']}s]",
                flush=True,
            )
        rows[name] = records.stamped(row)
        records.write("protocol_results", name, rows[name])
    print(table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
