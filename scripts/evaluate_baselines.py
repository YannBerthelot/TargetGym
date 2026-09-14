"""Score the shipped controllers with the reach-and-hold protocol.

Run with ``uv run python scripts/evaluate_baselines.py [--envs ...] [--seeds N]``.
For every plant with a PID (and an MPC where one exists) it runs
``target_gym.eval.evaluate_controller`` on the test episode and writes
``src/target_gym/data/protocol_results.json``: the long-run cost after burn-in
split into tracking and running cost, the reach cost per target change, the
failure rate, and the normalised expert advantage of the MPC against the PID,
``NEA = (PID - MPC) / (PID - rho_floor)``. ``--table`` prints the room map
(the markdown table in the audit) from the file without re-running anything.

Unlike the recorded baselines (``record_baselines.py``, one return per
episode), this is the number the benchmark is about: what a controller pays
per step once it is holding, and what a target change costs it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from target_gym.eval import evaluate_controller, nea  # noqa: E402
from target_gym.registry import REGISTRY  # noqa: E402

OUT = ROOT / "src" / "target_gym" / "data" / "protocol_results.json"


def _fmt(x):
    if x is None or x != x:
        return "—"
    return f"{x:.3g}"


def table(rows: dict) -> str:
    lines = [
        "| plant | floor ρ* | PID gain (track / run) | MPC gain (track / run) | NEA(MPC) | PID hold | MPC hold | PID reach | MPC reach | fail |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, r in rows.items():
        pid, mpc = r.get("pid"), r.get("mpc")
        floor = r["rho_floor"]

        def cell(m):
            if not m:
                return "—"
            return f"{_fmt(m['gain'])} ({_fmt(m.get('tracking'))} / {_fmt(m.get('running'))})"

        n = nea(mpc["gain"], pid["gain"], floor) if (pid and mpc) else float("nan")
        fail = max((m["failure_rate"] for m in (pid, mpc) if m), default=float("nan"))
        lines.append(
            f"| `{name}` | {_fmt(floor)} | {cell(pid)} | {cell(mpc)} | {_fmt(n)} | "
            f"{_fmt(pid['hold']) if pid else '—'} | {_fmt(mpc['hold']) if mpc else '—'} | "
            f"{_fmt(pid['reach_cost']) if pid else '—'} | {_fmt(mpc['reach_cost']) if mpc else '—'} | {_fmt(fail)} |"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--envs", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument(
        "--table", action="store_true", help="print the room map from the file"
    )
    ap.add_argument("--out", default=str(OUT), help="merge rows into this JSON")
    args = ap.parse_args()
    out_path = pathlib.Path(args.out)
    rows = json.loads(out_path.read_text()) if out_path.exists() else {}
    if args.table:
        print(table(rows))
        return 0
    names = args.envs or [n for n, s in REGISTRY.items() if s.has_pid]
    for name in names:
        spec = REGISTRY[name]
        p = spec.make_test_params()
        row = {
            "rho_floor": float(getattr(p, "rho_floor", float("nan"))),
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
        if "pid" in row and "mpc" in row:
            row["mpc"]["nea"] = nea(
                row["mpc"]["gain"], row["pid"]["gain"], row["rho_floor"]
            )
        rows[name] = row
        out_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
