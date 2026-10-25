"""Which recorded rows are stale, and folding re-recorded rows back in.

The CI recorder (``.github/workflows/record-baselines.yml``) runs one job per
environment and uses this to plan and to collect::

    uv run python scripts/stale_records.py list [--all]     # JSON list to record
    uv run python scripts/stale_records.py extract ENV DIR  # ENV's baseline row
    uv run python scripts/stale_records.py merge DIR...     # fold rows back in

``merge`` writes only rows that changed beyond their timings, so a weekly run
that reproduces every number leaves the data files untouched and commits
nothing.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "target_gym" / "data"
BASELINES = DATA / "baseline_returns.json"
PROTOCOL = DATA / "protocol_results.json"
TIMING = {"seconds", "generated"}


def stale(baselines: dict, protocol: dict, current: dict) -> list[str]:
    """Environments whose row in either file is missing or has moved."""
    return sorted(
        name
        for name, fp in current.items()
        if any(
            rows.get(name, {}).get("fingerprint") != fp
            for rows in (baselines, protocol)
        )
    )


def _content(row):
    if isinstance(row, dict):
        return {k: _content(v) for k, v in row.items() if k not in TIMING}
    return row


def merge(rows: dict, fragments: dict) -> tuple[dict, list[str]]:
    """``rows`` with every fragment that differs from it beyond timing."""
    changed = sorted(
        n for n, row in fragments.items() if _content(row) != _content(rows.get(n))
    )
    return {**rows, **{n: fragments[n] for n in changed}}, changed


def _read(path: pathlib.Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _write(path: pathlib.Path, rows: dict) -> None:
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").add_argument("--all", action="store_true")
    ex = sub.add_parser("extract")
    ex.add_argument("env")
    ex.add_argument("dir", type=pathlib.Path)
    sub.add_parser("merge").add_argument("dirs", nargs="+", type=pathlib.Path)
    args = ap.parse_args()

    if args.cmd == "list":
        from target_gym.provenance import baseline_fingerprint
        from target_gym.registry import REGISTRY

        specs = {n: s for n, s in REGISTRY.items() if s.has_pid}
        current = {n: baseline_fingerprint(s) for n, s in specs.items()}
        baselines = _read(BASELINES)
        # The recorder keeps no baseline row for a plant without an MPC.
        baselines.update(
            {
                n: {"fingerprint": current[n]}
                for n, s in specs.items()
                if s.make_mpc is None
            }
        )
        names = (
            sorted(current) if args.all else stale(baselines, _read(PROTOCOL), current)
        )
        print(json.dumps(names))
    elif args.cmd == "extract":
        args.dir.mkdir(parents=True, exist_ok=True)
        _write(args.dir / "baseline.json", {args.env: _read(BASELINES)[args.env]})
    else:
        moved = set()
        for path, name in ((BASELINES, "baseline.json"), (PROTOCOL, "protocol.json")):
            fragments = {}
            for d in args.dirs:
                fragments.update(_read(d / name))
            rows, changed = merge(_read(path), fragments)
            if changed:
                if "_meta" in rows:
                    stamp = _dt.datetime.now(_dt.timezone.utc).isoformat(
                        timespec="seconds"
                    )
                    rows["_meta"] = {**rows["_meta"], "generated": stamp}
                _write(path, rows)
            moved.update(changed)
        print(" ".join(sorted(moved)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
