"""Which recorded rows are stale, and folding re-recorded rows back in.

The CI recorder (``.github/workflows/record-baselines.yml``) runs one job per
environment and uses this to plan and to collect::

    uv run python scripts/stale_records.py list [--all]     # JSON list to record
    uv run python scripts/stale_records.py extract ENV DIR  # copy ENV's rows to DIR
    uv run python scripts/stale_records.py merge DIR...     # fold rows back in

``merge`` writes only the tasks whose rows moved beyond the tolerance
(``target_gym.records.same``), and only their files, so a weekly run that
reproduces every number, on whichever CPU the runner has, commits nothing.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from target_gym import records

#: The kinds of record a re-record writes.
KINDS = ("baseline_returns", "protocol_results")


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


def merge(rows: dict, fragments: dict) -> tuple[dict, list[str]]:
    """``rows`` with every fragment that differs from it beyond the tolerance
    and the recording's own metadata (``records.same``)."""
    changed = sorted(
        n for n, row in fragments.items() if not records.same(rows.get(n), row)
    )
    return {**rows, **{n: fragments[n] for n in changed}}, changed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").add_argument("--all", action="store_true")
    ex = sub.add_parser("extract")
    ex.add_argument("env")
    ex.add_argument("dir", type=pathlib.Path)
    sub.add_parser("merge").add_argument("dirs", nargs="+", type=pathlib.Path)
    args = ap.parse_args(argv)

    if args.cmd == "list":
        from target_gym.provenance import baseline_fingerprint
        from target_gym.registry import REGISTRY

        specs = {n: s for n, s in REGISTRY.items() if s.has_pid}
        current = {n: baseline_fingerprint(s) for n, s in specs.items()}
        baselines = records.load("baseline_returns")
        # The recorder keeps no baseline row for a plant without an MPC.
        baselines.update(
            {
                n: {"fingerprint": current[n]}
                for n, s in specs.items()
                if s.make_mpc is None
            }
        )
        names = (
            sorted(current)
            if args.all
            else stale(baselines, records.load("protocol_results"), current)
        )
        print(json.dumps(names))
    elif args.cmd == "extract":
        for kind in KINDS:
            source = records.path(kind, args.env)
            if source.exists():
                target = args.dir / kind / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
    else:
        moved = set()
        for kind in KINDS:
            fragments = {
                p.stem: json.loads(p.read_text())
                for d in args.dirs
                for p in sorted((d / kind).glob("*.json"))
            }
            _, changed = merge(records.load(kind), fragments)
            for name in changed:
                records.write(kind, name, fragments[name])
            moved.update(changed)
        print(" ".join(sorted(moved)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
