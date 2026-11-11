"""Which recorded rows are stale, and folding re-recorded rows back in.

The CI recorder (``.github/workflows/record-baselines.yml``) runs one job per
environment and uses this to plan and to collect::

    uv run python scripts/stale_records.py list [--all]     # JSON jobs to record
    uv run python scripts/stale_records.py extract ENV DIR  # copy ENV's rows to DIR
    uv run python scripts/stale_records.py merge DIR...     # fold rows back in

A row has two parts, the expert's and the oracle's, each with its own
fingerprint (``target_gym.provenance``), so ``list`` names the stale parts of
each environment and its job re-records those alone (``--part expert`` or
``--part oracle`` to record_baselines.py and evaluate_baselines.py, neither
when both are stale). ``list`` also prints each charter record's stale parts
(``target_gym.charter``) to stderr as ``{"charter": {task: parts}}``, so stdout
stays the jobs; a task with no record lists every part.

``merge`` writes only the tasks whose rows moved beyond the tolerance
(``target_gym.records.same``), and only their files, so a weekly run that
reproduces every number, on whichever CPU the runner has, commits nothing.

``migrate`` moved the rows recorded under the single ``fingerprint`` to the two
parts' fingerprints without re-recording anything (2026-11-03), after
checking that every such fingerprint matched the tree.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from target_gym import records

#: The kinds of record a re-record writes.
KINDS = ("baseline_returns", "protocol_results", "null_policies", "credibility")


def stale(baselines: dict, protocol: dict, current: dict) -> list[tuple[str, str]]:
    """``(environment, part)`` pairs whose fingerprint is missing or has moved
    in either file. ``current`` is ``{environment: {part: fingerprint}}``; an
    environment without an oracle has no baseline row."""
    return sorted(
        (name, part)
        for name, parts in current.items()
        for part, fp in parts.items()
        if any(
            rows.get(name, {}).get(f"{part}_fingerprint") != fp
            for rows in ((baselines, protocol) if "oracle" in parts else (protocol,))
        )
    )


def jobs(pairs) -> list[dict]:
    """One recording job per environment, naming its stale parts."""
    parts: dict[str, list[str]] = {}
    for name, part in sorted(pairs):
        parts.setdefault(name, []).append(part)
    return [{"env": n, "parts": " ".join(p)} for n, p in parts.items()]


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
    sub.add_parser("migrate")
    args = ap.parse_args(argv)

    if args.cmd in ("list", "migrate"):
        from target_gym.provenance import fingerprints
        from target_gym.registry import REGISTRY

        current = {n: fingerprints(s) for n, s in REGISTRY.items() if s.has_pid}
    if args.cmd == "list":
        pairs = (
            [(n, part) for n, parts in current.items() for part in parts]
            if args.all
            else stale(
                records.load("baseline_returns"),
                records.load("protocol_results"),
                current,
            )
        )
        print(json.dumps(jobs(pairs)))
        from target_gym import charter

        print(json.dumps({"charter": charter.stale_report()}), file=sys.stderr)
    elif args.cmd == "migrate":
        for kind in KINDS:
            for name, row in records.load(kind).items():
                if "fingerprint" in row:
                    row = {k: v for k, v in row.items() if k != "fingerprint"}
                    new = {f"{p}_fingerprint": fp for p, fp in current[name].items()}
                    records.write(kind, name, {**row, **new})
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
