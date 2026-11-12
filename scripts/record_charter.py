"""Fill or refresh parts of a task's charter record (docs/proposals/task-charter.md,
J1 to J4; schema and rules in ``target_gym.charter``).

    uv run python scripts/record_charter.py --tasks cstr [--parts j2,mpc] [--force]

Parts: ``reference`` (J1: the ladder's ``climb``, or the shipped expert on the
expert-reference tasks), ``nulls`` (each of ``null_set`` over the ladder seeds),
``mpc`` (the task's oracle, ``shipped_policy("mpc")``), ``ceiling`` (read from
``data/oracle_ceilings``, which ``pc_gym.foresight`` and the tasks' ceiling scripts
write through ``charter.write_ceiling``; a file measured on another plant reads
"not measured"), ``j2`` (the kept reference and every null on the commit's hash
seeds, and the family sweep, ``scripts/family_sweep.py``, on the task's
``ours.json``) and ``anchor`` (J4: the reference on the
scoring seeds). J3's lines run on the ladder seeds 100 to 109, as the ceilings do.
``nulls``, ``j2`` and ``anchor`` read the recorded reference.

A part whose fingerprint is current is skipped unless ``--force``. Each part run
is written at once and appended to ``data/charter/log.jsonl``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from target_gym import charter  # noqa: E402

NM = charter.NOT_MEASURED


def git_commit() -> tuple[str, bool]:
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)

    status = git("status", "--porcelain", "--untracked-files=no").stdout
    return git("rev-parse", "HEAD").stdout.strip(), bool(status.strip())


class Task:
    """What every part of one task's run shares: the ladder module, its row, the
    scenario weights and the record so far."""

    def __init__(self, name: str, record: dict):
        import reference_ladder as L

        from target_gym import scenarios
        from target_gym.registry import REGISTRY

        self.L, self.name, self.record = L, name, record
        self.spec = REGISTRY[name]
        self.weights = scenarios.weights(self.spec)
        self.expert = name in charter.EXPERT_REFERENCE
        self.row = L.Row(0, None) if self.expert else L.ROWS[name]()

    def line(self, result: dict, converged=None) -> dict:
        return charter.line(result["scenarios"], self.weights, converged)

    def kept(self) -> str:
        ref = self.record.get("reference")
        if not ref or not ref.get("kept"):
            raise SystemExit(f"{self.name}: record a reference with a kept rung first")
        return ref["kept"]

    def policy(self):
        from target_gym.benchmark import shipped_policy

        kept = self.kept()
        if self.expert:
            return shipped_policy("pid")
        return self.L.Reference(kept, self.L.build(self.name, self.row)[kept])

    def nulls(self, seeds=None) -> tuple[dict, dict]:
        runs, broken = {}, {}
        for null in self.L.null_set(self.name):
            try:
                runs[null.name] = self.L.run_suite(
                    self.name, self.row, null, seeds=seeds
                )
            except Exception as err:  # a stale table's shape, as check_nulls says
                broken[null.name] = f"{type(err).__name__}: {err}"
        return runs, broken


def part_reference(t: Task) -> dict:
    from target_gym.benchmark import shipped_policy

    if t.expert:
        res = t.L.run_suite(t.name, t.row, shipped_policy("pid"))
        out = {"kept": "expert", "rungs": {"expert": res}, "left_out": {}}
    else:
        out = t.L.climb(t.name)
    kept = out["rungs"].get(out["kept"]) if out["kept"] else None
    rungs = {
        k: {"J": t.line(v)["J"], "trips": v["trips"]} for k, v in out["rungs"].items()
    }
    return {
        "kind": "expert" if t.expert else "ladder",
        "kept": out["kept"],
        "rungs": rungs,
        "left_out": out["left_out"],
        "line": t.line(kept) if kept else NM,
    }


def part_nulls(t: Task) -> dict:
    runs, broken = t.nulls()
    return {"lines": {k: t.line(v) for k, v in runs.items()}, "broken": broken}


def part_mpc(t: Task) -> dict:
    from target_gym.benchmark import shipped_policy

    why = charter.MPC_NOT_MEASURED.get(t.name)
    if why or t.spec.make_mpc is None:
        return {"line": NM, "why": why or "the task ships no oracle"}
    return {"line": t.line(t.L.run_suite(t.name, t.row, shipped_policy("mpc")))}


def part_ceiling(t: Task) -> dict:
    from target_gym.provenance import environment_fingerprint

    stored = charter.load_ceiling(t.name)
    part = charter.ceiling_part(stored, environment_fingerprint(t.spec), t.weights)
    source = charter.CEILINGS / f"{t.name}.json"
    return part | {"source": str(source.relative_to(charter.REPO)) if stored else None}


def part_j2(t: Task) -> dict:
    commit, dirty = git_commit()
    seeds = charter.j2_seeds(commit, t.weights)
    ref = t.L.run_suite(t.name, t.row, t.policy(), seeds=seeds)
    runs, broken = t.nulls(seeds)
    costs = {s: v["costs"] for s, v in ref["scenarios"].items()}
    wins = {
        k: charter.null_wins(costs, {s: v["costs"] for s, v in r["scenarios"].items()})
        for k, r in runs.items()
    }
    n = sum(len(v) for v in seeds.values())
    trips = sum(v["trips"] for v in ref["scenarios"].values())
    import family_sweep

    table = family_sweep.table(t.name)
    draws = None if table is None else family_sweep.run(t.name, table)
    sweep = charter.sweep_result(draws)
    return {
        "commit": commit,
        "dirty": dirty,
        "noise": t.L.NOISE,
        "seeds": seeds,
        "reference_trips": trips,
        "reference_trips_by_scenario": {
            s: v["trips"] for s, v in ref["scenarios"].items()
        },
        "null_wins": wins,
        "null_share": {k: w / n for k, w in wins.items()},
        "null_trips": {k: r["trips"] for k, r in runs.items()},
        "broken": broken,
        "sweep": sweep,
        "sweep_draws": draws,
        **charter.verdict(n, trips, wins, sweep),
    }


def part_anchor(t: Task) -> dict:
    seeds = list(charter.ANCHOR_SEEDS)
    res = t.L.run_suite(t.name, t.row, t.policy(), seeds=seeds)
    return {"seeds": seeds, "line": t.line(res)}


PARTS = {
    "reference": part_reference,
    "nulls": part_nulls,
    "mpc": part_mpc,
    "ceiling": part_ceiling,
    "j2": part_j2,
    "anchor": part_anchor,
}


def summary(part: str, block: dict) -> dict:
    if part == "j2":
        keys = ("reference_trips", "null_wins", "sweep", "accepted", "reasons")
        return {k: block[k] for k in keys}
    ln = block.get("line")
    if part == "nulls":
        return {k: v["J"] for k, v in block["lines"].items()}
    return {"J": ln["J"], "trips": ln["trips"]} if isinstance(ln, dict) else ln


def record(name: str, parts, force: bool) -> list[str]:
    """Run ``parts`` on ``name``, skipping current ones unless ``force``; returns
    the record's schema errors."""
    now = charter.current(name)
    rec = charter.load(name) or {}
    stale = charter.stale_parts(rec, now)
    if rec.get("version") != now["version"]:
        rec = {}
    t = Task(name, rec)
    rec.update(task=name, version=now["version"], weights=t.weights)
    commit, dirty = git_commit()
    for part in [p for p in charter.PARTS if p in parts]:
        if part not in stale and not force:
            print(f"{name} {part}: current, skipped", flush=True)
            continue
        block = PARTS[part](t)
        stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        )
        rec[part] = {
            **block,
            "fingerprint": now["fingerprints"][part],
            "generated": stamp,
        }
        charter.write(name, rec)
        charter.append_log(
            {"task": name, "part": part, "commit": commit, "dirty": dirty}
            | {"generated": stamp, "version": now["version"]}
            | {
                "fingerprint": now["fingerprints"][part],
                "summary": summary(part, block),
            }
        )
        print(f"{name} {part}: {json.dumps(summary(part, block))}", flush=True)
    return charter.validate(rec)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--parts", default=",".join(charter.PARTS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    parts = [p for p in args.parts.split(",") if p]
    unknown = sorted(set(parts) - set(charter.PARTS))
    if unknown:
        ap.error(f"unknown parts {unknown}; choose from {charter.PARTS}")
    bad = 0
    for name in args.tasks:
        errors = record(name, parts, args.force)
        for e in errors:
            print(f"{name}: schema: {e}", file=sys.stderr)
        bad += bool(errors)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
