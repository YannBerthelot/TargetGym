"""The recorded data, one JSON file per task.

Each kind of record has a directory of its own under ``data/``, holding one
file per task (``env_versions`` one per version, ``<task>-vN.json``)::

    data/baseline_returns/<task>.json    scripts/record_baselines.py
    data/protocol_results/<task>.json    scripts/evaluate_baselines.py
    data/hold_measurements/<task>.json   scripts/measure_hold.py
    data/env_versions/<task>-vN.json     scripts/stamp_env_versions.py

They used to be one file per kind, which every recording PR rewrote, so each
merge broke every other open one. One file per task means re-recording a task
touches that task's files only.

A recording run notes the CPU it ran on (``stamped``): GitHub's runners do not
all have the same one, and the last bits of a long rollout differ between
them. Compare two recordings with ``same``, never with ``==``.

Not hashed into any fingerprint (``provenance``), so a change here cannot
stale a record.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import pathlib
import platform
import subprocess

__all__ = ["DATA", "KINDS", "load", "load_row", "path", "same", "stamped", "write"]

#: Inside the package, so it resolves from a checkout and an installed wheel.
DATA = pathlib.Path(__file__).resolve().parent / "data"

KINDS = ("baseline_returns", "protocol_results", "hold_measurements", "env_versions")

#: What only describes a recording run, left out when two are compared.
METADATA = frozenset({"seconds", "generated", "cpu"})

#: Relative tolerance between two recordings of one row. #97's CI re-record
#: moved plane3d_racetrack's PID hold from 12.661 to 12.676 (1.1e-3) and its
#: time in band by one step in 440 (6.3e-3) on a runner with another CPU; 1 %
#: clears both, and a dependency that moves a cost by more still shows.
REL_TOL = 1e-2
#: Absolute floor: costs this small are float noise around an exact zero
#: (first_order's expert holds at 3e-12).
ABS_TOL = 1e-12


def path(kind: str, stem: str) -> pathlib.Path:
    """The file holding ``stem``'s row of ``kind``."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind of record {kind!r}; one of {KINDS}")
    return DATA / kind / f"{stem}.json"


def load(kind: str) -> dict:
    """Every row of ``kind``, ``{stem: row}`` in name order."""
    directory = path(kind, "_").parent
    return {p.stem: json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))}


def load_row(kind: str, stem: str) -> dict:
    """``stem``'s row of ``kind``, or ``{}`` if it has none."""
    p = path(kind, stem)
    return json.loads(p.read_text()) if p.exists() else {}


def write(kind: str, stem: str, row: dict) -> pathlib.Path:
    """Write ``stem``'s row of ``kind`` and nothing else."""
    p = path(kind, stem)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    return p


def same(a, b) -> bool:
    """Whether two recordings of a row agree: the same keys and strings, and
    every number within ``REL_TOL`` (or ``ABS_TOL``), ``METADATA`` left out."""
    if isinstance(a, dict) and isinstance(b, dict):
        keys = (a.keys() | b.keys()) - METADATA
        return all(k in a and k in b and same(a[k], b[k]) for k in keys)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if _number(a) and _number(b):
        return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)
    return a == b


def _number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def stamped(row: dict) -> dict:
    """``row`` with the CPU it was recorded on and when."""
    now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    return {**row, "cpu": cpu_model(), "generated": now}


def cpu_model() -> str:
    """This machine's CPU model as the system names it."""
    try:
        if platform.system() == "Linux":
            for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            brand = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            return brand.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine() or "unknown"
