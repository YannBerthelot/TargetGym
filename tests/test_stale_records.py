"""``scripts/stale_records.py``: which records the CI recorder must redo, and
how it folds the per-environment rows its matrix jobs return back in.

The recording workflow (``.github/workflows/record-baselines.yml``) runs one
job per environment and commits the result, so these two decisions are what
stand between a re-record and a wrong commit: re-recording too little leaves a
stale row that fails every PR, and treating a timing-only change as a change
would open a bot PR every week for nothing.
"""

from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "stale_records_under_test", ROOT / "scripts" / "stale_records.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stale_lists_each_moved_or_missing_part_in_either_file():
    """A (task, part) pair is stale when that part's fingerprint is missing
    or has moved in either file. A task without an oracle has only its expert
    part, and no baseline row."""
    sr = _load()
    both = {"expert_fingerprint": "e1", "oracle_fingerprint": "o1"}
    current = {
        "a": {"expert": "e1", "oracle": "o1"},
        "b": {"expert": "e1", "oracle": "o2"},
        "c": {"expert": "e2", "oracle": "o1"},
        "d": {"expert": "e1", "oracle": "o1"},
        "e": {"expert": "e1"},
    }
    baselines = {"a": both, "b": both, "c": both, "d": {"expert_fingerprint": "e1"}}
    protocol = {n: both for n in "abcde"}
    # b's oracle and c's expert moved; d's baseline row has no oracle
    # fingerprint; e has no oracle, so no baseline row is expected.
    assert sr.stale(baselines, protocol, current) == [
        ("b", "oracle"),
        ("c", "expert"),
        ("d", "oracle"),
    ]


def test_the_recorder_runs_one_job_per_task_on_its_stale_parts_only():
    sr = _load()
    pairs = [("b", "oracle"), ("c", "expert"), ("c", "oracle")]
    assert sr.jobs(pairs) == [
        {"env": "b", "parts": "oracle"},
        {"env": "c", "parts": "expert oracle"},
    ]


def test_merge_reports_only_rows_that_changed_beyond_timing():
    sr = _load()
    rows = {
        "_meta": {"generated": "then"},
        "a": {"fingerprint": "f1", "pid_returns": [1.0], "seconds": 10.0},
        "b": {"fingerprint": "f2", "pid_returns": [2.0], "seconds": 10.0},
    }
    fragments = {
        "a": {"fingerprint": "f1", "pid_returns": [1.0], "seconds": 99.0},
        "b": {"fingerprint": "f9", "pid_returns": [2.0], "seconds": 10.0},
        "c": {"fingerprint": "f3", "pid_returns": [3.0], "seconds": 1.0},
    }
    merged, changed = sr.merge(rows, fragments)
    assert changed == ["b", "c"]
    assert merged["b"]["fingerprint"] == "f9" and merged["c"]["pid_returns"] == [3.0]
    assert merged["a"]["seconds"] == 10.0, "a timing-only change is not written"
    assert rows["b"]["fingerprint"] == "f2", "the input is not mutated"


def test_merge_compares_nested_protocol_rows_without_their_timings():
    sr = _load()
    rows = {"a": {"fingerprint": "f1", "pid": {"gain": 1.0, "seconds": 2.0}}}
    same = {"a": {"fingerprint": "f1", "pid": {"gain": 1.0, "seconds": 7.0}}}
    moved = {"a": {"fingerprint": "f1", "pid": {"gain": 1.5, "seconds": 2.0}}}
    assert sr.merge(rows, same)[1] == []
    assert sr.merge(rows, moved)[1] == ["a"]
