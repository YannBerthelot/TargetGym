"""The checks a ``re-record`` fixes run in a CI job of their own, ``records``.

They fail whenever a change leaves a recorded row stale, which says nothing
about whether the change is right. Kept apart, a stale record fails the
``records`` check alone, and a failing test job always means a real bug.
"""

from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "python-app.yml"

#: Every test that reads the rows the ``re-record`` label writes
#: (``baseline_returns.json`` and ``protocol_results.json``), by file.
RECORD_CHECKS = {
    "tests/experts/test_mpc_baselines.py": {
        "test_recorded_baseline_still_describes_this_tree",
        "test_mpc_controls_at_least_as_well_as_the_pid",
    },
    "tests/test_reward_contract.py": {
        "test_recorded_protocol_row_still_describes_this_tree",
        "test_mpc_does_not_beat_a_measured_floor",
        "test_the_recorded_mpc_cost_is_what_the_mpc_measures_now",
    },
}


def _marked_records(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any("mark.records" in ast.unparse(d) for d in node.decorator_list)
    }


def test_the_record_checks_and_only_they_are_marked():
    for rel, names in RECORD_CHECKS.items():
        assert _marked_records(ROOT / rel) == names, rel


def test_ci_runs_the_record_checks_in_their_own_job():
    runs = [line for line in WORKFLOW.read_text().splitlines() if "run: " in line]
    runs = [line for line in runs if "pytest" in line]
    own = [line for line in runs if "-m records" in line]
    assert len(own) == 1, own
    rest = [line for line in runs if line not in own]
    assert rest and all("not records" in line for line in rest), rest
