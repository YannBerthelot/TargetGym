"""The horizon files (H13): a compact summary per task in the package, and
the full evidence outside it, in git but not in the wheel.

Contract (docs/proposals/oracle-horizon-protocol.md, H13): the summary in
``data/oracle_horizons/<task>.json`` holds T_c, N, M, P and the pair and seed
that set each when the task is compliant, none of them when it is not, the
class per CV and the label counts, in under a kilobyte; it names the evidence
file, which lives under ``evidence/`` at the repository root.
"""

import json
import pathlib

import pytest

from target_gym import records

ROOT = pathlib.Path(__file__).resolve().parents[2]
SUMMARIES = sorted((records.DATA / "oracle_horizons").glob("*.json"))
HORIZON = ("T_c_steps", "N", "M", "P", "T_c_set_by", "P_set_by")


@pytest.mark.parametrize("path", SUMMARIES, ids=lambda p: p.stem)
def test_a_summary_is_compact_and_its_evidence_is_outside_the_package(path):
    row = json.loads(path.read_text())
    assert path.stat().st_size < 1024
    assert {"compliant", "classes", "counts", "protocol", "cpu", "generated"} <= set(
        row
    )
    assert all(k in row for k in HORIZON) == row["compliant"]
    assert not any(k in row for k in HORIZON) or row["compliant"]
    evidence = ROOT / row["evidence"]
    assert evidence.is_file() and records.DATA.parent not in evidence.parents
    assert "pairs" in json.loads(evidence.read_text())
