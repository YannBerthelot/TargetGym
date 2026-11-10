"""The horizon files (H13): a compact summary per task in the package, and
the full evidence outside it, written by the run but kept out of git and the
wheel (owner, 2026-11-10).

Contract (docs/proposals/oracle-horizon-protocol.md, H13): the summary in
``data/oracle_horizons/<task>.json`` holds T_c, N, M, P and the pair and seed
that set each when the task is compliant, none of them when it is not, the
class per CV and the label counts, in under a kilobyte; it names the evidence
file, under ``evidence/`` at the repository root. A checkout has the evidence
only where the measuring script ran, so its contents are checked when present.
"""

import json
import pathlib

import pytest

from target_gym import records

ROOT = pathlib.Path(__file__).resolve().parents[2]
SUMMARIES = sorted((records.DATA / "oracle_horizons").glob("*.json"))
HORIZON = ("T_c_steps", "N", "M", "P", "T_c_set_by", "P_set_by")


@pytest.mark.parametrize("path", SUMMARIES, ids=lambda p: p.stem)
def test_a_summary_is_compact_and_names_evidence_outside_the_package(path):
    row = json.loads(path.read_text())
    assert path.stat().st_size < 1024
    assert {"compliant", "classes", "counts", "protocol", "cpu", "generated"} <= set(
        row
    )
    assert all(k in row for k in HORIZON) == row["compliant"]
    assert not any(k in row for k in HORIZON) or row["compliant"]
    evidence = ROOT / row["evidence"]
    assert evidence.parent == ROOT / "evidence" / "oracle_horizons"
    assert records.DATA.parent not in evidence.parents
    if evidence.is_file():
        assert "pairs" in json.loads(evidence.read_text())
