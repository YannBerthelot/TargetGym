"""The record command and the readiness it gives a freeze (task protocol, items
4 and 5).

Contract, from the protocol and the brief of 2026-11-28:

- a task is ready to freeze when its record holds every baseline (doing nothing,
  the expert, the observation-only MPC, PPO and SAC), recorded on the parts the
  tree has now, and the MPC or a learned agent beats the expert with an interval
  above zero; a freeze refuses a scored task that is not;
- ``tg.score`` takes one policy per scenario, for an MPC built on each
  scenario's model, and that is the same comparison as one policy everywhere;
- the record command scores each line with ``tg.score`` on the scoring seeds,
  and the expert beats doing nothing (item 6).

The end-to-end tests run on pH against a stand-in reference, the shipped expert
frozen on today's plant into a temporary folder, because the references on the
integration branch were frozen before PR 306 re-stamped the plants.
"""

import importlib
import pathlib
import sys

import pytest

import target_gym as tg
from target_gym import provenance, reference, v1
from target_gym.registry import REGISTRY

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
record_v1 = importlib.import_module("record_v1")

PARTS = {"plant": "a", "scoring": "b", "reference": "c"}


def _line(lo, hi):
    return {"saving": (lo + hi) / 2, "saving_ci": [lo, hi]}


def _complete(**overrides):
    lines = {line: _line(-0.1, 0.1) for line in v1.LINES}
    lines["mpc"] = _line(0.05, 0.2)
    lines.update(overrides)
    return {"parts": dict(PARTS), "lines": lines}


def test_a_complete_current_record_with_room_is_ready():
    assert v1.ready(_complete(), PARTS) == []


def test_room_from_a_learned_agent_counts():
    rec = _complete(mpc=_line(-0.1, 0.2), sac=_line(0.01, 0.3))
    assert v1.ready(rec, PARTS) == []


def test_an_interval_that_reaches_zero_is_no_room():
    rec = _complete(mpc=_line(0.0, 0.2))
    assert v1.ready(rec, PARTS) == [
        "no room: neither the MPC nor a learned agent beats the expert"
    ]


def test_a_missing_baseline_and_a_moved_part_keep_a_task_back():
    rec = _complete()
    del rec["lines"]["sac"]
    moved = {**PARTS, "plant": "z"}
    assert v1.ready(rec, moved) == [
        "plant changed since the record",
        "sac not recorded",
    ]


def test_a_task_with_no_record_is_not_ready():
    assert v1.ready(None, PARTS) == ["no v1 record: run scripts/record_v1.py"]


@pytest.fixture
def stand_in_reference(tmp_path, monkeypatch):
    """pH's shipped expert as its reference, frozen on today's plant."""
    spec = REGISTRY["ph_neutralization"]
    frozen = {
        "task": "ph_neutralization",
        "kind": "expert",
        "rung": "expert",
        "params": {},
        "stamp": provenance.environment_fingerprint(spec),
        "expert": provenance.expert_fingerprint(spec),
        "format": 1,
    }
    folder = tmp_path / "reference"  # apart from any manifest a test writes
    reference.write("ph_neutralization", frozen, folder)
    build, load = reference.build, reference.load
    monkeypatch.setattr(reference, "build", lambda task, root=folder: build(task, root))
    monkeypatch.setattr(reference, "load", lambda task, root=folder: load(task, root))
    return folder


SEEDS = (200, 201)


@pytest.mark.medium
def test_one_policy_per_scenario_is_the_same_comparison(stand_in_reference):
    ref = tg.make_reference("ph_neutralization")
    one = tg.score("ph_neutralization", ref, seeds=SEEDS)
    each = tg.score(
        "ph_neutralization",
        {s: ref for s in tg.list_scenarios("ph_neutralization")},
        seeds=SEEDS,
    )
    assert each["scenarios"] == one["scenarios"]
    with pytest.raises(ValueError, match="policies for"):
        tg.score("ph_neutralization", {"normal": ref}, seeds=SEEDS)


@pytest.mark.medium
def test_the_record_scores_doing_nothing_and_the_expert(stand_in_reference, tmp_path):
    rec = record_v1.record(
        "ph_neutralization", ["nothing", "expert"], seeds=SEEDS, root=tmp_path / "rec"
    )
    nothing, expert = rec["lines"]["nothing"], rec["lines"]["expert"]
    assert expert["saving"] == 0.0 and expert["J"] == pytest.approx(
        expert["J_reference"]
    )
    assert nothing["read_state"] and nothing["J"] > expert["J"]  # item 6
    assert rec["parts"] == v1.manifest("ph_neutralization")["parts"]
    assert expert["commit"] == record_v1._git("rev-parse", "HEAD")
    again = v1.record("ph_neutralization", tmp_path / "rec")
    assert again["lines"].keys() == {"nothing", "expert"}
    assert "mpc not recorded" in v1.ready(again, rec["parts"])


@pytest.mark.medium
def test_a_task_its_record_does_not_show_ready_is_not_frozen(
    stand_in_reference, tmp_path
):
    with pytest.raises(v1.NotReady, match="no v1 record"):
        v1.freeze("ph_neutralization", root=tmp_path, records=tmp_path / "none")
