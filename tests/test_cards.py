"""Task cards (task protocol, item 8): one page per task, generated.

Contract, from the brief of 2026-11-28: a card says the job and the decision,
the score with its scenarios and their weights, the baselines' numbers, which
prices are sourced and where, what the task does not model, and that the room is
an upper bound. Its numbers come from the tree (the scoring definition, the v1
record, the scored configuration), its prose from the inputs the task's session
writes, and the Gymnasium ID appears once the task is frozen. Every card on disk
is the one its inputs and the tree give.
"""

import importlib
import json
import pathlib
import sys

import pytest

import target_gym as tg
from target_gym import reference, v1

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
cards = importlib.import_module("generate_cards")

TASK = "compressor_surge"
INPUTS = {
    "job": "A station operator holds the header pressure with the drive speed.",
    "decision": "Every 0.1 s, the speed command and the recycle command.",
    "unit": "kJ",
    "prices": [
        {"field": "zone_price", "trades": "a second outside the zone", "unit": "kJ/s"},
        {
            "field": "energy_price",
            "trades": "a kJ of the compressor's work",
            "unit": "-",
            "source": "the job's own unit",
        },
    ],
    "not_modelled": ["the drive's inertia", "a multistage map"],
}


def _line(saving, lo, hi, trips=0):
    return {
        "J": 3.0 * (1 - saving),
        "saving": saving,
        "saving_ci": [lo, hi],
        "scenarios": [{"policy_trips": trips}],
    }


@pytest.fixture
def recorded(tmp_path):
    rec = {
        "task": TASK,
        "stamp": "compressor_surge-v5",
        "commit": "0123456789abcdef",
        "parts": v1.manifest(TASK)["parts"],
        "lines": {
            "nothing": _line(-36.0, -40.0, -32.0),
            "expert": _line(0.0, 0.0, 0.0),
            "mpc": _line(0.123, 0.05, 0.2, trips=1),
        },
    }
    (tmp_path / f"{TASK}.json").write_text(json.dumps(rec))
    return tmp_path


def test_a_card_says_the_job_the_score_and_the_baselines(recorded, tmp_path):
    text = cards.card(TASK, INPUTS, records=recorded, manifests=tmp_path / "none")
    assert INPUTS["job"] in text and INPUTS["decision"] in text
    for name, weight in v1.scenario_names(TASK).items():
        assert f"| `{name}` | {weight:.3f} |" in text
    assert "seeds 200 to 209" in text
    # The MPC's saving of 0.123 within [0.05, 0.2], one trip, as percentages.
    assert (
        "| MPC on the observation | 2.631 | +12.3 % | [+5.0 %, +20.0 %] | 1 |" in text
    )
    assert "| PPO, 30 runs | not recorded |" in text
    assert "upper bound" in text
    rung = reference.load(TASK)["rung"]
    assert f"the reference ladder's `{rung}` rung" in text
    for item in INPUTS["not_modelled"]:
        assert f"- {item}" in text


def test_a_card_says_which_prices_are_sourced(recorded, tmp_path):
    text = cards.card(TASK, INPUTS, records=recorded, manifests=tmp_path / "none")
    scored = tg.make(TASK)[1]
    assert f"| `zone_price` | {scored.zone_price:g} | kJ/s |" in text
    assert "a second outside the zone | ours, not sourced |" in text
    assert "| the job's own unit |" in text
    assert "1 of 2 prices have a published source." in text
    assert "A task waits until every one does" in text


def test_the_gymnasium_id_appears_once_frozen(recorded, tmp_path):
    before = cards.card(TASK, INPUTS, records=recorded, manifests=tmp_path / "none")
    assert "TargetGym/compressor_surge-v1" not in before
    manifests = tmp_path / "frozen"
    manifests.mkdir()
    entry = {**v1.manifest(TASK), "frozen": "2026-12-01", "commit": "fedcba9876"}
    (manifests / f"{TASK}.json").write_text(json.dumps(entry))
    after = cards.card(TASK, INPUTS, records=recorded, manifests=manifests)
    assert "Frozen as `TargetGym/compressor_surge-v1` on 2026-12-01" in after
    assert 'gymnasium.make("TargetGym/compressor_surge-v1")' in after


def test_a_price_that_is_no_parameter_is_refused(recorded, tmp_path):
    wrong = {**INPUTS, "prices": [{"field": "no_such", "trades": "-", "unit": "-"}]}
    with pytest.raises(SystemExit, match="no_such"):
        cards.card(TASK, wrong, records=recorded, manifests=tmp_path / "none")


@pytest.mark.parametrize("source", sorted(cards.CARDS.glob("*.json")), ids=str)
def test_every_card_is_current(source):
    text = cards.card(source.stem, json.loads(source.read_text()))
    assert (
        source.with_suffix(".md").read_text() == text
    ), "run scripts/generate_cards.py"
