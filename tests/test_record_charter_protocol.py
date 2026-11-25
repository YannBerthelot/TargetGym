"""The charter record of a task scored on stream runs (the reactor, owner,
2026-11-24): what ``scripts/record_charter.py`` and ``scripts/reactor_ceiling.py``
must do with ``target_gym.scoring_runs``'s protocol.

Contract:
- the record's weights are the protocol's folded ones, four scenarios;
- the stop rule reads the reference and ceiling lines: a scenario on its first
  10 seeds whose room interval contains 1 takes the extra seeds; one already on 20,
  or whose room is clear of 1, does not;
- a scenario the ceiling part named takes seeds 100 to 119; the others keep 100 to
  109;
- the ceiling file is assembled only when every scenario has its own seeds: an
  extended scenario missing its extra seeds writes nothing.
By hand: reference 10 a step on every seed against a ceiling of 8 is a room of
1.25 with no spread, clear of 1; a ceiling of 8 and 12 in turn is a room near 1.
"""

import importlib
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
record = importlib.import_module("record_charter")
ceiling = importlib.import_module("reactor_ceiling")

FOUR = ("baseload", "edf_daily", "fast_return", "krsko_slow")
FIRST, MORE = list(range(100, 110)), list(range(100, 120))


def _line(per: dict) -> dict:
    return {"scenarios": {s: {"costs": c} for s, c in per.items()}}


def test_the_reactors_record_weighs_its_four_kept_scenarios():
    t = record.Task("reactor", {})
    assert set(t.weights) == set(FOUR)
    assert sum(t.weights.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "ceiling_costs, reference_costs, expected",
    [
        ([8.0] * 10, [10.0] * 10, []),
        ([8.0, 12.0] * 5, [10.0] * 10, ["edf_daily"]),
        ([8.0, 12.0] * 10, [10.0] * 20, []),  # already on its extra seeds
    ],
    ids=["clear_of_one", "holds_one", "already_extended"],
)
def test_the_stop_rule_names_scenarios_whose_room_interval_holds_one(
    ceiling_costs, reference_costs, expected
):
    clear = {s: ([10.0] * 10, [8.0] * 10) for s in FOUR}
    clear["edf_daily"] = (reference_costs, ceiling_costs)
    ref = _line({s: r for s, (r, _) in clear.items()})
    ceil = _line({s: c for s, (_, c) in clear.items()})
    weights = {s: 0.25 for s in FOUR}
    assert record.extended(ref, ceil, weights, MORE) == expected


def test_no_line_measured_extends_nothing():
    assert record.extended(record.NM, _line({}), {"baseload": 1.0}, MORE) == []


def test_a_scenario_the_ceiling_part_named_takes_the_extra_seeds():
    t = record.Task("reactor", {"ceiling": {"extended": ["baseload"]}})
    assert t.seeds_for("baseload") == MORE
    assert t.seeds_for("edf_daily") == FIRST


def _piece(scenario, seeds):
    n = len(seeds)
    return {scenario: {"seeds": list(seeds), "costs": [1.0] * n,
                       "trips": [0] * n, "converged": [True] * n}}  # fmt: skip


def test_an_extended_scenario_without_its_extra_seeds_writes_no_ceiling():
    weights = {s: 0.25 for s in FOUR}
    seeds = {s: FIRST for s in FOUR} | {"baseload": MORE}
    pieces = [_piece(s, FIRST) for s in FOUR]
    with pytest.raises(SystemExit):
        ceiling.assemble(pieces, weights, seeds)
    pieces.append(_piece("baseload", range(110, 120)))
    stored = ceiling.assemble(pieces, weights, seeds)
    assert len(stored["scenarios"]["baseload"]["ceiling_costs"]) == 20
    assert len(stored["scenarios"]["edf_daily"]["ceiling_costs"]) == 10


# The planner line in pieces (review, 2026-11-24): the same 7-day runs as the
# ceiling, the planner on the mean path (no outside world foreseen), merged into
# the record's mpc part. By hand: costs 2 and 4 on baseload (weight 0.5) and 10 on
# every other scenario (0.5 / 3 each) give J = 0.5 x 3 + 0.5 x 10 = 6.5.


def test_the_planner_line_plans_on_the_mean_path_and_the_ceiling_sees_the_world():
    mpc_params, mpc_seed = ceiling.planner("edf_daily", "mpc")
    ceil_params, ceil_seed = ceiling.planner("edf_daily", "ceiling")
    assert mpc_seed is None and ceil_seed == 0
    assert float(mpc_params.demand_sigma) == 0.0  # the mean path: no wobble drawn
    assert float(ceil_params.demand_sigma) > 0.0
    assert int(mpc_params.max_steps_in_episode) == 7 * 2880
    assert int(ceil_params.max_steps_in_episode) == 7 * 2880


def test_the_planner_line_is_merged_from_its_pieces():
    weights = {"baseload": 0.5} | {s: 0.5 / 3 for s in FOUR[1:]}
    seeds = {s: [100, 101] for s in FOUR}
    pieces = [_piece(s, [100, 101]) for s in FOUR[1:]]
    for s in FOUR[1:]:
        pieces[FOUR.index(s) - 1][s]["costs"] = [10.0, 10.0]
    pieces.append({"baseload": {"seeds": [100, 101], "costs": [2.0, 4.0],
                                "trips": [0, 0], "converged": [True, True]}})  # fmt: skip
    line = record.line_from_pieces(pieces, weights, seeds)
    assert line["J"] == pytest.approx(6.5)
    assert line["converged"] is True
    with pytest.raises(SystemExit):
        record.line_from_pieces(pieces[1:], weights, seeds)
