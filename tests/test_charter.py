"""The charter record's pure rules (target_gym.charter), from
docs/proposals/task-charter.md:

- J2: seeds from a hash of the commit, 30 or 5 per scenario if more, every
  scenario at least 5, the rest by weight; accepted when the reference trips on at
  most 1 seed in every 30, no null beats it on more than half the seeds, and the
  sweep (B7) holds.
- J3: four lines, each per scenario and on J, or "not measured".
- J4: J = sum over scenarios of w_s times the mean cost over that scenario's seeds.
- A part is stale when its fingerprint moved, every part when the env stamp moved.

Assumed where the charter is silent: the rest goes by largest remainder (ties to
the heavier scenario); a tie between null and reference is not a win.
"""

from __future__ import annotations

import pytest

from target_gym import charter
from target_gym.charter import NOT_MEASURED as NM

W = {"normal": 0.75, "storm": 0.25}


def _line(costs: dict, trips: dict | None = None) -> dict:
    trips = trips or {s: 0 for s in costs}
    return charter.line(
        {s: {"costs": c, "trips": trips[s]} for s, c in costs.items()}, W
    )


def _record(**over) -> dict:
    commit = "abc123"
    seeds = charter.j2_seeds(commit, W)
    j2 = {"commit": commit, "seeds": seeds, "reference_trips": 0}
    j2 |= {"null_wins": {"hold": 3}, "null_share": {"hold": 0.1}, "sweep": NM}
    j2 |= charter.verdict(30, 0, {"hold": 3}, NM)
    rec = {
        "task": "toy",
        "version": {"env": "toy-v1", "stamp": "s"},
        "weights": W,
        "reference": {"kind": "ladder", "kept": "pi", "line": _line({"normal": [1.0], "storm": [2.0]})},
        "nulls": {"lines": {"hold": _line({"normal": [3.0], "storm": [3.0]})}, "broken": {}},
        "mpc": {"line": NM},
        "ceiling": {"line": NM, "source": None},
        "j2": j2,
        "anchor": {"seeds": list(charter.ANCHOR_SEEDS), "line": _line({"normal": [1.0], "storm": [2.0]})},
    }  # fmt: skip
    for p in charter.PARTS:
        rec[p]["fingerprint"] = f"fp-{p}"
    return rec | over


def _current(**moved) -> dict:
    fps = {p: f"fp-{p}" for p in charter.PARTS} | moved
    return {"version": {"env": "toy-v1", "stamp": "s"}, "fingerprints": fps}


# --- J2 seeds -----------------------------------------------------------------


def test_three_scenarios_share_30_seeds_with_at_least_5_each():
    counts = charter.seed_counts({"a": 0.8, "b": 0.15, "c": 0.05})
    # 15 left after 5 each: 12, 2.25, 0.75 -> 12, 2, 0 and the last seed to c (.75).
    assert counts == {"a": 17, "b": 7, "c": 6}


def test_the_rest_goes_by_weight_by_largest_remainder():
    # 30 seeds, 20 as 5 each, 10 left: 6.2, 2.3, 1.0, 0.5 -> 6, 2, 1, 0 and one to d.
    counts = charter.seed_counts({"a": 0.62, "b": 0.23, "c": 0.10, "d": 0.05})
    assert counts == {"a": 11, "b": 7, "c": 6, "d": 6}


def test_seven_scenarios_take_5_each_over_30():
    weights = {f"s{i}": w for i, w in enumerate((0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05))}
    assert charter.seed_counts(weights) == {s: 5 for s in weights}


def test_hash_seeds_are_fixed_by_the_commit():
    a = charter.j2_seeds("deadbeef", W)
    assert a == charter.j2_seeds("deadbeef", dict(reversed(W.items())))
    assert a != charter.j2_seeds("deadbeee", W)
    assert {s: len(v) for s, v in a.items()} == charter.seed_counts(W)
    for v in a.values():
        assert len(set(v)) == len(v)
        assert min(v) >= charter.J2_SEED_FLOOR
        assert not set(v) & (set(charter.ANCHOR_SEEDS) | set(range(100, 110)))


# --- J2 verdict ---------------------------------------------------------------


@pytest.mark.parametrize(
    "n, trips, accepted", [(30, 1, True), (30, 2, False), (60, 2, True), (35, 2, False)]
)
def test_the_reference_may_trip_on_one_seed_in_every_30(n, trips, accepted):
    assert charter.verdict(n, trips, {}, True)["accepted"] is accepted


@pytest.mark.parametrize("wins, accepted", [(15, True), (16, False)])
def test_no_null_may_beat_the_reference_on_more_than_half_the_seeds(wins, accepted):
    v = charter.verdict(30, 0, {"hold": 0, "I-only": wins}, True)
    assert v["accepted"] is accepted


@pytest.mark.parametrize("sweep", [False, NM])
def test_acceptance_needs_the_sweep_to_hold(sweep):
    v = charter.verdict(30, 0, {}, sweep)
    assert not v["accepted"] and len(v["reasons"]) == 1


def test_a_null_wins_on_the_seeds_it_costs_strictly_less():
    ref = {"normal": [1.0, 2.0, 3.0], "storm": [5.0]}
    null = {"normal": [0.5, 2.0, 4.0], "storm": [4.0]}
    assert charter.null_wins(ref, null) == 2


# --- J3 / J4 lines ------------------------------------------------------------


def test_J_is_the_weighted_sum_of_scenario_means():
    ln = _line({"normal": [1.0, 3.0], "storm": [4.0]}, {"normal": 1, "storm": 2})
    assert ln["J"] == pytest.approx(0.75 * 2.0 + 0.25 * 4.0)
    assert ln["trips"] == 3 and ln["converged"] is None


def test_the_best_null_is_the_cheapest_on_J_and_none_reads_not_measured():
    rec = _record()
    rec["nulls"]["lines"]["table"] = _line({"normal": [0.5], "storm": [0.5]})
    lines = charter.j3(rec)
    assert lines["best_null"]["name"] == "table"
    assert lines["mpc"] == NM and lines["reference"]["J"] == pytest.approx(1.25)
    assert charter.j3({})["best_null"] == NM


def _ceiling(capped=0):
    sc = lambda c: {
        "ceiling_costs": c,
        "ceiling_trips": 0,  # noqa: E731
        "solver": {"status": [0, 2], "capped": capped, "fell_back": 0},
    }
    return {"scenarios": {"normal": sc([1.0, 2.0]), "storm": sc([2.0, 2.0])}}  # fmt: skip


def test_a_stored_ceiling_reads_as_a_line_and_says_if_it_converged():
    ln = charter.ceiling_line(_ceiling(), W)
    assert ln["J"] == pytest.approx(0.75 * 1.5 + 0.25 * 2.0) and ln["converged"]
    assert charter.ceiling_line(_ceiling(capped=1), W)["converged"] is False


def test_a_ceiling_missing_a_scenario_is_not_measured():
    stored = _ceiling()
    del stored["scenarios"]["storm"]
    assert charter.ceiling_line(stored, W) == NM


# --- schema -------------------------------------------------------------------


def test_a_well_formed_record_validates():
    assert charter.validate(_record()) == []


def test_a_J_that_is_not_the_weighted_mean_fails():
    rec = _record()
    rec["reference"]["line"]["J"] += 1.0
    assert charter.validate(rec)


def test_an_acceptance_that_does_not_follow_from_its_numbers_fails():
    rec = _record()
    rec["j2"]["accepted"] = True
    assert charter.validate(rec)


def test_j2_seeds_must_be_the_commits():
    rec = _record()
    rec["j2"]["seeds"]["normal"][0] += 1
    assert charter.validate(rec)


def test_a_part_without_its_fingerprint_fails():
    rec = _record()
    del rec["mpc"]["fingerprint"]
    assert charter.validate(rec) == ["mpc: missing 'fingerprint'"]


# --- staleness ----------------------------------------------------------------


def test_no_record_is_stale_in_every_part():
    assert charter.stale_parts(None, _current()) == list(charter.PARTS)


def test_a_moved_env_stamp_stales_every_part():
    now = _current() | {"version": {"env": "toy-v2", "stamp": "t"}}
    assert charter.stale_parts(_record(), now) == list(charter.PARTS)


def test_a_moved_fingerprint_stales_that_part_alone():
    assert charter.stale_parts(_record(), _current()) == []
    assert charter.stale_parts(_record(), _current(mpc="new")) == ["mpc"]


def test_charter_tasks_are_the_ladder_rows_less_fixtures_plus_reactor(tmp_path):
    for stem in ("cstr", "first_order", "_shared"):
        (tmp_path / f"{stem}.py").write_text("")
    assert charter.charter_tasks(tmp_path) == ["cstr", "reactor"]


def test_the_log_only_grows(tmp_path):
    log = tmp_path / "log.jsonl"
    charter.append_log({"run": 1}, log)
    charter.append_log({"run": 2}, log)
    assert log.read_text().splitlines() == ['{"run": 1}', '{"run": 2}']


# --- the ceiling's plant (J3) -------------------------------------------------
# A ceiling file holds the stamp of the plant it ran on (provenance's
# environment_fingerprint, as rl_results does). A ceiling measured on another
# plant, or one that cannot say which, is stale and is not read as a J3 line.

_STORED = {
    "scenarios": {
        "normal": {"ceiling_costs": [0.5], "ceiling_trips": 0, "solver": {"converged": 1}},
        "storm": {"ceiling_costs": [1.0], "ceiling_trips": 0, "solver": {"converged": 1}},
    }
}  # fmt: skip


def test_a_ceiling_measured_on_another_plant_is_stale():
    now = _current() | {"ceiling_stamp": "old"}
    assert charter.stale_parts(_record(), now) == ["ceiling"]


def test_a_ceiling_that_carries_no_stamp_is_stale():
    now = _current() | {"ceiling_stamp": None}
    assert charter.stale_parts(_record(), now) == ["ceiling"]


def test_a_ceiling_measured_on_this_plant_is_current():
    assert charter.stale_parts(_record(), _current() | {"ceiling_stamp": "s"}) == []


def test_a_ceiling_from_another_plant_is_not_read_as_a_line():
    part = charter.ceiling_part({**_STORED, "stamp": "old"}, "s", W)
    assert part["line"] == NM
    assert "another plant" in part["why"]


def test_a_written_ceiling_is_read_back_on_its_own_plant(tmp_path):
    p = charter.write_ceiling("toy", _STORED, "s", tmp_path)
    part = charter.ceiling_part(charter.load_ceiling("toy", tmp_path), "s", W)
    # J = 0.75 * 0.5 + 0.25 * 1.0
    assert part["line"]["J"] == pytest.approx(0.625)
    assert p == tmp_path / "toy.json"


def test_no_ceiling_file_is_not_measured():
    assert charter.ceiling_part(None, "s", W)["line"] == NM


# --- the sweep (B7) in J2 -----------------------------------------------------


def test_a_task_without_a_sweep_fails_j2_on_it():
    sweep = charter.sweep_result(None)
    assert charter.verdict(30, 0, {"hold": 3}, sweep) == {
        "accepted": False,
        "reasons": ["sweep not measured"],
    }


def test_a_sweep_that_holds_on_every_draw_lets_j2_judge():
    sweep = charter.sweep_result([{"holds": True}] * 8)
    assert charter.verdict(30, 0, {"hold": 3}, sweep) == {
        "accepted": True,
        "reasons": [],
    }


def test_one_draw_that_fails_fails_the_sweep():
    sweep = charter.sweep_result([{"holds": True}] * 7 + [{"holds": False}])
    assert charter.verdict(30, 0, {"hold": 3}, sweep)["reasons"] == ["sweep fails"]
