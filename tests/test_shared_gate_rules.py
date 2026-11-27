"""The shared gate rules every scored task is judged by (owner, 2026-11-27,
"Shared rules, then ship"), from the coordinating session's relay of the
owner's text, not from the code:

1. Ceiling. The ceiling is the cheapest converged plan known per seed: the
   foresight runs (from the hold, the reference and, where the task has an
   MPC, the MPC's moves) and the MPC line's own run. Converged means the kept
   plans had no capped or failed solve, the two best foresight starts agree
   within 1 % of the kept J, and, where the task has a floor, the kept J is
   within 3 % above it. The MPC line joins the cheapest-plan choice, not the
   agreement test.
2. Nulls. A null that beats the reference by under 1 % of the reference's
   cost is "not held" instead of failing: on a scenario's median in the
   sweep, and on a seed's cost when J2 counts null wins. A win of 1 % or more
   fails as before.
3. Fresh artefacts. A stored artefact a gate reads that fails its own
   staleness check (a null table fitted to another expert) is rebuilt by its
   recipe before the gates run; a current one is left alone.

Assumed: with one foresight start there is nothing to agree, so agreement
passes; a tie (0 % edge) is "not held".
"""

from __future__ import annotations

import pytest

from target_gym import charter

W = {"normal": 0.75, "storm": 0.25}


# --- 1. ceiling ---------------------------------------------------------------


@pytest.mark.parametrize(
    "starts, floor, converged",
    [
        # A start lost in a bad basin does not count: the two best agree to 0.5 %.
        ({"hold": 5e4, "reference": 1.035, "mpc": 1.030}, None, True),
        # The two best 1.5 % apart.
        ({"hold": 5e4, "reference": 1.045, "mpc": 1.030}, None, False),
        # Agree, 2.9 % above a floor; then 3.5 % above it.
        ({"reference": 1.030, "mpc": 1.029}, 1.0, True),
        ({"reference": 1.036, "mpc": 1.035}, 1.0, False),
        # One start: nothing to agree.
        ({"reference": 7.0}, None, True),
    ],
)
def test_a_ceiling_converges_when_its_two_best_starts_agree_near_any_floor(
    starts, floor, converged
):
    assert charter.stop_rule(starts, floor) is converged


def _stored(stop=None, capped=0):
    sc = lambda c: {  # noqa: E731
        "seeds": [100, 101],
        "ceiling_costs": c,
        "ceiling_trips": 0,
        "solver": {"status": [0, 0], "capped": capped, "fell_back": 0},
    }
    out = {"scenarios": {"normal": sc([1.0, 2.0]), "storm": sc([2.0, 2.0])}}
    return out | ({"stop_rule": {"converged": stop}} if stop is not None else {})


def test_a_ceiling_whose_starts_disagree_is_not_converged():
    assert charter.ceiling_line(_stored(stop=True), W)["converged"] is True
    assert charter.ceiling_line(_stored(stop=False), W)["converged"] is False
    assert charter.ceiling_line(_stored(stop=True, capped=1), W)["converged"] is False


def _line(costs, trips=None):
    trips = trips or {s: 0 for s in costs}
    per = {s: {"costs": c, "trips": trips[s]} for s, c in costs.items()}
    return charter.line(per, W)


def test_the_ceiling_takes_the_cheaper_of_foresight_and_the_mpc_line_per_seed():
    foresight = _line({"normal": [1.0, 2.0], "storm": [2.0, 2.0]})
    mpc = _line({"normal": [1.5, 1.0], "storm": [2.5, 2.0]})
    best = charter.best_known(foresight | {"converged": True}, mpc, W)
    assert best["scenarios"]["normal"]["costs"] == [1.0, 1.0]
    assert best["scenarios"]["storm"]["costs"] == [2.0, 2.0]
    # foresight on a tie
    assert best["picked"] == {
        "normal": ["foresight", "mpc"],
        "storm": ["foresight"] * 2,
    }
    assert best["J"] == pytest.approx(0.75 * 1.0 + 0.25 * 2.0)
    # the MPC line does not enter the agreement: convergence is foresight's
    assert best["converged"] is True
    unconverged = foresight | {"converged": False}
    assert charter.best_known(unconverged, mpc, W)["converged"] is False


def test_a_tripped_seed_never_beats_an_untripped_one():
    foresight = _line({"normal": [3.0], "storm": [2.0]})
    mpc = _line({"normal": [1.0], "storm": [2.0]}, trips={"normal": 1, "storm": 0})
    best = charter.best_known(foresight, mpc, W)
    assert best["scenarios"]["normal"]["costs"] == [3.0]
    assert best["picked"]["normal"] == ["foresight"]


def test_an_mpc_line_without_a_ceiling_or_on_other_seeds_is_not_mixed_in():
    foresight = _line({"normal": [1.0, 2.0], "storm": [2.0, 2.0]})
    other = _line({"normal": [0.1], "storm": [0.1]})
    assert charter.best_known(foresight, other, W)["J"] == foresight["J"]
    assert charter.best_known(foresight, charter.NOT_MEASURED, W)["J"] == foresight["J"]
    assert charter.best_known(charter.NOT_MEASURED, other, W) == charter.NOT_MEASURED


# --- 2. nulls -----------------------------------------------------------------


@pytest.mark.parametrize(
    "reference, null, edge",
    [
        (1.0, 1.2, "held"),
        (1.0, 0.995, "not held"),  # 0.5 % better
        (1.0, 1.0, "not held"),  # a tie
        (1.0, 0.99, "fails"),  # exactly 1 %
        (1.0, 0.5, "fails"),
    ],
)
def test_a_null_within_1_percent_of_the_reference_is_not_held(reference, null, edge):
    assert charter.null_edge(reference, null) == edge


def test_j2_counts_only_wins_of_1_percent_or_more_and_reports_the_rest():
    ref = {"normal": [1.0, 1.0, 1.0], "storm": [2.0]}
    null = {"normal": [0.995, 0.98, 1.2], "storm": [1.999]}
    assert charter.null_wins(ref, null) == 1
    assert charter.null_not_held(ref, null) == 2


def _ladder_draw(kept, null):
    rungs = {"pi": kept | {"trips": 0}}
    return {"kept": "pi", "rungs": rungs, "nulls": {"table": null}, "holds": False}


def test_a_sweep_draw_beaten_by_under_1_percent_holds_with_that_scenario_not_held():
    kept = {"normal": 0.0684, "storm": 0.0672}
    near = _ladder_draw(kept, {"normal": 0.0680, "storm": 0.276})  # 0.6 % on normal
    far = _ladder_draw(kept, {"normal": 0.0650, "storm": 0.276})  # 5 % on normal
    assert charter.sweep_result([near] * 3) is True
    assert charter.sweep_not_held([near] * 3) == ["normal"]
    assert charter.sweep_result([near, far]) is False


def test_a_ladder_draw_with_no_kept_rung_or_no_null_fails():
    kept = {"normal": 1.0, "storm": 1.0}
    assert charter.sweep_result([_ladder_draw(kept, {}) | {"nulls": {}}]) is False
    assert charter.sweep_result([_ladder_draw(kept, kept) | {"kept": None}]) is False


def _reactor_row(r, n):
    return {"reference_median": r, "null_median": n, "beats_null": r < n}


def test_the_reactor_s_sweep_draws_follow_the_same_margin():
    def draw(**sc):
        return {"at_rest_cost": 0.0, "scenarios": sc, "holds": False}

    near = draw(base=_reactor_row(1.0, 0.996), daily=_reactor_row(1.0, 9.0))
    assert charter.sweep_result([near]) is True
    assert charter.sweep_result([draw(base=_reactor_row(1.0, 0.98))]) is False
    # a draw the recipe could not even be run on fails, whatever the margins
    refused = {"scenarios": {}, "holds": False, "tune_refused": "tripped"}
    assert charter.sweep_result([refused]) is False


def test_a_draw_with_no_medians_keeps_its_own_verdict():
    assert charter.sweep_result([{"holds": True}] * 2) is True
    assert charter.sweep_result([{"holds": True}, {"holds": False}]) is False


# --- 3. fresh artefacts -------------------------------------------------------


def test_a_null_table_fitted_to_another_expert_is_stale_and_a_current_one_is_not():
    stale = charter.stale_artefacts({"expert_fingerprint": "old"}, "new")
    assert stale == ["null_policies"]
    assert charter.stale_artefacts({"expert_fingerprint": "new"}, "new") == []
    assert charter.stale_artefacts(None, "new") == ["null_policies"]
