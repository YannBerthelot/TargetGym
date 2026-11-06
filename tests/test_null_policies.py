"""The null-policy gate (docs/proposals/industrial-fixes.md, check 1). The policies
are ``target_gym.null_policies``, and ``scripts/record_credibility.py`` fits them.

A task that fails the gate is a strict xfail naming the finding, and the fix removes
its own line, so a fix that works goes red until it does.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

from target_gym import records
from target_gym.null_policies import (
    EXCEPTIONS,
    EXPERT_FIGURES,
    NO_FIGURE,
    Null,
    credibility_row,
    gate,
    null_policies,
    run_on_seeds,
    wins_like_the_expert,
)
from target_gym.registry import REGISTRY


def _script():
    path = pathlib.Path(__file__).parent.parent / "scripts" / "record_credibility.py"
    spec = importlib.util.spec_from_file_location("record_credibility_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIT = _script()
fit_table, i_gains, simc_i_gain = FIT.fit_table, FIT.i_gains, FIT.simc_i_gain


#: What an aircraft expert loses to, until PR 9. TECS's ``time_const`` of 2 pulls
#: past the limit load on a height capture, and each trip costs far more than a
#: null policy's error. With the protocol's start of 5 all seven tasks pass.
_STALE_TECS = "PR 9: TECS's time_const 2 trips the new envelope; at 5 this passes"

#: The tasks a null policy beats today, each a finding of the industrial review.
#: The PR that fixes one removes its line.
KNOWN = {
    "cstr": "the expert was tuned on an instant analyser; PR 9 retunes it",
    "distillation": "the expert was tuned on instant analysers; PR 9 retunes it",
    "ph_neutralization": "the expert's gains predate the probe, so it cycles; PR 9 "
    "retunes them",
    "plane": _STALE_TECS,
    "plane3d_circle": _STALE_TECS,
    "plane3d_figure8": _STALE_TECS,
    "plane3d_heading": _STALE_TECS,
    "plane3d_racetrack": _STALE_TECS,
    "plane_energy": _STALE_TECS,
    "plane_sine": _STALE_TECS,
    "reactor": "PR 9: an integrator on Tavg beats the rod program's dead band",
    "unstable_cstr": "the expert's gains predate the thermowell and the valve, so it "
    "trips; PR 9 retunes them",
}


@pytest.mark.parametrize(
    "ratio, win_rate, fails",
    [
        (1.2, 0.6, True),
        (1.2, 0.5, False),  # 5 of 10 wins is not enough
        (1.0, 0.9, False),  # a tie on the ratio
        (0.9, 0.8, False),  # most seeds, but dearer overall
    ],
)
def test_a_null_policy_fails_a_task_with_six_wins_and_a_ratio_above_1(
    ratio, win_rate, fails
):
    assert wins_like_the_expert(ratio, win_rate) is fails


def test_the_table_recovers_a_quadratic_map_of_one_target():
    targets = np.linspace(-1.0, 3.0, 9)[:, None]
    actions = 1.0 + 2.0 * targets + targets**2  # at t = 2: 1 + 4 + 4 = 9
    np.testing.assert_allclose(fit_table(targets, actions)(np.array([[2.0]])), [[9.0]])


def test_a_target_that_never_moves_gets_the_mean_action():
    targets, actions = np.full((4, 1), 5.0), np.array([[1.0], [2.0], [3.0], [6.0]])
    np.testing.assert_allclose(fit_table(targets, actions)(np.array([[5.0]])), [[3.0]])


def test_the_i_gain_is_simcs_with_the_lag_counted_as_delay():
    """k 2, lag 8 steps and delay 2: 1 / (2 * 2 * (8 + 2)) = 0.025 per step."""
    assert simc_i_gain(2.0, 8.0, 2.0) == pytest.approx(0.025)


def test_an_invalid_bump_test_tunes_no_loop():
    """All 10 of ``reactor``'s fits are invalid, ``cstr``'s are all valid. Those
    fits read ``n``, from before the rebuild; PR 9 measures Tavg's."""
    assert i_gains("reactor", 1, 1, 1.0) is None
    assert i_gains("cstr", 1, 1, 0.25)[0, 0] != 0.0


def test_a_task_with_no_bump_test_has_no_i_only_null():
    assert i_gains("plane", 2, 1, 1.0) is None


def test_a_null_reads_its_table_and_gains_from_the_record(tmp_path, monkeypatch):
    """The stored row is all a null needs, since the bump tests are outside the
    wheel. ``u = 0.1 + 2 t`` with a target of 0.25 is 0.6, on a gain of 0.5."""
    row = {
        "table": {"mean": [0.0], "scale": [1.0], "coef": [[0.1], [2.0], [0.0]]},
        "i_gains": [[0.5]],
    }
    monkeypatch.setattr(records, "DATA", tmp_path)
    records.write("null_policies", "first_order", row)
    policies = {p.name: p for p in null_policies("first_order")}
    assert {"table", "I-only"} <= set(policies)
    spec = REGISTRY["first_order"]
    env, params = spec.make_env(), spec.make_test_params()
    task = FIT.task_info("first_order", env, params, 1)
    obs = np.zeros((1, task.value_index[0] + task.target_index[0] + 2))
    obs[0, list(task.target_index)] = 0.25
    table = policies["table"]
    action, _ = table.act(table.reset(None, obs, task), obs, 0)
    np.testing.assert_allclose(action, [[0.6]])
    trim = policies["I-only"]
    action, carry = trim.act(trim.reset(None, obs, task), obs, 0)
    np.testing.assert_allclose(carry, [[0.125]])  # 0.25 of error at 0.5


def test_a_task_with_no_stored_table_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(records, "DATA", tmp_path)
    with pytest.raises(ValueError, match="first_order"):
        null_policies("first_order")


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_task_has_a_credibility_record_of_its_nulls(name):
    row = records.load_row("credibility", name)
    assert set(row["nulls"]) == {policy.name for policy in null_policies(name)}
    assert row["exception"] == EXCEPTIONS.get(name)
    assert row["expert_figure"] == EXPERT_FIGURES.get(name, NO_FIGURE)


def test_a_task_with_no_sourced_figure_says_so():
    """Silence must not read as a pass: check 2 needs a figure or the owner's
    written exception. The rods' +-1.5 F could not be re-opened, so ``reactor``
    has none."""
    for name in ("four_tank", "reactor"):
        row = credibility_row(name, {})
        assert row["expert_figure"] == {"status": "no sourced figure"}


def test_a_sourced_figure_says_where_it_was_read_and_how():
    row = credibility_row("plane", {})
    assert row["expert_figure"]["status"] == "read, primary"
    assert row["expert_figure"]["value"] == pytest.approx(19.812)
    assert credibility_row("glass_furnace", {})["expert_figure"]["status"] == (
        "read, secondary"
    )


def test_the_record_names_each_null_that_wins_and_the_exception():
    """A win needs both a ratio above 1 and 6 of 10 seeds, as the gate's rule."""
    results = {"table": (1.2, 0.6), "constant 0.5": (1.2, 0.5)}
    row = credibility_row("unstable_cstr", results)
    assert row["null_wins"] == ["table"]
    assert row["nulls"]["constant 0.5"] == {"ratio": 1.2, "win_rate": 0.5}
    assert "tau_j" in row["exception"]
    assert credibility_row("cstr", results)["exception"] is None


def test_the_trim_removes_the_error_a_table_leaves():
    """``first_order`` with a table that is wrong by a constant: the integrator
    reads the error and takes it out, and it stops at the box."""
    name = "first_order"
    params = REGISTRY[name].make_test_params()
    gains = i_gains(name, 1, 1, float(params.delta_t))
    wrong = Null("off", lambda o: np.zeros((len(o), 1)), gains)
    out = run_on_seeds(wrong, name, (0,))
    assert abs(out["value"][0, -1, 0] - out["target"][0, -1, 0]) < 0.05
    assert np.abs(out["action"]).max() <= 1.0 + 1e-6


@pytest.mark.large
@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            n,
            marks=pytest.mark.xfail(strict=True, reason=KNOWN[n]) if n in KNOWN else (),
        )
        for n in sorted(REGISTRY)
    ],
)
def test_no_null_policy_passes_for_the_expert(name):
    fails = {k: v for k, v in gate(name).items() if wins_like_the_expert(*v)}
    assert not fails, f"{name}: (ratio, win rate) {fails}"
