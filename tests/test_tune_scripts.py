"""What ``scripts/tune_*.py`` share, and the process recipes' own rules.

Contract, from docs/proposals/expert-protocols-noise.md: a bump is one run with
its step in it (N1), a loop nobody holds stays closed through it and records
its output, and a run at a level writes that level's file and no other. From
expert-protocols-process.md section 9: a test starts from no stored gain (R1),
and a compressor line that fails the schedule check climbs the ladder.
"""

import importlib
import json
import pathlib
import sys

import numpy as np
import pytest

from target_gym import provenance
from target_gym.identification import BASELINE
from target_gym.registry import REGISTRY

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
process = importlib.import_module("tune_process_experts")
industrial = importlib.import_module("tune_industrial_experts")


def test_a_recipe_reads_the_model_it_is_stepped_on():
    # From rest at the target, k 2 and tau 0.5, with a half step of delay.
    spec = REGISTRY["first_order"]
    p = spec.make_test_params()
    rest = process.manual(lambda o, xp: xp.reshape(o[1] / 2.0, (1,)))
    plant = process.Plant(
        spec, p, [lambda o, s, q: o[0]], [(None, None)], controller=rest
    )
    inst = process.Instrument.of(spec.make_env(), p, "x")
    loop = process.read_loop(plant, "x", 0, 0, 0.1, 60, inst, room=plant.room(0))
    assert loop["k"] == pytest.approx(2.0, rel=0.02)
    assert loop["tau1"] == pytest.approx(0.5, rel=0.05)
    assert loop["theta"] == pytest.approx(0.025, abs=0.05)


def test_an_industrial_step_holds_through_the_baseline():
    class Hold:
        channel = 0

        def reset(self):
            self.seen = []

        def __call__(self, expert, obs, t, d):
            self.seen.append((t, d))
            return np.array([0.0], np.float32) + d

    act = Hold()
    t, dy, dt = industrial.step_response("first_order", 100, act, 150, 60, 0.1)
    assert dt == 0.05 and t.shape == dy.shape == (60,)
    assert [d for t, d in act.seen if t < BASELINE] == [0.0] * (150 + BASELINE)
    assert [d for t, d in act.seen if t >= BASELINE] == [0.1] * 60


def test_a_closed_test_keeps_the_expert_in_auto_and_records_what_it_sent():
    test = industrial._Closed(0, 0)
    _, dy, _ = industrial.step_response("first_order", 100, test, 150, 60, 0.1)
    assert len(test.sent) == BASELINE + 60
    # The expert fights the step: its output moves back, so x barely does.
    assert max(abs(dy)) < 0.2 * (1 - np.exp(-6.0)) / 2


@pytest.mark.parametrize("script", [process, industrial])
def test_a_level_writes_its_own_file(script, tmp_path, monkeypatch):
    shipped = tmp_path / "pid_gains.json"
    shipped.write_text('{"first_order": {"Kp": 1.0}}\n')
    monkeypatch.setattr(provenance, "GAINS_PATH", shipped)
    monkeypatch.setattr(script, "LEVEL", 2.0)
    monkeypatch.setattr(script, "ROOT", tmp_path)
    script.write("cstr", {"Kp": 3.0})
    assert json.loads(shipped.read_text()) == {"first_order": {"Kp": 1.0}}
    stored = json.loads(provenance.gains_path(2).read_text())
    assert stored["cstr"]["Kp"] == 3.0 and list(stored) == ["cstr"]


def test_the_first_order_recipe_answers_the_noise_level(monkeypatch):
    # k 2, tau 0.5, delay half a step of 0.05 (C4), so theta 0.025.
    # Level 0.5, band 6 x 1.1e-3 x 0.5 = 0.0033: Kc = 0.5 / (2 x 0.05) = 5, and
    # 5 x 0.0033 is inside the dead band 0.02, so no filter.
    monkeypatch.setattr(process, "LEVEL", 0.5)
    quiet = process.tune_first_order()
    assert quiet["Tf"] == 0.0
    assert quiet["Kp"] == pytest.approx(5.0, rel=0.05)
    assert quiet["noise_band"] == pytest.approx(0.0033, rel=0.1)
    # Level 2, band 0.0132. The first Ti is min(0.5, 4 x 0.05) = 0.2, so the
    # filter's cap is 0.02. a = 1 - e^-2.5 = 0.918 leaves sqrt(a / (2 - a)) =
    # 0.921 of the noise, past the dead band at Kc = 0.5 / (2 x 0.09) = 2.78, so
    # Kc = 0.02 / (0.0132 x 0.921) = 1.65.
    monkeypatch.setattr(process, "LEVEL", 2.0)
    loud = process.tune_first_order()
    assert loud["Tf"] == pytest.approx(0.02, rel=0.05)
    assert loud["Kp"] == pytest.approx(1.65, rel=0.1)


def test_a_bump_tests_length_is_the_recipes_and_not_the_tasks():
    # The column's episode is 400 steps and its tests stay at the 200 they ran on.
    assert REGISTRY["distillation"].make_test_params().max_steps_in_episode == 400
    assert process._params("distillation").max_steps_in_episode == 200


@pytest.mark.parametrize("name", sorted(process.TEST_STEPS))
def test_every_recipe_fixes_its_own_tests_length(name):
    assert process._params(name).max_steps_in_episode == process.TEST_STEPS[name]


@pytest.mark.parametrize("name", sorted(industrial.TEST_STEPS))
def test_every_energy_recipe_fixes_its_own_tests_length(name):
    assert industrial._params(name).max_steps_in_episode == industrial.TEST_STEPS[name]


def test_a_cycle_read_leaves_the_crowns_own_swing_out():
    # A step of nothing, noise off: the crown moves about 2.5 K through a reversal
    # by itself, and against the same phase three reversals back only 0.2 K.
    from target_gym.glass_furnace.env import settled_point

    spec = REGISTRY["glass_furnace"]
    p = spec.make_test_params(
        sensor_noise_scale=0.0,
        max_steps_in_episode=400,
        m_pull_noise_std=0.0,
        lhv_noise_std=0.0,
        target_T_crown_trim=0.0,
    )
    rest = process.manual(lambda o, xp: xp.reshape(settled_point(o[4])[0], (1,)))
    plant = process.Plant(
        spec, p, [lambda o, s, q: o[0]], [(None, None)], controller=rest
    )
    inst = process.Instrument.of(spec.make_env(), p, "T_crown_meas")
    held = plant.bump(0, 0, 0, 0.0, 40, inst)
    cycled = plant.bump(0, 0, 0, 0.0, 40, inst, at=(150, 2))
    assert np.abs(held.dy).max() > 1.0
    assert np.abs(cycled.dy).max() < 0.25


@pytest.mark.medium
# battery left with v4: its expert plans a baseline from the reported energy
# and has no tuned gains (energy/battery/PHYSICS.md, section 9).
@pytest.mark.parametrize("name", ["boiler_drum", "glass_furnace", "hvac"])
def test_an_energy_recipe_reproduces_what_is_stored(name):
    stored = json.loads(provenance.GAINS_PATH.read_text())[name]
    out = industrial.TUNERS[name]()
    for key, value in out.items():
        if not isinstance(value, dict):
            assert stored[key] == pytest.approx(value, rel=1e-6), key


@pytest.mark.medium
@pytest.mark.parametrize("name", sorted(process.TUNERS))
def test_a_recipe_reproduces_what_is_stored(name):
    # Fixed tests, no stored gain read: a run gives the entry it wrote, to the 6
    # decimals ``write`` keeps.
    stored = json.loads(provenance.GAINS_PATH.read_text())[name]
    out = process.TUNERS[name]()
    for key, value in out.items():
        if not isinstance(value, dict):
            assert float(value) == pytest.approx(stored[key], abs=1.5e-6), key


def _no_store(*args, **kwargs):
    raise AssertionError("a recipe read the gains it replaces")


def test_a_run_reads_no_stored_gain(monkeypatch):
    # R1: run at a level with nothing stored, and a run after --write, agree.
    monkeypatch.setattr(
        REGISTRY["first_order"].__class__, "make_pid_controller", _no_store
    )
    assert process.tune_first_order()["noise_band"] > 0


def test_the_inner_pd_is_simc_on_the_datasheet():
    # Slope 50000 / (1000 x 0.239 x 100) = 2.092 K/min per K. Lag 1/6 + 0.05 =
    # 0.2167 min. Delay 0.05 + 1/30 + 0.025 = 0.1083 min. Kp 1 / (2.092 x 0.2167),
    # Kd Kp x 0.2167: 2.206 and 0.478.
    Kp, Kd = process._datasheet_pd(REGISTRY["unstable_cstr"].make_test_params())
    assert Kp == pytest.approx(2.206, abs=0.002)
    assert Kd == pytest.approx(0.478, abs=0.001)


def _ladder(monkeypatch, verdicts):
    """A recipe whose line passes or fails the schedule as told."""
    from target_gym.compressor_surge.experts import DEFAULT_GAINS

    seen = []

    def pair(line):
        seen.append(line)
        return {
            **DEFAULT_GAINS,
            "surge_line": line,
            "override_line": line / 2,
            "model": {},
        }

    monkeypatch.setattr(process, "_compressor_pair", pair)
    monkeypatch.setattr(process, "_compressor_rga", lambda line: 1.0)
    monkeypatch.setattr(process, "_schedule_check", lambda g: verdicts[len(seen) - 1])
    return seen


def test_a_line_that_trips_on_the_schedule_climbs_the_ladder(monkeypatch):
    seen = _ladder(monkeypatch, [(0.09, 2), (0.09, 0)])
    assert process.tune_compressor_surge()["surge_line"] == 0.125
    assert seen == [0.10, 0.125]


def test_a_margin_under_the_override_line_climbs_the_ladder(monkeypatch):
    _ladder(monkeypatch, [(0.049, 0), (0.0626, 0)])
    assert process.tune_compressor_surge()["surge_line"] == 0.125


def test_a_ladder_that_runs_out_ships_its_last_rung(monkeypatch):
    # C9: a poor result ships, and the record says so.
    _ladder(monkeypatch, [(0.0, 3)] * 3)
    out = process.tune_compressor_surge()
    assert out["surge_line"] == 0.15
    assert out["model"]["schedule"] == {"least margin": 0.0, "trips": 3}
