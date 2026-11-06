"""What ``scripts/tune_*.py`` share, on ``first_order``: the only recipe a test
may run, since the others wait for their plants (industrial-fixes.md, PR 9).

Contract, from docs/proposals/expert-protocols-noise.md: a bump is one run with
its step in it (N1), a loop nobody holds stays closed through it and records
its output, and a run at a level writes that level's file and no other.
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


def test_a_process_bump_is_one_run_with_its_step_inside():
    # From rest at 0, raw 0.1 is u = 0.2 and x settles at 0.2 with tau 0.5 s:
    # after 60 steps of 0.05 s, 0.2 (1 - e^-6). Noise sd 1.1e-3, so 3 sd.
    _, _, step, reset = process._plant("first_order")
    state = reset(process.jax.random.PRNGKey(100))[1].replace(x=process.jnp.float32(0))
    dy = process._bump(step, state, lambda _: 0.0, lambda _: 0.1, 60)[:, 0]
    assert dy.shape == (60,)
    assert dy[-1] == pytest.approx(0.2 * (1 - np.exp(-6.0)), abs=3 * 1.1e-3 + 1e-3)


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


def test_a_run_at_a_level_settles_on_the_shipped_expert(monkeypatch):
    # Nothing is stored at 2 yet, which must not stop the run that stores it.
    with pytest.raises(KeyError):
        REGISTRY["first_order"].make_pid_controller(2)
    monkeypatch.setattr(process, "LEVEL", 2.0)
    assert process.tune_first_order()["noise_band"] > 0
