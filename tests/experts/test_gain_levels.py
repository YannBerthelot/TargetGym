"""Gains keyed by task and noise level.

Contract, from docs/proposals/expert-protocols-noise.md ("Gains per noise
level"), docs/proposals/industrial-fixes.md ("Expert protocols, amended") and
D8 of docs/proposals/expert-protocols-process.md:

- ``pid_gains.json`` is level 1, the shipped gains: no level and level 1 both
  read it, and neither moves when another level is stored or read.
- ``make_expert(name, sensor_noise_scale=level)`` and
  ``REGISTRY[name].make_pid_controller(level)`` read that level's gains only.
- A level with no stored gains raises and names the task and the level. There
  is no fallback, whether the level, its file, the task or one key is missing.
- A level's gains are part of that level's fingerprint, and not of another's.
- ``gain()`` raises on a missing key.
"""

import json

import jax
import numpy as np
import pytest

import target_gym as tg
from target_gym import provenance
from target_gym.experts import pid
from target_gym.registry import REGISTRY


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A private copy of the shipped file, and a way to store a level beside it."""
    shipped = tmp_path / "pid_gains.json"
    shipped.write_text(provenance.GAINS_PATH.read_text())
    monkeypatch.setattr(provenance, "GAINS_PATH", shipped)

    def write(level, entries):
        provenance.gains_path(level).write_text(json.dumps(entries))

    write.shipped = json.loads(shipped.read_text())
    return write


SCALE = 1.02  # small enough for unstable_cstr's build guard to accept


def _scaled(entry):
    """An entry with every number scaled, so a level's gains are told apart."""
    if isinstance(entry, dict):
        return {k: _scaled(v) for k, v in entry.items()}
    return SCALE * entry if isinstance(entry, (int, float)) else entry


def _leaves(controller):
    return [float(x) for x in jax.tree_util.tree_leaves(controller.params)]


@pytest.mark.parametrize("task", ["first_order", "boiler_drum"])
def test_no_level_and_level_one_are_the_shipped_gains(task):
    shipped = _leaves(REGISTRY[task].make_pid_controller())
    assert _leaves(REGISTRY[task].make_pid_controller(1)) == shipped
    assert (
        _leaves(REGISTRY[task].make_pid_controller(sensor_noise_scale=1.0)) == shipped
    )


@pytest.mark.parametrize(
    "task, field, key",
    [
        ("first_order", "Kp", "Kp"),  # read through gain()
        ("boiler_drum", "Kp_pressure", "Kp_pressure"),  # load_gains().get(task)
        ("unstable_cstr", "Kp_T", "Kp_T"),  # the task's own load_gains()
    ],
)
def test_a_stored_level_gives_that_levels_gains(store, task, field, key):
    store(2, {task: _scaled(store.shipped[task])})
    controller = REGISTRY[task].make_pid_controller(2)
    assert getattr(controller.params, field) == pytest.approx(
        SCALE * store.shipped[task][key], rel=1e-6
    )


def test_make_expert_acts_on_the_level_it_was_given(store):
    # A pure P loop: action = Kp (target - x) = 0.1 (1 - 0).
    store(0.5, {"first_order": {"Kp": 0.1, "Ki": 0.0, "Kd": 0.0}})
    expert = tg.make_expert("first_order", sensor_noise_scale=0.5)
    action, _ = expert.step(expert.reset(), np.array([0.0, 1.0], np.float32))
    assert float(np.ravel(action)[0]) == pytest.approx(0.1, rel=1e-6)


def test_the_batched_form_ajax_calls_keeps_working_with_and_without_a_level(store):
    # The P loop's action, 0.1 (1 - 0), once per batch row; the shipped form
    # keeps the shape it has always had.
    store(2, {"first_order": {"Kp": 0.1, "Ki": 0.0, "Kd": 0.0}})
    spec = REGISTRY["first_order"]
    obs = np.array([[0.0, 1.0], [0.0, 1.0]], np.float32)
    policy = spec.make_pid_controller(2).batched()
    action, _ = policy(policy.init_state(2), obs)
    np.testing.assert_allclose(np.ravel(action), [0.1, 0.1], rtol=1e-6)

    policy = spec.make_pid_controller().batched()
    action, _ = policy(policy.init_state(2), obs)
    assert np.shape(action) == (2, 1)


def test_the_shipped_gains_come_back_after_a_level(store):
    store(2, {"first_order": {"Kp": 0.1, "Ki": 0.0, "Kd": 0.0}})
    before = _leaves(REGISTRY["first_order"].make_pid_controller())
    REGISTRY["first_order"].make_pid_controller(2)
    assert _leaves(REGISTRY["first_order"].make_pid_controller()) == before
    assert pid.load_gains()["first_order"] == store.shipped["first_order"]


def test_a_failed_build_at_a_level_leaves_the_shipped_gains_in_place(store):
    # The entry passes the key check, and its value then fails to convert.
    store(2, {"first_order": {"Kp": "not a number", "Ki": 0.0, "Kd": 0.0}})
    with pytest.raises(ValueError):
        REGISTRY["first_order"].make_pid_controller(2)
    assert pid.load_gains()["first_order"] == store.shipped["first_order"]


@pytest.mark.parametrize("level", [0, 0.7, 3])
def test_a_level_outside_the_study_raises(store, level):
    with pytest.raises(KeyError, match="first_order"):
        REGISTRY["first_order"].make_pid_controller(level)


def test_a_level_with_no_file_raises_for_every_task_form():
    # Nothing is stored at 2 yet, so no task may answer from the shipped gains.
    with pytest.raises(KeyError, match=r"first_order.*2"):
        REGISTRY["first_order"].make_pid_controller(2)
    with pytest.raises(KeyError, match=r"first_order.*2"):
        tg.make_expert("first_order", sensor_noise_scale=2)


def test_a_level_without_the_task_raises_even_if_its_gains_are_in_code(store):
    # hvac stores nothing in the file, so an entry is how a level vouches for it.
    store(2, {"first_order": {"Kp": 0.1, "Ki": 0.0, "Kd": 0.0}})
    with pytest.raises(KeyError, match="hvac"):
        REGISTRY["hvac"].make_pid_controller(2)


def test_a_level_missing_one_gain_raises_instead_of_using_a_default(store):
    # boiler_drum reads _p.get(key, default), which would otherwise hide it.
    entry = _scaled(store.shipped["boiler_drum"])
    del entry["Kp_pressure"]
    store(2, {"boiler_drum": entry})
    with pytest.raises(KeyError, match="Kp_pressure"):
        REGISTRY["boiler_drum"].make_pid_controller(2)


def test_a_level_needs_the_other_entries_the_task_reads(store):
    # plane_sine's expert reads the plane entry as well as its own.
    store(2, {"plane_sine": {}})
    with pytest.raises(KeyError, match="lacks 'plane'"):
        REGISTRY["plane_sine"].make_pid_controller(2)


def test_a_level_has_its_own_fingerprint(store):
    spec = REGISTRY["first_order"]
    shipped = provenance.expert_fingerprint(spec)
    store(2, {"first_order": {"Kp": 0.1, "Ki": 0.0, "Kd": 0.0}})
    first = provenance.expert_fingerprint(spec, 2)
    assert provenance.expert_fingerprint(spec) == shipped, "level 2 moved level 1"
    assert first != shipped

    store(2, {"first_order": {"Kp": 0.2, "Ki": 0.0, "Kd": 0.0}})
    assert provenance.expert_fingerprint(spec, 2) != first
    assert provenance.expert_fingerprint(spec) == shipped


def test_equal_gains_at_two_levels_are_still_two_fingerprints(store):
    # The expert is the same but it is scored on other noise.
    entry = store.shipped["first_order"]
    store(0.5, {"first_order": entry})
    store(2, {"first_order": entry})
    spec = REGISTRY["first_order"]
    assert provenance.expert_fingerprint(spec, 0.5) != provenance.expert_fingerprint(
        spec, 2
    )


def test_a_level_with_no_gains_has_no_fingerprint():
    with pytest.raises(KeyError):
        provenance.expert_fingerprint(REGISTRY["first_order"], 2)


def test_level_one_fingerprint_is_the_shipped_one():
    spec = REGISTRY["first_order"]
    assert provenance.expert_fingerprint(spec, 1) == provenance.expert_fingerprint(spec)


def test_gain_raises_on_a_key_the_store_does_not_hold():
    with pytest.raises(KeyError, match="no_such_gain"):
        pid.gain("cstr", "no_such_gain", 1.5)
    with pytest.raises(KeyError, match="no_such_task"):
        pid.gain("no_such_task", "Kp")


def test_battery_keeps_the_gains_it_ran_on_as_literals():
    # 0.5, 0.02 and 0.12 are the defaults its factory passed to gain() before
    # gain() raised; the store had no battery entry, so those were the gains.
    params = REGISTRY["battery"].make_pid_controller().params
    assert (params.Kp, params.Ki, params.guard_margin) == (0.5, 0.02, 0.12)
