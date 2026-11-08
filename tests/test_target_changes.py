"""When a new target becomes active (``benchmark.target_changes``), which is
where ``eval.run_episode`` and the benchmark runner start a target cycle.

Contract (``target_gym.eval``'s module docstring: a cycle runs from one target
change to the next; the user's request of 2026-10-27 that every scheduled
change start one):

- every step of a piecewise-constant schedule is a change, however small next
  to the target's value: a 90 m altitude step at 7300 m, a 4 C trim at 1590 C;
- a target that moves by a comparable amount every step, drifting (an
  Ornstein-Uhlenbeck demand) or smooth (a sine), makes no change;
- float rounding in a held target makes no change;
- a change in any output counts, and the first step is compared with the
  target at reset;
- a restart after a trip is a fresh draw of the plant, not a target change, so a
  policy that trips is held over the same steps as one that does not (the
  industrial-fixes plan, "One hold window for all").
"""

from __future__ import annotations

import numpy as np
import pytest

from target_gym import benchmark as B
from target_gym import eval as E
from target_gym.registry import REGISTRY
from target_gym.runners.runners import baseline_policy


def _changes(targets, first, restarts=None):
    targets = np.asarray(targets, float).reshape(len(targets), -1)
    return np.flatnonzero(B.target_changes(targets, np.atleast_1d(first), restarts))


def _steps(n, *at):
    flags = np.zeros(n, bool)
    flags[list(at)] = True
    return flags


def test_a_small_step_on_a_large_target_is_a_change():
    # 90 m at 7300 m is 1.2 % of the value: still a new level to reach.
    targets = [7300.0] * 3 + [7390.0] * 3
    np.testing.assert_array_equal(_changes(targets, 7300.0), [3])


def test_every_step_of_a_staircase_is_a_change():
    targets = [1.0, 1.0, 2.0, 2.0, 2.0, 1.5, 1.5]
    np.testing.assert_array_equal(_changes(targets, 1.0), [2, 5])


def test_a_drifting_target_makes_no_change_but_a_step_on_it_does():
    # A random walk with 0.01 increments; the step is 1.0, a hundred of them.
    walk = 1.0 + np.cumsum(np.random.default_rng(0).normal(0.0, 0.01, 500))
    assert len(_changes(walk, 1.0)) == 0
    stepped = walk + np.where(np.arange(500) >= 250, 1.0, 0.0)
    np.testing.assert_array_equal(_changes(stepped, 1.0), [250])


def test_a_smooth_target_makes_no_change():
    t = np.arange(480)
    assert len(_changes(1590.0 + 20.0 * np.sin(2 * np.pi * t / 480), 1590.0)) == 0


def test_float_rounding_in_a_held_target_makes_no_change():
    held = np.full(50, 1590.0)
    held[[10, 30]] *= 1.0 + 1e-9  # a recomputed target read back one ulp off
    assert len(_changes(held, 1590.0)) == 0


def test_a_change_in_any_output_counts_from_the_target_at_reset():
    first = [10.0, 1.0]
    targets = [[10.4, 1.0], [10.4, 1.0], [10.4, 1.2], [10.4, 1.2]]
    np.testing.assert_array_equal(_changes(targets, first), [0, 2])


# The scheduled setpoints of the scored configurations, from each task's
# schedule: the target at ``time`` t (steps taken) is level k for t in
# [k block, (k + 1) block), and entry i of an episode is the step that takes
# ``time`` from i to i + 1, so level k becomes active at entry k block - 1.
#   plane_energy (v4): target_steps = 8 treads over 1600 steps, 200 steps each
#     (``plane.env.commanded_altitude``); the eighth starts on the last step.
#   wind_turbine: N_TARGET_BLOCKS = 4 blocks of target_block_steps 100.
#   glass_furnace: N_SETPOINTS = 5 slots over 1600 steps, 320 each.
_TRIPS_ON_A_MOVE = pytest.mark.xfail(
    strict=True,
    reason="PR 9: the expert trips on step 1199 of seed 101, and a move on the step "
    "of a restart cannot be told from it",
)
SCHEDULES = [pytest.param("plane_energy", 200, 8, marks=_TRIPS_ON_A_MOVE)]
SCHEDULES += [("wind_turbine", 100, 3), ("glass_furnace", 320, 4)]


def test_a_restart_is_not_a_target_change():
    """The plant restarts at step 2 on a fresh target, and the schedule moves at
    step 4. Only the schedule's move starts a cycle."""
    targets = [1.0, 1.0, 5.0, 5.0, 7.0, 7.0]
    np.testing.assert_array_equal(_changes(targets, 1.0, _steps(6, 2)), [4])


def test_a_restart_catching_up_with_the_schedule_is_not_a_second_jump():
    """The trip at step 2 shows the reset's own target, 5, and the next step the
    schedule's level at the clock, 6. Neither is a move (``plane_energy``, seed
    100: changes at 637 and 638 for the one trip at 637)."""
    targets = [1.0, 1.0, 5.0, 6.0, 6.0, 6.0]
    assert len(_changes(targets, 1.0, _steps(6, 2))) == 0


def test_a_move_just_after_a_restart_is_still_a_move():
    """The restart takes steps 2 and 3. The schedule's move at step 4 follows
    them directly, and a rule counting runs of jumps would swallow it."""
    targets = [1.0, 1.0, 5.0, 6.0, 8.0, 8.0]
    np.testing.assert_array_equal(_changes(targets, 1.0, _steps(6, 2)), [4])


def test_a_ramp_cut_by_a_restart_is_still_one_move():
    """A ramp from step 2 to step 7 with a restart at step 4 is one move, at its
    start, not a second one after the restart."""
    targets = [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0]
    np.testing.assert_array_equal(_changes(targets, 0.0, _steps(12, 4)), [2])


def test_restarts_on_most_steps_do_not_set_the_drift():
    """The drift a move is measured against comes from the steps that did not
    restart. With 6 restarts of 3 in 12 steps the median step is 2, so the move
    of 1 at step 8 would pass for drift."""
    targets = [3.0, 6.0, 9.0, 9.0, 12.0, 15.0, 18.0, 18.0, 19.0, 19.0, 19.0, 19.0]
    restarts = _steps(12, 0, 1, 2, 4, 5, 6)
    np.testing.assert_array_equal(_changes(targets, 0.0, restarts), [8])


def test_a_run_that_restarts_on_every_step_has_no_move():
    assert len(_changes([1.0, 2.0, 3.0], 0.0, np.ones(3, bool))) == 0


def test_a_tripping_run_is_held_over_the_schedule():
    """Through ``run_policy``: a policy that trips over and over on the energy
    clearances keeps the schedule's 300 s clearances and no others, and its
    steps after a restart stay in its hold."""
    spec = REGISTRY["plane_energy"]
    params = spec.make_test_params()

    # Full nose-down at idle dives into the ground, the 2D aircraft's one trip
    # since Amendment 2 (its expert no longer trips), and restarts, over and over.
    def dive(obs):
        return np.array([-1.0, -1.0], np.float32)

    ep = E.run_episode(spec, params, dive, seed=100)
    assert ep.failed.sum() > 8
    assert set(np.flatnonzero(ep.target_change)) <= set(300 * np.arange(1, 6) - 1)


@pytest.mark.large
@pytest.mark.parametrize("name,block,n_changes", SCHEDULES)
def test_the_scorer_starts_a_cycle_at_every_scheduled_setpoint(name, block, n_changes):
    spec = REGISTRY[name]
    p = spec.make_test_params()
    expected = block * np.arange(1, n_changes + 1) - 1
    for seed in (100, 101):
        ep = E.run_episode(spec, p, baseline_policy(spec, "pid", p), seed=seed)
        np.testing.assert_array_equal(
            np.flatnonzero(ep.target_change), expected, err_msg=f"seed {seed}"
        )


@pytest.mark.large
def test_a_compressor_ramp_starts_one_cycle():
    """N_SETPOINT_BLOCKS = 4 blocks of 300 steps, each opening on a 100-step
    ramp. The target at ``time`` t + 1 first moves at t = 0, 300, 600 and 900."""
    spec = REGISTRY["compressor_surge"]
    p = spec.make_test_params()
    for seed in (100, 101):
        ep = E.run_episode(spec, p, baseline_policy(spec, "pid", p), seed=seed)
        np.testing.assert_array_equal(
            np.flatnonzero(ep.target_change),
            [0, 300, 600, 900],
            err_msg=f"seed {seed}",
        )


@pytest.mark.large
@pytest.mark.parametrize("name", ["plane_sine"])
def test_a_drifting_or_smooth_target_starts_no_cycle_in_the_scorer(name):
    # plane_sine: a sine altitude.
    spec = REGISTRY[name]
    p = spec.make_test_params()
    ep = E.run_episode(spec, p, baseline_policy(spec, "pid", p), seed=100)
    assert not ep.target_change.any()


def test_a_ramp_that_moves_on_under_half_the_steps_is_one_move():
    """Its median step is 0, so each ramp step would look like a new target,
    and a task would get hundreds of cycles from a few moves. A ramp is one move,
    counted where it starts. Here 20 steps from step 40, in 100."""
    targets = np.concatenate([np.zeros(40), np.linspace(0.05, 1.0, 20), np.ones(40)])[
        :, None
    ]
    assert list(_changes(targets, 0.0)) == [40]


def test_two_ramps_are_two_moves():
    up = np.linspace(0.1, 1.0, 10)
    targets = np.concatenate(
        [np.zeros(30), up, np.ones(30), up[::-1] - 0.1, np.zeros(20)]
    )[:, None]
    assert list(_changes(targets, 0.0)) == [30, 70]
