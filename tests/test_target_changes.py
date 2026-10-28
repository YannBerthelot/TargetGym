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
  target at reset.
"""

from __future__ import annotations

import numpy as np
import pytest

from target_gym import benchmark as B
from target_gym import eval as E
from target_gym.registry import REGISTRY
from target_gym.runners.runners import baseline_policy


def _changes(targets, first):
    targets = np.asarray(targets, float).reshape(len(targets), -1)
    return np.flatnonzero(B.target_changes(targets, np.atleast_1d(first)))


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
#   plane_energy: target_steps = 8 treads over 1200 steps, 150 steps each
#     (``plane.env.commanded_altitude``); the eighth starts on the last step.
#   compressor_surge: N_SETPOINT_BLOCKS = 4 blocks of setpoint_block_steps 300.
#   glass_furnace: N_SETPOINTS = 5 slots over 1600 steps, 320 each.
SCHEDULES = [("plane_energy", 150, 8), ("compressor_surge", 300, 3)]
SCHEDULES += [("glass_furnace", 320, 4)]


@pytest.mark.slow
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


@pytest.mark.slow
@pytest.mark.parametrize("name", ["reactor", "plane_sine"])
def test_a_drifting_or_smooth_target_starts_no_cycle_in_the_scorer(name):
    # reactor: an Ornstein-Uhlenbeck power demand; plane_sine: a sine altitude.
    spec = REGISTRY[name]
    p = spec.make_test_params()
    ep = E.run_episode(spec, p, baseline_policy(spec, "pid", p), seed=100)
    assert not ep.target_change.any()
