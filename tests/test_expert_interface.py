"""GAIN_NAMES says which params of each task's expert are gains.

Contract, from each law's docstring and the fix plan: a gain is a leaf of the law's
params with a value a multiplier can move, and the attitude gains leave the aircraft
experts. The realistic bug: a law changes and a row silently names a leaf nothing
reads, or one that is 0, so a gain multiplier does nothing.
"""

import jax
import pytest

from target_gym.experts import GAIN_NAMES
from target_gym.registry import REGISTRY


def leaves(task: str) -> dict:
    params = REGISTRY[task].make_pid_controller().params
    flat = jax.tree_util.tree_flatten_with_path(params)[0]
    return {
        ".".join(str(getattr(key, "name", key)) for key in path): leaf
        for path, leaf in flat
    }


@pytest.mark.parametrize("task", sorted(GAIN_NAMES))
def test_every_named_gain_is_a_leaf_with_a_value(task: str) -> None:
    values = leaves(task)
    for name in GAIN_NAMES[task]:
        assert values[name] != 0, f"a multiplier on {task}'s {name} moves nothing"


def test_the_gaps_stay_named() -> None:
    """The reactor's rod program and the kiln's rule base have no gains; a row
    gaining some means the law changed and the docstring's gaps are stale."""
    assert GAIN_NAMES["reactor"] == ()
    assert set(GAIN_NAMES["cement_kiln"]) == {"Kp_inner", "Ki_inner"}


def test_no_aircraft_row_names_a_gain_of_the_inner_loop() -> None:
    """PR 7 moves the attitude loops into the plant: a row naming one would break
    the arm that reads it on the day that lands."""
    aircraft = [task for task in GAIN_NAMES if task.startswith("plane")]
    assert len(aircraft) == 7
    for task in aircraft:
        assert not {
            "Kp_roll",
            "Kd_roll",
            "Kp_theta",
            "Ki_theta",
            "Kd_theta",
        }.intersection(GAIN_NAMES[task])
