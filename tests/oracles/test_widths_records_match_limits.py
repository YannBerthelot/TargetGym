"""Each oracle's widths record against the limits its model plans.

Contract (smoothing protocol, K11): ``limit_backoffs`` holds one back-off per
limit of the task's model, in ``TaskModel.limits`` order, or none where nothing
is backed off. The controller binds them to the limits pairwise, so a record
measured on an older model, with another count, puts its back-offs on limits
they were not measured for and drops the rest.
"""

import pathlib

import pytest

from target_gym import records
from target_gym.experts.mpc import plan_params
from target_gym.registry import REGISTRY

WIDTHS = pathlib.Path(records.__file__).parent / "data" / "oracle_widths"
TASKS = sorted(p.stem for p in WIDTHS.glob("*.json") if p.stem in REGISTRY)


@pytest.mark.parametrize("name", TASKS)
def test_a_widths_record_holds_one_backoff_per_planned_limit_or_none(name):
    spec = REGISTRY[name]
    if spec.make_mpc is None:
        pytest.skip("the task ships no oracle")
    oracle = spec.make_mpc(spec.make_env(), plan_params(spec, spec.make_test_params()))
    if not hasattr(oracle, "model"):
        pytest.skip("not on the shared solver: no limits to back off")
    back = records.load_row("oracle_widths", name).get("limit_backoffs") or []
    assert len(back) in (0, len(oracle.model.limits))
