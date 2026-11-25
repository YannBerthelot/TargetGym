"""pH's sensitivity run (owner, 2026-11-24): the plant's members.

Contract: the run repeats the ladder, its null check and the ceiling on the shipped
plant and on members that scale one group of unsourced numbers at a time, by 0.5 and by
1.5: the buffer flow, the two tanks' volumes (their residence time), the two upset
sizes. A member scales its numbers in every scenario, whatever the scenario sets them
to, so an upset a scenario switches off (size 0) stays off and the buffer a scenario
removes stays removed. The spans are stated, not sourced: this is a sensitivity, not
the charter's family sweep.
"""

import importlib.util
import pathlib

import pytest

from target_gym import registry, scenarios

ROOT = pathlib.Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def run():
    spec = importlib.util.spec_from_file_location(
        "ph_sensitivity", ROOT / "scripts" / "ph_sensitivity.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_members_are_the_shipped_plant_and_each_group_at_half_and_half_more(run):
    assert [name for name, _ in run.members()] == [
        "shipped",
        "buffer x0.5",
        "buffer x1.5",
        "residence x0.5",
        "residence x1.5",
        "upsets x0.5",
        "upsets x1.5",
    ]
    assert dict(run.members())["shipped"] == {}
    assert dict(run.members())["residence x0.5"] == {"V_s1": 0.5, "V": 0.5}


def _params(run, scales, scenario):
    spec = registry.get("ph_neutralization")
    by_name = {s.name: s for s in scenarios.suite(spec)}
    return scenarios.params_for(run.Scaled(spec, scales), by_name[scenario])


@pytest.mark.parametrize(
    "scenario, acid, flow",
    # Shipped sizes 0.2 where the scenario runs the upset, 0 where it does not.
    [("normal", 0.0, 0.0), ("acid_strength", 0.3, 0.0), ("bad_day", 0.3, 0.3)],
)
def test_a_member_scales_an_upset_where_the_scenario_runs_it(run, scenario, acid, flow):
    p = _params(run, {"acid_upset_size": 1.5, "flow_swing": 1.5}, scenario)
    assert float(p.acid_upset_size) == pytest.approx(acid)
    assert float(p.flow_swing) == pytest.approx(flow)


def test_a_member_halves_the_buffer_and_leaves_a_lost_buffer_lost(run):
    # Shipped 0.55 mL/s; the buffer-loss scenario sets it to 0.
    assert float(_params(run, {"q2_nominal": 0.5}, "normal").q2_nominal) == (
        pytest.approx(0.275)
    )
    assert float(_params(run, {"q2_nominal": 0.5}, "buffer_loss").q2_nominal) == 0.0


def test_the_shipped_member_is_the_plant_as_registered(run):
    spec = registry.get("ph_neutralization")
    assert run.Scaled(spec, {}).make_test_params() == spec.make_test_params()
