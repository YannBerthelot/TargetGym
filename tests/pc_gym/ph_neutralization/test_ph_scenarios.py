"""pH's scored suite after the evaluation cut (owner, 2026-11-24).

Contract: a scenario stays only if removing it would hide a named behaviour or
change which controller looks better. pH keeps three: the normal day (the edge
against the unmetered buffer), the bad day (acid strength and influent flow both
move, unannounced) and the buffer loss (the steep curve). The acid-only and
flow-only windows repeated the bad day and are folded into it: weights normal 0.80,
bad day 0.15 (each disturbance keeps its former share of time, 0.10 + 0.05), buffer
loss 0.05 as before. Sizes stay as shipped: both upsets 0.2 on the bad day, none on
the normal day, no buffer in the buffer loss.
"""

import pytest

from target_gym import registry, scenarios

SPEC = registry.get("ph_neutralization")


def test_the_suite_is_the_normal_day_the_bad_day_and_the_buffer_loss():
    assert [s.name for s in scenarios.suite(SPEC)] == [
        "normal",
        "bad_day",
        "buffer_loss",
    ]


def test_the_bad_day_carries_the_folded_upsets_share_of_time():
    assert scenarios.weights(SPEC) == pytest.approx(
        {"normal": 0.80, "bad_day": 0.15, "buffer_loss": 0.05}
    )


@pytest.mark.parametrize(
    "name, acid, flow, buffer",
    [
        ("normal", 0.0, 0.0, 0.55),
        ("bad_day", 0.2, 0.2, 0.55),
        ("buffer_loss", 0.0, 0.0, 0.0),
    ],
)
def test_each_scenario_keeps_its_shipped_sizes(name, acid, flow, buffer):
    s = {x.name: x for x in scenarios.suite(SPEC)}[name]
    p = scenarios.params_for(SPEC, s)
    assert float(p.acid_upset_size) == pytest.approx(acid)
    assert float(p.flow_swing) == pytest.approx(flow)
    assert float(p.q2_nominal) == pytest.approx(buffer)
