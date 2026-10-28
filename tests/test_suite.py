"""The capability suite (docs/capability-profile.md).

Contract, from the user's decisions of 2026-10-27:

- every task carries the capabilities of the table the user confirmed, and
  the docs page shows that table;
- per task, two radii: 0.5 at the expert and 1 at the ceiling (hold) or at the
  best known (reach), on the score's log scale, worse than the expert inside,
  clipped to [0, 1], and undefined where the task leaves no room;
- a capability is the mean of its tasks' defined radii;
- the expert is the 0.5 ring; the oracle's polygon comes from the records;
- the chart is two radars side by side, hold and reach.
"""

from __future__ import annotations

import math
import pathlib
import re

import matplotlib
import pytest

from target_gym import suite
from target_gym.registry import REGISTRY

matplotlib.use("Agg")

DOC = pathlib.Path(__file__).resolve().parent.parent / "docs" / "capability-profile.md"


def test_every_task_is_tagged_with_known_capabilities():
    assert set(suite.CAPABILITIES) == set(REGISTRY)
    used = {tag for tags in suite.CAPABILITIES.values() for tag in tags}
    assert used == set(suite.AXES)


def test_the_docs_table_shows_the_tags_the_suite_uses():
    by_label = {label: axis for axis, label in suite.LABELS.items()}
    section = DOC.read_text().split("## Why each task has its tags")[1]
    row = re.compile(r"^\| `(\w+)` \| ([^|]+) \|", re.MULTILINE)
    documented = {
        name: tuple(by_label[t.strip()] for t in tags.split(","))
        for name, tags in row.findall(section.split("\n## ")[0])
    }
    assert documented == suite.CAPABILITIES


def test_the_radius_puts_the_expert_at_half_and_the_best_at_one():
    assert suite.radius(0.0) == 0.5
    assert suite.radius(1.0) == 1.0
    assert suite.radius(-0.5) == 0.25
    assert suite.radius(-3.0) == 0.0
    assert suite.radius(2.0) == 1.0
    assert math.isnan(suite.radius(float("nan")))


def test_a_capability_is_the_mean_of_its_tasks_radii():
    """Task a holds at the ceiling (radius 1) and reaches with a ratio of 4
    where the best known is 16, half the log span (0.75). Task b holds half a
    span worse than the expert (0.25) and has no best known move. Capability
    x has both tasks, y only a."""
    tags = {"a": ("x", "y"), "b": ("x",)}
    results = {
        "a": {"score": 1.0, "transient_ratio": 4.0, "oracle_transient_ratio": 16.0},
        "b": {
            "score": -0.5,
            "transient_ratio": 1.0,
            "oracle_transient_ratio": float("nan"),
        },
    }
    p = suite.profile(results, tags=tags)
    assert p["hold"] == pytest.approx({"x": (1.0 + 0.25) / 2, "y": 1.0})
    assert p["reach"] == pytest.approx({"x": 0.75, "y": 0.75})


def test_the_expert_is_the_half_ring_and_the_oracle_comes_from_the_records():
    """Task a: the expert holds at 100 and the oracle at 10 against a ceiling
    of 1, half the two decades (radius 0.75); the oracle's moves cost a
    quarter of the expert's, so it is the best known (1). Task b: the expert
    already holds within the ceiling and no move is recorded, so neither
    radar has a point for it."""
    rows = {
        "a": {
            "ceiling": 1.0,
            "pid": {"hold_tracking": 100.0, "move_cost": 400.0},
            "mpc": {"hold_tracking": 10.0, "move_cost": 100.0},
        },
        "b": {
            "ceiling": 1.0,
            "pid": {"hold_tracking": 0.5},
            "mpc": {"hold_tracking": 0.1},
        },
    }
    refs = suite.references(rows, tags={"a": ("x",), "b": ("x", "y")})
    assert refs["expert"]["hold"]["x"] == 0.5
    assert refs["oracle"]["hold"]["x"] == pytest.approx(0.75)
    assert refs["expert"]["reach"]["x"] == 0.5
    assert refs["oracle"]["reach"]["x"] == pytest.approx(1.0)
    for who in ("expert", "oracle"):
        assert math.isnan(refs[who]["hold"]["y"])
        assert math.isnan(refs[who]["reach"]["y"])


def test_the_chart_is_a_hold_and_a_reach_radar_with_every_profile():
    flat = dict.fromkeys(suite.AXES, 0.5)
    fig = suite.plot_profile({"expert": {"hold": flat, "reach": flat}})
    polar = [ax for ax in fig.axes if ax.name == "polar"]
    assert [ax.get_title() for ax in polar] == ["Hold", "Reach"]
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert labels == ["expert"]


@pytest.mark.large
def test_the_expert_run_through_the_suite_sits_on_the_half_ring():
    """The expert scored against itself on four_tank: parity on every number,
    so radius 0.5 on both of the task's capabilities."""
    from target_gym.runners.runners import baseline_policy

    def make_policy(name):
        spec = REGISTRY[name]
        return baseline_policy(spec, "pid", spec.make_test_params())

    out = suite.run_suite(make_policy, seeds=2, tasks=["four_tank"])
    m = out["tasks"]["four_tank"]
    assert m["ratio"] == pytest.approx(1.0)
    assert m["transient_ratio"] == pytest.approx(1.0)
    for axis in suite.CAPABILITIES["four_tank"]:
        assert out["profile"]["hold"][axis] == pytest.approx(0.5)
