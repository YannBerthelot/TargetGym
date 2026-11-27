"""docs/rl-protocol.md gives numbers a training driver could copy, so the ones
the code can check are checked."""

from __future__ import annotations

import pathlib
import re

from target_gym.charter import FLAGSHIP
from target_gym.registry import REGISTRY, control_step_seconds

PROTOCOL = pathlib.Path(__file__).parents[2] / "docs" / "rl-protocol.md"
ROW = re.compile(
    r"^\| (\w+) \| ([\d.]+) s \| (\d+|not finite) \| (\d+|none) \| (\d+|open) "
    r"\| (0\.\d{5}|open) \|$",
    re.MULTILINE,
)


def test_the_discount_table_gives_each_scored_tasks_window_and_gamma():
    """Section 6, decided 2026-11-10 (long strata, S4), which retired
    ``gamma = 1 - 1/N``: ``gamma = 1 - 1/(10 T)`` with ``T`` the longer of the
    slow mode ``t_s`` and the longest preview, in the task's control steps. A slow
    mode that is not finite gives no ``T``: the row says so, never a number."""
    rows = {m[0]: m[1:] for m in ROW.findall(PROTOCOL.read_text())}
    assert set(rows) == set(FLAGSHIP)
    for name, (step, t_s, preview, T, gamma) in rows.items():
        spec = REGISTRY[name]
        assert float(step) == control_step_seconds(
            spec.make_env(), spec.make_test_params()
        )
        if t_s == "not finite":
            assert (T, gamma) == ("open", "open"), name
            continue
        assert int(T) == max(int(t_s), 0 if preview == "none" else int(preview)), name
        assert gamma == f"{1 - 1 / (10 * int(T)):.5f}", name


def test_the_protocol_names_the_reward_a_learner_trains_on():
    """The outside review's L5: the owner's measurement-noise decision 3
    (2026-11-05) has learners train on the measured reward while the score reads
    the true cost, and section 2 says so where a driver reads its settings."""
    section = PROTOCOL.read_text().split("## 2.")[1].split("## 3.")[0]
    assert 'info["measured_reward"]' in section
