"""docs/rl-protocol.md gives numbers a training driver could copy, so the ones
the code can check are checked."""

from __future__ import annotations

import pathlib
import re

from target_gym.registry import REGISTRY

PROTOCOL = pathlib.Path(__file__).parents[2] / "docs" / "rl-protocol.md"
ROW = re.compile(r"^\| (\w+) \| (\d+) \| .+ \| (0\.\d{5}) \|$", re.MULTILINE)


def test_the_discount_table_gives_each_tasks_own_episode_and_gamma():
    """Section 6: ``gamma = 1 - 1/N``, with N the environment's own episode."""
    stated = {
        name: (int(n), gamma) for name, n, gamma in ROW.findall(PROTOCOL.read_text())
    }
    episodes = {
        n: int(s.make_test_params().max_steps_in_episode) for n, s in REGISTRY.items()
    }
    assert stated == {n: (N, f"{1 - 1 / N:.5f}") for n, N in episodes.items()}
