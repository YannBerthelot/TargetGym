"""Freeze a task at v1: ``TargetGym/<task>-v1`` (task protocol, item 9).

    uv run python scripts/freeze_v1.py reactor

Run once per task, on a clean checkout of the commit it freezes at, when the
task has the protocol's nine items. It writes ``src/target_gym/data/v1/<task>.json``,
the manifest that registers the Gymnasium ID and arms the guard
(``tests/test_v1_freeze.py``). Commit that file on its own. A task already
frozen is refused: a change to it becomes ``<task>-v2`` beside it.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from target_gym import v1  # noqa: E402
from target_gym.registry import REGISTRY  # noqa: E402


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("task", choices=sorted(REGISTRY))
    args = parser.parse_args(argv)
    if _git("status", "--porcelain", "--untracked-files=no"):
        print("refused: the tree has uncommitted changes; freeze a commit")
        return 1
    entry = v1.freeze(args.task, commit=_git("rev-parse", "HEAD"))
    print(f"froze {entry['id']} at {entry['commit'][:10]} ({entry['stamp']})")
    for part, value in entry["parts"].items():
        print(f"  {part}: {value}")
    print(f"commit {v1.MANIFESTS.relative_to(ROOT) / (args.task + '.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
