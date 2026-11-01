"""Is version X.Y.Z ready to release from this commit?

The first job of ``.github/workflows/release.yml``, following
``~/.claude/RELEASING.md`` §2, step 1. It refuses a version that is not three
integers, one with no CHANGELOG entry, one already tagged, and a commit that
is not the head of the branch it is released from. That branch is main, or,
for a patch to an older minor, ``release/X.Y``. The package version itself
comes from the tag (hatch-vcs), so the build job checks the wheel it makes
against the input.

    python scripts/release_check.py 0.7.0
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys
from collections.abc import Iterable

ROOT = pathlib.Path(__file__).resolve().parents[1]
VERSION = re.compile(r"\d+\.\d+\.\d+")


def problems(
    version: str,
    changelog: str,
    tags: Iterable[str],
    commit: str,
    branch: str,
    branch_head: str,
) -> list[str]:
    """What blocks releasing ``version`` from ``commit`` on ``branch``; empty
    when nothing."""
    found = []
    if not VERSION.fullmatch(version):
        found.append(f"{version!r} is not a version of the form X.Y.Z")
    minor = version.rsplit(".", 1)[0]
    if branch not in ("main", f"release/{minor}"):
        found.append(f"{version} cannot be released from {branch}")
    # ``## 0.7.0`` or Keep a Changelog's ``## [0.7.0] - 2026-11-02``.
    entry = rf"^## (?:{re.escape(version)}|\[{re.escape(version)}\])(?: .*)?$"
    if not re.search(entry, changelog, re.MULTILINE):
        found.append(f"CHANGELOG.md has no '## [{version}]' entry")
    if f"v{version}" in set(tags):
        found.append(f"the tag v{version} already exists")
    if commit != branch_head:
        found.append(f"{commit[:12]} is not {branch}'s head ({branch_head[:12]})")
    return found


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version")
    version = parser.parse_args().version
    # The branch the workflow was started on (GITHUB_REF_NAME in Actions).
    branch = os.environ.get("GITHUB_REF_NAME") or _git(
        "rev-parse", "--abbrev-ref", "HEAD"
    )
    head = _git("ls-remote", "origin", f"refs/heads/{branch}").split()
    found = problems(
        version,
        (ROOT / "CHANGELOG.md").read_text(),
        _git("tag", "--list").splitlines(),
        _git("rev-parse", "HEAD"),
        branch,
        head[0] if head else "",
    )
    for problem in found:
        print(f"::error::{problem}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
