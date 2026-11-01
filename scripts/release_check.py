"""Is version X.Y.Z ready to release from this commit?

The first job of ``.github/workflows/release.yml``, following
``~/.claude/RELEASING.md`` §2, step 1. It refuses a version that is not three
integers, one with no CHANGELOG entry, one already tagged, and a commit that
is not main's head. The package version itself comes from the tag
(hatch-vcs), so the build job checks the wheel it makes against the input.

    python scripts/release_check.py 0.7.0
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
from collections.abc import Iterable

ROOT = pathlib.Path(__file__).resolve().parents[1]
VERSION = re.compile(r"\d+\.\d+\.\d+")


def problems(
    version: str, changelog: str, tags: Iterable[str], commit: str, main_head: str
) -> list[str]:
    """What blocks releasing ``version`` from ``commit``; empty when nothing."""
    found = []
    if not VERSION.fullmatch(version):
        found.append(f"{version!r} is not a version of the form X.Y.Z")
    # ``## 0.7.0`` or Keep a Changelog's ``## [0.7.0] - 2026-11-02``.
    entry = rf"^## (?:{re.escape(version)}|\[{re.escape(version)}\])(?: .*)?$"
    if not re.search(entry, changelog, re.MULTILINE):
        found.append(f"CHANGELOG.md has no '## [{version}]' entry")
    if f"v{version}" in set(tags):
        found.append(f"the tag v{version} already exists")
    if commit != main_head:
        found.append(f"{commit[:12]} is not main's head ({main_head[:12]})")
    return found


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version")
    version = parser.parse_args().version
    main_head = _git("ls-remote", "origin", "refs/heads/main").split()[0]
    found = problems(
        version,
        (ROOT / "CHANGELOG.md").read_text(),
        _git("tag", "--list").splitlines(),
        _git("rev-parse", "HEAD"),
        main_head,
    )
    for problem in found:
        print(f"::error::{problem}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
