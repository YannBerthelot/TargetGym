"""The release workflow's first gate: is this version ready to tag?

Contract (``~/.claude/RELEASING.md`` §1 and §2, step 1, written by the owner):
a release ``X.Y.Z`` goes ahead only when

* the version is plain semver, three integers;
* CHANGELOG.md has an entry headed for it, ``## X.Y.Z`` or Keep a
  Changelog's ``## [X.Y.Z] - date``, as this project's changelog is written;
* the tag ``vX.Y.Z`` does not exist yet;
* the commit to release is main's head.

``release_check.problems`` returns what blocks the release, empty when
nothing does. Each case below breaks one rule, so it must report exactly one
problem.
"""

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "_release_check", ROOT / "scripts" / "release_check.py"
)
release_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release_check)

CHANGELOG = """# Changelog

## [Unreleased]

## [0.7.0] - 2026-11-02

### Changed

- Something.
"""
TAGS = ["0.5.0", "v0.6.0", "v0.6.1"]
HEAD = "a" * 40


def _problems(**changes):
    args = dict(
        version="0.7.0", changelog=CHANGELOG, tags=TAGS, commit=HEAD, main_head=HEAD
    )
    args.update(changes)
    return release_check.problems(**args)


def test_a_ready_release_has_no_problems():
    assert _problems() == []


def test_a_plain_heading_counts_as_the_entry():
    assert _problems(changelog="# Changelog\n\n## 0.7.0\n\n- Something.\n") == []


@pytest.mark.parametrize("version", ["0.7", "v0.7.0", "0.7.0rc1", "0.7.0 ", "a.b.c"])
def test_a_version_that_is_not_three_integers_is_refused(version):
    assert len(_problems(version=version)) >= 1


@pytest.mark.parametrize(
    "changelog",
    [
        "# Changelog\n\n## [Unreleased]\n\n- Something.\n",
        # A longer version that merely contains this one.
        "# Changelog\n\n## [10.7.0] - 2026-11-02\n",
        "# Changelog\n\n## [0.7.00] - 2026-11-02\n",
        "# Changelog\n\n## 0.7.00\n",
        # A subsection, not an entry.
        "# Changelog\n\n### 0.7.0\n",
        # Mentioned in prose only.
        "# Changelog\n\nSee 0.7.0 below.\n",
    ],
    ids=[
        "unreleased-only",
        "longer-version",
        "padded",
        "padded-plain",
        "subsection",
        "prose",
    ],
)
def test_a_missing_changelog_entry_is_refused(changelog):
    assert len(_problems(changelog=changelog)) == 1


def test_an_existing_tag_is_refused():
    assert len(_problems(tags=[*TAGS, "v0.7.0"])) == 1


def test_a_commit_behind_main_is_refused():
    assert len(_problems(commit="b" * 40)) == 1
