"""The release workflow's first gate: is this version ready to tag?

Contract (``~/.claude/RELEASING.md`` §1 and §2, step 1, written by the owner):
a release ``X.Y.Z`` goes ahead only when

* the version is plain semver, three integers;
* CHANGELOG.md has an entry headed for it, ``## X.Y.Z`` or Keep a
  Changelog's ``## [X.Y.Z] - date``, as this project's changelog is written;
* the tag ``vX.Y.Z`` does not exist yet;
* the commit to release is the head of main, or, for a patch to an older
  minor, the head of that minor's ``release/X.Y`` branch (the owner chose a
  0.6.2 patch from ``release/0.6`` on 2026-11-01).

``release_check.problems`` returns what blocks the release, empty when
nothing does. Each case below breaks one rule, so it must report exactly one
problem.

Once on PyPI, the release gets a GitHub Release whose notes are its CHANGELOG
entry (the owner chose this on 2026-11-02). ``release_check.notes`` returns
that entry: the text under its heading, without the heading, up to the next
entry or the link definitions that close a Keep a Changelog file.
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
        version="0.7.0",
        changelog=CHANGELOG,
        tags=TAGS,
        commit=HEAD,
        branch="main",
        branch_head=HEAD,
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


def test_a_commit_behind_its_branch_is_refused():
    assert len(_problems(commit="b" * 40)) == 1


def test_a_patch_comes_from_its_minor_release_branch():
    changelog = "# Changelog\n\n## [0.6.2]\n\n- six is declared.\n"
    assert _problems(version="0.6.2", changelog=changelog, branch="release/0.6") == []


def test_a_release_branch_releases_only_its_own_minor():
    assert len(_problems(branch="release/0.6")) == 1


@pytest.mark.parametrize("branch", ["feature/gate", "release/0.6.x", "releases/0.7"])
def test_any_other_branch_is_refused(branch):
    assert len(_problems(branch=branch)) == 1


NOTES_CHANGELOG = """# Changelog

## [Unreleased]

## [0.7.0] - 2026-11-02

Grouped by what it changes for a user.

### Changed

- Something.

## [0.6.2] - 2026-11-01

### Fixed

- six is declared.

[Unreleased]: https://github.com/YannBerthelot/TargetGym/compare/v0.7.0...HEAD
"""


def test_release_notes_are_the_entry_under_its_heading():
    assert release_check.notes("0.7.0", NOTES_CHANGELOG) == (
        "Grouped by what it changes for a user.\n\n### Changed\n\n- Something.\n"
    )


def test_the_last_entry_ends_before_the_link_definitions():
    assert release_check.notes("0.6.2", NOTES_CHANGELOG) == (
        "### Fixed\n\n- six is declared.\n"
    )


@pytest.mark.parametrize("heading", ["## 0.7.0", "## [0.7.0]"])
def test_release_notes_read_every_heading_form(heading):
    changelog = f"# Changelog\n\n{heading}\n\n- Something.\n"
    assert release_check.notes("0.7.0", changelog) == "- Something.\n"


@pytest.mark.parametrize(
    "version, changelog",
    [("0.8.0", NOTES_CHANGELOG), ("0.7.0", "# Changelog\n\n## [0.7.00]\n\n- No.\n")],
    ids=["absent", "longer-version"],
)
def test_a_version_without_an_entry_has_no_notes(version, changelog):
    with pytest.raises(ValueError):
        release_check.notes(version, changelog)
