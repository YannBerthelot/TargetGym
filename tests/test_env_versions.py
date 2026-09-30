"""An environment cannot change without its version changing.

Versioning starts at the 0.6 release, where every environment ships as `v1`.
Nothing before that is versioned: the package had no users, so there were no
published results to keep meaningful.

What the version buys a reader is the ability to cite a number. `plane-v1` has
to mean one thing forever, so the interesting test is not that the name exists
but that the *thing behind it* has not moved. `src/target_gym/data/env_versions.json` records
the fingerprint each version was stamped at, and this fails when the tree
disagrees with it.

The fingerprint deliberately excludes the controllers: re-tuning a PID does not
change what the environment is, and should not force a version bump.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from target_gym.provenance import environment_fingerprint
from target_gym.registry import REGISTRY

PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "src"
    / "target_gym"
    / "data"
    / "env_versions.json"
)
STORED = json.loads(PATH.read_text()) if PATH.exists() else {}

_INSTRUCTION = (
    "Decide which of these it is, then run scripts/stamp_env_versions.py:\n"
    "  1. the behaviour changed -> bump `version` on the EnvSpec and re-stamp;\n"
    "  2. it cannot have changed (a rename, a refactor, a moved constant)\n"
    "     -> re-stamp in place with --note saying why it is behaviour-preserving.\n"
    "Re-stamping in place to turn this green without deciding is the one way\n"
    "this file can lie about what a published number means."
)


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_environment_is_versioned(name):
    """Every registered environment carries a version, so it can be cited."""
    spec = REGISTRY[name]
    assert isinstance(spec.version, int) and spec.version >= 1
    assert spec.versioned_name == f"{name}-v{spec.version}"


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_environment_matches_the_version_it_claims(name):
    """The tree still describes the version it says it is."""
    spec = REGISTRY[name]
    stamped = STORED.get(spec.versioned_name)
    assert stamped is not None, (
        f"{spec.versioned_name} has never been stamped. "
        f"Run scripts/stamp_env_versions.py."
    )
    assert stamped["fingerprint"] == environment_fingerprint(spec), (
        f"{spec.versioned_name} no longer matches what was stamped for it.\n"
        f"{_INSTRUCTION}"
    )


def test_no_version_is_silently_reused():
    """A bumped version must not inherit the fingerprint of the one before it.

    Catches the mistake this whole file exists to prevent: bumping the number
    but re-stamping the old fingerprint under it, which would publish two names
    for one environment and make the version meaningless.
    """
    by_fingerprint: dict[str, list[str]] = {}
    for name, entry in STORED.items():
        by_fingerprint.setdefault(entry["fingerprint"], []).append(name)
    collisions = {
        fp: sorted(names)
        for fp, names in by_fingerprint.items()
        # Two *different* environments sharing a fingerprint is impossible in
        # practice (their modules differ), so a collision means two versions of
        # the same one.
        if len({n.rsplit("-v", 1)[0] for n in names}) == 1 and len(names) > 1
    }
    assert not collisions, (
        f"these versions share a fingerprint, so the bump recorded no change: "
        f"{collisions}"
    )
