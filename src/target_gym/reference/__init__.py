"""The charter's reference controllers, shipped as data (charter J1).

Each scored task's reference is the rung of the reference ladder the charter kept
for it, or, on the reactor, its shipped expert (``charter.EXPERT_REFERENCE``).
Its numbers are frozen in ``data/reference/<task>.json`` by the ladder's export
command, run on the plants as they stand, for the scored suite::

    uv run python scripts/reference_ladder.py --freeze

:func:`build` rebuilds the controller from that file, with no bump test, and
``target_gym.make_reference`` hands it out. A file frozen on another plant is
refused: the comparator would not be the charter's. A re-record that changes a
task's kept rung changes the comparator every saving is quoted against, so the
CHANGELOG says so.

Kept out of the environment packages and every provenance fingerprint, like
``api.py``: what a reference is made of is not part of what a plant is.
"""

from __future__ import annotations

import pathlib

from target_gym.reference.frozen import FROZEN, assemble, changes, load, path, write

__all__ = ["FROZEN", "StaleReference", "assemble", "build", "changes", "load"]
__all__ += ["path", "write"]

#: The command that freezes a task's reference, for the messages below.
COMMAND = "scripts/reference_ladder.py --freeze"


class StaleReference(RuntimeError):
    """A frozen reference that no longer describes the plant or its expert."""


def build(task: str, root: pathlib.Path = FROZEN):
    """``(controller, frozen)``: task ``task``'s reference as a pure controller,
    rebuilt from its frozen file, and the file's contents."""
    from target_gym import provenance
    from target_gym.registry import get

    spec = get(task)
    frozen = load(task, root)
    if frozen is None:
        have = sorted(p.stem for p in pathlib.Path(root).glob("*.json"))
        raise KeyError(f"{task} has no frozen reference; these tasks do: {have}")
    stamp = provenance.environment_fingerprint(spec)
    if frozen["stamp"] != stamp:
        raise StaleReference(
            f"{task}'s reference was frozen on plant stamp {frozen['stamp']}, and "
            f"the plant is now {stamp}: freeze it again with {COMMAND}"
        )
    if frozen["kind"] == "expert":
        expert = provenance.expert_fingerprint(spec)
        if frozen["expert"] != expert:
            raise StaleReference(
                f"{task}'s reference is its expert as fingerprinted "
                f"{frozen['expert']}, and the expert is now {expert}: freeze it "
                f"again with {COMMAND}"
            )
        return spec.make_pid_controller(), frozen
    return assemble(_row(task), frozen["law"], spec), frozen


def _row(task: str):
    """Task ``task``'s ladder row, as the ladder builds it. The rows live in the
    repository's ``scripts/ladder_rows`` until they move into the package, so
    this needs a source checkout until then."""
    import importlib.util
    import sys

    if "reference_ladder" not in sys.modules:
        from target_gym.charter import SCRIPTS

        script = SCRIPTS / "reference_ladder.py"
        if not script.is_file():
            raise FileNotFoundError(
                f"{script} is absent: the ladder's rows ship in the repository's "
                "scripts/ for now, so rebuilding a ladder rung needs a source "
                "checkout"
            )
        module_spec = importlib.util.spec_from_file_location("reference_ladder", script)
        assert module_spec is not None and module_spec.loader is not None
        module = importlib.util.module_from_spec(module_spec)
        sys.modules["reference_ladder"] = module  # the rows import it by name
        try:
            module_spec.loader.exec_module(module)
        except BaseException:
            del sys.modules["reference_ladder"]
            raise
    return sys.modules["reference_ladder"].ROWS[task]()
