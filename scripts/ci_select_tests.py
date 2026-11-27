"""The tests a pull request into the integration branch runs.

    uv run python scripts/ci_select_tests.py --base HEAD^1 -- -q -n 2
    uv run python scripts/ci_select_tests.py --changed src/target_gym/hvac/env.py -- -q

Owner's rule (2026-11-24): a pull request runs the test folders of the tasks
whose sources it changes, the test files it changes and the cross-task files;
the whole tier runs on the push. A task's sources are its package and the files
its spec declares (``provenance``), so a change to a file every task depends on
selects every task, and that runs everything. So does a path no rule here
recognises: the selection may run too much, never silently too little.

The other top-level test files run too, filtered to the touched tasks' names
(``-k``), since most of them are parametrised by task.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = "src/target_gym/"

#: Run whole on every selected pull request: the checks that span the tasks.
CROSS_TASK = (
    "tests/test_registry_rules.py",
    "tests/test_scaled.py",
    "tests/test_env_versions.py",
    "tests/test_fingerprint_parts.py",
    "tests/test_scoring.py",
    "tests/test_env_conformance.py",
    "tests/test_flagship.py",
    "tests/test_current_rows.py",
    "tests/test_ceiling_foresight.py",
)
DOCS = ("tests/test_docs.py", "tests/test_docs_redirects.py")
LADDER = ("tests/experts/test_ladder.py", "tests/experts/test_reference_ladder.py")
#: Shared code with a test folder of its own, run beside the tasks it reaches.
OWN_FOLDER = {"experts": "tests/experts", "oracles": "tests/oracles"}
DOCUMENTS = (".md", ".ipynb", ".cff", ".gif", ".png", ".svg", ".mp4")
DOCUMENT_ROOTS = (
    "docs/",
    "notebooks/",
    "videos/",
    "figures/",
    "evidence/",
    "overrides/",
)
EVERYTHING_NAMES = {"pyproject.toml", "uv.lock", "Makefile", "conftest.py"}


@dataclasses.dataclass(frozen=True)
class Selection:
    everything: bool
    paths: tuple = ()
    tasks: tuple = ()

    @property
    def keyword(self) -> str | None:
        """The ``-k`` filter for the other top-level files, or ``None``."""
        return " or ".join(self.tasks) if self.tasks and not self.everything else None


@functools.cache
def _tasks() -> dict[str, tuple[str, frozenset]]:
    """Each task's package folder and its source files, as repo-relative paths."""
    from target_gym import provenance
    from target_gym.registry import REGISTRY

    out = {}
    for name, spec in REGISTRY.items():
        module = type(spec.make_env()).__module__.rsplit(".", 1)[0]
        files = {*provenance.expert_sources(spec), *provenance.oracle_sources(spec)}
        rel = frozenset(p.resolve().relative_to(ROOT).as_posix() for p in files)
        out[name] = ("src/" + module.replace(".", "/") + "/", rel)
    return out


def test_folder(task: str) -> str:
    """The test folder that mirrors a task's package."""
    return "tests/" + _tasks()[task][0][len(SRC) :].rstrip("/")


def _naming(script: str) -> list[str]:
    """The test files whose text names a script."""
    found = []
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        if script in path.read_text():
            found.append(path.relative_to(ROOT).as_posix())
    return found


def _one(path: str) -> tuple[bool, set, set, bool]:
    """``(everything, tasks, paths, spans_tasks)`` for one changed path."""
    name = pathlib.PurePosixPath(path).name
    document = path.endswith(DOCUMENTS) or path.startswith(DOCUMENT_ROOTS)
    if name in EVERYTHING_NAMES or path.startswith(".github/"):
        return True, set(), set(), True
    if path.startswith("tests/"):
        if name.startswith("test_") and name.endswith(".py"):
            return False, set(), {path}, False
        return False, set(), set(), True  # pinned data: the cross-task files
    if path.startswith(SRC + "reference/rows/"):
        task = {name[:-3]} & set(_tasks())
        return False, task, set(LADDER), True
    if path.startswith("scripts/"):
        if name == "ci_select_tests.py":
            return True, set(), set(), True
        return False, set(), set(_naming(name)), False
    if path.startswith(SRC + "data/"):
        stem = name.split(".")[0]
        if stem in _tasks():  # a record kept per task
            return False, {stem}, set(), True
        # A stamp is checked by the cross-task files; a file holding every
        # task's rows (the gains) cannot be attributed, so it runs everything.
        return "/env_versions/" not in path, set(), set(), True
    if path.startswith(SRC):
        inside = {t for t, (folder, _) in _tasks().items() if path.startswith(folder)}
        if document:
            return False, set(), set(DOCS), bool(inside)
        tasks = inside | {t for t, (_, files) in _tasks().items() if path in files}
        own = OWN_FOLDER.get(path[len(SRC) :].split("/")[0])
        return not tasks, tasks, {own} if own else set(), True
    if document or name == "mkdocs.yml":
        return False, set(), set(DOCS), False
    return True, set(), set(), True


def select(changed: list[str]) -> Selection:
    """What to run for a set of changed paths (see the module docstring)."""
    everything, tasks, paths, spans = not changed, set(), set(), False
    for path in changed:
        all_, t, p, s = _one(path)
        everything, tasks, paths, spans = (
            everything or all_,
            tasks | t,
            paths | p,
            spans or s,
        )
    if everything or len(tasks) == len(_tasks()):
        return Selection(True)
    folders = {test_folder(t) for t in tasks}
    # A changed test file inside a selected folder is already run by it.
    paths = {p for p in paths if not any(p.startswith(f + "/") for f in folders)}
    cross = set(CROSS_TASK) if spans or tasks else set()
    return Selection(
        False, tuple(sorted(folders | paths | cross)), tuple(sorted(tasks))
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    given = ap.add_mutually_exclusive_group(required=True)
    given.add_argument("--base", help="the commit the change is read against")
    given.add_argument("--changed", nargs="+", help="the changed paths themselves")
    ap.add_argument("pytest_args", nargs="*", help="passed to every pytest run")
    args = ap.parse_args()
    changed = args.changed
    if changed is None:
        diff = ["git", "diff", "--name-only", args.base, "HEAD"]
        done = subprocess.run(
            diff, cwd=ROOT, capture_output=True, text=True, check=True
        )
        changed = done.stdout.split("\n")
    chosen = select([c for c in changed if c])
    pytest = [sys.executable, "-m", "pytest", *args.pytest_args]
    if chosen.everything:
        print("selection: the whole tier", flush=True)
        return subprocess.run(pytest, cwd=ROOT).returncode
    print(
        f"selection: tasks {list(chosen.tasks)}, paths {list(chosen.paths)}", flush=True
    )
    runs = [[*pytest, *chosen.paths]] if chosen.paths else []
    if chosen.keyword:
        top = (ROOT / "tests").glob("test_*.py")
        others = sorted(p.relative_to(ROOT).as_posix() for p in top)
        rest = [p for p in others if p not in chosen.paths]
        runs.append([*pytest, "-k", chosen.keyword, *rest])
    # 5 is pytest's "no test collected", which a filter may rightly give.
    codes = [subprocess.run(run, cwd=ROOT).returncode for run in runs]
    return next((c for c in codes if c not in (0, 5)), 0)


if __name__ == "__main__":
    sys.exit(main())
