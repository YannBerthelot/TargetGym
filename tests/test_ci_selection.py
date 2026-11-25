"""Which tests a pull request into the integration branch runs.

Contract, from the owner's decision of 2026-11-24 and the registry's declared
sources, not from the script:

- A pull request runs the test folders of the tasks whose sources it changes,
  the changed test files, and the cross-task files (registry rules, version
  stamps, fingerprints, scoring, conformance). The whole tier runs on the push.
- A task's sources are its own package and the files its spec declares
  (``fingerprint_sources``): a change to the 2D aircraft's dynamics reaches the
  3D aircraft and the patrol, which fly on them.
- A change that can reach every task runs the whole tier: the shared physics,
  the shared controller code, the registry, the packaging, the workflow, the
  test configuration.
- A path no rule recognises runs the whole tier. Selection may run too much,
  never silently too little.
"""

import importlib.util
import pathlib
import subprocess
import sys

import pytest

from target_gym.registry import REGISTRY

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "ci_select_tests", ROOT / "scripts" / "ci_select_tests.py"
)
selection = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = selection  # its dataclass looks its module up
_spec.loader.exec_module(selection)

CROSS_TASK = {
    "tests/test_registry_rules.py",
    "tests/test_env_versions.py",
    "tests/test_fingerprint_parts.py",
    "tests/test_scoring.py",
    "tests/test_env_conformance.py",
}


def test_a_change_in_one_task_runs_its_folder_and_the_cross_task_files():
    chosen = selection.select(["src/target_gym/hvac/env.py"])

    assert not chosen.everything
    assert chosen.tasks == ("hvac",)
    assert "tests/hvac" in chosen.paths
    assert CROSS_TASK <= set(chosen.paths)


def test_a_nested_package_maps_to_its_nested_test_folder():
    chosen = selection.select(["src/target_gym/pc_gym/cstr/env.py"])

    assert chosen.tasks == ("cstr",)
    assert "tests/pc_gym/cstr" in chosen.paths


def test_a_change_reaches_the_tasks_that_declare_the_file():
    """The 3D aircraft and the patrol declare ``plane/dynamics.py`` among their
    sources (registry), so a change there runs their folders too."""
    chosen = selection.select(["src/target_gym/plane/dynamics.py"])

    assert not chosen.everything
    assert {"plane", "plane_sine", "plane3d_heading", "patrol"} <= set(chosen.tasks)
    assert {"tests/plane", "tests/plane3d", "tests/patrol"} <= set(chosen.paths)
    assert "hvac" not in chosen.tasks


@pytest.mark.parametrize(
    "path",
    [
        "src/target_gym/base.py",
        "src/target_gym/integration.py",
        "src/target_gym/reward.py",
        "src/target_gym/utils.py",
        "src/target_gym/experts/pid.py",
        "src/target_gym/experts/ladder.py",
        "src/target_gym/data/pid_gains.json",
        "src/target_gym/registry.py",
        "src/target_gym/benchmark.py",
        "pyproject.toml",
        "uv.lock",
        ".github/workflows/python-app.yml",
        "tests/conftest.py",
        "scripts/ci_select_tests.py",
    ],
)
def test_a_change_that_can_reach_every_task_runs_everything(path):
    assert selection.select([path]).everything


def test_the_shared_solver_reaches_the_tasks_that_stand_on_it():
    """``oracles/`` is in the oracle sources of the tasks with ``oracle_data``
    (provenance), and has a test folder of its own."""
    chosen = selection.select(["src/target_gym/oracles/shooting.py"])

    assert not chosen.everything
    assert set(chosen.tasks) == {n for n, s in REGISTRY.items() if s.oracle_data}
    assert "tests/oracles" in chosen.paths


def test_one_far_reaching_file_among_task_files_still_runs_everything():
    chosen = selection.select(["src/target_gym/hvac/env.py", "src/target_gym/base.py"])

    assert chosen.everything


def test_a_path_no_rule_recognises_runs_everything():
    assert selection.select(["some/new/place/thing.py"]).everything
    assert selection.select(["src/target_gym/a_new_module.py"]).everything


def test_a_changed_test_file_is_run_itself():
    changed = sorted((ROOT / "tests" / "hvac").glob("test_*.py"))[0]
    rel = changed.relative_to(ROOT).as_posix()

    chosen = selection.select([rel])

    assert not chosen.everything
    assert rel in chosen.paths
    assert "tests/hvac" not in chosen.paths


def test_a_rendering_file_reaches_its_task_by_its_folder():
    """Drawing code is in no task's declared sources (it cannot change a
    return), yet the task's folder holds its render tests."""
    chosen = selection.select(["src/target_gym/pc_gym/cstr/rendering.py"])

    assert not chosen.everything
    assert "tests/pc_gym/cstr" in chosen.paths


def test_documents_alone_run_the_docs_tests_and_no_task():
    chosen = selection.select(["docs/index.md", "README.md", "mkdocs.yml"])

    assert not chosen.everything
    assert chosen.tasks == ()
    assert set(chosen.paths) == {"tests/test_docs.py", "tests/test_docs_redirects.py"}


def test_a_physics_contract_runs_the_docs_tests_and_the_cross_task_files():
    """A task page is generated from its PHYSICS.md, and the stamp does not
    hash it: no task folder, the docs tests, the cross-task files."""
    chosen = selection.select(["src/target_gym/hvac/PHYSICS.md"])

    assert not chosen.everything
    assert "tests/test_docs.py" in chosen.paths
    assert "tests/hvac" not in chosen.paths


def test_a_version_stamp_runs_the_cross_task_files():
    chosen = selection.select(["src/target_gym/data/env_versions/hvac-v6.json"])

    assert not chosen.everything
    assert CROSS_TASK <= set(chosen.paths)


def test_a_tasks_record_reaches_that_task():
    chosen = selection.select(["src/target_gym/data/oracle_ceilings/hvac.json"])

    assert not chosen.everything
    assert chosen.tasks == ("hvac",)


def test_a_ladder_row_reaches_its_task_and_the_ladder_tests():
    chosen = selection.select(["scripts/ladder_rows/hvac.py"])

    assert not chosen.everything
    assert chosen.tasks == ("hvac",)
    assert "tests/experts/test_reference_ladder.py" in chosen.paths


def test_a_script_runs_the_test_files_that_name_it():
    """``tests/test_env_versions.py`` loads ``scripts/stamp_env_versions.py``."""
    chosen = selection.select(["scripts/stamp_env_versions.py"])

    assert not chosen.everything
    assert "tests/test_env_versions.py" in chosen.paths


def test_no_source_file_is_silently_left_unrun():
    """Every tracked source file either runs everything or reaches a task, and
    every path selected exists."""
    tracked = subprocess.run(
        ["git", "ls-files", "src/target_gym/*.py", "src/target_gym/**/*.py"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()  # fmt: skip
    assert len(tracked) > 100
    for path in tracked:
        chosen = selection.select([path])
        assert chosen.everything or chosen.tasks, path
        missing = [p for p in chosen.paths if not (ROOT / p).exists()]
        assert not missing, (path, missing)


def test_every_task_has_a_test_folder_the_selection_can_name():
    for name in REGISTRY:
        folder = selection.test_folder(name)
        assert (ROOT / folder).is_dir(), (name, folder)


def test_the_task_filter_names_the_touched_tasks():
    """The other top-level test files run filtered to the touched tasks."""
    chosen = selection.select(["src/target_gym/hvac/env.py"])

    assert chosen.keyword == "hvac"
    assert selection.select(["docs/index.md"]).keyword is None


def test_ci_selects_on_a_pull_request_and_keeps_the_whole_tier_on_the_push():
    """The workflow's test job: one step runs the selection, only on a pull
    request into the integration branch and against the merge commit's first
    parent; the whole tier with its coverage gate runs everywhere else."""
    import yaml

    workflow = ROOT / ".github" / "workflows" / "python-app.yml"
    steps = yaml.safe_load(workflow.read_text())["jobs"]["test"]["steps"]
    selecting = [s for s in steps if "ci_select_tests.py" in s.get("run", "")]
    whole = [s for s in steps if "--cov=target_gym" in s.get("run", "")]

    assert len(selecting) == 1 and len(whole) == 1
    assert selecting[0]["if"] == "github.base_ref == 'int/amendment3'"
    assert "github.base_ref != 'int/amendment3'" in whole[0]["if"]
    assert "--base HEAD^1" in selecting[0]["run"]
    assert "not records" in selecting[0]["run"]
    assert steps[0]["with"]["fetch-depth"] >= 2


def _conftest():
    spec = importlib.util.spec_from_file_location(
        "tg_conftest", ROOT / "tests" / "conftest.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_shards_split_the_tests_into_disjoint_parts_that_cover_them_all():
    """A pull request's tests run on several runners at once (owner, 2026-11-24):
    every test runs on exactly one of them, whatever the order of collection."""
    in_shard = _conftest().in_shard
    ids = [f"tests/test_x.py::test_{i}[case{i % 7}]" for i in range(300)]

    parts = [{t for t in ids if in_shard(t, k, 3)} for k in (1, 2, 3)]

    assert set().union(*parts) == set(ids)
    assert sum(len(p) for p in parts) == len(ids)
    assert all(len(p) > 50 for p in parts)  # 100 each expected; none starved
    assert parts[0] == {t for t in reversed(ids) if in_shard(t, 1, 3)}


def test_one_shard_of_one_is_every_test():
    assert _conftest().in_shard("tests/test_x.py::test_a", 1, 1)


def test_ci_shards_the_selected_run_and_requires_one_aggregate_check():
    """On a pull request into the integration branch the test job runs as three
    shards, and one job, 'tests pass', is what the branch requires."""
    import yaml

    jobs = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "python-app.yml").read_text()
    )["jobs"]
    matrix = jobs["test"]["strategy"]["matrix"]

    assert "[1, 2, 3]" in matrix["shard"] or "[1,2,3]" in matrix["shard"]
    assert "TG_SHARD" in jobs["test"]["env"]
    assert jobs["tests-pass"]["name"] == "tests pass"
    assert jobs["tests-pass"]["needs"] == "test"


def _workflow_jobs():
    import yaml

    return yaml.safe_load(
        (ROOT / ".github" / "workflows" / "python-app.yml").read_text()
    )["jobs"]


def test_a_push_to_the_integration_branch_runs_the_whole_tier_in_shards():
    """Owner, 2026-11-25: the push after a merge runs the whole tier on every
    Python, split in three shards like a pull request, since unsplit it ran past
    the job's 60 minutes on three of four Pythons."""
    matrix = _workflow_jobs()["test"]["strategy"]["matrix"]

    assert "refs/heads/int/amendment3" in matrix["shard"]
    assert "[1, 2, 3]" in matrix["shard"]


def test_sharded_coverage_is_combined_and_gated_once():
    """A shard covers a third of the tests, so no shard can hold the coverage
    gate: each one uploads its data and 'tests pass' combines them and enforces
    the project's floor, on the push as on the pull request."""
    jobs = _workflow_jobs()
    steps = jobs["test"]["steps"]
    covered = [s for s in steps if "--cov=target_gym" in s.get("run", "")]
    gate = jobs["tests-pass"]

    assert len(covered) == 1
    assert "--cov-fail-under" in covered[0]["run"]  # waived per shard only
    assert any("upload-artifact" in s.get("uses", "") for s in steps)
    assert "refs/heads/int/amendment3" in gate["if"]
    runs = " ".join(s.get("run", "") for s in gate["steps"])
    assert "coverage combine" in runs and "coverage report" in runs


def test_a_solo_test_never_shares_a_runner_with_another_worker():
    """Owner, 2026-11-25: a test whose compile fills a runner (the four-wingman
    formation rollout, about 11 GB of a 16 GB runner) runs alone. Every
    parallel run of the tier leaves it out, and one step runs it on one
    worker."""
    import tomllib

    steps = _workflow_jobs()["test"]["steps"]
    parallel = [s["run"] for s in steps if "-n auto" in s.get("run", "")]
    alone = [s for s in steps if '-m "solo' in s.get("run", "")]
    markers = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["pytest"][
        "ini_options"
    ]["markers"]

    assert parallel and all("not solo" in run for run in parallel), parallel
    assert len(alone) == 1 and "-n 0" in alone[0]["run"]
    assert any(m.startswith("solo:") for m in markers)
    source = (ROOT / "tests" / "formation" / "test_formation_marl.py").read_text()
    assert "mark.solo" in source


def test_a_push_run_is_never_cancelled_by_a_later_push():
    """Two merges into int an hour apart cancelled each push run before its
    medium tier finished (2026-11-25, runs 45623244455 and 45634414414), so
    the tier never reported. A pull request's run still gives way to its next
    push; a branch push waits its turn instead."""
    import yaml

    group = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "python-app.yml").read_text()
    )["concurrency"]
    cancel = str(group["cancel-in-progress"])
    assert "pull_request" in cancel and cancel.startswith("${{"), cancel
