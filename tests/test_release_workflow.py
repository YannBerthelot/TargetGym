"""The release workflow keeps the gates ``~/.claude/RELEASING.md`` requires.

These read the workflow files, so they fail only when someone edits one. That
is worth it for a mistake that cannot be undone: a version on PyPI cannot be
replaced and a leaked token cannot be recalled. A step that can simply be run
again after a failure has no test here.

Contract (RELEASING.md §2 and §4, written by the owner, and the PyPI trusted
publisher the owner configured on 2026-10-28: repository TargetGym, workflow
``release.yml``, environment ``pypi``):

* ``release.yml`` is started by hand only, with the version as input.
* Nothing is tagged or published before the gates pass: the tag needs the
  check, the slow tier and the package gate, and publishing needs the tag.
* Publishing is trusted publishing from the protected ``pypi`` environment:
  an OIDC token, and no PyPI token anywhere in the workflows.
* Tags come only from ``release.yml``; nothing publishes on a GitHub release.
* The package gate builds without ``[tool.uv.sources]``, installs the wheel
  on the oldest and the newest supported Python, and runs each extra on its
  own.

These read the workflow files: what they assert is what GitHub will run.
"""

import pathlib
import re
import tomllib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def _load(name):
    return yaml.safe_load((WORKFLOWS / name).read_text())


def _triggers(workflow):
    # YAML 1.1 reads a bare ``on`` key as the boolean True.
    return workflow.get("on", workflow.get(True))


def _upstream(jobs, name):
    """Every job ``name`` waits for, directly or through another job."""
    seen, todo = set(), [name]
    while todo:
        needs = jobs[todo.pop()].get("needs", [])
        for job in [needs] if isinstance(needs, str) else needs:
            if job not in seen:
                seen.add(job)
                todo.append(job)
    return seen


def test_a_release_is_started_by_hand_with_its_version():
    triggers = _triggers(_load("release.yml"))
    assert set(triggers) == {"workflow_dispatch"}
    assert triggers["workflow_dispatch"]["inputs"]["version"]["required"] is True


def test_nothing_is_tagged_or_published_before_the_gates_pass():
    jobs = _load("release.yml")["jobs"]
    assert {"check", "slow", "package"} <= _upstream(jobs, "tag")
    assert jobs["package"]["uses"] == "./.github/workflows/package.yml"
    assert "tag" in _upstream(jobs, "publish")
    assert "publish" in _upstream(jobs, "verify")


def test_publishing_is_trusted_publishing_from_the_pypi_environment():
    publish = _load("release.yml")["jobs"]["publish"]
    assert publish["environment"] == "pypi"
    assert publish["permissions"]["id-token"] == "write"


def test_no_workflow_holds_a_pypi_token():
    for path in WORKFLOWS.glob("*.yml"):
        text = path.read_text()
        for token in ("PYPI_TOKEN", "UV_PUBLISH_TOKEN", "TWINE_PASSWORD"):
            assert token not in text, f"{path.name} uses {token}"


def test_only_the_release_workflow_tags_and_nothing_publishes_on_a_release():
    for path in WORKFLOWS.glob("*.yml"):
        assert "release" not in (_triggers(_load(path.name)) or {}), path.name
        if path.name != "release.yml":
            text = path.read_text()
            assert "git tag" not in text and "gh release create" not in text


def test_the_package_gate_installs_the_wheel_as_users_get_it():
    package = _load("package.yml")
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]

    build = "\n".join(step.get("run", "") for step in package["jobs"]["build"]["steps"])
    assert "uv build --no-sources" in build

    matrix = package["jobs"]["installed"]["strategy"]["matrix"]
    # The oldest and the newest Python of requires-python.
    oldest, ceiling = re.fullmatch(
        r">=3\.(\d+),<3\.(\d+)", pyproject["requires-python"]
    ).groups()
    assert matrix["python"] == [f"3.{oldest}", f"3.{int(ceiling) - 1}"]
    extras = set(pyproject["optional-dependencies"]) - {"all"}
    assert {"core", "docs", *extras} <= set(matrix["check"])
