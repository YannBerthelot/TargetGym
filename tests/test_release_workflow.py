"""The release workflow keeps the gates ``~/.claude/RELEASING.md`` requires.

Contract (RELEASING.md §2 and §4, written by the owner, and the PyPI trusted
publisher the owner configured on 2026-10-28: repository TargetGym, workflow
``release.yml``, environment ``pypi``):

* ``release.yml`` is started by hand only, with the version as input.
* Nothing is tagged or published before the gates pass: the tag needs the
  check, the slow tier and the package gate, and publishing needs the tag.
* Publishing is trusted publishing from the protected ``pypi`` environment:
  an OIDC token, and no PyPI token anywhere in the workflows.
* Tags come only from ``release.yml``; nothing publishes on a GitHub release.
* Once a release is on PyPI, its documentation is deployed as the site's
  version ``X.Y``, aliased ``latest`` (RELEASING.md §2, step 8).
* Once a release is on PyPI, it gets a GitHub Release for its tag, with its
  CHANGELOG entry as notes and the files PyPI received; only a release from
  main is marked Latest (RELEASING.md §2, step 7, added 2026-11-02).
* The package gate builds without ``[tool.uv.sources]``, installs the wheel
  on the oldest and the newest supported Python, and runs each extra on its
  own. It also runs on pull requests that change the packaging.

These read the workflow files: what they assert is what GitHub will run.
"""

import pathlib
import tomllib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def _load(name):
    return yaml.safe_load((WORKFLOWS / name).read_text())


def _triggers(workflow):
    # YAML 1.1 reads a bare ``on`` key as the boolean True.
    return workflow.get("on", workflow.get(True))


def _script(job):
    return "\n".join(step.get("run", "") for step in job.get("steps", []))


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
    # The oldest and the newest Python of requires-python: if it changes, the
    # gate's interpreters change with it.
    assert pyproject["requires-python"] == ">=3.11,<3.15"
    assert matrix["python"] == ["3.11", "3.14"]
    extras = set(pyproject["optional-dependencies"]) - {"all"}
    assert {"core", "docs", *extras} <= set(matrix["check"])


def test_the_package_gate_runs_on_packaging_changes_and_for_the_release():
    triggers = _triggers(_load("package.yml"))
    assert "workflow_call" in triggers
    assert {"pyproject.toml", "uv.lock"} <= set(triggers["pull_request"]["paths"])
    release = _load("release.yml")["jobs"]["package"]
    assert release["uses"] == "./.github/workflows/package.yml"


def test_a_published_release_deploys_its_docs_as_latest():
    jobs = _load("release.yml")["jobs"]
    assert "publish" in _upstream(jobs, "docs")
    docs = jobs["docs"]
    assert docs["uses"] == "./.github/workflows/docs-deploy.yml"
    # "latest" for a release from main, nothing for a patch to an older minor.
    assert "'latest'" in docs["with"]["alias"]
    assert "refs/heads/main" in docs["with"]["alias"]
    assert "workflow_call" in _triggers(_load("docs-deploy.yml"))


def test_a_published_release_gets_a_github_release_with_its_files():
    jobs = _load("release.yml")["jobs"]
    [name] = [n for n, job in jobs.items() if "gh release create" in _script(job)]
    assert "publish" in _upstream(jobs, name)
    job = jobs[name]
    # The files the package job built and PyPI received, not a rebuild.
    assert any(step.get("with", {}).get("name") == "dist" for step in job["steps"])
    assert "release_check.py --notes" in _script(job)
    # Latest only for a release from main, as with the docs' alias.
    assert "refs/heads/main" in yaml.safe_dump(job)


def test_a_rerun_publish_deploys_the_artifact_its_own_attempt_uploaded():
    """Re-running the publish job uploads a second Pages artifact into the
    same run, and deploy-pages refuses a run with two of one name ("Multiple
    artifacts named github-pages", run 41657239439). Each attempt names its
    own and deploys exactly that one."""
    steps = _load("docs-deploy.yml")["jobs"]["publish"]["steps"]
    upload = next(
        s for s in steps if s.get("uses", "").startswith("actions/upload-pages")
    )
    deploy = next(
        s for s in steps if s.get("uses", "").startswith("actions/deploy-pages")
    )
    assert "github.run_attempt" in upload["with"]["name"]
    assert deploy["with"]["artifact_name"] == upload["with"]["name"]
