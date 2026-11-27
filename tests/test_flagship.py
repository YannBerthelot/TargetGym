"""The scored suite, the charter's flagship (docs/proposals/task-charter.md,
Words and J5).

Contract, from the owner's decisions of 2026-11-24 and 2026-11-27 and not from
the code:

- the scored suite is reactor, hvac, compressor_surge and ph_neutralization,
  stated in one place (``charter.FLAGSHIP``, also ``tg.FLAGSHIP``): the first
  suite of 2026-11-24 less cstr, which left it for 0.8.0 on 2026-11-27
  ("Registered, not scored");
- every other task ships registered, not scored: it runs through ``tg.make``,
  its page says it is under review and its charter lines are not measured, it
  is out of the results and moves no radar spoke, and its records are not held
  current, so the records job goes green with only the four recorded;
- ``tg.list_envs()`` still gives every registered name;
- the old-format rows of the tasks outside the suite stay in the repository,
  and no page prints a number from them;
- the README shows the four in a table and the others by family;
- the capability radar is out of the docs for this release, with a line saying
  it returns when it is rebuilt on the saving over the reference; its code
  stays.

A task's status is one of four: scored (in the flagship), Extra (the
registry's reason, unchanged), fixture (``charter.FIXTURES``), or under review.
"""

from __future__ import annotations

import importlib.util
import inspect
import math
import os
import pathlib
import re
import subprocess
import sys

import pytest

import target_gym as tg
from target_gym import charter, registry, suite

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

#: The owner's choice of 2026-11-24, less cstr (2026-11-27).
FOUR = {"reactor", "hvac", "compressor_surge", "ph_neutralization"}
OTHERS = set(registry.REGISTRY) - FOUR


def _script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", ROOT / "scripts" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _names(text: str) -> list[str]:
    """Every registered task name ``text`` writes in backticks, in order."""
    return [n for n in re.findall(r"`(\w+)`", text) if n in registry.REGISTRY]


# --- the list and the status ------------------------------------------------


def test_the_flagship_is_the_four_tasks_of_the_decisions():
    assert len(charter.FLAGSHIP) == 4
    assert set(charter.FLAGSHIP) == FOUR
    assert tg.FLAGSHIP == charter.FLAGSHIP


def test_a_scored_task_is_registered_and_neither_extra_nor_a_fixture():
    for name in charter.FLAGSHIP:
        assert registry.get(name).extra is None, name
        assert name not in charter.FIXTURES, name


@pytest.mark.parametrize(
    "name, status",
    [
        ("reactor", "scored"),
        ("ph_neutralization", "scored"),
        ("cstr", "under review"),  # the owner, 2026-11-27
        # The owner, 2026-11-24: glass_furnace joins when its four pieces are
        # done, cement_kiln ships as it is, not scored.
        ("glass_furnace", "under review"),
        ("cement_kiln", "under review"),
        ("plane", "under review"),
        ("four_tank", "extra"),
        ("battery", "extra"),
        ("first_order", "fixture"),
    ],
)
def test_a_task_has_the_status_its_place_gives_it(name, status):
    assert charter.status(name) == status


def test_every_task_has_one_status_and_four_are_scored():
    statuses = {name: charter.status(name) for name in registry.REGISTRY}
    assert set(statuses.values()) <= {"scored", "under review", "extra", "fixture"}
    assert {n for n, s in statuses.items() if s == "scored"} == FOUR


def test_an_unregistered_name_has_no_status():
    with pytest.raises(KeyError):
        charter.status("no_such_task")


# --- outside the suite a task still runs ------------------------------------


def test_list_envs_still_gives_every_registered_task():
    assert tg.list_envs() == registry.env_names()
    assert FOUR < set(tg.list_envs())


@pytest.mark.parametrize("name", ["plane", "four_tank", "first_order", "hvac"])
def test_a_task_of_any_status_is_made_the_same_way(name):
    env, params = tg.make(name)
    assert env is registry.get(name).make_env()
    assert int(params.max_steps_in_episode) > 0


# --- the pages --------------------------------------------------------------

STATUS_ROW = {
    "scored": "Scored",
    "under review": "Under review, not scored",
    "extra": "Extra, never scored",
    "fixture": "Test fixture, never scored",
}


def _page(name: str) -> str:
    return (DOCS / "environments" / f"{name}.md").read_text()


@pytest.mark.parametrize("name", sorted(registry.REGISTRY))
def test_a_task_page_states_its_status(name):
    row = re.search(r"^\| Status \| (.+) \|$", _page(name), re.MULTILINE)
    assert row, f"{name}: no Status row"
    assert row.group(1) == STATUS_ROW[charter.status(name)]


@pytest.mark.parametrize("name", sorted(registry.REGISTRY))
def test_a_page_says_under_review_and_not_measured_exactly_when_its_task_is(name):
    text = _page(name)
    noted = '!!! note "Under review"' in text
    assert noted == (charter.status(name) == "under review")
    if noted:
        note = text.split('!!! note "Under review"')[1].split("\n\n")[0]
        assert "not scored" in note and "not measured" in note


@pytest.mark.parametrize("name", sorted(OTHERS))
def test_a_page_outside_the_suite_prints_no_recorded_number(name):
    """Its records are not held current, so none of their numbers is shown:
    neither the recorded returns nor a charter line."""
    text = _page(name)
    baselines = text.split("\n## Baselines\n")[1].split("\n## ")[0]
    assert baselines.strip().startswith("Not measured")
    assert "| PID |" not in text and "Recorded over" not in text
    assert "\n## Reference and ceiling\n" not in text


@pytest.mark.parametrize("name", sorted(FOUR))
def test_a_scored_page_shows_the_four_charter_lines(name):
    """Charter J3: the best null policy, the reference, the MPC and the
    ceiling, each with its cost or "not measured"."""
    section = _page(name).split("\n## Reference and ceiling\n")[1].split("\n## ")[0]
    lines = re.findall(r"^\| (\w[\w ]*?) \|", section, re.MULTILINE)
    assert lines[1:] == ["Best null policy", "Reference", "MPC", "Ceiling"]


def test_a_scored_page_says_not_measured_for_an_mpc_line_its_record_lacks(monkeypatch):
    """The owner, 2026-11-24: a scored task's MPC line may read "not measured"
    in this release. The page then says so with the record's reason, and still
    gives the other three lines and the room: the reference costs 2 per step
    and the ceiling 1, so the room is 2."""
    weights = {"normal": 1.0}

    def line(cost):
        return charter.line({"normal": {"costs": [cost, cost], "trips": 0}}, weights)

    record = {
        "weights": weights,
        "nulls": {"lines": {"hold": line(8.0)}},
        "reference": {"line": line(2.0)},
        "mpc": {"line": charter.NOT_MEASURED, "why": "the task ships no oracle"},
        "ceiling": {"line": line(1.0)},
    }
    # The page generator imports the reference generator beside it.
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    section = _script("generate_env_pages").charter_section(record)
    rows = {row["Line"]: row["Cost per step"] for row in _table(section)}
    assert rows == {
        "Best null policy": "8",
        "Reference": "2",
        "MPC": "not measured",
        "Ceiling": "1",
    }
    assert "the task ships no oracle" in section
    assert "is 2." in section


@pytest.mark.parametrize("not_held", [["feed_buildup"], []])
def test_a_scored_page_says_which_scenarios_its_reference_does_not_hold(
    monkeypatch, not_held
):
    """The shared null margin (2026-11-27): the page lists the scenarios where a
    null beats the reference by under 1 % in the sweep, from the record's J2, and
    says nothing of the kind where there are none."""
    weights = {"normal": 1.0}
    line = charter.line({"normal": {"costs": [2.0, 2.0], "trips": 0}}, weights)
    record = {
        "task": "cstr",
        "weights": weights,
        "reference": {"line": line},
        "j2": {"sweep_not_held": not_held},
    }
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    section = _script("generate_env_pages").charter_section(record)
    assert ("not held" in section) == bool(not_held)
    for scenario in not_held:
        assert f"`{scenario}` is not held" in section
        assert "under 1 %" in section


def _table(text: str) -> list[dict]:
    """Every row of every Markdown table in ``text``, keyed by its header."""
    rows, header = [], None
    for line in text.splitlines():
        if not line.startswith("|"):
            header = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
        elif not set(cells[0]) <= set("-: "):
            rows.append(dict(zip(header, cells)))
    return rows


def test_the_environment_reference_gives_each_task_its_status():
    text = (DOCS / "environments.md").read_text()
    shown = {
        row["Environment"].strip("`"): row["Status"]
        for row in _table(text)
        if "Status" in row
    }
    assert shown == {n: STATUS_ROW[charter.status(n)] for n in registry.REGISTRY}
    intro = text.split("\n## ")[0]
    assert set(_names(intro.split("scored")[1].split(".")[0])) == FOUR


def test_the_readme_shows_the_four_in_a_table_and_the_others_by_family():
    section = (ROOT / "README.md").read_text().split("\n## Environments\n")[1]
    section = section.split("\n## ")[0]
    scored, others = section.split("\n### Registered, not scored\n")
    scored = scored.split("\n### The scored suite\n")[1]
    assert [_names(row["Task"])[0] for row in _table(scored)] == list(charter.FLAGSHIP)
    families = _table(others)
    assert {row["Family"].strip("*") for row in families} <= set(
        registry.GROUPS.values()
    )
    listed = [n for row in families for n in _names(row["Tasks"])]
    assert sorted(listed) == sorted(OTHERS)


def test_the_docs_home_names_the_four():
    blocks = (DOCS / "index.md").read_text().split("\n\n")
    (block,) = [b for b in blocks if "are scored" in b]
    assert set(_names(block)) == FOUR


# --- the results page and the radar -----------------------------------------


@pytest.fixture(scope="module")
def results():
    return _script("generate_baseline_table")


@pytest.fixture
def as_if_current(monkeypatch):
    """Every record read as current. The pages publish only rows that match
    the tree (owner, 2026-11-25), so between a plant change and its re-record
    they may publish none; the scored-only rule is checked apart from that."""
    from target_gym import records

    monkeypatch.setattr(records, "current_parts", lambda row, fps: dict(row))
    monkeypatch.setattr(charter, "stale_parts", lambda record, current: [])


def test_the_results_page_has_a_row_for_scored_tasks_only(results, as_if_current):
    page = results.outputs()[results.PAGE]
    body, unscored = page.split("\n## Not scored\n")
    linked = set(re.findall(r"^\| \[`(\w+)`\]\(environments/", body, re.MULTILINE))
    plain = {row["task"] for row in _table(body) if "task" in row}
    assert linked and linked <= FOUR
    assert plain == FOUR
    assert set(_names(unscored)) == OTHERS


def test_the_readme_chart_draws_scored_tasks_only(results, as_if_current):
    svg = results.outputs()[results.IMAGES[False]]
    drawn = {n for n in registry.REGISTRY if f">{n}<" in svg}
    assert drawn and drawn <= FOUR


def test_a_task_outside_the_suite_moves_no_spoke():
    high = {"score": 0.9, "transient_ratio": 3.0, "oracle_transient_ratio": 3.0}
    tags = suite.CAPABILITIES["plane"]
    assert tags
    profile = suite.profile({"plane": high})
    for which in ("hold", "reach"):
        assert all(math.isnan(profile[which][axis]) for axis in tags)


def test_a_scored_task_moves_its_spokes():
    high = {"score": 0.9, "transient_ratio": 3.0, "oracle_transient_ratio": 3.0}
    tags = suite.CAPABILITIES["hvac"]
    assert tags
    profile = suite.profile({"hvac": high})
    for axis in tags:
        assert profile["hold"][axis] == pytest.approx(suite.radius(0.9))


def test_the_suite_runs_the_scored_tasks_by_default():
    default = inspect.signature(suite.run_suite).parameters["tasks"].default
    assert tuple(default) == charter.FLAGSHIP


def test_the_capability_radar_is_out_of_the_built_docs(results):
    """The owner, 2026-11-24: left out of this release. The page leaves the
    site and the chart is not built; the protocol page says when it returns."""
    config = (ROOT / "mkdocs.yml").read_text()
    excluded = config.split("exclude_docs:")[1].split("\nnav:")[0]
    assert "capability-profile.md" in excluded
    assert "capability-profile.md" not in config.split("\nnav:")[1]
    assert not [p for p in results.outputs() if "capability" in p]
    published = [ROOT / "README.md", *DOCS.glob("*.md"), *DOCS.glob("*/*.md")]
    linking = [
        p.name
        for p in published
        if p.name != "capability-profile.md" and "capability-profile" in p.read_text()
    ]
    assert not linking, linking
    protocol = " ".join((DOCS / "rl-protocol.md").read_text().split())
    assert "capability profile" in protocol
    assert "returns when it is rebuilt on the saving over the reference" in protocol


# --- the records ------------------------------------------------------------

RECORD_FILES = (
    "tests/test_charter_records.py",
    "tests/experts/test_mpc_baselines.py",
    "tests/test_reward_contract.py",
)
#: The checks every scored task has a case in: its charter record is present,
#: current and measured, and its protocol row describes this tree.
HELD_CURRENT = {
    "test_each_charter_task_has_a_current_valid_record",
    "test_a_scored_task_has_its_lines_measured_and_is_accepted",
    "test_recorded_protocol_row_still_describes_this_tree",
}
#: A baseline row compares the PID with the MPC, so it exists only where both
#: ship. A scored task may ship no MPC in this release (owner, 2026-11-24).
WITH_MPC = {name for name in FOUR if registry.get(name).has_mpc}


@pytest.mark.medium
def test_the_records_job_covers_the_four_and_only_them():
    """What CI's ``records`` job collects: no case for a task outside the
    suite, and one per scored task wherever a record is held current."""
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", "records"]
        + ["-n", "0", "-p", "no:randomly", *RECORD_FILES],
        capture_output=True,
        text=True,
        cwd=ROOT,
        # The records job runs unsharded, so its collection must too: a
        # ``TG_SHARD`` inherited from a sharded CI run would keep a third.
        env={
            **{k: v for k, v in os.environ.items() if k != "TG_SHARD"},
            "PYTHONPATH": str(ROOT / "src"),
        },
    )
    cases: dict[str, set[str]] = {}
    for test, task in re.findall(r"::(\w+)\[([\w-]+)\]", out.stdout):
        cases.setdefault(test, set()).add(task)
    assert HELD_CURRENT <= set(cases), out.stdout[-2000:] + out.stderr[-2000:]
    for test, tasks in cases.items():
        assert tasks <= FOUR, test
        if test in HELD_CURRENT:
            assert tasks == FOUR, test
    assert cases["test_recorded_baseline_still_describes_this_tree"] == WITH_MPC


def test_the_stale_list_names_scored_tasks_only(capsys):
    """The ``re-record`` label records what ``stale_records.py list`` names:
    with ``--all``, every held row, stale or not."""
    import json

    assert _script("stale_records").main(["list", "--all"]) in (0, None)
    jobs = json.loads(capsys.readouterr().out)
    assert {job["env"] for job in jobs} == FOUR


def test_the_charter_stale_report_covers_the_flagship_by_default():
    default = inspect.signature(charter.stale_report).parameters["tasks"].default
    assert tuple(default) == charter.FLAGSHIP


def test_a_recorder_run_without_names_records_the_scored_tasks():
    """The protocol scores every scored task's PID; the baseline returns
    compare the PID with the MPC, so they cover the scored tasks with both."""
    assert set(_script("evaluate_baselines").default_envs()) == FOUR
    assert set(_script("record_baselines").default_envs()) == WITH_MPC
