"""The generated pages publish only records the tree still describes.

Contract, from the owner's decision of 2026-11-25 (picker, "Pages skip stale
rows"), not from the code:

- a generated page shows a recorded row only while its fingerprint or version
  stamp matches the tree as it stands;
- a stale row is not deleted: it stays in ``data/`` and shows again once it is
  re-recorded;
- the two parts of an old-format row (``records.PARTS``: the expert's and the
  oracle's) are judged apart, each by its own fingerprint; the oracle's numbers
  are read against the expert's, so a row whose expert part is stale shows
  nothing;
- a charter record's parts are judged by ``charter.stale_parts``: a stale
  part's line reads "not current" and gives no number.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re

import pytest

from target_gym import charter, provenance, records
from target_gym.registry import REGISTRY

ROOT = pathlib.Path(__file__).resolve().parents[1]
FPS = {"expert": "e1", "oracle": "o1"}


def _script(name: str, monkeypatch):
    # The page generator imports the reference generator beside it.
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        f"{name}_current_rows", ROOT / "scripts" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(expert="e1", oracle="o1") -> dict:
    return {
        "steps": 10,
        "seeds": 2,
        "expert_fingerprint": expert,
        "pid_returns": [-5038.0, -5038.0],
        "oracle_fingerprint": oracle,
        "mpc_returns": [-7.0, -7.0],
        "mpc_trips": 0,
    }


# --- old-format rows ----------------------------------------------------------


def test_a_row_matching_the_tree_is_kept_whole():
    assert records.current_parts(_row(), FPS) == _row()


def test_a_row_whose_oracle_moved_keeps_only_the_expert_part():
    kept = records.current_parts(_row(oracle="o0"), FPS)
    assert kept["pid_returns"] == [-5038.0, -5038.0]
    assert kept["steps"] == 10
    assert not set(kept) & records.PARTS["oracle"]


def test_a_row_whose_expert_moved_shows_nothing():
    assert records.current_parts(_row(expert="e0"), FPS) == {}


def test_a_task_with_no_oracle_keeps_its_expert_row():
    row = {k: v for k, v in _row().items() if k in records.PARTS["expert"]}
    assert records.current_parts(row, {"expert": "e1"}) == row


# --- charter lines --------------------------------------------------------------

WEIGHTS = {"normal": 1.0}


def _line(cost):
    return charter.line({"normal": {"costs": [cost, cost], "trips": 0}}, WEIGHTS)


def _record() -> dict:
    return {
        "weights": WEIGHTS,
        "nulls": {"lines": {"hold": _line(8.0)}},
        "reference": {"line": _line(2.0)},
        "mpc": {"line": _line(1.5)},
        "ceiling": {"line": _line(1.0)},
    }


def _costs(section: str) -> dict:
    rows = re.findall(r"^\| ([^|]+?) \| [^|]+ \| ([^|]+?) \|", section, re.MULTILINE)
    return dict(rows[1:])


def test_a_stale_charter_line_reads_not_current_and_gives_no_number(monkeypatch):
    """The reference and the ceiling are stale: their lines give no cost, and
    no room is computed from them (the room needs both, so it is not shown)."""
    gen = _script("generate_env_pages", monkeypatch)
    section = gen.charter_section(_record(), stale=("reference", "ceiling"))
    assert _costs(section) == {
        "Best null policy": "8",
        "Reference": "not current",
        "MPC": "1.5",
        "Ceiling": "not current",
    }
    assert "is 2." not in section
    assert "re-recorded" in section


def test_a_current_charter_record_shows_every_line(monkeypatch):
    gen = _script("generate_env_pages", monkeypatch)
    section = gen.charter_section(_record(), stale=())
    assert _costs(section) == {
        "Best null policy": "8",
        "Reference": "2",
        "MPC": "1.5",
        "Ceiling": "1",
    }
    assert "is 2." in section


# --- task pages -----------------------------------------------------------------


@pytest.mark.parametrize("current", [True, False])
def test_a_task_page_shows_its_baseline_row_only_while_it_is_current(
    monkeypatch, current
):
    """cstr's page published v7's "PID -5038" under v8: a row recorded on
    another tree is not shown, one recorded on this tree is."""
    gen = _script("generate_env_pages", monkeypatch)
    spec = REGISTRY["cstr"]
    fps = provenance.fingerprints(spec)
    row = _row(fps["expert"] if current else "an earlier tree", fps.get("oracle"))
    row["steps"] = int(spec.make_test_params().max_steps_in_episode)
    text, _ = gen.page("cstr", spec, {"cstr": row})
    baselines = text.split("\n## Baselines\n")[1].split("\n## ")[0]
    assert ("-5038" in baselines) == current
    if not current:
        assert "re-recorded" in baselines


@pytest.mark.parametrize("name", charter.FLAGSHIP)
def test_a_scored_page_gives_no_cost_for_a_stale_charter_line(name):
    """On the committed pages: each J3 line whose record part is stale reads
    "not current"."""
    part = {"Best null policy": "nulls", "Reference": "reference"}
    part |= {"MPC": "mpc", "Ceiling": "ceiling"}
    stale = set(charter.stale_parts(charter.load(name), charter.current(name)))
    text = (ROOT / "docs" / "environments" / f"{name}.md").read_text()
    section = text.split("\n## Reference and ceiling\n")[1].split("\n## ")[0]
    costs = _costs(section)
    for label, p in part.items():
        if p in stale:
            assert costs[label] == "not current", (name, label)


# --- results page ---------------------------------------------------------------


@pytest.fixture(scope="module")
def results_page():
    mp = pytest.MonkeyPatch()
    try:
        yield _script("generate_baseline_table", mp)._page()
    finally:
        mp.undo()


def _fingerprints(name: str) -> dict:
    return provenance.fingerprints(REGISTRY[name])


def test_the_results_chart_embeds_only_current_rows(results_page):
    block = re.search(
        r'<script type="application/json" id="tg-results-data">(.*?)</script>',
        results_page,
        re.S,
    )
    recorded = records.load("protocol_results")
    for row in json.loads(block.group(1)):
        fps, rec = _fingerprints(row["name"]), recorded[row["name"]]
        assert rec["expert_fingerprint"] == fps["expert"], row["name"]
        if row["oracle"] is not None:
            assert rec["oracle_fingerprint"] == fps.get("oracle"), row["name"]


def _section(page: str, heading: str) -> str:
    """The section under ``heading``; "" when the page leaves it out, as it
    does a table with no current row."""
    if f"\n## {heading}\n" not in page:
        return ""
    return page.split(f"\n## {heading}\n")[1].split("\n## ")[0]


def _tasks(section: str) -> list[str]:
    """The tasks a results table has rows for: each row opens on its link."""
    return re.findall(r"^\| \[`(\w+)`\]", section, re.M)


@pytest.mark.parametrize("heading", ["Room to grow", "Whole episode"])
def test_a_protocol_table_lists_only_current_rows(results_page, heading):
    recorded = records.load("protocol_results")
    for name in _tasks(_section(results_page, heading)):
        fp = _fingerprints(name)["expert"]
        assert recorded[name]["expert_fingerprint"] == fp, name


def test_the_returns_table_lists_only_current_rows(results_page):
    recorded = records.load("baseline_returns")
    for name in _tasks(_section(results_page, "Episode returns")):
        kept = records.current_parts(recorded[name], _fingerprints(name))
        assert kept == recorded[name], name


def test_the_charter_table_gives_no_cost_for_a_stale_part(results_page):
    """A record with no current part has no row; a stale part reads "not
    current"."""
    if "\n## Charter lines\n" not in results_page:
        return
    section = _section(results_page, "Charter lines")
    columns = ("nulls", "reference", "mpc", "ceiling")
    for cells in re.findall(r"^\| (\w+) \| (.+) \|$", section, re.M):
        name, values = cells[0], [c.strip() for c in cells[1].split("|")]
        if name not in REGISTRY:
            continue
        stale = charter.stale_parts(charter.load(name), charter.current(name))
        assert set(stale) != set(charter.PARTS), name
        for part, value in zip(columns, values):
            if part in stale:
                assert value == "not current", (name, part)
