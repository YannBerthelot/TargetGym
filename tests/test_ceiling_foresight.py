"""Each scored task names, in plain words, the hidden facts its ceiling reads.

Contract, from the owner's decision of 2026-11-25 (picker, "Report foresight,
name it plainly"), not from the code:

- the foresight ceiling stays the room's top;
- every scored task's page, in its reference and ceiling section, states which
  hidden facts the ceiling reads, so readers know part of the room may be out
  of reach of a policy that reads only the observations;
- the charter's room words say the same;
- the decision's own examples: the compressor's true surge line, the pH plant's
  hidden acid strength and buffer flow, the weather ahead (hvac) and the load
  ahead (reactor); the cstr's page already says its policy never reads the
  composition.
"""

from __future__ import annotations

import pathlib

import pytest

from target_gym import charter

ROOT = pathlib.Path(__file__).resolve().parents[1]

NAMED = {
    "compressor_surge": ("surge line",),
    "ph_neutralization": ("acid strength", "buffer flow"),
    "hvac": ("outdoor temperature", "ahead"),
    "reactor": ("load", "ahead"),
    "cstr": ("composition",),
}


def _section(name: str) -> str:
    text = (ROOT / "docs" / "environments" / f"{name}.md").read_text()
    return text.split("\n## Reference and ceiling\n")[1].split("\n## ")[0]


def test_every_scored_task_lists_what_its_ceiling_reads():
    assert set(charter.CEILING_READS) == set(charter.FLAGSHIP)
    assert all(charter.CEILING_READS[t] for t in charter.FLAGSHIP)


@pytest.mark.parametrize("name", charter.FLAGSHIP)
def test_a_scored_page_names_the_ceilings_hidden_facts(name):
    section = _section(name)
    for fact in charter.CEILING_READS[name]:
        assert fact in section, (name, fact)
    for words in NAMED[name]:
        assert words in section, (name, words)
    assert "may be out of reach" in section, name


def test_the_charter_says_the_room_may_be_out_of_reach():
    text = (ROOT / "docs" / "proposals" / "task-charter.md").read_text()
    words = text.split("- **Room:**")[1].split("\n- **")[0]
    assert "hidden" in words and "out of reach" in words
