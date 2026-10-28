"""The results chart: each task's oracle and ceiling against its expert.

Contract (from the user's request, 2026-10-27): the docs and the README show,
for every task, how many times cheaper the oracle holds than the expert and
where the instrument-resolution ceiling sits, on a log scale with the expert
at 0. The numbers are the recorded results, never transcribed by hand
(scripts/generate_baseline_table.py's docstring), so the page and the images
are regenerated from ``protocol_results.json``.
"""

import importlib.util
import json
import pathlib
import re

import pytest

from target_gym import registry

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def gen():
    path = ROOT / "scripts" / "generate_baseline_table.py"
    spec = importlib.util.spec_from_file_location("_generate_baseline_table", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(name, expert, oracle, ceiling):
    return {
        "ceiling": ceiling,
        "pid": {"hold_tracking": expert, "gain": expert},
        "mpc": {"hold_tracking": oracle, "gain": oracle},
    }


def test_the_chart_shows_the_recorded_results(gen):
    """Every registered task, with its recorded expert hold, oracle hold and
    ceiling, as protocol_results.json has them."""
    data = json.loads((ROOT / "src/target_gym/data/protocol_results.json").read_text())
    rows = {r["name"]: r for r in gen.chart_rows(data)}
    assert list(rows) == list(registry.REGISTRY)
    for name, recorded in data.items():
        assert rows[name]["expert"] == recorded["pid"]["hold_tracking"]
        assert rows[name]["oracle"] == recorded["mpc"]["hold_tracking"]
        assert rows[name]["ceiling"] == recorded["ceiling"]


def test_the_page_embeds_the_same_rows(gen):
    """The interactive chart reads its data from the page, so the page must
    carry exactly the rows the chart is built from."""
    data = json.loads((ROOT / "src/target_gym/data/protocol_results.json").read_text())
    block = re.search(
        r'<script type="application/json" id="tg-results-data">(.*?)</script>',
        gen._page(),
        re.S,
    )
    assert json.loads(block.group(1)) == gen.chart_rows(data)


def test_the_image_places_each_task_against_its_expert(gen):
    """Worked by hand: an oracle 100x cheaper than its expert is -2 decades
    and a ceiling 1000x below it is -3; a ceiling twice the expert's cost is
    log10(2) = +0.30; an oracle at exactly zero cost cannot sit on a log axis
    and is drawn past its left edge."""
    data = {
        "plane": _row("plane", expert=100.0, oracle=1.0, ceiling=0.1),
        "cstr": _row("cstr", expert=1.0, oracle=0.0, ceiling=2.0),
    }
    svg = gen.results_svg(gen.chart_rows(data))
    marks = {
        m.group(1): m.group(0)
        for m in re.finditer(r'<g data-task="([a-z0-9_]+)"[^>]*>', svg)
    }
    assert set(marks) == {"plane", "cstr"}
    assert 'data-oracle="-2.00"' in marks["plane"]
    assert 'data-ceiling="-3.00"' in marks["plane"]
    assert 'data-ceiling="0.30"' in marks["cstr"]
    assert 'data-oracle="off-axis"' in marks["cstr"]


@pytest.mark.parametrize("dark", [False, True])
def test_both_images_are_standalone_svg(gen, dark):
    """GitHub shows the README images as plain images: no script, and a
    background of their own so the light and dark variants read on either
    theme."""
    data = json.loads((ROOT / "src/target_gym/data/protocol_results.json").read_text())
    svg = gen.results_svg(gen.chart_rows(data), dark=dark)
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert "<script" not in svg
    assert re.search(r"<rect[^>]*class=\"bg\"", svg)
