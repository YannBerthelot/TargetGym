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
import math
import pathlib
import posixpath
import re

import pytest

from target_gym import records, registry

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
    data = records.load("protocol_results")
    rows = {r["name"]: r for r in gen.chart_rows(data)}
    assert list(rows) == list(registry.REGISTRY)
    for name, recorded in data.items():
        assert rows[name]["expert"] == recorded["pid"]["hold_tracking"]
        assert rows[name]["oracle"] == recorded["mpc"]["hold_tracking"]
        assert rows[name]["ceiling"] == recorded["ceiling"]


def test_the_page_embeds_the_same_rows(gen):
    """The interactive chart reads its data from the page, so the page must
    carry exactly the rows the chart is built from."""
    data = records.load("protocol_results")
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
    data = records.load("protocol_results")
    svg = gen.results_svg(gen.chart_rows(data), dark=dark)
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert "<script" not in svg
    assert re.search(r"<rect[^>]*class=\"bg\"", svg)


def test_the_capability_image_draws_the_recorded_profiles(gen):
    """Worked by hand: on plane (disturbance) the expert holds at 100 and the
    oracle at 10 against a ceiling of 1, half the two decades, so the oracle
    sits at radius 0.75 on Hold and the expert on its 0.5 ring; the oracle's
    moves cost a quarter of the expert's, so it is the best known (radius 1)
    on Reach."""
    row = _row("plane", expert=100.0, oracle=10.0, ceiling=1.0)
    row["pid"]["move_cost"], row["mpc"]["move_cost"] = 400.0, 100.0
    svg = gen.capability_svg({"plane": row})
    points = {
        (m["radar"], m["who"], m["axis"]): m["r"]
        for m in re.finditer(
            r'data-radar="(?P<radar>\w+)" data-who="(?P<who>\w+)" '
            r'data-axis="(?P<axis>\w+)" data-r="(?P<r>[\d.]+)"',
            svg,
        )
    }
    assert points == {
        ("hold", "expert", "disturbance"): "0.50",
        ("hold", "oracle", "disturbance"): "0.75",
        ("reach", "expert", "disturbance"): "0.50",
        ("reach", "oracle", "disturbance"): "1.00",
    }


@pytest.mark.parametrize("dark", [False, True])
def test_the_capability_image_is_standalone_svg(gen, dark):
    data = records.load("protocol_results")
    svg = gen.capability_svg(data, dark=dark)
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert "<script" not in svg
    assert re.search(r"<rect[^>]*class=\"bg\"", svg)


@pytest.mark.parametrize("direction", [math.inf, -math.inf])
def test_the_capability_image_is_the_same_on_every_platform(
    gen, monkeypatch, direction
):
    """The docs check regenerates the image on Linux and compares it with the
    one committed from a Mac, and their maths libraries can differ in the last
    bit of sin, cos and log. Nudging every such result one unit in the last
    place, either way, must leave the image unchanged."""
    data = records.load("protocol_results")
    drawn = gen.capability_svg(data)
    for name in ("sin", "cos", "log"):
        real = getattr(math, name)
        monkeypatch.setattr(
            math, name, lambda x, f=real: math.nextafter(f(x), direction)
        )
    assert gen.capability_svg(data) == drawn


SITE = "https://yannberthelot.github.io/TargetGym/"


def test_the_readme_chart_is_one_the_docs_build_publishes(gen):
    """The chart images are built with the site, not committed, and PyPI
    shows the README without the repository's files: so the README links
    each image on the published site, each is a file the docs build writes,
    and its link to the interactive version lands on the page's heading."""
    readme = (ROOT / "README.md").read_text()
    published = gen.outputs()
    images = re.findall(r'(?:src|srcset)="([^"]*results-vs-expert[^"]*)"', readme)
    assert len(images) == 2
    for url in images:
        assert url.startswith(SITE) and url[len(SITE) :] in published, url
    assert f"{SITE}baseline-results/#at-a-glance" in readme
    assert "\n## At a glance\n" in published[gen.PAGE]


def test_every_image_a_page_embeds_is_one_the_site_publishes(gen):
    """MkDocs serves ``docs/<page>.md`` at ``<page>/`` (``index.md`` at its
    directory) and leaves raw HTML alone, so the browser resolves a relative
    ``src`` or ``srcset`` from that directory. Each must land on a file the
    site publishes: one under ``docs/``, one the build generates, or a
    task's clip, which the deploy renders at ``videos/<task>/pid_output.gif``
    (``docs-deploy.yml``; the clips are not committed)."""
    docs = ROOT / "docs"
    clips = {f"videos/{name}/pid_output.gif" for name in registry.REGISTRY}
    published = set(gen.outputs()) | clips
    missing = []
    for page in sorted(docs.rglob("*.md")):
        rel = page.relative_to(docs)
        served = rel.parent if rel.stem == "index" else rel.with_suffix("")
        for url in re.findall(r'(?:src|srcset)="([^"]+)"', page.read_text()):
            url = url.split()[0]
            if url.startswith(("http://", "https://", "/", "data:")):
                continue
            target = posixpath.normpath(posixpath.join(served.as_posix(), url))
            if not ((docs / target).is_file() or target in published):
                missing.append(f"{rel}: {url}")
    assert not missing, missing
