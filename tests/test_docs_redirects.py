"""Old unversioned links keep working once the site is versioned.

Contract (the owner's 0.7.0 plan, 2026-10-28: "Add redirects for old
unversioned links, such as the README chart link to
/baseline-results/#at-a-glance"): mike serves each version under its own
directory (``dev/``, ``0.7/``, ``latest/``), so a page that lived at
``/<page>/`` before 0.7.0 would 404. ``docs_redirects.stubs`` maps each page
of the target version to a page at its old path that sends the reader to
the same page of that version, keeping the anchor. It never writes over a
version's own directory or mike's root files.
"""

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "_docs_redirects", ROOT / "scripts" / "docs_redirects.py"
)
docs_redirects = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(docs_redirects)

VERSIONS = ["dev", "0.7", "latest"]


def test_a_page_redirects_to_the_same_page_of_the_target_version():
    stubs = docs_redirects.stubs(["baseline-results/index.html"], "latest", VERSIONS)
    html = stubs["baseline-results/index.html"]
    # One level deep, so the target is one level up, then into latest/.
    assert "../latest/baseline-results/" in html
    # The anchor (#at-a-glance) survives the redirect.
    assert "location.hash" in html


def test_a_nested_page_climbs_to_the_root_first():
    stubs = docs_redirects.stubs(["physics/cstr/index.html"], "latest", VERSIONS)
    assert "../../latest/physics/cstr/" in stubs["physics/cstr/index.html"]


def test_the_root_and_the_versions_are_left_to_mike():
    pages = ["index.html", "dev/index.html", "latest/api/index.html", "404.html"]
    assert docs_redirects.stubs(pages, "latest", VERSIONS) == {}
