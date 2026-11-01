"""Redirect pages for the unversioned URLs the site had before mike.

Until 0.7.0 the site was one version, at ``/TargetGym/<page>/``. With mike
each version lives in its own directory (``dev/``, ``X.Y/``, ``latest/``) and
the root only redirects to the default version, so every deeper old link
(``/TargetGym/baseline-results/#at-a-glance`` among them) would 404.

The docs deploy workflow runs this on the site it publishes: for every page
of the target version it writes a page at the old path that sends the reader
to the same page of that version, anchor included. Nothing here is committed
to ``gh-pages``; the stubs live only in the published artifact.

    python scripts/docs_redirects.py SITE_DIR
"""

from __future__ import annotations

import json
import pathlib
import sys

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Redirecting</title>
<link rel="canonical" href="{target}">
<meta http-equiv="refresh" content="0; url={target}">
<script>location.replace("{target}" + location.search + location.hash);</script>
</head>
<body><a href="{target}">This page moved to {target}</a></body>
</html>
"""


def stubs(pages: list[str], target: str, versions: list[str]) -> dict[str, str]:
    """The redirect page to write at each old path, keyed by that path.

    ``pages`` are the target version's pages relative to its directory
    (``baseline-results/index.html``). The root page, other files and any
    path inside a version's own directory are left alone.
    """
    out = {}
    for page in pages:
        parts = page.split("/")
        if len(parts) < 2 or parts[-1] != "index.html" or parts[0] in versions:
            continue
        depth = len(parts) - 1
        url = "../" * depth + target + "/" + "/".join(parts[:-1]) + "/"
        out[page] = TEMPLATE.format(target=url)
    return out


def main(site: pathlib.Path) -> None:
    entries = json.loads((site / "versions.json").read_text())
    versions = [e["version"] for e in entries]
    versions += [alias for e in entries for alias in e.get("aliases", [])]
    target = "latest" if (site / "latest").is_dir() else entries[0]["version"]
    pages = [
        p.relative_to(site / target).as_posix()
        for p in sorted((site / target).rglob("index.html"))
    ]
    written = 0
    for path, html in stubs(pages, target, versions).items():
        dest = site / path
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(html)
        written += 1
    print(f"{written} redirect pages to {target}/")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
