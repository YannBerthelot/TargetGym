"""MkDocs build hooks.

Two rewrites, both done before MkDocs resolves links.

**Includes.** The per-environment physics contracts live next to the code they
describe, in ``src/target_gym/<env>/PHYSICS.md``, and the contributing guide
lives at the repository root. The stub pages that surface them on the site name
the file with a ``--8<--`` line, which this hook expands. ``pymdownx.snippets``
would also do that, but during markdown *conversion* -- after
``on_page_markdown`` -- so a hook wanting to touch the included text would see
only the directive.

**Links.** Those documents, and the pages that reference them, link by
repository path (``../src/target_gym/plane/PHYSICS.md``, ``../CONTRIBUTING.md``).
Those are the correct links for someone reading the Markdown on GitHub, and
meaningless once the files are flattened onto the site. Rewriting them here
means the sources stay right for both readers, rather than being made wrong in
the repository to suit the site.
"""

import pathlib
import re

_INCLUDE = re.compile(r'^--8<--\s+"([^"]+)"\s*$', re.MULTILINE)
# ``../plane3d/PHYSICS.md``, ``../src/target_gym/pc_gym/cstr/PHYSICS.md``, ...
# The environment's directory name is the physics page's slug.
_PHYSICS_LINK = re.compile(r"\((?:\.\./)+(?:[\w/]+/)?(\w+)/PHYSICS\.md(#[^)]*)?\)")
_CONTRIBUTING_LINK = re.compile(r"\((?:\.\./)+CONTRIBUTING\.md(#[^)]*)?\)")


def on_page_markdown(markdown, page, config, files):
    root = pathlib.Path(config.config_file_path).parent

    def _inline(match):
        target = root / match.group(1)
        if not target.is_file():
            raise FileNotFoundError(
                f"{page.file.src_uri}: included file not found: {match.group(1)}"
            )
        return target.read_text()

    markdown = _INCLUDE.sub(_inline, markdown)

    # Physics pages sit beside each other; every other page reaches them
    # through the physics/ directory.
    in_physics = page.file.src_uri.startswith("physics/")
    prefix = "" if in_physics else "physics/"
    up = "../" if in_physics else ""
    markdown = _PHYSICS_LINK.sub(
        lambda m: f"({prefix}{m.group(1)}.md{m.group(2) or ''})", markdown
    )
    return _CONTRIBUTING_LINK.sub(
        lambda m: f"({up}contributing.md{m.group(1) or ''})", markdown
    )
