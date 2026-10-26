"""Find documentation that no longer describes the code.

    uv run python scripts/check_doc_drift.py            # report
    uv run python scripts/check_doc_drift.py --check    # CI: exit 1 on findings

This suite's contracts are what a reader consults *instead of* the code, so a
stale one does not merely age, it actively misinforms. Reviewing all twenty-one
environments found this to be the most common defect in the repository, and
three of that review's own findings were wrong because of it: a deviation
claiming post-stall lift decays to zero when the fix had long been implemented,
a docstring saying sub-step rewards are summed when the code takes their mean,
and a deviation crediting a reward change to a band the reward does not read.

Four checks, all mechanical. None of them needs judgement, which is the point:
the judgement calls are the ones that rot.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "target_gym"

#: Comments where a second ``#`` is deliberate rather than a mangled merge.
DOUBLE_HASH_ALLOW = ("# noqa", "# type:", "# pragma", "# fmt:")

#: ``#rrggbb`` and ``#rgb`` are colours, not the start of a second comment.
HEX_COLOUR = re.compile(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})\b")

#: Backticked identifiers in a PHYSICS.md that are prose or belong to another
#: package, and so are not expected to resolve locally.
SYMBOL_ALLOW = {
    "compute_reward",  # every contract names it; defined per package
    "step_env",
    "get_obs",
    "reset_env",
    "check_is_terminal",
    "target_gym",
    "log_scaled_reward",
    "precision_floor",
    "max_steps_in_episode",
    "delta_t",
}


def _registry_names() -> set[str]:
    """Environment names are identifiers in prose, not symbols in a package."""
    text = (SRC / "registry.py").read_text()
    return set(re.findall(r'name="([a-z0-9_]+)"', text))


IDENT = re.compile(r"`([a-z][a-z0-9]*(?:_[a-z0-9]+)+|[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)`")


def _packages() -> list[pathlib.Path]:
    return sorted({p.parent for p in SRC.rglob("PHYSICS.md")})


#: Words that mark a mention as a deliberate reference to something gone.
HISTORICAL = (
    "used to",
    "no longer",
    "removed",
    "absorbed",
    "predates",
    "left over",
    "since",
    "formerly",
    "was ",
)


def _is_historical(text: str, name: str) -> bool:
    """True if every mention of *name* sits in a sentence about the past."""
    import re as _re

    for m in _re.finditer(_re.escape(f"`{name}`"), text):
        window = text[max(0, m.start() - 300) : m.end() + 150].lower()
        if not any(h in window for h in HISTORICAL):
            return False
    return True


def check_symbols() -> list[str]:
    """A contract naming a symbol its own package no longer defines."""
    out = []
    for pkg in _packages():
        doc = pkg / "PHYSICS.md"
        source = "\n".join(p.read_text() for p in pkg.rglob("*.py"))
        # A contract may legitimately name a test that asserts it, a field of
        # the recorded baselines, or an integrator, none of which live in the
        # package itself.
        for other in (ROOT / "tests", ROOT / "scripts", SRC / "integration.py"):
            source += "\n" + (
                other.read_text()
                if other.is_file()
                else "\n".join(q.read_text() for q in other.rglob("*.py"))
            )
        # Shared physics lives one level up for the aircraft, and the
        # formation's slot geometry in the patrol; allow both.
        for extra in (SRC / "plane", SRC / "plane3d", SRC / "patrol", SRC / "utils.py"):
            if extra.exists():
                source += "\n" + (
                    extra.read_text()
                    if extra.is_file()
                    else "\n".join(p.read_text() for p in extra.rglob("*.py"))
                )
        for name in sorted(set(IDENT.findall(doc.read_text()))):
            if name in SYMBOL_ALLOW or name in _registry_names():
                continue
            if re.search(rf"\b{re.escape(name)}\b", source):
                continue
            # Prose that says a thing *used to* exist is doing its job, so look
            # at the sentence the name sits in rather than only at whether the
            # symbol resolves.
            if _is_historical(doc.read_text(), name):
                continue
            out.append(
                f"{doc.relative_to(ROOT)}: names `{name}`, which no longer exists"
            )
    return out


def check_reward_claims() -> list[str]:
    """A parameter whose comment claims the reward uses it, when it does not."""
    out = []
    for path in sorted(SRC.rglob("env.py")):
        text = path.read_text()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        rewards = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and "reward" in n.name
        ]
        if not rewards:
            continue
        body = "\n".join(ast.get_source_segment(text, fn) or "" for fn in rewards)
        lines = text.splitlines()
        for n in ast.walk(tree):
            if not (isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)):
                continue
            name = n.target.id
            # the declaration's own line plus any ``#:`` block above it
            k = n.lineno - 1
            comment = lines[k]
            j = k - 1
            while j >= 0 and lines[j].lstrip().startswith(("#:", "#")):
                comment = lines[j] + "\n" + comment
                j -= 1
            # A claim about what the reward *does with this parameter*, not a
            # passing mention of the reward. Without this the check fires on
            # every comment that explains itself by reference to the reward,
            # which is most of the good ones.
            claims = (
                "tracking reward reaches",
                "reward reaches",
                "reward halves",
                "band for the reward",
                "reward uses",
                "reward is scaled by",
                "tracking band",
                "slot reward",
            )
            low = comment.lower()
            if not any(c in low for c in claims):
                continue
            if re.search(rf"\b{re.escape(name)}\b", body):
                continue
            if any(
                d in low for d in ("not read by", "not a reward", "controller constant")
            ):
                continue  # the comment already says it is not the reward's
            out.append(
                f"{path.relative_to(ROOT)}:{n.lineno}: `{name}` is described in terms of "
                "the reward, but compute_reward does not read it"
            )
    return out


def check_episode_comments() -> list[str]:
    """A comment above ``max_steps_in_episode`` quoting a different number."""
    out = []
    for path in (ROOT / "src/target_gym/registry.py",):
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            m = re.search(r'"max_steps_in_episode":\s*(\d+)', line)
            if not m:
                continue
            value = int(m.group(1))
            block, j = [], i - 1
            while j >= 0 and lines[j].lstrip().startswith("#"):
                block.append(lines[j])
                j -= 1
            joined = "\n".join(block)
            # A comment that deliberately cites what a value *used to* be is
            # doing its job, not drifting.
            if any(
                h in joined.lower()
                for h in (
                    "used to",
                    "this comment read",
                    "was ",
                    "before the",
                    "raised it",
                )
            ):
                continue
            quoted = re.findall(r"(\d[\d_]*)\s*steps", joined)
            for q in quoted:
                if int(q.replace("_", "")) != value:
                    out.append(
                        f"{path.relative_to(ROOT)}:{i + 1}: comment says {q} steps, "
                        f"the value is {value}"
                    )
                    break
    return out


def check_double_hash() -> list[str]:
    """A comment containing a second ``#`` -- the signature of a merged comment."""
    out = []
    for path in sorted(SRC.rglob("*.py")):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if "#" not in line:
                continue
            comment = line[line.index("#") :]
            if any(a in comment for a in DOUBLE_HASH_ALLOW):
                continue
            if stripped.startswith("#"):
                continue  # a whole-line comment may legitimately contain '#'
            if HEX_COLOUR.sub("", comment).count("#") > 1:
                out.append(
                    f"{path.relative_to(ROOT)}:{i}: two '#' in one comment: {comment.strip()}"
                )
    return out


CHECKS = (
    ("symbols named in a contract that no longer exist", check_symbols),
    ("parameters described as the reward's that it does not read", check_reward_claims),
    ("episode-length comments disagreeing with the value", check_episode_comments),
    ("comments that look merged", check_double_hash),
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="exit 1 if anything is found")
    args = ap.parse_args()

    total = 0
    for title, fn in CHECKS:
        found = fn()
        total += len(found)
        print(f"\n{title}: {len(found)}")
        for line in found:
            print(f"  {line}")

    print(f"\n{total} finding(s)")
    return 1 if (args.check and total) else 0


if __name__ == "__main__":
    sys.exit(main())
