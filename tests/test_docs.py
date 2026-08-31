"""The documentation is checked, not merely written.

Two failure modes this guards against, both of which happen quietly:

* ``docs/environments.md`` states shapes and baselines for eighteen
  environments. It is generated from the registry, and drifts the moment
  someone adds an environment or a baseline without regenerating it.
* Code in the docs rots. Every runnable example here is executed, so a renamed
  argument or a changed return arity fails the suite rather than greeting a
  new user.

A fenced block is executed unless its first line is ``# doc: skip`` -- used for
snippets that write files, install packages, or need a GPU.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
GENERATOR = ROOT / "scripts" / "generate_env_reference.py"

BLOCK = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)


def _runnable_blocks(path: pathlib.Path) -> list[str]:
    return [
        code
        for code in BLOCK.findall(path.read_text())
        if not code.lstrip().startswith("# doc: skip")
    ]


def _doc_cases() -> list[tuple[str, str]]:
    cases = []
    for md in sorted(DOCS.glob("*.md")):
        for i, code in enumerate(_runnable_blocks(md)):
            cases.append(pytest.param(code, id=f"{md.name}:{i}"))
    return cases


def test_environment_reference_is_in_sync_with_the_registry():
    """docs/environments.md must match what the generator produces."""
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, (
        f"{result.stdout}{result.stderr}\n"
        "Run: python scripts/generate_env_reference.py"
    )


@pytest.mark.parametrize("code", _doc_cases())
def test_documented_example_runs(code, tmp_path, monkeypatch):
    """Every runnable example in docs/ executes without raising."""
    monkeypatch.chdir(tmp_path)
    exec(compile(code, "<doc>", "exec"), {"__name__": "__doc_example__"})


def test_every_environment_has_a_physics_contract():
    """The README claims every environment carries a PHYSICS.md. Keep it true.

    The claim is made in four places and is the project's main quality
    argument, so it is asserted rather than trusted -- a new environment
    without a contract fails here instead of quietly weakening the claim.
    """
    from target_gym.registry import REGISTRY

    missing = []
    for name, spec in REGISTRY.items():
        module = type(spec.make_env()).__module__
        for suffix in (".env_jax", ".marl"):
            module = module.removesuffix(suffix)
        if not (ROOT / "src" / module.replace(".", "/") / "PHYSICS.md").exists():
            missing.append(name)
    assert not missing, f"environments without a PHYSICS.md: {missing}"
