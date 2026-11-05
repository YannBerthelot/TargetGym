"""A record's two parts, the expert's rows and the oracle's, each have their own
fingerprint (``provenance.expert_fingerprint``, ``provenance.oracle_fingerprint``).

Contract (the build plan's PR A, 2026-11-03): retuning an expert re-records
only expert rows, and rebuilding an oracle only oracle rows. Both cover the
environment's own non-controller modules, the shared physics, the parameter
values, the scored burn-in and the fixed settle. On top, the expert part
covers experts/pid.py, the task's expert modules and its gains entries; the
oracle part experts/mpc.py, the task's oracle modules, its noise fields, and
any expert code or gains the oracle still reads. An oracle on the shared
solver (``oracle_data``, the build plan's PR C1) has the solver's modules in
place of experts/mpc.py, and the values of its horizon and widths files. No
controller edit moves the environment's version stamp.
"""

from __future__ import annotations

import ast
import json
import pathlib

import pytest

from target_gym import provenance, records
from target_gym.registry import REGISTRY

SRC = provenance._ROOT
FINGERPRINTS = {
    "expert": provenance.expert_fingerprint,
    "oracle": provenance.oracle_fingerprint,
    "stamp": provenance.environment_fingerprint,
}


def _moved(spec, change) -> set[str]:
    """Which fingerprints of ``spec`` move when ``change()`` is applied."""
    before = {k: f(spec) for k, f in FINGERPRINTS.items()}
    change()
    return {k for k, f in FINGERPRINTS.items() if f(spec) != before[k]}


@pytest.mark.parametrize(
    "name,rel,parts",
    [
        ("cstr", "pc_gym/cstr/oracle.py", {"oracle"}),
        # cstr's oracle is on the shared solver, the reactor's not yet.
        ("cstr", "experts/mpc.py", set()),
        ("reactor", "experts/mpc.py", {"oracle"}),
        ("cstr", "oracles/shooting.py", {"oracle"}),
        ("reactor", "oracles/shooting.py", set()),
        ("cstr", "pc_gym/cstr/experts.py", {"expert"}),
        ("cstr", "experts/pid.py", {"expert"}),
        ("cstr", "pc_gym/cstr/env.py", {"expert", "oracle", "stamp"}),
        # Its oracle no longer falls back to the pair.
        ("compressor_surge", "compressor_surge/experts.py", {"expert"}),
        # An oracle on the shared solver reads no expert code.
        ("wind_turbine", "energy/wind_turbine/experts.py", {"expert"}),
        # Oracles that still read expert code: their part moves with it.
        ("plane", "experts/pid.py", {"expert", "oracle"}),
        ("formation", "patrol/oracle.py", {"oracle"}),
    ],
)
def test_an_edit_stales_only_the_parts_that_read_the_file(
    monkeypatch, name, rel, parts
):
    target, read = SRC / rel, pathlib.Path.read_text

    def edited(path, *args, **kwargs):
        text = read(path, *args, **kwargs)
        return text + "\n_edited = 1\n" if path == target else text

    change = lambda: monkeypatch.setattr(
        pathlib.Path, "read_text", edited
    )  # noqa: E731
    assert _moved(REGISTRY[name], change) == parts


@pytest.mark.parametrize(
    "name,key,parts",
    [
        ("cstr", "cstr", {"expert"}),
        ("unstable_cstr", "unstable_cstr", {"expert"}),
        ("wind_turbine", "wind_turbine", {"expert"}),
        # The formation oracle's lead flies the 3D heading gains; its expert
        # does not read them.
        ("formation", "plane3d_heading", {"oracle"}),
        # Patrol's scripted lead flies task constants since PR A.
        ("patrol", "plane3d_heading", set()),
    ],
)
def test_a_retune_stales_only_the_parts_that_read_the_key(
    tmp_path, monkeypatch, name, key, parts
):
    gains = json.loads(provenance.GAINS_PATH.read_text())
    path = tmp_path / "pid_gains.json"
    path.write_text(json.dumps(gains))
    monkeypatch.setattr(provenance, "GAINS_PATH", path)
    retuned = {**gains, key: {**gains[key], "retuned": 1.0}}
    assert _moved(REGISTRY[name], lambda: path.write_text(json.dumps(retuned))) == parts


@pytest.mark.parametrize(
    "rel", ["oracle_horizons/cstr.json", "oracle_widths/cstr.json"]
)
def test_an_oracles_data_stales_its_part_by_value_only(monkeypatch, rel):
    """A new value in the horizon or widths file moves the oracle part; a
    new run stamp (when, on which CPU) moves nothing."""
    target, read = SRC / "data" / rel, pathlib.Path.read_text
    row = json.loads(read(target))

    def write(change):
        text = json.dumps({**row, **change})
        monkeypatch.setattr(
            pathlib.Path,
            "read_text",
            lambda path, *a, **k: text if path == target else read(path, *a, **k),
        )

    spec = REGISTRY["cstr"]
    assert _moved(spec, lambda: write({"generated": "later", "cpu": "other"})) == set()
    assert _moved(spec, lambda: write({"P": 1, "sites": {"new": 1}})) == {"oracle"}


def _module_file(module: str) -> pathlib.Path | None:
    if not module.startswith("target_gym."):
        return None
    rel = pathlib.Path(*module.split(".")[1:])
    for path in (SRC / rel.with_suffix(".py"), SRC / rel / "__init__.py"):
        if path.is_file():
            return path
    return None


def _imported_files(path: pathlib.Path) -> set[pathlib.Path]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        else:
            continue
        found |= {f for f in map(_module_file, names) if f is not None}
    return found


def _expert_side(path: pathlib.Path) -> bool:
    return path == SRC / "experts" / "pid.py" or path.name.startswith("experts")


@pytest.mark.parametrize("name", [n for n, s in REGISTRY.items() if s.has_mpc])
def test_the_oracle_part_covers_the_expert_code_its_oracle_imports(name):
    """Followed through every controller module the oracle's own module
    imports. ``oracle_reads`` lists exactly those files, so the later PRs that
    cut an oracle loose from its expert drop the entry too."""
    from tests.test_registry_rules import _factory

    spec = REGISTRY[name]
    todo, seen, reached = [_module_file(_factory(spec.make_mpc)[1])], set(), set()
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        for dep in _imported_files(path):
            if _expert_side(dep):
                reached.add(dep)
            if _expert_side(dep) or dep.name.startswith(("oracle", "experts")):
                todo.append(dep)
    assert reached <= set(provenance.oracle_sources(spec)), name
    assert {SRC / rel for rel in spec.oracle_reads} == reached, name


def test_migration_moves_each_row_to_both_fingerprints_and_keeps_the_rest(
    tmp_path, monkeypatch
):
    """The split re-records nothing: every row recorded under the single
    fingerprint gets the tree's two, and every other key stays as it was."""
    from tests.test_stale_records import _load

    monkeypatch.setattr(records, "DATA", tmp_path)
    old = {
        "fingerprint": "f1",
        "pid_returns": [1.0],
        "mpc_returns": [2.0],
        "cpu": "x",
    }
    records.write("baseline_returns", "cstr", old)
    records.write("protocol_results", "cstr", {"fingerprint": "f1", "pid": {}})
    assert _load().main(["migrate"]) == 0
    spec = REGISTRY["cstr"]
    want = {
        "expert_fingerprint": provenance.expert_fingerprint(spec),
        "oracle_fingerprint": provenance.oracle_fingerprint(spec),
    }
    rest = {k: v for k, v in old.items() if k != "fingerprint"}
    assert records.load_row("baseline_returns", "cstr") == {**rest, **want}
    assert records.load_row("protocol_results", "cstr") == {"pid": {}, **want}
