"""Registry rules: the 21 tasks registered before the new task families are
pinned, and every task added after them follows the rules that keep its
records honest.

Pinned: every field of the 21 specs (``tests/data/pinned_specs.json``, taken
before any task was added) and their order, since a task's position in
REGISTRY is its benchmark seed index. New tasks are appended after them.

A task added after the 21 keeps its controllers in its own package's
``experts.py`` (``experts/pid.py`` and ``experts/mpc.py`` are in every task's
baseline fingerprint), declares any physics it imports from another package in
``fingerprint_sources``, has a name that cannot collide with another task's
gains keys, ships as ``-v2`` with a ``compute_reward_v1`` of its own, has a
hold row, and, if it ships a PID, has a row in ``TUNERS`` in
scripts/tune_pid.py. The checks loop over the added tasks and assert once, so
they pass trivially while none exists.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import pathlib

import pytest

from target_gym import provenance, registry

ROOT = pathlib.Path(__file__).resolve().parent.parent
PINNED_SPECS_PATH = ROOT / "tests" / "data" / "pinned_specs.json"
SRC = ROOT / "src" / "target_gym"

PINNED_21 = [
    "plane",
    "plane_energy",
    "plane_sine",
    "plane3d_heading",
    "plane3d_circle",
    "plane3d_racetrack",
    "plane3d_figure8",
    "patrol",
    "patrol_bearing_only",
    "cstr",
    "first_order",
    "four_tank",
    "ph_neutralization",
    "distillation",
    "glass_furnace",
    "reactor",
    "hvac",
    "cement_kiln",
    "boiler_drum",
    "wind_turbine",
    "battery",
]

_FACTORY_MODULES = {
    "_pid.<locals>.make": "target_gym.experts.pid",
    "_mpc.<locals>.make": "target_gym.experts.mpc",
}


def _factory(f):
    """``None``, or what a controller factory builds: its qualname, the module
    it imports from, and the name it looks up there, read from its closure."""
    if f is None:
        return None
    cells = dict(
        zip(f.__code__.co_freevars, (c.cell_contents for c in f.__closure__ or ()))
    )
    module = cells.get("module", _FACTORY_MODULES.get(f.__qualname__))
    return [f.__qualname__, module, cells.get("factory_name")]


def _pinned_snapshot() -> list[dict]:
    """Every field of the 21 pinned specs that could change what they are,
    including the ones no fingerprint sees (the controller factories, the
    conformance allowances)."""
    out = []
    for name in PINNED_21:
        s = registry.get(name)
        env = s.make_env()
        out.append(
            {
                "name": s.name,
                "group": s.group,
                "version": s.version,
                "display_name": registry.display_name(name),
                "params_cls": [s.params_cls._module, s.params_cls._name],
                "env_factory": s.env_factory.__qualname__,
                "env_class": [type(env).__module__, type(env).__qualname__],
                "make_pid": _factory(s.make_pid),
                "make_mpc": _factory(s.make_mpc),
                "test_params": s.test_params,
                "tuned_gains_key": s.tuned_gains_key,
                "baselines_note": s.baselines_note,
                "disturbance_fields": s.disturbance_fields,
                "disturbance_overrides": s.disturbance_overrides,
                "effectiveness_overrides": s.effectiveness_overrides,
                "expert_degraded": s.expert_degraded,
                "mpc_degraded": s.mpc_degraded,
                "noise_fields": s.noise_fields,
            }
        )
    return out


def test_the_21_specs_are_pinned():
    """``tests/data/pinned_specs.json`` was taken before any task was added.
    Changing one of the 21 specs means regenerating it in the same change, so
    the edit shows in review:

        uv run python -c "import json; from tests.test_registry_rules import
        _pinned_snapshot as s; open('tests/data/pinned_specs.json', 'w').write(
        json.dumps(s(), indent=2) + '\\n')"
    """
    pinned = json.loads(PINNED_SPECS_PATH.read_text())
    now = json.loads(json.dumps(_pinned_snapshot()))
    assert [r["name"] for r in now] == [r["name"] for r in pinned]
    diffs = [
        f"{new['name']}.{field}: {old.get(field)!r} -> {new[field]!r}"
        for old, new in zip(pinned, now)
        for field in new
        if old.get(field) != new[field]
    ]
    assert not diffs, "pinned specs changed:\n" + "\n".join(diffs)


def test_new_tasks_are_appended_after_the_21():
    """A task's position in REGISTRY is its benchmark seed index, and
    TargetFoundation derives seeds the same way, so new tasks go after the 21
    and never between them."""
    assert list(registry.REGISTRY)[:21] == PINNED_21
    names = [s.name for s in registry._SPECS]
    twice = sorted({n for n in names if names.count(n) > 1})
    assert not twice, f"names registered more than once: {twice}"


def _added() -> list:
    """The tasks registered after the 21."""
    return [s for s in registry.all_specs() if s.name not in PINNED_21]


def _package(spec) -> str:
    """The module path of the package ``spec``'s environment is defined in."""
    return type(spec.make_env()).__module__.rsplit(".", 1)[0]


def _package_dir(spec) -> pathlib.Path:
    return SRC.parent / pathlib.Path(*_package(spec).split("."))


def test_a_new_name_is_no_prefix_of_another():
    """Gains keys are collected by name prefix (``baseline_fingerprint`` takes
    every key that starts with the task's name), and runners.py and
    measure_hold.py treat names starting with "plane" or "patrol" as aircraft.
    So a new name must not start with, or be the start of, any other
    registered name, and must not start with either prefix."""
    names = registry.env_names()
    problems = set()
    for e in (s.name for s in _added()):
        for n in names:
            if n != e and (n.startswith(e) or e.startswith(n)):
                problems.add(f"{min(e, n)} and {max(e, n)}: one starts with the other")
        if e.startswith(("plane", "patrol")):
            problems.add(f"{e}: starts with 'plane' or 'patrol'")
    assert not problems, "\n".join(sorted(problems))


def test_a_new_task_ships_as_v2_with_its_own_v1_reward():
    """A new task is first stamped as ``-v2``, and its package's env.py defines
    ``compute_reward_v1`` itself, since TargetFoundation looks it up by that
    name. That ``compute_reward`` honours ``reward_version`` is already checked
    for every spec by test_reward_contract.py."""
    stamps = json.loads((SRC / "data" / "env_versions.json").read_text())
    problems = []
    for s in _added():
        if f"{s.name}-v1" in stamps:
            problems.append(f"{s.name}: stamped as -v1; a new task ships as -v2")
        env_py = _package_dir(s) / "env.py"
        body = ast.parse(env_py.read_text()).body if env_py.exists() else []
        if not any(
            isinstance(node, ast.FunctionDef) and node.name == "compute_reward_v1"
            for node in body
        ):
            problems.append(
                f"{s.name}: {env_py.relative_to(ROOT)} defines no compute_reward_v1"
            )
    assert not problems, "\n".join(problems)


def test_controllers_stay_out_of_the_shared_files():
    """A new task's controllers live in its own experts.py. experts/pid.py and
    experts/mpc.py are in every task's baseline fingerprint, so a controller
    added there would stale the records of all 21 (the baseline test is what
    enforces that; this names the cause)."""
    shared = [SRC / "experts" / f for f in ("pid.py", "mpc.py", "__init__.py")]
    texts = {path: path.read_text() for path in shared}
    problems = []
    for s in _added():
        problems += [
            f"{s.name} is named in {path.relative_to(ROOT)}"
            for path, text in texts.items()
            if s.name in text
        ]
    assert not problems, "\n".join(problems)


# What a package added after the 21 may import from target_gym besides itself
# and its declared fingerprint_sources: the shared plumbing, which every
# fingerprint already covers.
_SHARED_MODULES = {
    "target_gym.base",
    "target_gym.utils",
    "target_gym.integration",
    "target_gym.reward",
}
# And, in experts*.py only, the shared controller code.
_CONTROLLER_MODULES = {
    "target_gym.experts",
    "target_gym.experts.pid",
    "target_gym.experts.mpc",
}


def _imports(tree) -> list[tuple[ast.Import | ast.ImportFrom, bool]]:
    """Every import in ``tree``, and whether it sits inside a function body."""
    found = []

    def visit(node, in_function):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                found.append((child, in_function))
            visit(
                child,
                in_function
                or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)),
            )

    visit(tree, False)
    return found


def _may_import(module: str, package: str, allowed: set[str]) -> bool:
    if module != "target_gym" and not module.startswith("target_gym."):
        return True  # not this library
    return module == package or module.startswith(f"{package}.") or module in allowed


def test_a_new_package_imports_only_what_it_declares():
    """Imported physics must be fingerprinted, so a package added after the 21
    imports only itself, the shared plumbing and the modules its spec declares
    in ``fingerprint_sources`` (its experts*.py may also import the shared
    controller code). Relative imports are refused because the check reads
    absolute names only. A file not named experts* imports its package's
    controllers only inside a function, as the env_jax files of the 21 do: the
    version stamp leaves experts* out, so physics that depended on them could
    change with no version bump."""
    problems = []
    for s in _added():
        package = _package(s)
        declared = {
            "target_gym." + rel.removesuffix(".py").replace("/", ".")
            for rel in s.fingerprint_sources
        }
        for path in sorted(_package_dir(s).glob("*.py")):
            if path.name.startswith("rendering"):
                continue
            is_experts = path.name.startswith("experts")
            allowed = _SHARED_MODULES | declared
            if is_experts:
                allowed |= _CONTROLLER_MODULES
            for node, in_function in _imports(ast.parse(path.read_text())):
                where = f"{path.relative_to(ROOT)}:{node.lineno}"
                if isinstance(node, ast.ImportFrom) and node.level:
                    problems.append(f"{where}: relative import; write {package}...")
                    continue
                # Each imported name, with the modules it may stand for.
                # ``from a import b`` passes if either ``a`` or ``a.b`` is
                # allowed, so ``from target_gym.pc_gym.cstr import env`` counts
                # as importing the declared pc_gym/cstr/env.py.
                if isinstance(node, ast.Import):
                    imported = [(a.name, [a.name]) for a in node.names]
                else:
                    module = node.module or ""
                    imported = [
                        (
                            f"{module}.{a.name}" if module == "target_gym" else module,
                            [module, f"{module}.{a.name}"],
                        )
                        for a in node.names
                    ]
                refused = {
                    name
                    for name, modules in imported
                    if not any(_may_import(m, package, allowed) for m in modules)
                }
                problems += [
                    f"{where}: imports {name}; declare it in fingerprint_sources "
                    "or stop importing it"
                    for name in sorted(refused)
                ]
                reached = [m for _, modules in imported for m in modules]
                if (
                    not is_experts
                    and not in_function
                    and any(n.startswith(f"{package}.experts") for n in reached)
                ):
                    problems.append(
                        f"{where}: imports its package's controllers outside a "
                        "function"
                    )
    assert not problems, "\n".join(problems)


def test_the_21_hash_what_they_hashed():
    """The 21 declare no extra sources and hold no experts* file, so both
    fingerprints of the 21 read the same files as before. A new task's own
    experts.py is in its baseline fingerprint and out of its version stamp."""
    problems = []
    for s in registry.all_specs():
        sources = provenance._env_sources(s)
        stamped = provenance._env_sources(s, controllers=False)
        if s.name in PINNED_21:
            if s.fingerprint_sources:
                problems.append(f"{s.name}: declares {s.fingerprint_sources}")
            if stamped != sources:
                dropped = [
                    str(p.relative_to(ROOT)) for p in sources if p not in stamped
                ]
                problems.append(f"{s.name}: the version stamp would drop {dropped}")
            continue
        experts = _package_dir(s) / "experts.py"
        if experts.exists():
            if experts not in sources:
                problems.append(
                    f"{s.name}: experts.py is not in its baseline fingerprint"
                )
            if experts in stamped:
                problems.append(f"{s.name}: experts.py is in its version stamp")
    assert not problems, "\n".join(problems)


def test_every_mpc_task_and_every_new_task_has_a_hold_row():
    """``eval.hold_settings`` returns a burn-in of 0 for a task with no row in
    hold_measurements.json, without a word, and the protocol would then score
    the approach as if it were the hold. Every task with an MPC has a row
    (patrol_bearing_only has neither), and so does every new task."""
    rows = json.loads((SRC / "data" / "hold_measurements.json").read_text())
    missing = [
        s.name
        for s in registry.all_specs()
        if (s.has_mpc or s.name not in PINNED_21) and s.name not in rows
    ]
    assert not missing, (
        f"no row in src/target_gym/data/hold_measurements.json for {missing}. "
        "Run scripts/measure_hold.py --envs with those names."
    )


def _tuner_rows() -> set[str]:
    """The keys of ``TUNERS`` in scripts/tune_pid.py, read from its source.
    Importing the script would put ``src`` on ``sys.path`` as a side effect."""
    path = ROOT / "scripts" / "tune_pid.py"
    for node in ast.parse(path.read_text()).body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "TUNERS" for t in node.targets)
            and isinstance(node.value, ast.Dict)
        ):
            return {
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
    raise AssertionError("scripts/tune_pid.py defines no TUNERS dict literal")


def test_every_new_task_with_a_pid_has_a_tuner_row():
    """A new task's gains are set by naming it to scripts/tune_pid.py, which
    refuses a name with no row in ``TUNERS``. So every task added after the 21
    that ships a PID has a row there."""
    rows = _tuner_rows()
    missing = [s.name for s in _added() if s.has_pid and s.name not in rows]
    assert not missing, f"no TUNERS row in scripts/tune_pid.py for {missing}"


def test_allowlists_name_real_tasks():
    """An allowlist entry for a task that is not registered exempts nothing,
    and usually means the task was renamed."""
    import tests.test_env_conformance as C

    unknown = sorted(set(C.KNOWN_OPEN_LOOP_UNSTABLE) - set(registry.env_names()))
    assert not unknown, f"KNOWN_OPEN_LOOP_UNSTABLE names unregistered tasks: {unknown}"


def test_declared_sources_enter_both_fingerprints():
    """A declared source is hashed into both fingerprints, and a declared file
    that does not exist is an error, not a silent skip."""
    base = registry.REGISTRY["first_order"]
    declared = dataclasses.replace(base, fingerprint_sources=("pc_gym/cstr/env.py",))
    assert provenance._ROOT / "pc_gym/cstr/env.py" in provenance._env_sources(declared)
    for fingerprint in (
        provenance.environment_fingerprint,
        provenance.baseline_fingerprint,
    ):
        assert fingerprint(declared) != fingerprint(base), fingerprint.__name__
        typo = dataclasses.replace(base, fingerprint_sources=("no/such/file.py",))
        with pytest.raises(FileNotFoundError, match="no/such/file.py"):
            fingerprint(typo)
