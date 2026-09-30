"""The registry's tiers: the 21 core tasks every default accessor returns, and
the extended tasks that only code asking for them sees.

The tests here protect the 21. Whatever is added beside them, the default
accessors, the benchmark seeds, the fingerprints and every field of the 21
specs stay what they were.

They also hold every extended task to the layout, naming, import and record
rules that keep it apart from the 21. Those checks loop over the specs and
assert once, unparametrised, so they pass trivially while no extended task is
registered and apply to the first one with no edit here. The paths no core
task reaches are exercised with a fake extended spec that is never registered
(the ``fake`` fixture).
"""

from __future__ import annotations

import ast
import dataclasses
import json
import pathlib
import subprocess
import sys

import jax
import jax.numpy as jnp
import pytest

from target_gym import benchmark, provenance, registry

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORE_SPECS_PATH = ROOT / "tests" / "data" / "core_specs.json"
SRC = ROOT / "src" / "target_gym"

CORE_21 = [
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


def _core_snapshot() -> list[dict]:
    """Every field of the 21 core specs that could change what they are,
    including the ones no fingerprint sees (the controller factories, the
    conformance allowances)."""
    out = []
    for name in CORE_21:
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
    """``tests/data/core_specs.json`` was taken on main before the tiers were
    added. Changing a core spec means regenerating it in the same change, so
    the edit shows in review:

        uv run python -c "import json; from tests.test_registry_tiers import
        _core_snapshot as s; open('tests/data/core_specs.json', 'w').write(
        json.dumps(s(), indent=2) + '\\n')"
    """
    pinned = json.loads(CORE_SPECS_PATH.read_text())
    now = json.loads(json.dumps(_core_snapshot()))
    assert [r["name"] for r in now] == [r["name"] for r in pinned]
    diffs = [
        f"{new['name']}.{field}: {old.get(field)!r} -> {new[field]!r}"
        for old, new in zip(pinned, now)
        for field in new
        if old.get(field) != new[field]
    ]
    assert not diffs, "core specs changed:\n" + "\n".join(diffs)


# ---------------------------------------------------------------------------
# Every tier, checked as a whole
# ---------------------------------------------------------------------------


def _package_dir(spec) -> pathlib.Path:
    """An extended task's package: ``src/target_gym/extended/<name>/``."""
    return SRC / "extended" / spec.name


def test_the_defaults_are_the_21():
    """Every default accessor returns the 21, in their order. Adding a core
    task means editing CORE_21, which makes it a visible decision."""
    assert list(registry.REGISTRY) == CORE_21
    assert registry.env_names() == CORE_21
    assert [s.name for s in registry.all_specs()] == CORE_21
    assert list(registry.GROUPS) == ["aircraft", "process", "industrial", "energy"]
    odd = [
        s.name
        for s in registry.all_specs()
        if s.tier != "core" or s.seed_index is not None
    ]
    assert not odd, f"default specs that are not plain core specs: {odd}"


def test_benchmark_seeds_are_pinned():
    """A core task's benchmark seed index is its position in CORE_21, which
    TargetFoundation relies on. An extended task declares its own, from 1000
    up and increasing, so adding one never shifts another task's episodes."""
    problems = [
        f"{name}: seed index {registry.task_seed_index(name)}, pinned at {i}"
        for i, name in enumerate(CORE_21)
        if registry.task_seed_index(name) != i
    ]
    extended = list(registry.all_specs("extended"))
    problems += [
        f"{s.name}: seed_index {s.seed_index!r}, expected an int of 1000 or more"
        for s in extended
        if not isinstance(s.seed_index, int) or s.seed_index < 1000
    ]
    indices = [s.seed_index for s in extended]
    comparable = all(isinstance(i, int) for i in indices)
    if comparable and any(a >= b for a, b in zip(indices, indices[1:])):
        problems.append(
            f"extended seed indices {indices} do not increase in registration order"
        )
    assert not problems, "\n".join(problems)


def test_tiers_are_valid_and_appended():
    """Extended specs come after the 21 in ``_SPECS``, never between them: a
    core task's position is its benchmark seed index."""
    specs = registry._SPECS
    problems = [
        f"{s.name}: unknown tier {s.tier!r}, expected one of {registry.TIERS}"
        for s in specs
        if s.tier not in registry.TIERS
    ]
    tiers = [s.tier for s in specs]
    if "extended" in tiers and "core" in tiers[tiers.index("extended") :]:
        problems.append("a core spec follows an extended one in _SPECS")
    names = [s.name for s in specs]
    twice = sorted({n for n in names if names.count(n) > 1})
    if twice:
        problems.append(f"names registered more than once: {twice}")
    assert not problems, "\n".join(problems)


def test_every_spec_is_in_a_group_of_its_own_tier():
    """A core spec's group is in GROUPS and an extended spec's is in
    EXTENDED_GROUPS. The two never share a key, and no extended group is
    empty."""
    problems = []
    for s in registry.all_specs("all"):
        groups = registry.GROUPS if s.tier == "core" else registry.EXTENDED_GROUPS
        if s.group not in groups:
            problems.append(
                f"{s.name} ({s.tier}): group {s.group!r} is not one of {sorted(groups)}"
            )
    shared = sorted(set(registry.GROUPS) & set(registry.EXTENDED_GROUPS))
    if shared:
        problems.append(f"groups in both GROUPS and EXTENDED_GROUPS: {shared}")
    empty = [
        g for g in registry.EXTENDED_GROUPS if not list(registry.specs_in_group(g))
    ]
    if empty:
        problems.append(f"extended groups with no task: {empty}")
    assert not problems, "\n".join(problems)


def test_no_extended_name_is_a_prefix_of_another():
    """Gains keys are collected by name prefix (``baseline_fingerprint`` takes
    every key that starts with the task's name), and runners.py and
    measure_hold.py treat names starting with "plane" or "patrol" as aircraft.
    So an extended name must not start with, or be the start of, any other
    registered name, and must not start with either prefix."""
    names = registry.env_names("all")
    problems = set()
    for e in registry.env_names("extended"):
        for n in names:
            if n != e and (n.startswith(e) or e.startswith(n)):
                problems.add(f"{min(e, n)} and {max(e, n)}: one starts with the other")
        if e.startswith(("plane", "patrol")):
            problems.add(f"{e}: starts with 'plane' or 'patrol'")
    assert not problems, "\n".join(sorted(problems))


def test_an_extended_task_ships_as_v2_with_its_own_v1_reward():
    """An extended task is first stamped as ``-v2``, and its package's env.py
    defines ``compute_reward_v1`` itself, since TargetFoundation looks it up by
    that name. Core tasks are exempt, because plane3d and patrol name theirs
    differently. That ``compute_reward`` honours ``reward_version`` is already
    checked for every spec by test_reward_contract.py."""
    stamps = json.loads((SRC / "data" / "env_versions.json").read_text())
    problems = []
    for s in registry.all_specs("extended"):
        if f"{s.name}-v1" in stamps:
            problems.append(f"{s.name}: stamped as -v1; an extended task ships as -v2")
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


def test_an_extended_task_lives_in_its_own_package():
    """Env class in ``extended/<name>/env_jax.py``, params class in
    ``extended/<name>/env.py``, and a PHYSICS.md beside them. No core spec
    points into ``target_gym.extended``."""
    problems = []
    for s in registry.all_specs("all"):
        env_module = type(s.make_env()).__module__
        params_module = s.params_cls._module
        if s.tier == "core":
            problems += [
                f"{s.name}: a core task defined in {m}"
                for m in (env_module, params_module)
                if m.startswith("target_gym.extended.")
            ]
            continue
        package = f"target_gym.extended.{s.name}"
        if env_module != f"{package}.env_jax":
            problems.append(
                f"{s.name}: env class in {env_module}, not {package}.env_jax"
            )
        if params_module != f"{package}.env":
            problems.append(
                f"{s.name}: params class in {params_module}, not {package}.env"
            )
        if not (_package_dir(s) / "PHYSICS.md").exists():
            problems.append(
                f"{s.name}: no {_package_dir(s).relative_to(ROOT)}/PHYSICS.md"
            )
    assert not problems, "\n".join(problems)


def test_controllers_stay_out_of_the_shared_files():
    """An extended task's controllers live in its own experts.py. experts/pid.py
    and experts/mpc.py are in every task's baseline fingerprint, so a
    controller added there would stale the records of all 21 (the baseline
    test is what enforces that; this names the cause)."""
    shared = [SRC / "experts" / f for f in ("pid.py", "mpc.py", "__init__.py")]
    texts = {path: path.read_text() for path in shared}
    problems = []
    for s in registry.all_specs("extended"):
        problems += [
            f"{s.name} is named in {path.relative_to(ROOT)}"
            for path, text in texts.items()
            if s.name in text
        ]
        try:
            if s.has_pid and s.make_pid() is None:
                problems.append(f"{s.name}: make_pid returned None")
            if s.has_mpc and s.make_mpc(s.make_env(), s.make_test_params()) is None:
                problems.append(f"{s.name}: make_mpc returned None")
        except Exception as e:  # reported with the rest below
            problems.append(f"{s.name}: building its controllers raised {e!r}")
    assert not problems, "\n".join(problems)


# What any extended package may import from target_gym besides itself and its
# declared fingerprint_sources: the shared plumbing, which every fingerprint
# already covers.
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


def test_an_extended_package_imports_only_what_it_declares():
    """Imported physics must be fingerprinted, so an extended package imports
    only itself, the shared plumbing and the modules its spec declares in
    ``fingerprint_sources`` (its experts*.py may also import the shared
    controller code). Relative imports are refused because the check reads
    absolute names only. A file not named experts* imports its package's
    controllers only inside a function, as the core env_jax files do: the
    version stamp leaves experts* out, so physics that depended on them could
    change with no version bump."""
    problems = []
    for s in registry.all_specs("extended"):
        package = f"target_gym.extended.{s.name}"
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


def test_importing_target_gym_runs_no_extended_code():
    """``import target_gym``, its star import and the default accessors load
    no extended module, so a broken extended task cannot reach a process that
    has not asked for it."""
    code = (
        "import json, sys, target_gym\n"
        "from target_gym import *\n"
        "from target_gym import registry\n"
        "list(registry.REGISTRY); list(registry.all_specs())\n"
        "print(json.dumps(sorted(m for m in sys.modules"
        " if m.startswith('target_gym.extended'))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT
    )
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert not loaded, f"importing target_gym loaded extended modules: {loaded}"


def test_the_21_hash_what_they_hashed():
    """A core spec declares no extra sources, and no core package holds an
    experts* file, so both fingerprints of the 21 read the same files as
    before the tiers. An extended task's experts.py is in its baseline
    fingerprint and out of its version stamp."""
    problems = []
    for s in registry.all_specs("all"):
        if s.tier == "core" and s.fingerprint_sources:
            problems.append(f"{s.name}: a core spec declares {s.fingerprint_sources}")
            continue
        sources = provenance._env_sources(s)
        stamped = provenance._env_sources(s, controllers=False)
        if s.tier == "core":
            if stamped != sources:
                dropped = [
                    str(p.relative_to(ROOT)) for p in sources if p not in stamped
                ]
                problems.append(f"{s.name}: the version stamp would drop {dropped}")
            continue
        experts = provenance._ROOT / "extended" / s.name / "experts.py"
        if experts not in sources:
            problems.append(f"{s.name}: experts.py is not in its baseline fingerprint")
        if experts in stamped:
            problems.append(f"{s.name}: experts.py is in its version stamp")
    assert not problems, "\n".join(problems)


def test_every_mpc_task_and_every_extended_task_has_a_hold_row():
    """``eval.hold_settings`` returns a burn-in of 0 for a task with no row in
    hold_measurements.json, without a word, and the protocol would then score
    the approach as if it were the hold. Every task with an MPC has a row
    (patrol_bearing_only has neither), and so does every extended task."""
    rows = json.loads((SRC / "data" / "hold_measurements.json").read_text())
    missing = [
        s.name
        for s in registry.all_specs("all")
        if (s.has_mpc or s.tier == "extended") and s.name not in rows
    ]
    assert not missing, (
        f"no row in src/target_gym/data/hold_measurements.json for {missing}. "
        "Run scripts/measure_hold.py --envs with those names."
    )


def test_no_suite_drops_a_task():
    """The spec-parametrised suites run every tier. One that went back to the
    default accessors would silently stop checking the extended tasks."""
    import tests.experts.test_mpc_baselines as MB
    import tests.test_env_conformance as C
    import tests.test_env_versions as V
    import tests.test_reward_contract as RC

    names = registry.env_names("all")
    mpc = [s.name for s in registry.all_specs("all") if s.has_mpc]
    problems = [
        f"{suite} runs {got}, expected {want}"
        for suite, got, want in (
            ("test_env_conformance.py", C.SPEC_IDS, names),
            ("test_reward_contract.py", RC.IDS, names),
            ("test_env_versions.py", V.NAMES, sorted(names)),
            ("experts/test_mpc_baselines.py", MB.MPC_ENVS, mpc),
        )
        if got != want
    ]
    assert not problems, "\n".join(problems)


def test_no_suite_drops_an_extended_task(fake):
    """The same, with an extended task registered. Fresh copies of the four
    suites are loaded while the fake is in ``_SPECS``, so a suite that went
    back to the default accessors fails here even while no real extended task
    exists."""
    import importlib.util

    lists = {
        "test_env_conformance.py": "SPEC_IDS",
        "test_reward_contract.py": "IDS",
        "test_env_versions.py": "NAMES",
        "experts/test_mpc_baselines.py": "MPC_ENVS",
    }
    dropped = []
    for i, (rel, attr) in enumerate(lists.items()):
        loader = importlib.util.spec_from_file_location(
            f"_tier_probe_{i}", ROOT / "tests" / rel
        )
        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)
        if fake.name not in getattr(module, attr):
            dropped.append(f"{rel}: {attr} leaves out {fake.name}")
    assert not dropped, "\n".join(dropped)


def test_allowlists_name_real_tasks():
    """An allowlist entry for a task that is not registered exempts nothing,
    and usually means the task was renamed."""
    import tests.test_env_conformance as C

    names = set(registry.env_names("all"))
    unknown = sorted(set(C.KNOWN_OPEN_LOOP_UNSTABLE) - names)
    assert not unknown, f"KNOWN_OPEN_LOOP_UNSTABLE names unregistered tasks: {unknown}"


# ---------------------------------------------------------------------------
# A fake extended task
#
# The paths no core task reaches (tier filtering, get() across tiers, extended
# groups, task_seed_index, the benchmark lookup, declared fingerprint sources)
# are exercised on first_order under another name, appended to _SPECS for one
# test at a time and never registered.
# ---------------------------------------------------------------------------


@pytest.fixture
def fake(monkeypatch):
    spec = dataclasses.replace(
        registry.REGISTRY["first_order"],
        name="zz_fake_ext",
        group="zz_fake_group",
        tier="extended",
        seed_index=1000,
    )
    monkeypatch.setattr(registry, "_SPECS", registry._SPECS + (spec,))
    monkeypatch.setitem(registry.EXTENDED_GROUPS, "zz_fake_group", "Fake")
    yield spec
    registry._ENV_CACHE.pop("zz_fake_ext", None)


def _process_group() -> list[str]:
    return [s.name for s in registry.specs_in_group("process")]


def test_an_extended_task_leaves_the_defaults_alone(fake, monkeypatch):
    process = [n for n, s in registry.REGISTRY.items() if s.group == "process"]
    assert registry.env_names() == CORE_21
    assert [s.name for s in registry.all_specs()] == CORE_21
    assert fake.name not in registry.REGISTRY
    assert _process_group() == process
    # A second fake given a core group by mistake still stays out of it.
    stray = dataclasses.replace(fake, name="zz_fake_stray", group="process")
    monkeypatch.setattr(registry, "_SPECS", registry._SPECS + (stray,))
    assert _process_group() == process


def test_an_extended_task_is_found_by_asking_for_it(fake):
    assert registry.env_names("all") == CORE_21 + ["zz_fake_ext"]
    assert registry.env_names("extended") == ["zz_fake_ext"]
    assert list(registry.all_specs("extended")) == [fake]
    assert registry.get("zz_fake_ext") is fake
    assert list(registry.specs_in_group("zz_fake_group")) == [fake]
    assert registry.task_seed_index("zz_fake_ext") == 1000


def test_the_benchmark_runs_an_extended_task_it_is_given(fake):
    """Neither the lookup nor the seed index raises for a task outside
    REGISTRY, and its key comes from its declared seed_index."""
    seen = {}

    class Zeros:
        name = "zeros"

        def reset(self, key, obs, task):
            seen[task.name] = key
            return jnp.zeros((task.n_episodes, task.n_act))

        def act(self, carry, obs, t):
            return carry, carry

    rows = benchmark.run_policy_on_benchmark(
        Zeros(), tasks=[fake.name], n_episodes=1, n_steps=1, verbose=False
    )
    assert "metrics" in rows[fake.name]
    task_key = jax.random.fold_in(jax.random.PRNGKey(0), 1000)
    # run_policy hands the policy fold_in(task_key, 1)
    assert (seen[fake.name] == jax.random.fold_in(task_key, 1)).all()


def test_declared_sources_enter_both_fingerprints(fake):
    declared = dataclasses.replace(fake, fingerprint_sources=("pc_gym/cstr/env.py",))
    assert provenance._ROOT / "pc_gym/cstr/env.py" in provenance._env_sources(declared)
    for fingerprint in (
        provenance.environment_fingerprint,
        provenance.baseline_fingerprint,
    ):
        assert fingerprint(declared) != fingerprint(fake), fingerprint.__name__
        typo = dataclasses.replace(fake, fingerprint_sources=("no/such/file.py",))
        with pytest.raises(FileNotFoundError, match="no/such/file.py"):
            fingerprint(typo)
