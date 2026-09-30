"""The registry's tiers: the 21 core tasks every default accessor returns, and
the extended tasks that only code asking for them sees.

The tests here protect the 21. Whatever is added beside them, the default
accessors, the benchmark seeds, the fingerprints and every field of the 21
specs stay what they were.
"""

from __future__ import annotations

import json
import pathlib

from target_gym import registry

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORE_SPECS_PATH = ROOT / "tests" / "data" / "core_specs.json"

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
