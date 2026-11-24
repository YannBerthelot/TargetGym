"""Extra tasks (charter rule 5): kept in the registry, listed with their reason,
never scored. The benchmark is the registry minus them.

Contract, from the charter's Extra rule and Amendment 3's per-task table, not
from the code: four_tank is Extra (a lab rig with no industrial job); cstr,
unstable_cstr and compressor_surge are back in the benchmark. battery is Extra by
the owner's decision of 2026-11-15: it failed acceptance (J2), so it leaves the
core pool, the headline and the capability claims, with its code kept.
unstable_cstr is Extra by the owner's decision of the same day: its J2 passed,
but on 4 of its 5 scenarios the true-state oracle tracks exactly, so its room
measures a perfect state reading rather than control. It leaves the core pool
and the capability claims the same way. boiler_drum is Extra by the owner's
decision of 2026-11-24, as solved (charter J5): its reference costs nothing on
normal days and what room remains is worth GBP 90 to 170 a year. The same day
the owner set one rule for the radar: an Extra task carries no capability tag
and moves no spoke, four_tank included.
"""

import pytest

from target_gym import registry


def test_four_tank_is_extra_with_its_reason():
    spec = registry.get("four_tank")
    assert spec.extra and "lab" in spec.extra
    assert "four_tank" not in {s.name for s in registry.benchmark_specs()}


def test_battery_is_extra_with_its_reason():
    spec = registry.get("battery")
    assert spec.extra and "J2" in spec.extra
    assert "battery" not in {s.name for s in registry.benchmark_specs()}


def test_battery_moves_no_capability_spoke():
    """Out of the capability claims: like the first_order fixture, it keeps an
    entry in the suite and its score enters no aggregate."""
    from target_gym import suite

    assert suite.CAPABILITIES["battery"] == ()
    high = {"score": 0.9, "transient_ratio": 3.0, "oracle_transient_ratio": 3.0}
    hvac = {"score": 0.1, "transient_ratio": 1.0, "oracle_transient_ratio": 3.0}
    profile = suite.profile({"battery": high, "hvac": hvac})
    assert profile["hold"]["economic"] == pytest.approx(suite.radius(0.1))


def test_unstable_cstr_is_extra_with_its_reason():
    spec = registry.get("unstable_cstr")
    assert spec.extra and "oracle" in spec.extra
    assert "unstable_cstr" not in {s.name for s in registry.benchmark_specs()}


def test_unstable_cstr_moves_no_capability_spoke():
    """Out of the capability claims: its score enters no aggregate. Beside
    cstr, the nonlinearity spoke stands where cstr alone puts it."""
    from target_gym import suite

    assert suite.CAPABILITIES["unstable_cstr"] == ()
    high = {"score": 0.9, "transient_ratio": 3.0, "oracle_transient_ratio": 3.0}
    cstr = {"score": 0.1, "transient_ratio": 1.0, "oracle_transient_ratio": 3.0}
    profile = suite.profile({"unstable_cstr": high, "cstr": cstr})
    assert profile["hold"]["nonlinearity"] == pytest.approx(suite.radius(0.1))


def test_boiler_drum_is_extra_with_its_reason():
    spec = registry.get("boiler_drum")
    assert spec.extra and "solved" in spec.extra
    assert "boiler_drum" not in {s.name for s in registry.benchmark_specs()}


def test_an_extra_task_carries_no_capability_tag():
    """Owner's rule, 2026-11-24: an Extra task moves no spoke."""
    from target_gym import suite

    extra = [name for name, spec in registry.REGISTRY.items() if spec.extra]
    assert {"four_tank", "battery", "unstable_cstr", "boiler_drum"} <= set(extra)
    assert {name: suite.CAPABILITIES[name] for name in extra} == dict.fromkeys(
        extra, ()
    )


@pytest.mark.parametrize("name", ["cstr", "compressor_surge", "ph_neutralization"])
def test_the_process_tasks_brought_back_are_in_the_benchmark(name):
    assert registry.get(name).extra is None
    assert name in {s.name for s in registry.benchmark_specs()}


def test_an_extra_task_still_builds_and_steps():
    """Extra tasks stay in the repository and keep working."""
    import jax

    spec = registry.get("four_tank")
    env, p = spec.make_env(), spec.make_test_params()
    obs, s = env.reset_env(jax.random.PRNGKey(0), p)
    obs, s, r, *_ = env.step_env(
        jax.random.PRNGKey(1), s, env.action_space(p).sample(jax.random.PRNGKey(2)), p
    )
    assert obs.shape == env.obs_shape
