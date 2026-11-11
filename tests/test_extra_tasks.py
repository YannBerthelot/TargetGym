"""Extra tasks (charter rule 5): kept in the registry, listed with their reason,
never scored. The benchmark is the registry minus them.

Contract, from the charter's Extra rule and Amendment 3's per-task table, not
from the code: four_tank is Extra (a lab rig with no industrial job); cstr,
unstable_cstr and compressor_surge are back in the benchmark.
"""

import pytest

from target_gym import registry


def test_four_tank_is_extra_with_its_reason():
    spec = registry.get("four_tank")
    assert spec.extra and "lab" in spec.extra
    assert "four_tank" not in {s.name for s in registry.benchmark_specs()}


@pytest.mark.parametrize(
    "name", ["cstr", "unstable_cstr", "compressor_surge", "ph_neutralization"]
)
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
