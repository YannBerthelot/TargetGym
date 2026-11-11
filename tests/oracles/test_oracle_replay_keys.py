"""K8's replay of a plan is the env's step on the oracle's own params.

Contract (controller.Oracle, "the noise is zeroed; only a trip draws"): on the
planner's params (``plan_params``) a step's outcome does not depend on its
key, so the replay's fixed key, and the draws x64 changes
(``jax.random.normal`` gives other values under x64 than in float32), cannot
move it. And each oracle's record holds one back-off per limit of its model,
so no limit is dropped or handed another's back-off.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import records
from target_gym.experts.mpc import plan_params
from target_gym.oracles.controller import enable_x64, float64
from target_gym.registry import REGISTRY

pytestmark = pytest.mark.medium

#: The thermal and energy oracles. Other families' oracles fail these checks
#: on int/amendment3 (glass_furnace's step reads its key on the planner's
#: params; reactor's and unstable_cstr's records hold more back-offs than their
#: models have limits), reported to their owners rather than fixed here.
ORACLES = ["battery", "boiler_drum", "hvac", "wind_turbine"]


@pytest.mark.parametrize("name", ORACLES)
def test_a_step_on_the_oracles_params_does_not_depend_on_its_key_under_x64(name):
    spec = REGISTRY[name]
    env = spec.make_env()
    p = plan_params(spec, spec.make_test_params())
    _, st = env.reset_env(jax.random.PRNGKey(100), p)
    a = env.initial_action(st, p) if hasattr(env, "initial_action") else None
    with enable_x64():
        p64, s = float64(p), float64(st)
        a = jnp.zeros(env.action_space(p).shape) if a is None else float64(a)

        def run(seed):
            s1 = s
            for t in range(3):
                key = jax.random.fold_in(jax.random.PRNGKey(seed), t)
                s1 = env.step_env(key, s1, a, p64)[1]
            return s1

        one, two = run(0), run(1)
    for x, y in zip(jax.tree_util.tree_leaves(one), jax.tree_util.tree_leaves(two)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


@pytest.mark.parametrize("name", ORACLES)
def test_an_oracle_record_holds_one_backoff_per_limit_or_none(name):
    spec = REGISTRY[name]
    env = spec.make_env()
    p = plan_params(spec, spec.make_test_params())
    oracle = spec.make_mpc(env, p)
    if not hasattr(oracle, "model"):
        pytest.skip("not on the shared solver: no limits to back off")
    back = records.load_row("oracle_widths", name).get("limit_backoffs") or []
    assert len(back) in (0, len(oracle.model.limits))
