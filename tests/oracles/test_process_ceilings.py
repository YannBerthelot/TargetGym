"""The process family's ceilings (charter Words "Ceiling", J3): the foresight
oracle plans the task's own reward on the env's own step.

Contract, from the charter and the oracle rule (an oracle is the problem
definition, the true model, the true state and the future, solved by a
converged solver):

- the planned cost of a plan is the cost the env charges when the plan is
  replayed through the benchmark's own loop (``benchmark.run_policy``) on the
  same seed and scenario, so the ceiling is the task's score and nothing
  restated by hand;
- hand-checked: on the pH task one 5 s step with the stage-1 valve at half
  range (11 mL/s) and the trim shut costs the reagent it doses, 11 mL/s x 5 s
  = 0.055 L at 1 L per L, the discharge staying inside the permit;
- a plan of the solver never costs more than the plan it started from, and
  its certificate says whether it converged.
"""

import numpy as np
import pytest

from target_gym import scenarios
from target_gym.benchmark import episodes_of, run_policy, task_info
from target_gym.pc_gym import foresight
from target_gym.registry import REGISTRY

PROCESS = (
    "ph_neutralization",
    "distillation",
    "unstable_cstr",
    "compressor_surge",
    "cstr",
)
SEED = 100


def _short(name, steps, scenario=0):
    spec = REGISTRY[name]
    s = scenarios.suite(spec)[scenario]
    params = scenarios.params_for(
        spec, s, sensor_noise_scale=1.0, max_steps_in_episode=steps
    )
    return spec.make_env(), params


def _replayed(name, env, params, plan):
    """The env's mean cost per step of ``plan`` on seed ``SEED``, through the
    benchmark's loop, and whether it tripped."""
    rollout = run_policy(
        foresight.PlanPolicy(np.asarray(plan)[None]),
        env,
        params,
        task_info(name, env, params, 1),
        seeds=[SEED],
    )
    (episode,) = episodes_of(rollout)
    return float(episode.cost.mean()), bool(np.any(episode.failed))


@pytest.mark.parametrize("name", PROCESS)
def test_the_planned_cost_of_a_plan_is_the_envs_replayed_cost(name):
    env, params = _short(name, 8)
    nu = int(np.prod(env.action_space(params).shape))
    plan = np.random.default_rng(0).uniform(-0.3, 0.3, (8, nu))
    planned = foresight.plan_cost(env, params, SEED, plan)
    replayed, tripped = _replayed(name, env, params, plan)
    assert not tripped
    # The same float32 functions, unbatched against vmapped: rounding only.
    np.testing.assert_allclose(planned, replayed, rtol=1e-6, atol=1e-7)


def test_a_tracked_plan_on_the_unstable_plant_replays_at_its_planned_cost():
    from target_gym.pc_gym.unstable_cstr.oracle import CEILING_FEEDBACK

    name = "unstable_cstr"
    env, params = _short(name, 8, scenario=2)
    plan = np.random.default_rng(1).uniform(-0.5, 0.5, (8, 1))
    path = foresight.tracking(env, params, SEED, plan, CEILING_FEEDBACK)
    policy = foresight.TrackingPolicy(plan[None], [path], CEILING_FEEDBACK[0])
    replayed = foresight.replay(name, env, params, [SEED], plan[None], policy)
    np.testing.assert_allclose(
        replayed["costs"][0], foresight.plan_cost(env, params, SEED, plan), rtol=1e-6
    )


def test_one_ph_step_costs_the_reagent_it_doses():
    env, params = _short("ph_neutralization", 1)
    plan = np.array([[0.0, -1.0]])  # q3 at half of 22 mL/s, the trim shut
    assert foresight.plan_cost(env, params, SEED, plan) == pytest.approx(0.055)
    assert _replayed("ph_neutralization", env, params, plan)[0] == pytest.approx(
        0.055, rel=1e-6
    )


def test_a_solve_never_ends_above_its_start_and_reports_convergence():
    env, params = _short("distillation", 20)
    from target_gym.pc_gym.distillation.oracle import CEILING_TERMS, ceiling_margins

    start = np.zeros((20, 2))
    plan, cert = foresight.solve(
        env, params, SEED, ceiling_margins, CEILING_TERMS, start, maxiter=30
    )
    assert foresight.plan_cost(env, params, SEED, plan) <= foresight.plan_cost(
        env, params, SEED, start
    )
    assert {"status", "projected_gradient", "iterations", "capped"} <= set(cert)
