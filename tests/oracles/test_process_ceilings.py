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
  its certificate says whether it converged;
- the compressor's ceiling plans no state left of the surge line, where the
  model has no rotating stall (compressor_surge/PHYSICS.md, D1): its surge
  margin is the step's soft minimum of Phi less the line's 2W, in hundredths;
- its plan stays inside each actuator's reach. A command further than the
  drive or the valve travels in a step moves the plant no further than one at
  the reach, so the plan's cost does not see it: the margin charges the
  excess, in reaches (the oracle's T1r, as a margin). ``ceiling_start`` clips
  a plan into reach, and the plant runs the clipped plan as it runs the plan.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import scenarios
from target_gym.benchmark import episode_keys, episodes_of, run_policy, task_info
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


# ---------------------------------------------------------------------------
# The compressor's margins: the surge line, and each actuator's reach
# ---------------------------------------------------------------------------


def _compressor(steps=60):
    """The compressor on slow_approach, whose reset holds the recycle open
    (0.13 to 0.32), so the valve has room to close, with its first state and
    step keys on ``SEED`` as the benchmark's loop draws them."""
    from target_gym.compressor_surge import oracle as O

    env, params = _short("compressor_surge", steps, scenario=1)
    resets, keys = episode_keys(jax.random.PRNGKey(SEED), 1, steps)
    state = env.reset_env(resets[0], params)[1]
    return env, params, O, state, keys[:, 0]


def test_the_compressors_ceiling_may_not_plan_left_of_the_surge_line():
    """The line is the machine's own, which the ceiling knows: the cubic's
    peak, Phi = 2W (1 + shift), 0.5 on the map's machine and 0.525 on one
    whose line is 5 % right of its map. A hundredth of Phi left of it the
    surge margin is -1, on it 0, a hundredth right of it +1."""
    env, p, O, state, _ = _compressor()
    row = O.CEILING_ROWS.index("surge")

    def surge(phi, shift):
        at = state.replace(phi_min=jnp.float32(phi), line_shift=jnp.float32(shift))
        return float(O.ceiling_margins(at, p)[0][row])

    for shift, line in ((0.0, 0.5), (0.05, 0.525)):
        assert surge(line - 0.01, shift) == pytest.approx(-1.0, abs=1e-4)
        assert surge(line, shift) == pytest.approx(0.0, abs=1e-4)
        assert surge(line + 0.01, shift) == pytest.approx(1.0, abs=1e-4)


def test_a_move_past_an_actuators_reach_is_charged_its_excess_in_reaches():
    """One step from the reset. A step is 0.1 s: the drive travels 0.03 /s x
    0.1 s = 0.003 of rated, the valve opens 0.5 /s x 0.1 s = 0.05 and closes
    0.1 /s x 0.1 s = 0.01. Half a reach is inside it; three reaches is two
    past it. The margin allows a hundredth of a reach, for float32 rounding
    in where an actuator stops."""
    env, p, O, state, keys = _compressor()
    hold = np.asarray(env.initial_action(state, p), float)
    drive, valve = (O.CEILING_ROWS.index(n) for n in ("drive_reach", "valve_reach"))
    speed = 2.0 * 0.003 / float(p.N_max - p.N_min)  # one reach, raw

    def rows(move):
        after = env.step_env(keys[0], state, jnp.asarray(move, jnp.float32), p)[1]
        m = np.asarray(O.ceiling_margins(after, p)[0], float)
        return m[drive], m[valve]

    assert min(rows(hold)) >= 0.0
    assert min(rows(hold + [0.5 * speed, 2.0 * 0.5 * 0.05])) >= 0.0
    assert min(rows(hold - [0.5 * speed, 2.0 * 0.5 * 0.01])) >= 0.0

    d, v = rows(hold + [3.0 * speed, 0.0])
    assert d == pytest.approx(-2.0, abs=0.02) and v >= 0.0
    d, v = rows(hold - [3.0 * speed, 0.0])
    assert d == pytest.approx(-2.0, abs=0.02) and v >= 0.0
    d, v = rows(hold + [0.0, 2.0 * 3.0 * 0.05])
    assert v == pytest.approx(-2.0, abs=0.02) and d >= 0.0
    d, v = rows(hold - [0.0, 2.0 * 3.0 * 0.01])
    assert v == pytest.approx(-2.0, abs=0.02) and d >= 0.0


def test_a_plan_clipped_into_reach_runs_as_the_plan_does():
    """Sixty moves that jump several reaches a step, the valve never below
    where it starts (so the plant stays right of its line), and the same plan
    through ``ceiling_start``. The header pressure agrees at every step to
    1 Pa: far above float32 rounding in where an actuator stops (1e-7 of its
    travel), far below what one 0.01 s substep of valve travel moves it. And
    the clipped plan is inside reach at every step."""
    env, p, O, state, keys = _compressor()
    hold = np.asarray(env.initial_action(state, p), float)
    rng = np.random.default_rng(0)
    plan = hold + np.stack(
        [rng.uniform(-0.2, 0.2, 60), rng.uniform(0.0, 0.5, 60)], axis=1
    )
    start = O.ceiling_start(env, p, state, plan)
    drive, valve = (O.CEILING_ROWS.index(n) for n in ("drive_reach", "valve_reach"))

    def run(moves):
        def step(s, ka):
            k, a = ka
            s1 = env.step_env(k, s, a, p)[1]
            s1 = jax.tree_util.tree_map(lambda x, y: x.astype(y.dtype), s1, s)
            return s1, (s1.dp, O.ceiling_margins(s1, p)[0])

        return jax.lax.scan(step, state, (keys, jnp.asarray(moves, jnp.float32)))[1]

    dp_plan, margins_plan = run(plan)
    dp_start, margins_start = run(start)
    assert float(np.min(np.asarray(margins_plan)[:, [drive, valve]])) < -1.0
    np.testing.assert_allclose(np.asarray(dp_start), np.asarray(dp_plan), atol=1.0)
    assert float(np.min(np.asarray(margins_start)[:, [drive, valve]])) >= 0.0
