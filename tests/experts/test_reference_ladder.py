"""The ladder script on real plants.

Contract, from docs/proposals/task-charter.md J1 and J4: every rung a task lacks has a
written reason, and the bump tests read the plant. ``first_order`` is a lag whose
datasheet is K = 1 per unit of input and tau = 0.5 s, with the raw action spanning
2 units, so 2 per raw action (env.py). A loop is tuned at every operating point its
row names and kept as tuned where the plant is most sensitive, so adding a less
sensitive point changes nothing, and adding a more sensitive one tunes as if the plant
sat there alone.

Each task's row lives in its own file, ``scripts/ladder_rows/<task>.py``. A rung's cost
on a seed is the task's ``J``: each scenario's mean cost per step, weighted by its share
of operating time. A loop on a zone holds the setpoint ``k = 2`` standard deviations
inside the zone's cheap edge, never past its centre, the deviation measured with the
loop at the centre on the bump-test seeds.
"""

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.medium


@pytest.fixture(scope="module")
def ladder():
    spec = importlib.util.spec_from_file_location(
        "reference_ladder", ROOT / "scripts" / "reference_ladder.py"
    )
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules["reference_ladder"] = module
    spec.loader.exec_module(module)
    return module


def test_every_rung_or_point_a_task_lacks_has_a_reason(ladder):
    for name, make in ladder.ROWS.items():
        row = make()
        present = {
            "pi": bool(row.loops),
            "feedforward": row.balance is not None,
            "cascade": bool(row.cascades),
            "decoupler": row.decouple,
            "override": bool(row.limits),
            # A tuned loop is tested at each point the episode visits, or says why
            # it has one.
            "points": bool(row.points) or not row.loops,
        }
        missing = {rung for rung, there in present.items() if not there}
        assert missing <= set(
            row.why
        ), f"{name}: no reason for {missing - set(row.why)}"


def test_bump_test_reads_the_first_order_datasheet(ladder):
    from target_gym.identification import Instrument, Plant
    from target_gym.registry import REGISTRY

    spec = REGISTRY["first_order"]
    env = spec.make_env()
    row = ladder.ROWS["first_order"]()
    params = spec.make_test_params(
        sensor_noise_scale=1.0, max_steps_in_episode=row.steps
    )
    plant = Plant(
        spec,
        params,
        [ladder.wrap(row.loops[0].read)],
        [(None, None)],
        controller=ladder.manual(row.rest),
    )
    model = ladder.read_loop(
        plant,
        "x",
        0,
        0,
        0.1,
        60,
        Instrument.of(env, params, "x"),
        plant.room(0),
        "fopdt",
    )
    # 5 %: the noise band is about a thirtieth of the step's response.
    assert model["k"] == pytest.approx(2.0, rel=0.05)
    assert model["tau"] == pytest.approx(0.5, rel=0.05)


def test_the_kept_rung_of_the_fixture_trips_nothing(ladder):
    result = ladder.climb("first_order")
    assert result["kept"] in result["rungs"]
    assert result["rungs"][result["kept"]]["trips"] == 0


def first_actions(ladder, row):
    """The pi rung's actions on two fixed observations, for comparing tunings."""
    import numpy as np

    ctrl = ladder.build("first_order", row)["pi"]
    memory, out = ctrl.init_state, []
    for obs in ([1.0, 1.0], [0.95, 1.0]):  # small errors: no clipping
        obs = np.asarray(obs + [0.0] * 8, np.float32)
        action, memory = ctrl.step(ctrl.params, memory, obs, np)
        out.append(float(np.asarray(action)[0]))
    return out


@pytest.mark.parametrize(
    "points, alone",
    [
        # K 1.5 is more sensitive than the datasheet's 1: tune as if only there.
        (({"K": 1.5},), {"K": 1.5}),
        # K 0.5 is less: the datasheet's point keeps the tune.
        (({"K": 0.5},), {}),
    ],
)
def test_tuning_follows_the_most_sensitive_operating_point(ladder, points, alone):
    import dataclasses

    base = ladder.ROWS["first_order"]()
    many = dataclasses.replace(base, points=points)
    single = dataclasses.replace(base, test={**base.test, **alone})
    assert first_actions(ladder, many) == pytest.approx(first_actions(ladder, single))


def test_each_row_lives_in_its_own_file(ladder):
    rows = ROOT / "scripts" / "ladder_rows"
    files = {p.stem for p in rows.glob("*.py") if not p.stem.startswith("_")}
    assert set(ladder.ROWS) == files


def test_a_rungs_cost_weighs_each_scenario_by_its_share(ladder):
    # Weights 3 and 1 are shares 0.75 and 0.25, so a seed's cost run on both
    # scenarios is 0.75 times its cost on the first alone plus 0.25 times its cost
    # on the second alone, each a mean per step.
    import dataclasses

    import numpy as np

    from target_gym.registry import REGISTRY
    from target_gym.scenarios import Scenario

    base = REGISTRY["first_order"]
    normal = Scenario("normal", {}, weight=3.0)
    strong = Scenario("strong", {"K": 2.0}, weight=1.0)

    def on(*suite):
        return dataclasses.replace(base, scenarios=lambda: suite)

    row = ladder.ROWS["first_order"]()
    ctrl = ladder.build("first_order", row)["pi"]
    both = ladder.run_rung("first_order", row, "pi", ctrl, spec=on(normal, strong))
    first = ladder.run_rung("first_order", row, "pi", ctrl, spec=on(normal))
    second = ladder.run_rung("first_order", row, "pi", ctrl, spec=on(strong))
    expected = 0.75 * np.array(first["costs"]) + 0.25 * np.array(second["costs"])
    np.testing.assert_allclose(both["costs"], expected, rtol=1e-6)
    assert set(both["scenarios"]) == {"normal", "strong"}


def test_a_scenarios_cost_is_per_step_whatever_its_length(ladder):
    # A plant with no gain, starting at 0, stays at 0 whatever the loop does, so
    # each step costs the same and a 150-step episode costs per step what a 75-step
    # one does. An episode's total would double.
    import dataclasses

    import numpy as np

    from target_gym.registry import REGISTRY
    from target_gym.scenarios import Scenario

    still = {"K": 0.0, "initial_x_range": (0.0, 0.0)}
    short = Scenario("short", {**still, "max_steps_in_episode": 75})
    long = Scenario("long", {**still, "max_steps_in_episode": 150})
    spec = dataclasses.replace(REGISTRY["first_order"], scenarios=lambda: (short, long))
    row = ladder.ROWS["first_order"]()
    ctrl = ladder.build("first_order", row)["pi"]
    out = ladder.run_rung("first_order", row, "pi", ctrl, spec=spec)["scenarios"]
    np.testing.assert_allclose(out["short"]["costs"], out["long"]["costs"], rtol=1e-3)


def test_a_zone_loop_holds_two_sigma_inside_its_cheap_edge(ladder):
    # Zone 0.5 to 1.5 on x, cheap side high: the setpoint is 1.5 - 2 sigma, and
    # never below the centre 1.0. The pi rung then holds x there, whatever target
    # the observation carries.
    import dataclasses

    import numpy as np

    base = ladder.ROWS["first_order"]()
    x = dataclasses.replace(base.loops[0], zone=(0.5, 1.5, "high"))
    row = dataclasses.replace(base, loops=(x,), balance=None)
    rungs = ladder.build("first_order", row)
    sigma, setpoint = row.placed["x on u"]
    assert sigma > 0
    assert setpoint == pytest.approx(max(1.5 - 2.0 * sigma, 1.0))
    held = ladder.cv_traces("first_order", row, rungs["pi"], 0)
    second_half = np.concatenate([t[len(t) // 2 :] for t in held])
    assert np.mean(second_half) == pytest.approx(setpoint, abs=sigma)
    # sigma is x's own spread under the loop: on this linear plant the loop holds x
    # with the same spread at the setpoint as at the centre (20 %: ten seeds).
    assert np.std(second_half, ddof=1) == pytest.approx(sigma, rel=0.2)


def test_a_zones_sigma_is_read_on_the_scored_normal_day(ladder):
    # A row may quiet its bump tests (here, no sensor noise). The zone's spread is
    # still the normal day's, so it matches the spread of a row whose tests are
    # noisy (20 %: ten seeds), and is not the near-zero of a quiet test.
    import dataclasses

    base = ladder.ROWS["first_order"]()
    x = dataclasses.replace(base.loops[0], zone=(0.5, 1.5, "high"))
    noisy = dataclasses.replace(base, loops=(x,), balance=None)
    quiet = dataclasses.replace(noisy, test={"sensor_noise_scale": 0.0})
    ladder.build("first_order", noisy)
    ladder.build("first_order", quiet)
    assert quiet.placed["x on u"][0] > 0.0
    assert quiet.placed["x on u"][0] == pytest.approx(
        noisy.placed["x on u"][0], rel=0.2
    )


def test_a_paired_bump_cancels_what_both_runs_share(ladder):
    # Read against the same seed's unstepped run, a step of nothing reads nothing,
    # sensor noise included, and the datasheet's lag reads to within 1 %: K = 2
    # per raw action, tau = 0.5 s.
    import numpy as np

    from target_gym.identification import Instrument, Plant
    from target_gym.registry import REGISTRY

    spec = REGISTRY["first_order"]
    env = spec.make_env()
    row = ladder.ROWS["first_order"]()
    params = spec.make_test_params(
        sensor_noise_scale=1.0, max_steps_in_episode=row.steps
    )
    plant = Plant(
        spec,
        params,
        [ladder.wrap(row.loops[0].read)],
        [(None, None)],
        controller=ladder.manual(row.rest),
        paired=True,
    )
    inst = Instrument.of(env, params, "x")
    assert np.all(plant.bump(0, 0, 0, 0.0, 60, inst).dy == 0.0)
    model = ladder.read_loop(plant, "x", 0, 0, 0.1, 60, inst, plant.room(0), "fopdt")
    assert model["k"] == pytest.approx(2.0, rel=0.01)
    assert model["tau"] == pytest.approx(0.5, rel=0.01)


def test_j2_holds_each_episodes_reset_action_as_a_null(ladder):
    # Charter J2: a reference must beat doing nothing, and on a plant that starts at
    # rest (rule 17) doing nothing is holding the actuator where the reset put it.
    # On cstr's normal day with no sensor noise that hold keeps the reactor on its
    # setpoint; the reset's own action is the plant's, not the null's.
    import jax
    import numpy as np

    from target_gym import scenarios
    from target_gym.benchmark import episode_keys, run_policy, task_info
    from target_gym.registry import REGISTRY

    assert "hold reset" in [n.name for n in ladder.null_set("cstr")]
    spec = REGISTRY["cstr"]
    env = spec.make_env()
    (normal,) = [s for s in scenarios.suite(spec) if s.name == "normal"]
    params = scenarios.params_for(spec, normal, sensor_noise_scale=0.0)
    seeds = (100, 101)
    (hold,) = [n for n in ladder.null_set("cstr") if n.name == "hold reset"]
    hold.bind(env, params, seeds)
    task = task_info("cstr", env, params, len(seeds))
    out = run_policy(hold, env, params, task, seeds=seeds, n_steps=40)
    for i, s in enumerate(seeds):
        key = episode_keys(jax.random.PRNGKey(s), 1, 1)[0][0]
        held = np.asarray(env.reset_env(key, params)[1].actuator_held, float)
        np.testing.assert_allclose(out["action"][i], np.tile(held, (40, 1)))
    np.testing.assert_allclose(out["value"], out["target"], atol=1e-3)
