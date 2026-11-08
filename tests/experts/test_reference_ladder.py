"""The ladder script on real plants.

Contract, from docs/proposals/task-charter.md rule 3: every rung a task lacks has a
written reason, and the bump tests read the plant. ``first_order`` is a lag whose
datasheet is K = 1 per unit of input and tau = 0.5 s, with the raw action spanning
2 units, so 2 per raw action (env.py). A loop is tuned at every operating point its
row names and kept as tuned where the plant is most sensitive, so adding a less
sensitive point changes nothing, and adding a more sensitive one tunes as if the plant
sat there alone.
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
