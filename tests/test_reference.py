"""The charter's reference as shipped data: ``tg.make_reference``.

Contract (charter J1; outside review C1, 2026-11-27; owner's picker the same day,
"Add make_reference + score"): each scored task's kept rung is frozen as package
data, ``data/reference/<task>.json``, by ``scripts/reference_ladder.py --freeze``.
``tg.make_reference(name)`` builds it from that file with no bump test, and it
steps as ``tg.make_expert``'s expert does. It refuses a task with no frozen
reference and a file frozen on another plant. On the scoring seeds it costs, seed
by seed, what the ladder's own kept rung cost there, which the export measured
when it froze it. The reactor's reference is its shipped expert
(``charter.EXPERT_REFERENCE``).

The tests marked ``records`` read the shipped files against the tree, as the
recorded rows' tests do, so they fail on a plant changed since the last freeze.
"""

from __future__ import annotations

import importlib.util
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import target_gym as tg
from target_gym import charter, reference

LADDER = [t for t in charter.FLAGSHIP if t not in charter.EXPERT_REFERENCE]


def test_every_scored_task_ships_a_frozen_reference():
    for name in charter.FLAGSHIP:
        frozen = reference.load(name)
        assert frozen is not None, f"{name}: no data/reference/{name}.json"
        assert frozen["task"] == name
        kind = "expert" if name in charter.EXPERT_REFERENCE else "ladder"
        assert frozen["kind"] == kind


def test_a_task_with_no_frozen_reference_is_refused_with_those_that_have_one():
    with pytest.raises(KeyError, match="cstr"):
        tg.make_reference("plane")


def test_an_unregistered_task_is_refused():
    with pytest.raises(KeyError, match="registered"):
        tg.make_reference("plnae")


@pytest.mark.parametrize("name", ["cstr", "reactor"])
def test_a_reference_frozen_on_another_plant_is_refused(name, tmp_path):
    frozen = reference.load(name)
    frozen["stamp"] = "0" * 16
    reference.write(name, frozen, tmp_path)
    with pytest.raises(reference.StaleReference, match="--freeze"):
        reference.build(name, tmp_path)


@pytest.mark.records
@pytest.mark.parametrize("name", charter.FLAGSHIP)
def test_each_frozen_reference_is_current_and_keeps_the_records_rung(name):
    frozen = reference.load(name)
    now = charter.current(name)
    assert frozen["stamp"] == now["version"]["stamp"], "frozen on another plant"
    assert frozen["fingerprint"] == now["fingerprints"]["reference"]
    assert frozen["rung"] == charter.load(name)["reference"]["kept"]


@pytest.mark.records
def test_the_reactors_reference_is_its_shipped_expert():
    ref = tg.make_reference("reactor")
    expert = tg.make_expert("reactor")
    assert ref.rung == "expert"
    env, params = tg.make("reactor")
    obs, _ = env.reset(jax.random.PRNGKey(0), params)
    a, _ = ref.step(ref.reset(), obs)
    b, _ = expert.step(expert.reset(), obs)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.fixture(scope="module")
def ladder():
    if "reference_ladder" in sys.modules:
        return sys.modules["reference_ladder"]
    spec = importlib.util.spec_from_file_location(
        "reference_ladder", charter.SCRIPTS / "reference_ladder.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["reference_ladder"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.records
@pytest.mark.parametrize("name", LADDER)
def test_make_reference_costs_what_the_ladders_rung_cost_on_the_scoring_seeds(
    name, ladder
):
    # The oracle is the export's run of the ladder's own kept rung on the scoring
    # seeds. Two of them, in every scenario: one is enough to tell another
    # controller, and every scenario reads its own parameters. The same run on
    # another machine's CPU agrees to 3e-7 of a cost or better (the outside
    # review's determinism check, 2026-11-27), hence the 1e-5.
    frozen = reference.load(name)
    controller, _ = reference.build(name)
    seeds = tuple(frozen["anchor"]["seeds"][:2])
    run = ladder.run_suite(
        name,
        ladder.Row(0, None, score=reference.changes(frozen)),
        ladder.Reference(frozen["rung"], controller),
        seeds=seeds,
    )
    for scenario, costs in frozen["anchor"]["scenarios"].items():
        np.testing.assert_allclose(
            run["scenarios"][scenario]["costs"], costs[:2], rtol=1e-5, err_msg=scenario
        )


@pytest.mark.records
@pytest.mark.parametrize("name", LADDER)
def test_a_reference_steps_like_an_expert_under_jit_and_vmap(name):
    ref = tg.make_reference(name)
    env, params = tg.make(name)[0], ref.params()
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    obs, _ = jax.vmap(env.reset, in_axes=(0, None))(keys, params)
    batched, _ = jax.vmap(ref.step)(ref.reset(3), obs)
    one, _ = ref.step(ref.reset(), obs[1])
    np.testing.assert_allclose(np.asarray(batched[1]), np.asarray(one), rtol=1e-6)
    low, high = env.action_space(params).low, env.action_space(params).high
    assert bool(jnp.all((batched >= low) & (batched <= high)))


def _row(**changes):
    """A one-loop row as a ladder row reads: obs 0 held toward obs 1 by MV 0, its
    feedforward 0.1 plus a tenth of obs 2."""
    from types import SimpleNamespace

    loop = SimpleNamespace(
        name="x on u", read=lambda o, xp: o[0], target=lambda o, xp: o[1]
    )
    fields = dict(
        loops=(loop,),
        cascades=(),
        limits=(),
        valves=(),
        balance=lambda o, xp: xp.reshape(0.1 + 0.1 * o[2], (1,)),
        rest=None,
        placed={},
    )
    return SimpleNamespace(**{**fields, **changes})


LAW = {
    "rung": "pi",
    "loops": [{"name": "x on u", "kc": 2.0, "ti": 4.0, "alpha": 1.0}],
    "mix": [[1.0]],
    "bias": [0.1],
    "setpoints": {},
    "start": False,
    "dt": 1.0,
}


@pytest.mark.parametrize(
    "law, want",
    # One step of the ladder's PI from rest (experts/ladder.py): the reading is
    # 0.2, so e = 0.5 - 0.2 = 0.3, the integral e * dt = 0.3, and the action
    # 0.1 + 2 * (0.3 + 0.3 / 4) = 0.85. A placed setpoint of 0.6 replaces the
    # target: e = 0.4, 0.1 + 2 * (0.4 + 0.4 / 4) = 1.1, held at the actuator's 1.
    # The row's balance at obs 2 = 2 is 0.1 + 0.2 = 0.3, and with kc 1 the PI adds
    # 0.3 + 0.3 / 4: 0.675.
    [
        (LAW, 0.85),
        ({**LAW, "setpoints": {"x on u": 0.6}}, 1.0),
        ({**LAW, "bias": "balance", "loops": [{**LAW["loops"][0], "kc": 1.0}]}, 0.675),
    ],
)
def test_a_law_on_its_row_steps_as_the_ladders_pi(law, want):
    controller = reference.assemble(_row(), law)
    obs = jnp.array([0.2, 0.5, 2.0], jnp.float32)
    action, _ = controller.step(None, controller.init_state, obs, jnp)
    np.testing.assert_allclose(np.asarray(action), [want], rtol=1e-6)


def test_a_bound_setpoint_shows_to_the_row_while_its_rung_steps_alone():
    # A feedforward that reads the placed setpoint, as pH's does: inside the step it
    # sees this rung's 0.6, and the row shows nothing once the step is done.
    seen = []
    row = _row()

    def balance(o, xp):
        seen.append(row.placed.get("x on u"))
        return xp.zeros((1,), xp.float32)

    row.balance = balance
    law = {**LAW, "bias": "balance", "setpoints": {"x on u": 0.6}, "bound": ["x on u"]}
    controller = reference.assemble(row, law)
    controller.step(None, controller.init_state, jnp.zeros(3, jnp.float32), jnp)
    assert seen == [(None, 0.6)]
    assert row.placed == {}


def test_a_law_naming_other_loops_than_its_rows_is_refused():
    other = {**LAW, "loops": LAW["loops"] * 2, "mix": [[1.0, 1.0]]}
    with pytest.raises(ValueError, match="row changed"):
        reference.assemble(_row(), other)
