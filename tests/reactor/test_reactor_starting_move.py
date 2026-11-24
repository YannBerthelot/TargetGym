"""The reactor starts its actuator where the reset plant is (owner, 2026-11-24).

Contract. ``base.canonical_reset``: a task with an ``initial_action`` starts its
actuator where the plant is, so the action held after a reset is the plant's own
starting move, and the charter's do-nothing null (``reference_ladder.HoldReset``:
each episode's actuator held where its reset put it) does nothing. The reactor's
actions (``reactor/env.py``, Action): rod demand, -1 fully in and +1 fully out;
boron flow, -1 full dilution and +1 full boration; turbine loading rate, +1 loads
at its fastest, as far as the plan.

By hand: the reset parks the D bank at 225 of its 228 steps (Krsko fig. 3), 3 steps
in, a rod demand of 1 - 2 x 3 / 228 = 0.973684; it opens no boron flow, 0; and the
turbine follows its plan, +1. Held, that move passes no water, $0 of effluent, and
moves no rod, $0 of rod steps; on a core with no burnup the boron stays where it
is.

Before this the action held after a reset was the middle of the box, (0, 0, 0): a
rod demand of half the stroke, the same null as "constant 0.5", which the charter
record ran as the reactor's do-nothing.
"""

import importlib.util
import pathlib
import sys

import jax
import numpy as np
import pytest

from target_gym.reactor import Reactor, ReactorParams

ROOT = pathlib.Path(__file__).resolve().parents[2]
PARKED = 1.0 - 2.0 * 3.0 / 228.0  # the rod demand that leaves the D bank at 225
MOVE = [PARKED, 0.0, 1.0]
QUIET = ReactorParams(
    demand_sigma=0.0,
    sensor_noise_scale=0.0,
    burnup_range=(0.0, 0.0),
    carried_days_range=(0.0, 0.0),
    cycle_range=(0.0, 0.0),
    initial_tilt_share=0.0,
    initial_xenon_power_range=(1.0, 1.0),
    ramp_down_range=(1e6, 1e6),
)


@pytest.mark.parametrize("seed", [0, 7])
def test_a_reset_holds_its_rods_parked_no_boron_flow_and_the_turbine_on_its_plan(seed):
    env = Reactor()
    _, state = env.reset_env(jax.random.PRNGKey(seed), env.default_params)
    np.testing.assert_allclose(np.asarray(state.actuator_held), MOVE, atol=1e-6)


def test_holding_the_resets_move_passes_no_water_and_moves_no_rod():
    env = Reactor()
    _, s0 = env.reset_env(jax.random.PRNGKey(0), QUIET)

    def step(s, _):
        _, s1, _, _, info = env.step_env(
            jax.random.PRNGKey(1), s, s0.actuator_held, QUIET
        )
        return s1, (info["reward_terms"]["running"], info["tripped"])

    two_hours = 240
    s1, (running, tripped) = jax.jit(
        lambda s: jax.lax.scan(step, s, None, length=two_hours)
    )(s0)
    assert not bool(tripped.any())
    # No kilogram of effluent and no rod step. A single-precision demand near 1
    # is held to 3e-8, so the position it asks for is the park to 1.5e-8 of the
    # stroke, 3.4e-6 of a step: under a hundred-thousandth of a dollar.
    assert float(running.sum()) == pytest.approx(0.0, abs=1e-5)
    assert float(s1.rod_in) == pytest.approx(3.0 / 228.0, abs=1e-7)
    # No flow and no burnup: the boron is where the reset left it (it is held to
    # 1.2e-4 ppm near 1690).
    assert float(s1.boron) == pytest.approx(float(s0.boron), abs=2e-4)


@pytest.mark.medium
def test_the_charters_hold_reset_null_is_the_resets_move_not_the_middle_of_the_box():
    spec = importlib.util.spec_from_file_location(
        "reference_ladder", ROOT / "scripts" / "reference_ladder.py"
    )
    ladder = importlib.util.module_from_spec(spec)
    sys.modules["reference_ladder"] = ladder
    spec.loader.exec_module(ladder)
    from target_gym.registry import REGISTRY

    task = REGISTRY["reactor"]
    (hold,) = [n for n in ladder.null_set("reactor") if n.name == "hold reset"]
    hold.bind(task.make_env(), task.make_test_params(), (100, 101))
    held, _ = hold.act(None, None, 0)
    np.testing.assert_allclose(held, np.tile(MOVE, (2, 1)), atol=1e-6)
    assert not np.allclose(held, 0.0)  # (0, 0, 0) is "constant 0.5"
