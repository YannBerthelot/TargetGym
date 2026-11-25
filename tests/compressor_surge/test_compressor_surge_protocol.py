"""The compressor's evaluation protocol (owner's rulings E, F and A, 2026-11-24).

Contract, from the owner's picker answers relayed by the coordinating session,
not from the code:

E. The suite has three scenarios: normal, slow_approach and fast_approach.
   load_swings is dropped and its weight (0.15) goes 43 % to slow approach and
   57 % to normal, the mix the reference prices it like. So normal weighs
   0.7 + 0.57 x 0.15 = 0.7855, slow approach 0.1 + 0.43 x 0.15 = 0.1645, fast
   approach keeps 0.05.
F. The protocol's seeds spread the machines: seed s gets a machine in the tenth
   number s mod 10 of the shift range [-line_shift_max, +line_shift_max], so
   any ten consecutive seeds hold one machine per tenth. A plain reset, as a
   user or a training run calls it, still draws the machine uniformly over the
   whole range.
A. The ceiling is converged when its two starts (the reference's moves and the
   hold) agree within 1 % on J and the kept plan's J is within 3 % above the
   balance floor. The floor is the quasi-steady recycle energy for the
   episode's demand: the compressor on its machine's own surge line at the
   speed whose head there is the header pressure, the cheapest pressure in the
   zone.
"""

import jax
import numpy as np
import pytest

from target_gym.benchmark import run_policy, task_info
from target_gym.compressor_surge.env import CompressorSurgeParams
from target_gym.compressor_surge.env_jax import CompressorSurge
from target_gym.compressor_surge.oracle import CEILING_STARTS, balance_floor
from target_gym.pc_gym.foresight import ceiling_converged
from target_gym.registry import REGISTRY
from target_gym.scenarios import suite, weights


def test_the_suite_is_normal_slow_and_fast_with_load_swings_folded_in():
    spec = REGISTRY["compressor_surge"]
    assert [s.name for s in suite(spec)] == ["normal", "slow_approach", "fast_approach"]
    w = weights(spec)
    assert w["normal"] == pytest.approx(0.7855, abs=1e-12)
    assert w["slow_approach"] == pytest.approx(0.1645, abs=1e-12)
    assert w["fast_approach"] == pytest.approx(0.05, abs=1e-12)


class _Hold:
    """Holds the action at zero: the test reads only the reset's machine."""

    name = "hold"

    def reset(self, key, obs, task):
        return task.n_act

    def act(self, carry, obs, t):
        return np.zeros((obs.shape[0], carry), np.float32), carry


def _protocol_machines(seeds):
    spec = REGISTRY["compressor_surge"]
    env, p = spec.make_env(), spec.make_test_params()
    out = run_policy(
        _Hold(), env, p, task_info(spec.name, env, p, len(seeds)), n_steps=1,
        seeds=list(seeds), record={"shift": lambda s, q, tripped: s.line_shift},
    )  # fmt: skip
    return np.asarray(out["records"]["shift"])[:, 0], float(p.line_shift_max)


@pytest.mark.parametrize("first", [100, 200])
def test_ten_consecutive_protocol_seeds_hold_one_machine_per_tenth(first):
    seeds = range(first, first + 10)
    shift, top = _protocol_machines(seeds)
    tenth = np.floor((shift / top + 1.0) / 2.0 * 10.0).astype(int)
    np.testing.assert_array_equal(tenth, [s % 10 for s in seeds])


def test_the_placement_and_held_out_seeds_get_different_machines():
    """Same tenths, but the place inside each is drawn from the seed's key."""
    a, _ = _protocol_machines(range(100, 110))
    b, _ = _protocol_machines(range(200, 210))
    assert np.all(a != b)


def test_a_plain_reset_still_draws_the_machine_over_the_whole_range():
    """512 plain resets with the task's parameters: both sides of the range,
    beyond 4 % either way (uniform: P(none beyond 0.04 of 0.05 on one side) =
    0.9^512, nil)."""
    p = REGISTRY["compressor_surge"].make_test_params()
    keys = jax.random.split(jax.random.PRNGKey(0), 512)
    env = CompressorSurge()
    shift = np.asarray(jax.vmap(lambda k: env.reset_env(k, p)[1].line_shift)(keys))
    assert shift.min() < -0.04 and shift.max() > 0.04


def test_the_ceiling_starts_from_the_reference_moves_and_the_hold():
    assert set(CEILING_STARTS) == {"reference", "hold"}


@pytest.mark.parametrize(
    "starts, floor, converged",
    [
        # Agree to 0.31 %, 2.2 % above the floor (the 2026-11-24 numbers).
        ({"reference": 1.954, "hold": 1.960}, 1.912, True),
        # Agree to 1.8 %: not converged, however near the floor.
        ({"reference": 1.954, "hold": 1.990}, 1.912, False),
        # Agree exactly but 4.6 % above the floor.
        ({"reference": 2.000, "hold": 2.000}, 1.912, False),
        # Near the edges: 0.99 % apart and 2.9 % above the floor converge;
        # 1.02 % apart, or 3.1 % above the floor, do not.
        ({"reference": 1.029, "hold": 1.0392}, 1.0, True),
        ({"reference": 1.029, "hold": 1.0395}, 1.0, False),
        ({"reference": 1.031, "hold": 1.031}, 1.0, False),
        # Below the floor (the plenum's storage credit): converged.
        ({"reference": 0.99, "hold": 0.995}, 1.0, True),
    ],
)
def test_the_stop_rule_needs_both_starts_to_agree_and_sit_near_the_floor(
    starts, floor, converged
):
    assert ceiling_converged(starts, floor) is converged


def _floor_inputs():
    p = CompressorSurgeParams()
    opening = np.linspace(0.2, 0.95, 50)
    return opening, p


def test_the_floor_is_zero_when_the_consumers_take_more_than_the_line():
    """Fully open consumers take more than the compressor's surge flow at the
    header pressure, so no recycle is needed."""
    _, p = _floor_inputs()
    assert balance_floor(np.full(20, 1.0), p, shift=0.0) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("shift", [-0.05, 0.05])
def test_with_the_consumers_shut_the_floor_scales_with_the_line_s_flow(shift):
    """Consumers shut: all of the line's flow is recycled. Every machine's line
    has the map's peak head (contract 1 of the hidden line), so at a given
    pressure the speed is the same and the flow, hence the floor, scales as
    Phi = 2W (1 + shift): the ratio to the map's machine is 1 + shift."""
    _, p = _floor_inputs()
    shut = np.zeros(20)
    ratio = balance_floor(shut, p, shift=shift) / balance_floor(shut, p, shift=0.0)
    assert ratio == pytest.approx(1.0 + shift, rel=1e-4)


def test_a_machine_whose_line_sits_right_needs_more_recycle():
    """A line further right means more minimum flow at the same head, so more
    of it recycled at low demand: the floor rises with the shift."""
    opening, p = _floor_inputs()
    floors = [balance_floor(opening, p, shift=s) for s in (-0.05, 0.0, 0.05)]
    assert 0.0 < floors[0] < floors[1] < floors[2]
