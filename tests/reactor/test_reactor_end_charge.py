"""The reactor's scored runs charge the boron their core owes at the end (v1 item 2).

Contract, from the owner's answer of 2026-11-28 ("Charge it both ways"), written
before the code:

- the core's stock is the boron it owes: the dilution that would bring Tavg back on
  its program with the rods at their park, at the core's own coefficients where it
  stands in its cycle, both feedbacks, since at a held power the fuel's
  temperature moves with the coolant's. Tavg under its program holds reactivity on loan, which a
  dilution must later supply; rods in below their park hold a reserve, which a
  withdrawal supplies in place of a dilution;
- the line counts: a flow paid for at its injection that has not reached the core
  moves the boron it will move;
- its value is minus the effluent that dilution takes: feed and bleed of pure
  water through the primary's mass, M ln(B / (B - owed)) kg, at the effluent
  price;
- a scored run is charged the value its core lost from the reset to its last step,
  and credited the value it gained, spread over the run's steps;
- so a policy that stops diluting at the run's end, leaving its core owing the
  boron it skipped, no longer gains by it.

By hand, at the start of the cycle (alpha_fuel -3 and alpha_coolant -12 pcm/K,
boron worth -8 pcm/ppm) and 1690 ppm, with M = 2.5e5 kg at $0.0628/kg:

- Tavg 1 K under its program, rods parked: 15 pcm on loan, 1.875 ppm owed,
  2.5e5 ln(1690 / 1688.125) = 277.52 kg of water: worth -$17.428 (the family
  sweep's ``reactor_ladder.owed`` reads the same $17.43, test_reactor_day_end.py);
- Tavg on its program, rods a quarter in: their worth there is
  -0.015 (0.25 - sin(pi / 2) / (2 pi)) = -136.27 pcm against -0.02 pcm parked, so
  -17.031 ppm owed, 2.5e5 ln(1690 / 1707.031) = -2506.7 kg: worth +$157.42;
- on its program, rods parked, with full dilution in the line for the 19 steps
  still to arrive of its 20: each 30 s step of 10 kg/s moves 1690 ppm by
  1690 expm1(-10 x 30 / 2.5e5) = -2.0268 ppm, so -38.509 ppm owed, a reserve worth
  2.5e5 ln(1728.509 / 1690) x 0.0628 = +$353.73, about the 5700 kg already paid
  for at $0.0628 ($357.96).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import scoring_runs as S
from target_gym.reactor.env import ReactorParams
from target_gym.reactor.env_jax import Reactor

ENV = Reactor()
START = ReactorParams(cycle_range=(0.0, 0.0), carried_days_range=(0.0, 0.0))


def _core(tavg_under=0.0, rod_in=None, boron=1690.0):
    """The reset core at the start of its cycle, its Tavg ``tavg_under`` K under its
    program, its rods at ``rod_in`` (parked if None) and its boron at ``boron``."""
    _, state = ENV.reset_env(jax.random.PRNGKey(0), START)
    shift = (
        float(state.T_ref) - 0.5 * float(state.T_coolant + state.T_cold) + tavg_under
    )
    return state.replace(
        T_coolant=state.T_coolant - shift,
        T_cold=state.T_cold - shift,
        rod_in=jnp.float32(START.rod_park if rod_in is None else rod_in),
        boron=jnp.float32(boron),
    )


def test_a_core_on_its_program_with_its_rods_parked_owes_nothing():
    assert float(ENV.stored_value(_core(), START)) == pytest.approx(0.0, abs=1e-3)


def test_a_kelvin_under_the_program_owes_the_dilution_that_lifts_it():
    # Float32 temperatures near 580 K: a few tenths of a mK of Tavg, so 1e-3.
    value = float(ENV.stored_value(_core(tavg_under=1.0), START))
    assert value == pytest.approx(-17.428, rel=1e-3)


def test_rods_in_below_their_park_hold_a_reserve_worth_its_dilution():
    value = float(ENV.stored_value(_core(rod_in=0.25), START))
    assert value == pytest.approx(157.42, rel=1e-4)


def test_boron_on_its_way_counts_as_the_boron_it_will_move():
    state = _core()
    queue = state.boron_queue.at[: int(state.boron_delay) - 1].set(-1.0)
    value = float(ENV.stored_value(state.replace(boron_queue=queue), START))
    assert int(state.boron_delay) == 20
    assert value == pytest.approx(353.73, rel=1e-3)


def test_the_scored_runs_charge_the_core_and_credit_it_both_ways():
    """The reactor's protocol names the plant's stock and credits a gain (the owner's
    "both ways"), where hvac's evaluation only charges a loss."""
    protocol = S.PROTOCOLS["reactor"]
    assert protocol.edge_charge == "stored_value"
    assert protocol.edge_credit is True


def _stops_at_the_end(ctrl, steps, quiet):
    """The reference, with its boron flow shut for the run's last ``quiet`` steps."""

    def act(memory, obs):
        inner, t = memory
        u, inner = ctrl.step(ctrl.params, inner, obs)
        u = jnp.where(t >= steps - quiet, u.at[1].set(0.0), u)
        return u, (inner, t + 1)

    return act, (ctrl.init_state, jnp.int32(0))


@pytest.mark.medium
def test_stopping_dilution_at_the_runs_end_no_longer_pays():
    """A day's baseload run, the reference against the same with its boron flow shut
    for the last 2 h. As scored with no end, the stop saved the reference 0.9 % of
    a baseload day's cost on the 7-day runs (v1 task pages, item 2's probe): the core
    ends owing the boron it skipped. Charged that boron at the effluent its dilution
    takes, the stop saves nothing to speak of: under 0.05 % of the run's cost on
    each seed, where the boron the stop skips is worth some 0.6 % of a day's.
    (What is left is the price's curvature: the charge prices the owed dilution at
    the end's boron, the run paid for its dilutions at theirs.)"""
    from target_gym.registry import REGISTRY

    steps, quiet, seeds = 2880, 240, [5, 6]
    ctrl = REGISTRY["reactor"].make_pid_controller()
    act = lambda m, o: ctrl.step(ctrl.params, m, o)  # noqa: E731
    base = S.run_functional("reactor", "baseload", act, ctrl.init_state, seeds, steps)
    stop, memory = _stops_at_the_end(ctrl, steps, quiet)
    held = S.run_functional("reactor", "baseload", stop, memory, seeds, steps)
    assert base["trips"] == [0, 0] and held["trips"] == [0, 0]
    saved = 1.0 - np.asarray(held["costs"]) / np.asarray(base["costs"])
    assert np.all(saved < 5e-4), saved
