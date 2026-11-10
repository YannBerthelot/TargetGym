"""The formation hold: the expert of both patrol tasks and the formation.

Flight-tested formation autopilots (AFIT's, NASA's F/A-18 Autonomous
Formation Flight) close the slot error in the lead's frame, along track on
speed, across it on course and vertically on height, over the aircraft's own
autopilot. These tests hold the shipped experts to what those trials
achieved, on held-out seeds their tuning never saw.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import target_gym as tg
from target_gym.eval import scored_burn_in
from target_gym.formation.env import slot_errors
from target_gym.patrol.env import slot_error, wrap_angle

HELD_OUT = tuple(range(200, 210))

# The contract, set before measuring (docs/proposals/scoring-and-experts.md,
# limits agreed with the user 2026-10-27):
# - every wingman's RMS 3D slot error after the scored burn-in is at most
#   10 ft (3.05 m), NASA's AFF requirement for two F/A-18s, one sigma
#   (NASA/TM-2002-210729). The tasks fly Hornets since 2026-11-10 (owner), so
#   AFF's number is the task's own; it used to be scaled by span to the A320's,
#   8.5 m;
# - the range-and-bearing follower is held to the same: what its estimator costs
#   is what that task measures;
# - the formation's lead holds the 3D heading task's spec
#   (tests/plane3d/test_plane3d_guidance.py): altitude within 2 m and
#   heading within the AHRS resolution (``heading_precision_floor``,
#   0.5 deg), both mean after the burn-in;
# - no trip.
SLOT_RMS_M = 10.0 * 0.3048
ALTITUDE_M = 2.0


def _finding(reason):
    """A held-out case the expert misses, kept as an expected failure as
    agreed before measuring and reported to the user 2026-10-28. Strict, so
    a fix shows up as XPASS and the mark has to come off."""
    return pytest.mark.xfail(strict=True, reason=f"finding: {reason}")


_STALE = _finding("the wingmen hold a median 7.7 m, one seed 428 m (2026-11-10)")
_STALE_TEAM = _finding("PR 9: the wingmen's trips restart the lead, and TECS is stale")
#: Measured 2026-11-10 on the Hornet's direct hold: a median 3.6 m, 5.8 m at worst.
_NEAR = _finding("the Hornet's hold misses 10 ft on these seeds, by up to 2.7 m")
_SENSOR = _finding("the estimator's hold misses 10 ft: a median 6.2 m, 4.1 m at best")
#: Held-out seeds the shipped experts hold (XPASS on 2026-11-10, marks removed).
PATROL_HOLDS = (200, 201, 202, 209)
LEAD_HOLDS = (200,)


@functools.cache
def _setup(name):
    env, params = tg.make(name)
    if name == "formation":

        def measure(state):
            lead = state.lead
            return (
                slot_errors(state),
                jnp.abs(lead.target_altitude - lead.z),
                jnp.abs(wrap_angle(lead.psi - lead.target_heading)),
            )

    else:

        def measure(state):
            return slot_error(state)[None], jnp.zeros(()), jnp.zeros(())

    return env, params, jax.jit(env.step_env), jax.jit(measure)


@functools.cache
def _episode(name, seed):
    """One held-out episode of the shipped expert on the scored configuration,
    after the scored burn-in: ``(tripped, RMS slot error per wingman, the
    lead's mean altitude error, its mean heading error)``."""
    env, params, step, measure = _setup(name)
    expert = tg.make_expert(name).host()
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset_env(key, params)
    tripped, rows = False, []
    for _ in range(int(params.max_steps_in_episode)):
        key, sub = jax.random.split(key)
        obs, state, _, _, info = step(sub, state, expert(np.asarray(obs)), params)
        tripped = tripped or bool(info["tripped"])
        rows.append([np.asarray(x) for x in measure(state)])
    slot, alt, hdg = (np.array(x) for x in zip(*rows))
    b = scored_burn_in(name, params)
    rms = np.sqrt((slot[b:] ** 2).mean(axis=0))
    return tripped, rms, alt[b:].mean(), hdg[b:].mean()


@pytest.mark.large
@pytest.mark.parametrize(
    "name, seed",
    [pytest.param("patrol", s, marks=()) for s in PATROL_HOLDS]
    + [
        pytest.param("patrol", s, marks=_NEAR)
        for s in HELD_OUT
        if s not in PATROL_HOLDS
    ]
    + [pytest.param("patrol_range_bearing", s, marks=_SENSOR) for s in HELD_OUT],
)
def test_the_follower_holds_its_slot(name, seed):
    tripped, rms, _, _ = _episode(name, seed)
    assert not tripped
    assert rms.max() <= SLOT_RMS_M, f"RMS slot error {rms.max():.2f} m"


@pytest.mark.large
@pytest.mark.parametrize("seed", [pytest.param(s, marks=_STALE) for s in HELD_OUT])
def test_every_formation_wingman_holds_its_slot(seed):
    tripped, rms, _, _ = _episode("formation", seed)
    assert not tripped
    assert rms.max() <= SLOT_RMS_M, f"RMS slot errors {np.round(rms, 2)} m"


@pytest.mark.large
@pytest.mark.parametrize(
    "seed",
    [pytest.param(s, marks=() if s in LEAD_HOLDS else _STALE_TEAM) for s in HELD_OUT],
)
def test_the_formation_lead_holds_altitude_and_heading(seed):
    _, _, alt, hdg = _episode("formation", seed)
    floor = _setup("formation")[1].heading_precision_floor
    assert alt <= ALTITUDE_M, f"mean altitude error {alt:.2f} m"
    assert hdg <= floor, f"mean heading error {hdg:.4f} rad"
