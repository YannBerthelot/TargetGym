"""The pH task against its plant record (REALITY.md, COLD_REVIEW.md tests 1 to 5).

Contract, from the sources in REALITY.md and not from the code:
- mixing and measurement put 7 to 10 s between a reagent move and the reading
  [P7, p. 2]; the apparent dead time lies in 5 to 20 s;
- buffering shifts the titration curve [P1, P2] by at least 0.1 pH somewhere;
- no discharge below pH 5.0 [P9]: the plant diverts effluent when the tank is
  too low [P3, pp. 4-5], so every such step is flagged;
- no single excursion outside the permit range over 60 min [P8]: 60 min or
  less is no violation, past it is, within one 5 s step;
- inside the permit band the permit pays nothing for precision, and reagent is
  the cost [P5, p. 2; P3, p. 4], so the run on less reagent scores at least as
  well.
Tolerances are the audit's, fixed before any run.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.pc_gym.ph_neutralization.env import (
    PHParams,
    solve_pH,
    steady_state_invariants,
)
from target_gym.pc_gym.ph_neutralization.env_jax import PHNeutralization

STEP_S = 5.0
#: [P8]: no single excursion over 60 minutes.
EXCURSION_LIMIT_S = 3600.0
#: [P7, p. 2]: the example product spec, pH 7 plus or minus 1.
BAND = (6.0, 8.0)


def quiet(**kw):
    """No buffer drift, no acid upsets, no instrument noise."""
    return PHParams(q2_noise_std=0.0, sensor_noise_scale=0.0, acid_upset_size=0.0, **kw)


@pytest.fixture(scope="module")
def env():
    return PHNeutralization()


@pytest.fixture(scope="module")
def step(env):
    return jax.jit(env.step_env)


def raw_for(q3, p):
    return jnp.array([2.0 * (q3 - p.q3_min) / (p.q3_max - p.q3_min) - 1.0])


def steady_q3_for(pH_goal, q2, p):
    """Base flow whose steady state sits at ``pH_goal``, by bisection on the
    public steady-state functions (pH rises with base flow)."""
    lo, hi = float(p.q3_min), float(p.q3_max)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(solve_pH(*steady_state_invariants(mid, q2, p), p)) < pH_goal:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_a_reagent_move_reaches_the_reading_after_the_mixing_delay(env, step):
    """[P7, p. 2]: about 7 s of mixing delay, 10 s assumed per tank."""
    p = quiet()
    _, s = env.reset_env(jax.random.PRNGKey(0), p)
    q3 = steady_q3_for(7.0, float(s.q2), p)
    s = s.replace(**dict(zip(("Wa", "Wb"), steady_state_invariants(q3, s.q2, p))))
    s = s.replace(pH=solve_pH(s.Wa, s.Wb, p), q3=q3)
    key = jax.random.PRNGKey(1)
    # A fresh history at the steady state, then a step of +0.5 mL/s.
    for _ in range(10):
        obs, s, *_ = step(key, s, raw_for(q3, p), p)
    base = float(obs[0])
    moved_at = None
    for t in range(1, 20):
        obs, s, *_ = step(key, s, raw_for(q3 + 0.5, p), p)
        if abs(float(obs[0]) - base) > 0.005:
            moved_at = t * STEP_S
            break
    assert moved_at is not None and 5.0 <= moved_at <= 20.0, moved_at


def test_buffering_shifts_the_titration_curve_by_a_readable_amount():
    """[P1]; [P2]: buffer changes shift the curve. At least 0.1 pH somewhere,
    ten times the 0.01 pH resolution."""
    p = quiet()
    q3 = jnp.linspace(1.0, float(p.q3_max), 200)
    bare = jax.vmap(lambda q: solve_pH(*steady_state_invariants(q, 0.0, p), p))(q3)
    buffered = jax.vmap(
        lambda q: solve_pH(*steady_state_invariants(q, p.q2_nominal, p), p)
    )(q3)
    assert float(jnp.max(jnp.abs(buffered - bare))) >= 0.1


def test_every_step_below_the_sewer_floor_is_diverted(env, step):
    """[P9]: no discharge below pH 5.0; [P3, pp. 4-5]: effluent is diverted on
    low tank pH. Excite: valve shut, no buffer, until the tank settles low."""
    p = quiet().replace(q2_nominal=0.0)
    _, s = env.reset_env(jax.random.PRNGKey(0), p)
    s = s.replace(q2=jnp.asarray(0.0))
    key = jax.random.PRNGKey(2)
    below = flagged = 0
    for _ in range(240):
        _, s, _, _, info = step(key, s, jnp.array([-1.0]), p)
        terms = env.reward_terms(s, p)
        low = float(s.pH) < 5.0
        below += low
        flagged += low and float(terms["diversion"]) > 0.0
        # A step in range is never charged as diverted.
        if not low:
            assert float(terms["diversion"]) == 0.0
    assert below > 0, "the excitation never took the tank below pH 5"
    assert flagged == below


def _excursion_run(env, step, p, release_after_s):
    """Valve full open from a steady state at pH 7, back to the steady valve
    once the excursion has run ``release_after_s``. With ``continuing`` off the
    returned state is the plant's own, so a trip is read on the state that
    made it. Returns per step whether it tripped, and the run of consecutive
    seconds outside the band through that step."""
    p = p.replace(continuing=False)
    _, s = env.reset_env(jax.random.PRNGKey(0), p)
    q3 = steady_q3_for(7.0, float(s.q2), p)
    Wa, Wb = steady_state_invariants(q3, s.q2, p)
    s = s.replace(Wa=Wa, Wb=Wb, pH=solve_pH(Wa, Wb, p), q3=q3)
    key = jax.random.PRNGKey(3)
    run, runs, trips = 0.0, [], []
    for _ in range(int(80 * 60 / STEP_S)):
        action = raw_for(q3, p) if run >= release_after_s else jnp.array([1.0])
        _, s, _, _, info = step(key, s, action, p)
        pH = float(s.pH)
        outside = not (BAND[0] <= pH <= BAND[1])
        run = run + STEP_S if outside else 0.0
        runs.append(run)
        trips.append(bool(info["tripped"]))
        if trips[-1]:
            break
    return np.array(runs), np.array(trips)


def test_an_excursion_of_under_an_hour_is_no_violation(env, step):
    """[P8]: 60 min or less outside the permit range is allowed."""
    p = quiet()
    runs, trips = _excursion_run(env, step, p, release_after_s=56 * 60.0)
    assert runs.max() >= 55 * 60.0, "the excitation did not approach the limit"
    assert runs.max() <= EXCURSION_LIMIT_S
    assert not trips.any()


def test_an_excursion_past_an_hour_is_a_violation_within_one_step(env, step):
    """[P8]: no single excursion over 60 min. The trip fires on the step the
    run first passes 60 min, within one 5 s step."""
    p = quiet()
    runs, trips = _excursion_run(env, step, p, release_after_s=np.inf)
    assert trips.any(), "a 70 min excursion never tripped"
    at = int(np.argmax(trips))
    # The step that tripped is past the limit, the one before is not.
    assert runs[at] > EXCURSION_LIMIT_S - STEP_S
    assert runs[at] <= EXCURSION_LIMIT_S + STEP_S
    assert not trips[:at].any()


def test_inside_the_band_less_reagent_scores_at_least_as_well(env, step):
    """[P5, p. 2]: overdosing is costly; [P3, p. 4]: set points nearer the
    limit cut reagent. Two constant valves at nominal buffer, settling at pH
    6.5 and 7.0, both inside the band."""
    p = quiet()
    _, s0 = env.reset_env(jax.random.PRNGKey(0), p)
    q2 = float(s0.q2)
    key = jax.random.PRNGKey(4)

    def episode_return(pH_goal):
        q3 = steady_q3_for(pH_goal, q2, p)
        Wa, Wb = steady_state_invariants(q3, q2, p)
        s = s0.replace(Wa=Wa, Wb=Wb, pH=solve_pH(Wa, Wb, p), q3=q3)
        total, lo, hi = 0.0, np.inf, -np.inf
        for _ in range(240):
            _, s, r, _, _ = step(key, s, raw_for(q3, p), p)
            total += float(r)
            lo, hi = min(lo, float(s.pH)), max(hi, float(s.pH))
        assert BAND[0] <= lo and hi <= BAND[1]
        return total

    assert episode_return(6.5) >= episode_return(7.0)
