"""The pH task against its plant family (REALITY.md, COLD_REVIEW.md, Amendment 3).

Contract, from the sources in REALITY.md and the owner's Amendment 3, not from
the code:
- mixing and measurement put 7 to 10 s between a reagent move and the reading
  [P7, p. 2]; the apparent dead time on the stage-1 probe lies in 5 to 20 s;
- buffering shifts the titration curve [P1, P2] by at least 0.1 pH somewhere;
- staged: two tanks in series [P3; P5, p. 2; P7], each with a retention time
  of at least 10 min at nominal flow [P4, p. 2];
- no discharge below pH 5.0 [P9]: the plant diverts effluent when the tank is
  too low [P3, pp. 4-5], so every such step is diverted into a finite basin,
  and a basin that overruns is a violation;
- no single excursion outside the permit range over 60 min [P8], diverted or
  not: 60 min or less is no violation, past it is, within one 5 s step;
- inside the permit band the permit pays nothing for precision, and reagent is
  the cost [P5, p. 2; P3, p. 4], so the run on less reagent scores at least as
  well;
- the do-nothing policy (no reagent) is clearly worse than holding the band:
  at least five times the cost of a perfect hold, and it trips (the review's
  finding, Amendment 3);
- influent flow swings, and its meter reads it [P3, Figure 2].
Tolerances are the audit's, or stated where ours, fixed before any run.
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
    """No buffer drift, no acid or flow upsets, no instrument noise."""
    off = dict(
        q2_noise_std=0.0, sensor_noise_scale=0.0, acid_upset_size=0.0, flow_swing=0.0
    )
    return PHParams(**{**off, **kw})


@pytest.fixture(scope="module")
def env():
    return PHNeutralization()


@pytest.fixture(scope="module")
def step(env):
    return jax.jit(env.step_env)


def raw_for(q3, p, trim=0.0):
    """Both valves: the stage-1 reagent valve at ``q3``, the stage-2 trim at
    ``trim`` (mL/s)."""
    return jnp.array(
        [
            2.0 * (q3 - p.q3_min) / (p.q3_max - p.q3_min) - 1.0,
            2.0 * trim / p.q3_trim_max - 1.0,
        ]
    )


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


def settled(env, p, pH_goal, key=0):
    """A reset plant moved to the steady state of ``pH_goal`` in both stages,
    the trim valve shut; returns the state and its stage-1 valve."""
    _, s = env.reset_env(jax.random.PRNGKey(key), p)
    q3 = steady_q3_for(pH_goal, float(s.q2), p)
    Wa, Wb = steady_state_invariants(q3, s.q2, p)
    pH = solve_pH(Wa, Wb, p)
    s = s.replace(Wa=Wa, Wb=Wb, pH=pH, Wa_s1=Wa, Wb_s1=Wb, pH_s1=pH, q3=q3, q3_trim=0.0)
    return s, q3


def test_each_stage_holds_at_least_ten_minutes_of_flow():
    """[P4, p. 2]: retention above 10 min; [P3; P7]: staged."""
    p = PHParams()
    flow = p.q1 + p.q2_nominal + 15.6  # mL/s at the nominal steady state
    assert p.V_s1 / flow >= 600.0 and p.V / flow >= 600.0


def test_a_reagent_move_reaches_the_reading_after_the_mixing_delay(env, step):
    """[P7, p. 2]: about 7 s of mixing delay, 10 s assumed per tank. On the
    stage-1 probe, the first the move reaches."""
    p = quiet()
    s, q3 = settled(env, p, 7.0)
    key = jax.random.PRNGKey(1)
    for _ in range(10):
        obs, s, *_ = step(key, s, raw_for(q3, p), p)
    base = float(obs[4])
    moved_at = None
    for t in range(1, 20):
        obs, s, *_ = step(key, s, raw_for(q3 + 2.0, p), p)
        if abs(float(obs[4]) - base) > 0.005:
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
    """[P9], [P3, pp. 4-5]. Excite: valves shut, no buffer, until the
    discharge settles low."""
    p = quiet(q2_nominal=0.0, continuing=False)
    _, s = env.reset_env(jax.random.PRNGKey(0), p)
    s = s.replace(q2=jnp.asarray(0.0))
    key = jax.random.PRNGKey(2)
    below = flagged = 0
    for _ in range(600):
        _, s, _, done, _ = step(key, s, jnp.array([-1.0, -1.0]), p)
        if bool(done):
            break
        terms = env.reward_terms(s, p)
        low = float(s.pH) < 5.0
        below += low
        flagged += low and float(terms["diversion"]) > 0.0
        if not low:
            assert float(terms["diversion"]) == 0.0
    assert below > 0, "the excitation never took the discharge below pH 5"
    assert flagged == below


def _excursion_run(env, step, p, release_after_s, low=False):
    """From a steady state at pH 7, both valves full open (or, ``low``, shut);
    once the excursion has run ``release_after_s``, the opposite valves until
    the discharge is back in band, then the steady valve.
    Returns per step whether it tripped, and the run of consecutive seconds
    the discharge has been outside the band through that step."""
    p = p.replace(continuing=False)
    s, q3 = settled(env, p, 7.0)
    key = jax.random.PRNGKey(3)
    push = jnp.array([-1.0, -1.0]) if low else jnp.array([1.0, 1.0])
    run, runs, trips = 0.0, [], []
    for _ in range(int(150 * 60 / STEP_S)):
        if run < release_after_s:
            action = push
        elif BAND[0] <= float(s.pH) <= BAND[1]:
            action = raw_for(q3, p)
        else:  # released: the opposite valves until back in band
            action = -push
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
    # Released at 45 min: two 10 min stages take minutes to bring the
    # discharge back, so the excursion ends between 45 and 60 min.
    runs, trips = _excursion_run(env, step, quiet(), release_after_s=45 * 60.0)
    assert runs.max() >= 45 * 60.0, "the excitation did not approach the limit"
    assert runs.max() <= EXCURSION_LIMIT_S
    assert not trips.any()


def test_an_excursion_past_an_hour_is_a_violation_within_one_step(env, step):
    """[P8]: no single excursion over 60 min."""
    runs, trips = _excursion_run(env, step, quiet(), release_after_s=np.inf)
    assert trips.any(), "a long excursion never tripped"
    at = int(np.argmax(trips))
    assert EXCURSION_LIMIT_S - STEP_S < runs[at] <= EXCURSION_LIMIT_S + STEP_S
    assert not trips[:at].any()


def test_diverting_does_not_stop_the_hour(env, step):
    """Amendment 3: the 60 min clock keeps running while effluent is diverted,
    so a discharge held below pH 5 still trips past the hour, or sooner if its
    basin overruns."""
    runs, trips = _excursion_run(
        env, step, quiet(q2_nominal=0.0), release_after_s=np.inf, low=True
    )
    assert trips.any(), "a discharge held below pH 5 never tripped"
    assert runs[int(np.argmax(trips))] <= EXCURSION_LIMIT_S + STEP_S


def test_an_overrun_basin_is_a_violation(env, step):
    """Amendment 3: the diversion basin is finite, and overrunning it is a
    violation. A basin of a quarter-hour's flow overruns before the hour."""
    p = quiet(q2_nominal=0.0, basin_volume=15.0 * 60 * 35.0 / 1000.0)
    runs, trips = _excursion_run(env, step, p, release_after_s=np.inf, low=True)
    assert trips.any()
    assert runs[int(np.argmax(trips))] < EXCURSION_LIMIT_S - 10 * 60.0


def test_inside_the_band_less_reagent_scores_at_least_as_well(env, step):
    """[P5, p. 2]: overdosing is costly; [P3, p. 4]: set points nearer the
    limit cut reagent. Two constant valves, settling at pH 6.5 and 7.0."""
    p = quiet()
    key = jax.random.PRNGKey(4)

    def episode_return(pH_goal):
        s, q3 = settled(env, p, pH_goal)
        total, lo, hi = 0.0, np.inf, -np.inf
        for _ in range(240):
            _, s, r, _, _ = step(key, s, raw_for(q3, p), p)
            total += float(r)
            lo, hi = min(lo, float(s.pH)), max(hi, float(s.pH))
        assert BAND[0] <= lo and hi <= BAND[1]
        return total

    assert episode_return(6.5) >= episode_return(7.0)


def test_adding_no_reagent_is_clearly_worse_than_holding_the_band(env, step):
    """The review measured zero reagent within 1.1 to 1.4 times a perfect hold.
    The fixed task must make it clearly worse: at least five times (ours) a
    hold at pH 7 on the steady valve, over the default episode, and it
    trips."""
    p = quiet()
    key = jax.random.PRNGKey(5)

    def episode(policy):
        s, q3 = settled(env, p, 7.0)
        total, tripped = 0.0, False
        for _ in range(int(p.max_steps_in_episode)):
            _, s, r, _, info = step(key, s, policy(q3), p)
            total += float(r)
            tripped = tripped or bool(info["tripped"])
        return -total, tripped

    hold, hold_trip = episode(lambda q3: raw_for(q3, p))
    none, none_trip = episode(lambda q3: jnp.array([-1.0, -1.0]))
    assert not hold_trip and none_trip
    assert none >= 5.0 * hold


def test_influent_flow_swings_are_metered_and_move_the_plant(env, step):
    """Amendment 3: influent flow swings; [P3, Figure 2]: the waste stream has
    a flow transmitter. With swings on, the metered flow leaves nominal and
    the discharge pH moves at a constant valve."""
    p = quiet(flow_swing=0.2)
    s, q3 = settled(env, p, 7.0)
    s = s.replace(
        flow_times=jnp.array([10, 10**6]), flow_levels=jnp.array([1.0, 1.2, 1.2])
    )
    key = jax.random.PRNGKey(6)
    flows, pHs = [], []
    for _ in range(400):
        obs, s, *_ = step(key, s, raw_for(q3, p), p)
        flows.append(float(obs[6]))
        pHs.append(float(s.pH))
    assert flows[-1] == pytest.approx(1.2 * p.q1, rel=1e-3)
    assert abs(pHs[-1] - 7.0) > 0.1
