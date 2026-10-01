"""The reach-and-hold protocol on synthetic cost series: the renewal
decomposition of Theorem 4, what the reach columns mean, and the split of each
target cycle into settle, hold and the anticipation of the next change."""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

from target_gym import eval as E

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _episode(cost, target_change, terms=None):
    n = len(cost)
    return E.Episode(
        cost=np.asarray(cost, float),
        in_band=np.ones(n, bool),
        target_change=np.asarray(target_change, bool),
        failed=np.zeros(n, bool),
        terms=terms or {},
    )


def test_gain_is_hold_plus_change_rate_times_reach_cost():
    """Constant hold cost 0.1, an excursion of +1 for 5 steps after each target
    change at rate p = 0.01: gain ~ 0.1 + p * 5, hold ~ 0.1, reach ~ 5."""
    rng = np.random.default_rng(0)
    T, p = 20000, 0.01
    tc = rng.random(T) < p
    cost = np.full(T, 0.1)
    for i in np.flatnonzero(tc):
        cost[i : i + 5] += 1.0
    m = E.evaluate([_episode(cost, tc)], burn_in=100, rho_floor=0.1, rho_ref=0.2)
    assert m["hold"] == pytest.approx(0.1, abs=0.01)
    assert m["reach_cost"] == pytest.approx(5.0, abs=0.5)
    assert m["gain"] == pytest.approx(0.1 + p * 5.0, abs=0.01)
    assert m["unsettled_fraction"] == 0.0
    assert m["nea"] == pytest.approx(0.5, abs=0.05)


def test_transient_cost_compares_controllers_that_reach_cost_cannot():
    """Reach B is the transient above the cycle's own hold level, over a
    transient detected against that level: a controller holding far off has
    its excursion swallowed (B small). The transient cost sums the first
    ``window`` steps of the cycle with nothing subtracted, and orders the two
    controllers by what the change actually cost."""
    tc = np.zeros(200, bool)
    tc[0] = tc[100] = True
    near = np.full(200, 1.0)
    far = np.full(200, 100.0)
    for c in (near, far):
        c[0:10] += 50.0
        c[100:110] += 50.0
    b_near, _ = E.reach_cost([_episode(near, tc)])
    b_far, _ = E.reach_cost([_episode(far, tc)])
    assert b_near == pytest.approx(500.0)
    assert b_far < b_near  # the 150 vs 100 excursion is "within hold" for far
    t_near, _ = E.transient_cost([_episode(near, tc)], window=20)
    t_far, _ = E.transient_cost([_episode(far, tc)], window=20)
    assert t_near == pytest.approx(500.0 + 20.0)
    assert t_far == pytest.approx(500.0 + 2000.0)


def test_a_cost_still_rising_at_the_end_reads_as_unsettled_with_negative_reach():
    """No hold reached in the window: the 'hold level' is above the transient,
    B comes out negative, and the cycle is counted as unsettled."""
    tc = np.zeros(100, bool)
    tc[0] = True
    rising = np.linspace(0.0, 10.0, 100)
    ep = _episode(rising, tc)
    b, _ = E.reach_cost([ep])
    assert b < 0.0
    assert E.unsettled_fraction([ep]) == 1.0
    m = E.evaluate([ep], burn_in=0)
    assert m["unsettled_fraction"] == 1.0 and m["reach_cost"] < 0.0


# ---------------------------------------------------------------------------
# The protocol as it was before the anticipation rule (main at 55e7bf9),
# frozen here as the reference a cycle with no anticipation must reproduce
# exactly.
# ---------------------------------------------------------------------------


def _old_settle(c):
    n = len(c)
    if n < 4:
        return 1
    level = c[n // 2 :].mean()
    ok = c <= 2.0 * level + 1e-12
    outside = np.flatnonzero(~ok[: n // 2])
    return int(min(max(outside[-1] + 1 if len(outside) else 1, 1), n // 2))


def _old_protocol(episodes, burn_in):
    """hold (and its terms), reach cost, transient cost and unsettled fraction,
    as ``evaluate`` computed them before anticipation was split off."""
    hold, terms, reach, transient = [], {}, [], []
    n_up = n_tot = 0
    for ep in episodes:
        m = np.zeros(len(ep.cost), bool)
        for a, b in E._cycles(ep):
            c = ep.cost[a:b]
            k = _old_settle(c)
            m[a + k : b] = True
            level = ep.cost[a + k : b].mean() if b - a > k else ep.cost[a:b].mean()
            reach.append((ep.cost[a : a + k] - level).sum())
            transient.append(ep.cost[a : min(a + max(int(burn_in), 1), b)].sum())
            if len(c) >= 4:
                n_tot += 1
                n_up += bool(c[len(c) // 2 :].mean() > c[: len(c) // 2].mean())
        m[:burn_in] = False
        if m.sum() >= 1:
            hold.append(ep.cost[m].mean())
            for key, series in ep.terms.items():
                terms.setdefault(key, []).append(series[m].mean())
    out = {
        "hold": float(np.mean(hold)),
        "hold_ci": E._ci(hold),
        "reach_cost": float(np.mean(reach)),
        "reach_cost_ci": E._ci(reach),
        "transient_cost": float(np.mean(transient)),
        "transient_cost_ci": E._ci(transient),
        "unsettled_fraction": n_up / max(n_tot, 1),
    }
    for key, vals in terms.items():
        out["hold_" + key] = float(np.mean(vals))
    return out


def _n_anticipating(episodes):
    return sum(
        s.hold_end < s.end
        for ep in episodes
        for s in E.cycle_segments(ep.cost, E._cycles(ep))
    )


def _assert_same_as_before(episodes, burn_in):
    new = E.evaluate(episodes, burn_in=burn_in)
    old = _old_protocol(episodes, burn_in)
    for key, value in old.items():
        if value != value:  # nan: both must be nan
            assert new[key] != new[key], key
        else:
            assert new[key] == value, (key, new[key], value)


def _anticipating_episode(n_cycles=20, length=100, ramp=None, hold=0.1):
    """Hold cost ``hold``, a +1 excursion over the 5 steps after each change,
    and a rise over the last steps of every cycle a change follows."""
    ramp = np.linspace(0.5, 4.0, 8) if ramp is None else np.asarray(ramp, float)
    T = n_cycles * length
    cost = np.full(T, hold)
    tc = np.zeros(T, bool)
    for i in range(n_cycles):
        a = i * length
        tc[a] = True
        cost[a : a + 5] += 1.0
        if i < n_cycles - 1:
            cost[a + length - len(ramp) : a + length] = ramp
    return _episode(cost, tc)


def test_a_rise_before_each_change_is_moved_to_that_changes_transient():
    """A controller that previews the schedule starts toward the next target
    before the change. Those steps leave the hold of the cycle they sit in and
    join the reach and transient cost of the change they prepare. The gain,
    which counts every step, does not move; the hold drops back to the level
    actually held; and the cycles no longer read as unsettled."""
    ramp = np.linspace(0.5, 4.0, 8)
    ep = _anticipating_episode(ramp=ramp)
    burn_in = 10
    segs = E.cycle_segments(ep.cost, E._cycles(ep))
    assert [s.end - s.hold_end for s in segs] == [8] * 19 + [0]
    assert all(s.hold_start - s.start == 5 for s in segs)

    old = _old_protocol([ep], burn_in)
    m = E.evaluate([ep], burn_in=burn_in)
    assert m["gain"] == E.gain([ep], burn_in)[0]
    assert m["hold"] == pytest.approx(0.1, rel=1e-12)
    assert old["hold"] > 0.12  # the rises counted as hold before
    excess = (ramp - 0.1).sum()
    assert m["reach_cost"] == pytest.approx(5.0 + excess * 19 / 20, rel=1e-12)
    assert m["transient_cost"] == pytest.approx(
        5 * 1.1 + 5 * 0.1 + ramp.sum() * 19 / 20, rel=1e-12
    )
    assert old["unsettled_fraction"] == pytest.approx(19 / 20)
    assert m["unsettled_fraction"] == 0.0
    # Theorem 4 still adds up, with the rises on the reach side
    p = 20 / len(ep.cost)
    assert m["gain"] == pytest.approx(m["hold"] + p * m["reach_cost"], rel=0.01)


def _hold_without(cost, cycles, antic):
    """The hold of one episode with the given anticipation of each cycle left
    out and the settle measured as the protocol measures it."""
    m = np.zeros(len(cost), bool)
    for (a, b), j in zip(cycles, antic):
        m[a + E._settle_of(cost[a : b - j], None) : b - j] = True
    return cost[m].mean()


def test_a_noisy_hold_with_a_rise_before_each_change():
    """The same on a noisy hold (cost ``0.1 * e**2``, Gaussian ``e``) with the
    error ramping to eight standard deviations over the last ten steps. Every
    rise is found, from its last six steps to at most four steps more than the
    rise, since a noise spike just before it can open the run. Its first steps
    cost no more than the noise does and may stay in the hold. The hold comes
    back to the noise level from three times it, and within 2% of the hold
    with exactly the rise left out."""
    rng = np.random.default_rng(3)
    n_cycles, length, H = 40, 120, 10
    e = rng.standard_normal(n_cycles * length)
    tc = np.zeros(len(e), bool)
    tc[::length] = True
    for i in range(n_cycles - 1):
        b = (i + 1) * length
        e[b - H : b] += 8.0 * np.arange(1, H + 1) / H
    ep = _episode(0.1 * e**2, tc)
    cycles = E._cycles(ep)
    found = [s.end - s.hold_end for s in E.cycle_segments(ep.cost, cycles)]
    assert found[-1] == 0
    assert all(6 <= j <= H + 4 for j in found[:-1]), found
    m = E.evaluate([ep], burn_in=0)
    exact = _hold_without(ep.cost, cycles, [H] * (n_cycles - 1) + [0])
    assert m["hold"] == pytest.approx(exact, rel=0.02)
    assert m["hold"] == pytest.approx(0.1, rel=0.1)
    assert _old_protocol([ep], 0)["hold"] > 3.0 * m["hold"]


def _ramp_with_dips(dips, H=20):
    """A rise from 0.02 to 2e4 over ``H`` steps, with the cost near zero on the
    steps ``dips`` before the change, as when the error crosses zero."""
    ramp = 0.02 * np.exp(np.arange(H) * np.log(1e6) / H)
    for d in dips:
        ramp[-d] = 1e-5
    return ramp


def _dip_episode(ramp, n_cycles=10, length=200, seed=4):
    rng = np.random.default_rng(seed)
    parts, tc = [], np.zeros(n_cycles * length, bool)
    for i in range(n_cycles):
        c = 0.005 * rng.standard_normal(length) ** 2
        c[:10] += 1e4 * np.exp(-np.arange(10) / 2.0)
        if i < n_cycles - 1:
            c[-len(ramp) :] = ramp
        parts.append(c)
        tc[i * length] = True
    return _episode(np.concatenate(parts), tc)


def test_a_dip_inside_the_rise_does_not_cut_it_short():
    """An error that crosses zero on its way to the next target costs almost
    nothing for a step. The unstable CSTR's MPC does this six steps before a
    change (a step costing 8e-5 among steps costing 7 to 162), and a rule
    that wanted every step above twice the level kept all but the last five
    steps of that rise in the hold, which was then 28 times what the MPC
    holds. A dip of one or two steps inside the run is now bridged, and the
    whole rise is found. The run can also take a few noise steps before the
    rise when they cost more than twice the level, here at most four and
    under one on average, and the hold is within 2% of the hold with exactly
    the rise left out. A dip of three steps ends the run there, the one limit
    of the bridge."""
    for dips in ((6,), (6, 7), (6, 15)):
        ep = _dip_episode(_ramp_with_dips(dips))
        cycles = E._cycles(ep)
        found = E.anticipations(ep.cost, cycles)
        assert all(20 <= j <= 24 for j in found[:-1]), (dips, found)
        assert np.mean(found[:-1]) <= 21.0, (dips, found)
        exact = _hold_without(ep.cost, cycles, [20] * 9 + [0])
        assert E.hold([ep], burn_in=0)[0] == pytest.approx(exact, rel=0.02)
        assert exact == pytest.approx(0.005, rel=0.1)
    ep = _dip_episode(_ramp_with_dips((6, 7, 8)))
    assert E.anticipations(ep.cost, E._cycles(ep))[:-1] == [5] * 9


def test_a_one_or_two_step_jump_before_the_change_is_found_when_huge():
    """An MPC with a short horizon moves one or two steps before the change.
    Runs that short are found only when every step costs a hundred times the
    level, which stationary noise essentially never does on the last step of
    a cycle (the rates below include it). A jump of 50 or 80 times the level
    stays in the hold."""
    for ramp, j in (([1e3], 1), ([1e3, 2e3], 2), ([5.0], 0), ([5.0, 8.0], 0)):
        ep = _anticipating_episode(ramp=ramp)
        found = E.anticipations(ep.cost, E._cycles(ep))
        assert found == [j] * 19 + [0], (ramp, found)
        if j:
            assert E.hold([ep], burn_in=10)[0] == pytest.approx(0.1, rel=1e-12)


# The noise laws of a stationary hold that must not read as anticipation, and
# the measured share of noise-only cycles in which the rule fired (seeded, so
# these are exact). "episode" is the rule as the protocol runs it, on 200
# episodes of ten cycles (1800 cycles searched per cell), each searched twice
# with the floor between. "single" is one cycle searched once with no floor,
# 1000 cycles per cell, the case of an episode with two cycles at worst.
#
#   cycle length              16              36              100
#                       episode single  episode single  episode single
#   e**2, Gaussian e       0     0.5%      0      0        0      0
#   |e|, Gaussian e        0     0         0      0        0      0
#   exponential            0     0         0      0        0      0
#   |t_3|                  0     0         0      0        0      0
#   t_3**2               0.06%   0.9%    0.11%  0.2%       0     0.3%
_NOISE = {
    "e**2": lambda r, s: r.standard_normal(s) ** 2,
    "|e|": lambda r, s: np.abs(r.standard_normal(s)),
    "exponential": lambda r, s: r.exponential(1.0, s),
    "|t3|": lambda r, s: np.abs(r.standard_t(3, s)),
    "t3**2": lambda r, s: r.standard_t(3, s) ** 2,
}


def _false_rate(cycles):
    return float(np.mean([E.anticipation(c) > 0 for c in cycles]))


def _episode_false_rate(costs, length):
    """Share of the cycles a change follows in which ``anticipations`` fires,
    each row of ``costs`` being one episode of cycles of ``length`` steps."""
    k = costs.shape[1] // length
    cycles = [(i * length, (i + 1) * length) for i in range(k)]
    return float(
        np.mean([j > 0 for c in costs for j in E.anticipations(c, cycles)[:-1]])
    )


@pytest.mark.parametrize("law", sorted(_NOISE))
def test_stationary_noise_is_not_read_as_anticipation(law):
    """A stationary noisy hold must give no anticipation in nearly every
    cycle. ``e**2`` exceeds twice its mean on 16% of steps, so a rule that
    looked only for steps above twice the level would fire on one cycle in
    six. Measured as the protocol runs the rule, it fires on at most 0.11% of
    noise-only cycles, and a single cycle searched with no floor on at most
    0.9% (table above). The bounds asserted are 0.3% in episodes, and for
    single cycles 2% at 16 steps and 0.5% beyond."""
    rng = np.random.default_rng(0)
    for n, bound in ((16, 0.02), (36, 0.005), (100, 0.005)):
        single = _false_rate(_NOISE[law](rng, (1000, n)))
        assert single <= bound, f"{law}, {n}-step cycle: {single:.2%} > {bound:.1%}"
        rate = _episode_false_rate(_NOISE[law](rng, (200, 10 * n)), n)
        assert rate <= 0.003, f"{law}, {n}-step cycles: {rate:.2%} > 0.3%"


def _ar1(rng, phi, rows, steps, burn=300):
    x = np.empty((rows, steps + burn))
    x[:, 0] = rng.standard_normal(rows)
    z = rng.standard_normal(x.shape) * np.sqrt(1 - phi**2)
    for t in range(1, x.shape[1]):
        x[:, t] = phi * x[:, t - 1] + z[:, t]
    return x[:, burn:]


def test_slowly_correlated_noise_is_read_as_anticipation_only_now_and_then():
    """The known limit, and what the floor does for it. Noise correlated over
    tens of steps (``e**2`` with ``e`` an AR(1) at 0.9, 0.98 or 0.99) makes
    long excursions, and one that happens to end at a change looks like
    anticipation from the cost alone. A single cycle searched with no floor
    reads it so in up to 6.9% of cycles (0.98, 36 steps). In episodes of ten
    cycles the floor, the mean level of the episode's cycles, brings this to
    at most 0.22% (measured 0.11%, 0.22% and 0 at 0.9 for 36, 100 and 200
    steps, and at most 0.11% at 0.98 and 0.99). A false detection moves that
    excursion from the hold to the next change's transient, which lowers the
    hold a little, and leaves the gain alone."""
    for phi in (0.9, 0.98, 0.99):
        rng = np.random.default_rng(0)
        for n in (36, 100, 200):
            single = _false_rate(_ar1(rng, phi, 1000, n) ** 2)
            assert single <= 0.10, f"AR {phi}, {n}-step cycle: {single:.1%}"
            rate = _episode_false_rate(_ar1(rng, phi, 100, 10 * n) ** 2, n)
            assert rate <= 0.01, f"AR {phi}, {n}-step cycles: {rate:.2%} > 1%"


# A night of the building as the shipped controllers hold it (test episodes,
# cost per step, three significant figures). The PID (seed 1, steps 87 to
# 122) holds inside the dead band until its heating cost rises over the last
# nine steps. The MPC (seed 0, steps 183 to 218) sits on a plateau of a few
# 1e-9 through the second half of the night.
_PID_NIGHT = [
    2.04e-02, 7.51e-03, 2.76e-03, 1.02e-03, 3.75e-04, 1.38e-04, 5.08e-05,
    1.87e-05, 6.88e-06, 2.53e-06, 9.33e-07, 3.44e-07, 1.26e-07, 4.66e-08,
    1.71e-08, 6.31e-09, 2.32e-09, 8.56e-10, 3.15e-10, 1.16e-10, 4.27e-11,
    1.57e-11, 5.79e-12, 2.13e-12, 7.85e-13, 2.89e-13, 1.06e-13, 1.05e-03,
    1.78e-03, 3.38e-03, 5.81e-03, 8.57e-03, 1.06e-02, 1.43e-02, 1.70e-02,
    1.81e-02,
]  # fmt: skip
_MPC_NIGHT = [
    1.83e-22, 6.75e-23, 2.49e-23, 9.15e-24, 3.37e-24, 1.24e-24, 4.57e-25,
    1.68e-25, 6.19e-26, 2.28e-26, 8.39e-27, 3.09e-27, 1.14e-27, 4.19e-28,
    1.54e-28, 5.68e-29, 2.09e-29, 7.69e-30, 2.83e-30, 1.04e-30, 3.84e-31,
    1.41e-31, 3.39e-09, 4.64e-09, 1.71e-09, 4.02e-09, 1.48e-09, 3.93e-09,
    4.84e-09, 1.78e-09, 4.05e-09, 4.88e-09, 5.19e-09, 5.30e-09, 1.95e-09,
    7.50e-09,
]  # fmt: skip


def _day_and_night(night, n_days=8, seed=6):
    """A building's schedule. Day cycles of 60 steps hold a noisy cost near
    1e-2 after a short transient, and every night is the 36-step ``night``."""
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(2 * n_days + 1):
        if i % 2 == 0:
            day = 0.01 * rng.standard_normal(60) ** 2
            day[:10] += 0.3
            parts.append(day)
        else:
            parts.append(np.asarray(night, float))
    tc = np.zeros(sum(len(c) for c in parts), bool)
    tc[np.cumsum([0] + [len(c) for c in parts[:-1]])] = True
    return _episode(np.concatenate(parts), tc)


def test_a_cycle_held_at_zero_cost_does_not_magnify_a_small_rise():
    """Against a level of zero, any cost is a tenfold excursion. On the
    building the PID's heating cost rises from 1e-3 to 1.8e-2 over the last
    nine steps of a night held at a cost near 1e-13, and the MPC's cost sits
    on a plateau of a few 1e-9 through the second half of a night held near
    1e-30. Neither is anticipation, and both were found by a rule compared
    with the cycle's own level only, as a single search with no floor still
    finds them here. The floor, the mean level of the episode's cycles, keeps
    both in the hold, and every number is what it was before. A rise that is
    large against the episode's typical level is still found."""
    for night in (_PID_NIGHT, _MPC_NIGHT):
        ep = _day_and_night(night)
        assert E.anticipation(night) > 0
        assert _n_anticipating([ep]) == 0
        _assert_same_as_before([ep], burn_in=0)
    pre_heat = np.array(_PID_NIGHT)
    pre_heat[-9:] = np.geomspace(0.1, 5.0, 9)
    ep = _day_and_night(pre_heat)
    assert E.anticipations(ep.cost, E._cycles(ep)) == [0, 9] * 8 + [0]


def _mixed_episode(rng, lengths, law):
    """Cycles of the given lengths, each a decaying transient after its change
    on top of a stationary noisy hold, with tracking and running terms."""
    parts, tc = [], []
    for n in lengths:
        t = np.arange(n)
        parts.append(0.1 * law(rng, n) + 5.0 * np.exp(-t / 4.0))
        flag = np.zeros(n, bool)
        flag[0] = True
        tc.append(flag)
    cost = np.concatenate(parts)
    tracking = 0.7 * cost
    return _episode(
        cost,
        np.concatenate(tc),
        terms={"tracking": tracking, "running": cost - tracking},
    )


@pytest.mark.parametrize("law", sorted(_NOISE))
def test_without_anticipation_every_number_is_unchanged(law):
    """Invariance. A cycle with no detected anticipation keeps its settle,
    level and hold steps, so an episode without any gives exactly the numbers
    the protocol gave before the rule, down to the last bit. Checked on noisy
    holds with transients, over cycle lengths from 3 to 400 steps."""
    rng = np.random.default_rng(1)
    lengths = [3, 7, 12, 15, 16, 20, 36, 60, 100, 250, 400] * 3
    episodes = [
        _mixed_episode(rng, rng.permutation(lengths), _NOISE[law]) for _ in range(3)
    ]
    assert _n_anticipating(episodes) == 0
    _assert_same_as_before(episodes, burn_in=50)


@pytest.mark.parametrize("name", ["battery", "hvac", "plane_energy"])
def test_real_pid_episodes_are_unchanged(name):
    """The same on the three plants whose test episodes hold more than one
    target: the shipped PID on the battery (dispatch blocks under white
    dispatch noise), the building (occupancy switches under weather, with a
    dead band at night) and the energy-managed aircraft (a ladder of speed
    and altitude targets), on the seeds ``evaluate_controller`` runs. The
    PIDs do not preview the schedule, no anticipation is found, and every
    protocol number is what the protocol gave before. The building's seeds 1
    and 2 are the ones where a rule without the floor found some."""
    from target_gym.registry import REGISTRY
    from target_gym.runners.runners import baseline_policy

    spec = REGISTRY[name]
    p = spec.make_test_params()
    episodes = [
        E.run_episode(spec, p, baseline_policy(spec, "pid", p), seed=s)
        for s in range(3)
    ]
    assert all(len(E._cycles(ep)) > 3 for ep in episodes)
    assert _n_anticipating(episodes) == 0
    burn_in = min(E.hold_settings(name)["burn_in"], int(p.max_steps_in_episode) // 2)
    _assert_same_as_before(episodes, burn_in)


def test_the_last_cycle_never_loses_steps():
    """No change follows the last cycle of an episode, so nothing at its end is
    anticipation, however it looks. Here both cycles end on the same rise; the
    first loses it to the change, the last keeps it as hold."""
    ep = _anticipating_episode(n_cycles=2)
    ep.cost[-8:] = np.linspace(0.5, 4.0, 8)
    first, last = E.cycle_segments(ep.cost, E._cycles(ep))
    assert first.end - first.hold_end == 8
    assert last.hold_end == last.end == len(ep.cost)
    assert E.hold_mask(ep, burn_in=0)[-8:].all()


def test_short_cycles_are_not_searched():
    """A cycle under 16 steps has too little hold to measure a level against,
    so it is never searched and scores as before, rise or not. At 16 steps the
    same rise is found."""
    ramp = np.linspace(0.5, 4.0, 4)
    short = _anticipating_episode(n_cycles=10, length=15, ramp=ramp)
    assert _n_anticipating([short]) == 0
    _assert_same_as_before([short], burn_in=0)
    at_16 = _anticipating_episode(n_cycles=10, length=16, ramp=ramp)
    assert _n_anticipating([at_16]) == 9


def test_a_cycle_still_settling_at_its_end_is_not_anticipation():
    """A controller that has not settled when the cycle ends is not
    anticipating. A cost still falling at the end sits below the level, a
    cost drifting up in a straight line never gets above twice the level of
    what precedes it, and a hump that comes back down before the change is
    not rising. All three keep their old numbers, and the drift still reads
    as unsettled."""
    t = np.arange(100)
    tc = np.zeros(300, bool)
    tc[::100] = True
    settling = _episode(np.tile(np.exp(-t / 60.0), 3), tc)
    drifting = _episode(np.tile(np.linspace(0.0, 10.0, 100), 3), tc)
    hump = np.full(100, 0.1)  # peaks two thirds in, still high at the change
    hump[30:] += 5.0 * np.sin(np.linspace(0.0, 0.9 * np.pi, 70))
    humped = _episode(np.tile(hump, 3), tc)
    for ep in (settling, drifting, humped):
        assert _n_anticipating([ep]) == 0
        _assert_same_as_before([ep], burn_in=0)
    assert E.unsettled_fraction([drifting]) == 1.0


def test_a_cycle_that_runs_away_at_its_end_is_charged_to_the_next_change():
    """What the rule cannot tell apart. A cost that blows up over the last
    steps before a change looks, from the cost alone, exactly like a
    controller leaving early for the next target, and it is treated as one.
    The blow-up leaves the hold and is charged to the next change's reach and
    transient cost, and a cycle that was unsettled only because of it no
    longer reads as unsettled. That is acceptable because those steps are not
    a hold under either reading, and the gain counts them in full whatever the
    split. The run takes at most half the cycle, so a run-away that started
    earlier keeps its first part in the hold of the cycle."""
    late = _anticipating_episode(n_cycles=2, ramp=0.25 * np.exp(np.arange(1, 13) / 2.0))
    first, _ = E.cycle_segments(late.cost, E._cycles(late))
    assert first.end - first.hold_end == 12
    assert E.hold([late], burn_in=0)[0] == pytest.approx(0.1, rel=1e-12)
    assert _old_protocol([late], 0)["unsettled_fraction"] == 0.5
    assert E.unsettled_fraction([late]) == 0.0

    early = _anticipating_episode(
        n_cycles=2, ramp=0.1 * np.exp(np.arange(1, 71) / 12.0)
    )
    first, _ = E.cycle_segments(early.cost, E._cycles(early))
    assert first.end - first.hold_end == 50  # capped at half the cycle
    assert E.hold_mask(early, burn_in=0)[30:50].all()
    for ep in (late, early):
        assert E.gain([ep], 0)[0] == ep.cost.mean()


def test_the_transient_window_reaches_as_far_before_the_change_as_after():
    """``transient_cost`` counts ``window`` steps after the change and at most
    ``window`` anticipation steps before it. With the whole anticipation
    counted, a plant without a burn-in (a one-step window) charged an MPC that
    moves 20 steps early for all 20 and the PID for its first step alone."""
    ramp = np.linspace(0.5, 4.0, 8)
    ep = _anticipating_episode(ramp=ramp)
    for window, before in ((1, ramp[-1:]), (3, ramp[-3:]), (10, ramp), (50, ramp)):
        after = 1.1 * min(window, 5) + 0.1 * max(min(window, 92) - 5, 0)
        t, _ = E.transient_cost([ep], window)
        assert t == pytest.approx(after + before.sum() * 19 / 20, rel=1e-12)


def _load_measure_hold():
    spec = importlib.util.spec_from_file_location(
        "measure_hold_under_test", ROOT / "scripts" / "measure_hold.py"
    )
    mh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mh)
    return mh


def test_measure_hold_leaves_out_the_same_steps():
    """``scripts/measure_hold.py`` measures the hold errors behind the floors
    with its own settle rule, and takes the anticipation from this module, so
    the two cannot drift apart. On an error series whose first output rises
    before each change, its hold mask drops the same steps the protocol does,
    and with no rise it is the mask it always computed. A fixed ``settle``
    replaces the measured one and leaves the anticipation as it is."""
    mh = _load_measure_hold()
    assert mh.anticipations is E.anticipations

    rng = np.random.default_rng(2)
    length, n_cycles = 80, 4
    targets = np.repeat(np.arange(n_cycles, dtype=float), length)[:, None]
    targets = np.hstack([targets, 2.0 * targets])
    e = 0.01 * np.abs(rng.standard_normal((n_cycles * length, 2)))
    quiet = e.copy()
    for i in range(1, n_cycles):
        e[i * length - 6 : i * length, 0] += np.linspace(0.2, 0.9, 6)
    m, settle, bounds, antic = mh.split_seed(e, targets, burn_in=20)
    cycles = list(zip(bounds[:-1].tolist(), bounds[1:].tolist()))
    assert antic == E.anticipations(e**2, cycles)
    assert all(6 <= j <= 10 for j in antic[:-1]) and antic[-1] == 0, antic
    for b, j in zip(bounds[1:-1], antic):
        assert not m[b - j : b].any()
    m5, settle5, _, antic5 = mh.split_seed(e, targets, burn_in=20, settle=5)
    assert settle5 == 5.0 and antic5 == antic
    assert (m5 == mh._hold_mask(len(e), 20, bounds, 5, antic)).all()

    m, settle, bounds, antic = mh.split_seed(quiet, targets, burn_in=20)
    assert antic == [0] * n_cycles
    lvl = quiet[mh._hold_mask(len(quiet), 20, bounds, 0)].mean(axis=0)
    old_settle = mh._settle(quiet, bounds, lvl)
    assert settle == old_settle
    old = mh._hold_mask(
        len(quiet), 20, bounds, int(min(old_settle, (len(quiet) - 20) // 4))
    )
    assert (m == old).all()


def test_measure_hold_is_as_sensitive_as_the_protocol():
    """The protocol searches a quadratic cost, and ``measure_hold`` searched
    the absolute errors with the same thresholds, which made it about ten
    times less sensitive, so a floor could keep an anticipation the protocol
    leaves out. An error rising to five noise standard deviations over the
    last twelve steps of each cycle was found in 86% of cycles on ``e**2``
    and in none on ``|e|``. ``measure_hold`` now searches the squared errors
    and finds what the protocol finds."""
    mh = _load_measure_hold()
    rng = np.random.default_rng(8)
    length, n_cycles, H = 150, 40, 12
    e = rng.standard_normal(length * n_cycles)
    for i in range(1, n_cycles):
        e[i * length - H : i * length] += 5.0 * np.arange(1, H + 1) / H
    targets = np.repeat(np.arange(n_cycles) % 2, length).astype(float)[:, None]
    errors = np.abs(e)[:, None]
    _, _, bounds, antic = mh.split_seed(errors, targets, burn_in=0)
    cycles = list(zip(bounds[:-1].tolist(), bounds[1:].tolist()))
    assert antic == E.anticipations(e**2, cycles)
    assert np.mean([j > 0 for j in antic[:-1]]) > 0.8
    assert np.mean([j > 0 for j in E.anticipations(np.abs(e), cycles)]) < 0.05
