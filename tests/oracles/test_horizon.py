"""Cooper's horizon rule (target_gym.oracles.horizon).

Contract: docs/proposals/oracle-horizon-protocol.md section 2 with the owner
decisions: T_c = max(0.1 tau, 0.5 theta), rounded down to whole steps, at
least one, the smallest over pairs; P = N + M; H_k = min(P, N_ep - k).
Expected values are worked by hand in the comments.
"""

import pytest

from target_gym.oracles.horizon import (
    cooper_horizons,
    episode_samples,
    horizon_at,
    ramp_samples,
    sample_steps,
    sample_time,
    task_horizon,
    task_ramp,
)


def fopdt(tau, theta):
    return {"model": "fopdt", "tau": tau, "theta": theta}


@pytest.mark.parametrize(
    "fit, T, expected",
    [
        # k = Int(3 / 1.5) + 1 = 3; N = Int(50 / 1.5) + 3 = 36; M = Int(10 / 1.5) + 3 = 9.
        (fopdt(10.0, 3.0), 1.5, (36, 9, 45)),
        # Section 6's first_order: tau 0.5 s, theta 0, T 0.05 s. k = 1,
        # N = Int(2.5 / 0.05) + 1 = 51, M = 10 + 1 = 11, P = 62.
        (fopdt(0.5, 0.0), 0.05, (51, 11, 62)),
        # Section 6's unstable_cstr in steps: k = Int(2.28) + 1 = 3,
        # N = Int(47.1) + 3 = 50, M = Int(9.42) + 3 = 12, P = 62.
        (fopdt(9.42, 2.28), 1.0, (50, 12, 62)),
        # theta / T = 0.6 / 0.2 = 3 exactly, though binary makes it 2.9999...:
        # k = 3 + 1 = 4, N = Int(17.5) + 4 = 21, M = Int(3.5) + 4 = 7.
        (fopdt(0.7, 0.6), 0.2, (21, 7, 28)),
        # Integrating: theta_e = 1 + 0.5 / 2 + 2 / 2 = 2.25, tau_CL = sqrt(10)
        # 2.25 = 7.115; k = Int(1 / 0.5) + 1 = 3; N = Int(71.15) + 3 = 74,
        # M = Int(14.23) + 3 = 17.
        ({"model": "integrating_lag", "tau2": 2.0, "theta": 1.0}, 0.5, (74, 17, 91)),
    ],
)
def test_cooper_horizons_on_hand_worked_fits(fit, T, expected):
    assert cooper_horizons(fit, T) == expected


def test_sample_time_is_the_larger_of_a_tenth_tau_and_half_theta():
    assert sample_time(fopdt(2.0, 3.0)) == pytest.approx(1.5)
    assert sample_time(fopdt(20.0, 3.0)) == pytest.approx(2.0)
    lag = {"model": "integrating_lag", "tau2": 4.0, "theta": 0.2}
    assert sample_time(lag) == pytest.approx(0.4)  # the lag stands for tau


def test_sample_time_takes_the_fastest_pair_rounded_down_to_whole_steps():
    assert sample_steps([1.5, 0.9, 2.0], dt=0.25) == 3  # 0.9 / 0.25 = 3.6
    assert sample_steps([0.15, 2.0], dt=0.05) == 3  # an exact multiple stays
    assert sample_steps([0.01, 5.0], dt=0.05) == 1  # never below one step


def test_task_horizon_samples_at_the_fast_pair_and_plans_for_the_slow_one():
    # T_c: max(1, 0) = 1 s and max(10, 2) = 10 s, so 1 step. At T = 1 s the
    # slow pair has k = 5, N = 500 + 5, M = 100 + 5, P = 610; the fast 62.
    fits = {("u1", "y", 100): fopdt(10.0, 0.0), ("u2", "y", 101): fopdt(100.0, 4.0)}
    tc, rows, top = task_horizon(fits, dt=1.0)
    assert (tc, top) == (1, ("u2", "y", 101))
    assert rows == {("u1", "y", 100): (51, 11, 62), ("u2", "y", 101): (505, 105, 610)}


@pytest.mark.parametrize(
    "P, k, expected",
    [(62, 0, 62), (62, 88, 62), (62, 89, 61), (62, 149, 1), (200, 0, 150)],
)
def test_horizon_shrinks_over_the_last_P_samples_of_the_episode(P, k, expected):
    assert horizon_at(P, n_ep=150, k=k) == expected


def test_no_horizon_past_the_episode_and_a_partial_last_sample_counts():
    with pytest.raises(ValueError):
        horizon_at(62, n_ep=150, k=150)
    assert episode_samples(150, 4) == 38  # 37 whole samples and 2 steps
    assert episode_samples(150, 3) == 50


def test_a_task_with_no_valid_pair_gets_no_protocol_horizon():
    # Amendment 2 (N3, replacing H20): no T_c, no P; the task keeps its oracle.
    assert task_horizon({}, dt=0.5) == (None, {}, None)


@pytest.mark.parametrize(
    "span, gain, rate, T, expected",
    [
        # The compressor by hand: 8000 Pa / (7206 Pa per unit x 2 x 0.03 / 0.35
        # units/s) = 6.478 s; at T = 0.1 s, 64.8 samples, rounded up.
        (8000.0, 7206.0, 2 * 0.03 / 0.35, 0.1, 65),
        (8000.0, -7206.0, 2 * 0.03 / 0.35, 0.1, 65),  # the gain's sign is no matter
        (10.0, 2.0, 1.0, 1.0, 5),  # 10 / 2 = 5 units at 1 unit/s: exactly 5
        (10.0, 2.0, 1.0, 0.3, 17),  # 5 s / 0.3 s = 16.7
    ],
)
def test_a_ramp_is_the_samples_a_rate_limit_takes_to_cross_the_target_range(
    span, gain, rate, T, expected
):
    assert ramp_samples(span, gain, rate, T) == expected


def test_a_task_ramp_is_the_longest_over_its_rate_limited_fits():
    """Two MVs on one CV, three fits. By hand, over a span of 12: "slow" at
    gain 3 and 4 units of rate 1 takes 4 s and 3 s; "fast" at gain 2 and rate
    10 takes 0.6 s. The longest is 4 s, 8 samples at T = 0.5. A CV or an MV
    with no rate limit does not count."""
    fits = {
        ("slow", "y", 100): {"k": 3.0},
        ("slow", "y", 101): {"k": -4.0},
        ("fast", "y", 100): {"k": 2.0},
        ("free", "y", 100): {"k": 0.1},
        ("slow", "z", 100): {"k": 0.1},
    }
    rates = {"slow": 1.0, "fast": 10.0}
    assert task_ramp(fits, "y", 12.0, rates, 0.5) == (8, ("slow", "y", 100))


def _at(tau):
    return {"model": "fopdt", "k": 1.0, "tau": tau, "theta": 0.0}


def test_a_quiet_fit_can_set_p():
    """Amendment 4. dt 100 s, a valid fit of tau 10000 s: T_c = 1000 s, 10 steps,
    N = 50 + 1, M = 10 + 1, P = 62. A quiet fit of tau 50000 s at that T_c: N =
    250 + 1, M = 50 + 1, P = 302, which takes the maximum."""
    tc, rows, top = task_horizon(
        {("u", "y", 100): _at(1e4)}, 100.0, quiet={("u", "y", 101, 1): _at(5e4)}
    )
    assert tc == 10
    assert top == ("u", "y", 101, 1)
    assert rows[top] == (251, 51, 302)


def test_a_quiet_fit_never_sets_t_c():
    """Tau 5000 s alone would give T_c = 500 s, 5 steps. It stays 10: H2 reads
    valid fits only, and the quiet fit's P of 26 + 6 = 32 leaves P at 62."""
    tc, rows, top = task_horizon(
        {("u", "y", 100): _at(1e4)}, 100.0, quiet={("u", "y", 101, 1): _at(5e3)}
    )
    assert tc == 10
    assert top == ("u", "y", 100)
    assert rows[("u", "y", 101, 1)] == (26, 6, 32)


def test_quiet_fits_alone_make_no_horizon():
    """No valid fit means no T_c, so the task stays non-compliant (N3)."""
    assert task_horizon({}, 100.0, quiet={("u", "y", 101, 1): _at(5e4)}) == (
        None,
        {},
        None,
    )


def test_an_integrating_fit_gives_no_ramp():
    """H21 divides by a settled gain. An integrating fit's k' is a slope, per
    second, so span / (|k'| rate) is no time at all: on the boiler's pressure it
    read as 973 samples, on the glass crown as 56726."""
    fits = {("u", "y", 100): {"k": 0.22, "model": "integrating_lag"}}
    assert task_ramp(fits, "y", 6.0, {"u": 0.014}, 2.0) == (0, None)


def test_a_task_with_no_rate_limited_fit_has_no_ramp():
    assert task_ramp({("u", "y", 100): {"k": 1.0}}, "y", 1.0, {}, 0.5) == (0, None)
