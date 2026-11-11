"""The reference ladder's law and its choice of rung.

Contract, from docs/proposals/task-charter.md rule 3 and textbook control:

- ``pi_gains``: SIMC with ``tc = theta`` (Skogestad 2003). FOPDT
  ``Kc = tau / (k * 2 * theta)``, ``Ti = min(tau, 8 * theta)``. Integrating
  ``Kc = 1 / (k * 2 * theta)``, ``Ti = 8 * theta``.
- ``decoupler``: ``G @ D`` is diagonal, each loop keeping the gain it was tuned on.
- ``most_sensitive``: of one loop's SIMC designs at several operating points, the
  one asking the smallest controller gain, so the loop stays stable where the plant
  is most sensitive. A tie keeps the earlier point.
- ``zone_setpoint`` (Amendment 3, rule 22): on a zone, the reference's setpoint is
  the cheap edge moved ``k = 2`` standard deviations inside, never past the zone's
  centre. With no cheap side it is the centre. An infinite edge can't be backed off
  from, and an infinite zone has no centre.
- ``cv_sigma``: the sample standard deviation of the controlled variable, pooled over
  seeds after the burn-in.
- ``null_verdict`` (J2): a null policy wins a seed by costing strictly less than the
  reference; a task fails when any null wins more than half the seeds.
- ``best_rung``: the lowest median cost among rungs that trip no seed; none if all
  trip. A tie goes to the lower rung, the simpler controller.
- ``ladder_controller``: position-form PI around a bias. The integral holds while
  the loop's MV is saturated or overridden. A cascade's outer output is the inner
  loop's setpoint. An override's select keeps the constraint on its side.

Expected values are worked by hand in each test.
"""

import numpy as np
import pytest

from target_gym.experts.ladder import (
    Cascade,
    Override,
    Pi,
    Rung,
    best_rung,
    cv_sigma,
    decoupler,
    ladder_controller,
    most_sensitive,
    null_verdict,
    pi_gains,
    zone_setpoint,
)


def read_cv(obs, xp):
    return xp.stack([obs[0]])


def read_target(obs, xp):
    return xp.stack([obs[1]])


def zero_bias(obs, xp):
    return xp.zeros(1)


def run(ctrl, observations):
    """The actions of ``ctrl`` on successive observations, on the host."""
    memory, out = ctrl.init_state, []
    for obs in observations:
        action, memory = ctrl.step(ctrl.params, memory, np.asarray(obs, float), np)
        out.append(np.asarray(action, float))
    return np.array(out)


def one_loop(pi, bias=zero_bias, **rung):
    return ladder_controller(
        Rung(name="test", loops=(pi,), mix=np.eye(1), **rung),
        read_cv,
        read_target,
        bias,
        dt=1.0,
    )


def test_simc_on_a_self_regulating_loop():
    # k 2, tau 10, theta 1: Kc = 10 / (2 * 2) = 2.5, Ti = min(10, 8) = 8.
    assert pi_gains(2.0, 10.0, 1.0) == pytest.approx((2.5, 8.0))


def test_simc_on_a_fast_loop_keeps_tau_as_ti():
    # k 1, tau 4, theta 1: Kc = 4 / 2 = 2, Ti = min(4, 8) = 4.
    assert pi_gains(1.0, 4.0, 1.0) == pytest.approx((2.0, 4.0))


def test_simc_on_an_integrating_loop():
    # k' 0.5, theta 2: Kc = 1 / (0.5 * 4) = 0.5, Ti = 8 * 2 = 16.
    assert pi_gains(0.5, None, 2.0) == pytest.approx((0.5, 16.0))


def test_simc_carries_the_gain_sign():
    assert pi_gains(-2.0, 10.0, 1.0)[0] == pytest.approx(-2.5)


def test_decoupler_diagonalises_the_gains_and_keeps_each_paired_gain():
    # Crossed pairing: loop 0 on MV 1, loop 1 on MV 0.
    G = np.array([[1.0, 3.0], [4.0, 1.0]])
    pairs = [1, 0]
    D = decoupler(G, pairs)
    # D's columns are loops, so the plant seen by the loops is G @ D, loop on loop.
    np.testing.assert_allclose(G @ D, np.diag([3.0, 4.0]), atol=1e-12)


def test_most_sensitive_point_is_the_one_asking_the_least_gain():
    # Same lags, gains 1 and 3: SIMC asks Kc 2.5 and 2.5 / 3, so the second.
    designs = [{"Kc": 2.5, "Ti": 8.0}, {"Kc": 2.5 / 3, "Ti": 8.0}]
    assert most_sensitive(designs) is designs[1]


def test_most_sensitive_compares_magnitudes_whatever_the_sign():
    designs = [{"Kc": -0.5}, {"Kc": 0.8}]
    assert most_sensitive(designs) is designs[0]


def test_most_sensitive_keeps_the_earlier_point_on_a_tie():
    designs = [{"Kc": 1.0, "Ti": 4.0}, {"Kc": 1.0, "Ti": 9.0}]
    assert most_sensitive(designs) is designs[0]


@pytest.mark.parametrize(
    "cheap, expected",
    # Zone 6 to 8, sigma 0.25: 2 sigma is 0.5 inside the cheap edge.
    [("low", 6.5), ("high", 7.5)],
)
def test_zone_setpoint_sits_two_sigma_inside_the_cheap_edge(cheap, expected):
    assert zone_setpoint(6.0, 8.0, 0.25, cheap) == pytest.approx(expected)


def test_zone_setpoint_never_crosses_the_centre():
    # sigma 0.75: 8 - 1.5 = 6.5 is past the centre 7, so the centre.
    assert zone_setpoint(6.0, 8.0, 0.75, "high") == pytest.approx(7.0)


def test_zone_setpoint_with_no_cheap_side_is_the_centre():
    assert zone_setpoint(6.0, 8.0, 0.25, None) == pytest.approx(7.0)


def test_a_one_sided_zone_backs_off_its_finite_edge():
    # A product spec x <= 0.01, sigma 0.001: 0.01 - 0.002 = 0.008.
    assert zone_setpoint(-np.inf, 0.01, 0.001, "high") == pytest.approx(0.008)


@pytest.mark.parametrize("cheap", ["low", None])
def test_an_infinite_edge_or_centre_is_an_error(cheap):
    with pytest.raises(ValueError):
        zone_setpoint(-np.inf, 0.01, 0.001, cheap)


def test_cv_sigma_pools_seeds_after_the_burn_in():
    # Samples 1, 3, 1, 3 once the 100s are dropped: mean 2, squared deviations
    # sum to 4, over n - 1 = 3, so sigma = sqrt(4 / 3).
    traces = [[100.0, 1.0, 3.0], [100.0, 1.0, 3.0]]
    assert cv_sigma(traces, burn_in=1) == pytest.approx((4 / 3) ** 0.5)


def test_a_null_wins_a_seed_only_by_costing_strictly_less():
    # Reference 1 on four seeds: a tie wins nothing, and 0.5, 0.5, 2, 2 wins two.
    verdict = null_verdict([1.0] * 4, {"tie": [1.0] * 4, "half": [0.5, 0.5, 2.0, 2.0]})
    assert verdict["wins"] == {"tie": 0, "half": 2}


def test_a_task_fails_j2_when_a_null_wins_more_than_half_the_seeds():
    # Two wins of four is half, not more: it passes. Three of four fails.
    assert not null_verdict([1.0] * 4, {"a": [0.5, 0.5, 2.0, 2.0]})["fails"]
    assert null_verdict([1.0] * 4, {"b": [0.5, 0.5, 0.5, 2.0]})["fails"]


def test_best_rung_is_the_cheapest_that_trips_no_seed():
    results = {
        "pi": {"costs": [5.0, 5.0, 5.0], "trips": 0},
        "feedforward": {"costs": [1.0, 1.0, 1.0], "trips": 1},
        "cascade": {"costs": [3.0, 4.0, 9.0], "trips": 0},
    }
    assert best_rung(results) == "cascade"  # median 4 beats 5, and 1 tripped


def test_best_rung_is_none_when_every_rung_trips():
    assert best_rung({"pi": {"costs": [1.0], "trips": 2}}) is None


def test_best_rung_breaks_a_tie_toward_the_lower_rung():
    results = {
        "pi": {"costs": [2.0], "trips": 0},
        "feedforward": {"costs": [2.0], "trips": 0},
    }
    assert best_rung(results) == "pi"


def test_pi_steps_as_proportional_plus_integral():
    # Kc 0.2, Ti 4, dt 1, error 1: 0.2 * 1.25 = 0.25, then 0.2 * 1.5 = 0.3.
    actions = run(one_loop(Pi(0, 0.2, 4.0)), [[0.0, 1.0], [0.0, 1.0]])
    np.testing.assert_allclose(actions[:, 0], [0.25, 0.3])


def test_bias_adds_to_the_output_and_a_zero_error_leaves_it():
    ctrl = one_loop(Pi(0, 2.0, 4.0), bias=lambda o, xp: xp.stack([0.25 * o[1]]))
    np.testing.assert_allclose(run(ctrl, [[0.4, 0.4]])[0], [0.1])


def test_integral_holds_while_the_output_is_saturated():
    # Error 10: output clips at 1, and the integral must not grow. When the error
    # falls to 0 the output is the integral alone, which has stayed at 0.
    ctrl = one_loop(Pi(0, 1.0, 1.0))
    actions = run(ctrl, [[0.0, 10.0]] * 3 + [[0.0, 0.0]])
    np.testing.assert_allclose(actions[:, 0], [1.0, 1.0, 1.0, 0.0])


def test_filter_starts_at_the_first_reading_and_then_lags():
    # alpha 0.5: first reading taken whole, the next halfway to the new reading.
    # Kc 1, Ti large enough to ignore: outputs are the filtered errors.
    ctrl = one_loop(Pi(0, 1.0, 1e9, alpha=0.5))
    actions = run(ctrl, [[0.2, 0.0], [0.4, 0.0]])
    np.testing.assert_allclose(actions[:, 0], [-0.2, -0.3], atol=1e-7)


def test_cascade_outer_output_is_the_inner_setpoint():
    # Outer: Kc 2, Ti huge, error 1 gives 2 above the inner rest 5, so setpoint 7.
    # Inner reading 6.5: error 0.5, Kc 0.2, Ti huge gives output 0.1.
    inner = Cascade(
        read=lambda o, xp: o[2],
        rest=lambda o, xp: 5.0,
        pi=Pi(0, 0.2, 1e9),
    )
    ctrl = one_loop(Pi(0, 2.0, 1e9), cascades=(inner,))
    actions = run(ctrl, [[0.0, 1.0, 6.5]])
    np.testing.assert_allclose(actions[0], [0.1], atol=1e-7)


def floor_override():
    """Keep ``obs[2]`` at or above 0.5, by MV 0 whose gain on it is positive."""
    return Override(
        read=lambda o, xp: o[2],
        limit=0.5,
        floor=True,
        mv=0,
        gain_sign=1.0,
        kc=1.0,
        ti=1e9,
    )


def test_override_leaves_the_output_alone_inside_the_limit():
    ctrl = one_loop(Pi(0, 1.0, 1e9), overrides=(floor_override(),))
    np.testing.assert_allclose(run(ctrl, [[0.0, 0.1, 0.9]])[0], [0.1])


def test_override_takes_the_output_past_the_limit():
    # The reading is 0.1 under its floor at 0.5, so 0.4 of error at Kc 1 raises the
    # MV above the 0.1 the main loop asks for: 0.1 + 0.4 from its reset at 0.1.
    ctrl = one_loop(Pi(0, 1.0, 1e9), overrides=(floor_override(),))
    actions = run(ctrl, [[0.0, 0.1, 0.9], [0.0, 0.1, 0.1]])
    np.testing.assert_allclose(actions[1], [0.5], atol=1e-6)


def test_integral_holds_while_an_override_has_the_output():
    # Ti 1, error 0.1: the integral would add 0.1 a step. Two overridden steps must
    # not add it, so the second step back inside reads 0.1 + 2 * 0.1, from its own
    # two steps only.
    ctrl = one_loop(Pi(0, 1.0, 1.0), overrides=(floor_override(),))
    actions = run(
        ctrl, [[0.0, 0.1, 0.1], [0.0, 0.1, 0.1], [0.0, 0.1, 0.9], [0.0, 0.1, 0.9]]
    )
    np.testing.assert_allclose(actions[3], [0.3], atol=1e-6)


def test_mix_routes_a_loop_to_its_paired_mv():
    def read(o, xp):
        return xp.stack([o[0], o[1]])

    def target(o, xp):
        return xp.stack([o[2], o[3]])

    ctrl = ladder_controller(
        Rung(
            name="crossed",
            loops=(Pi(0, 1.0, 1e9), Pi(1, 2.0, 1e9)),
            mix=np.array([[0.0, 1.0], [1.0, 0.0]]),
        ),
        read,
        target,
        lambda o, xp: xp.zeros(2),
        dt=1.0,
    )
    # Errors 0.1 and 0.2 give 0.1 and 0.4; crossed, MV 0 takes the second.
    np.testing.assert_allclose(run(ctrl, [[0.0, 0.0, 0.1, 0.2]])[0], [0.4, 0.1])
