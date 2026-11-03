"""Bump tests, the class rule and the fits (target_gym.identification).

Contract (docs/proposals/oracle-horizon-protocol.md H8 to H11, the expert
protocols C1 and C2): a fit recovers the parameters of the model it names
from that model's exact step response; H10 calls a response integrating when
its last quarter's mean slope is at least half its first quarter's; a bump
test on a plant of known dynamics returns that plant's step response.
"""

import numpy as np
import pytest

from target_gym.identification import (
    bump_test,
    fit_fopdt,
    fit_integrating_lag,
    identify,
    is_integrating,
)
from target_gym.registry import REGISTRY


def fopdt(t, k, tau, theta, du):
    return k * du * (1 - np.exp(-np.maximum(t - theta, 0.0) / tau))


@pytest.mark.parametrize(
    "k, tau, theta, du, dt",
    [
        (2.5, 4.0, 1.5, 0.1, 0.25),
        (-0.8, 10.0, 0.0, -0.05, 1.0),
        (3.0, 2.0, 1.3, 1.0, 0.5),
    ],
)
def test_fopdt_fit_recovers_the_model_it_was_simulated_from(k, tau, theta, du, dt):
    t = dt * np.arange(1, 121)
    fit = fit_fopdt(t, fopdt(t, k, tau, theta, du), du)
    np.testing.assert_allclose([fit["k"], fit["tau"]], [k, tau], rtol=1e-6)
    assert fit["theta"] == pytest.approx(theta, abs=1e-6)


def test_integrating_lag_fit_recovers_slope_lag_and_delay():
    # k' e^(-theta s) / (s (tau2 s + 1)), stepped by du: after the delay the
    # output is k' du (x - tau2 (1 - e^(-x / tau2))), x = t - theta.
    kp, tau2, theta, du = 0.3, 5.0, 2.0, 0.2
    t = 0.5 * np.arange(1, 201)
    x = np.maximum(t - theta, 0.0)
    dy = kp * du * (x - tau2 * (1 - np.exp(-x / tau2)))
    fit = fit_integrating_lag(t, dy, du)
    np.testing.assert_allclose([fit["k"], fit["tau2"]], [kp, tau2], rtol=1e-6)
    assert fit["theta"] == pytest.approx(theta, abs=1e-6)


def _first_order_seen_over(window_in_tau, n=400):
    # d[0] at the step, then 4q samples. For 1 - e^(-t / tau) the ratio of the
    # last quarter's rise to the first's is e^(-3 W / (4 tau)), which is 1/2 at
    # W = 4 ln 2 / 3 tau = 0.924 tau.
    return 1 - np.exp(-np.linspace(0.0, window_in_tau, n + 1))


@pytest.mark.parametrize(
    "signal, integrating",
    [
        (np.arange(401.0), True),  # a ramp
        (_first_order_seen_over(8.0), False),  # settled: ratio e^-6
        (_first_order_seen_over(0.8), True),  # ratio e^-0.6 = 0.55
        (_first_order_seen_over(1.2), False),  # ratio e^-0.9 = 0.41
        (np.zeros(401), False),  # no move is not a ramp
    ],
)
def test_class_rule_compares_the_last_quarter_slope_with_the_first(signal, integrating):
    assert is_integrating(signal) is integrating


def test_identify_fits_the_model_its_class_names():
    t = np.arange(1, 101.0)
    from target_gym.identification import Response

    ramp = identify(Response(0.5 * 0.1 * t, 1.0, 0.1))
    lag = identify(Response(fopdt(t, 2.0, 5.0, 0.0, 0.1), 1.0, 0.1))
    assert (ramp["class"], ramp["model"]) == ("integrating", "integrating_lag")
    assert (lag["class"], lag["model"]) == ("self-regulating", "fopdt")
    assert identify(Response(np.zeros(50), 1.0, 0.1))["class"] == "no response"


def test_bump_test_on_first_order_returns_its_documented_lag():
    # first_order is dx/dt = (K u - x) / tau with K 1, tau 0.5 s, and u in
    # [-2, 2] for a raw action in [-1, 1]: 2 per raw unit, no delay.
    spec = REGISTRY["first_order"]
    p = spec.make_test_params()
    out = bump_test(
        spec,
        p,
        {"x": lambda o, s, p: s.x},
        [0.1],
        n_settle=40,
        n_window=60,
        seeds=(100, 101),
    )
    for r in out[(0, "x")]:
        assert abs(r.du) == 0.1 and len(r.dy) == 60
        fit = fit_fopdt(r.t, r.dy, r.du)
        np.testing.assert_allclose([fit["k"], fit["tau"]], [2.0, 0.5], rtol=1e-3)
        assert fit["theta"] == pytest.approx(0.0, abs=1e-3)
