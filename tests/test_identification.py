"""Bump tests, fits, validity and retests (target_gym.identification).

Contract: docs/proposals/oracle-horizon-protocol.md, H8 to H11 as amended and
H14 to H19 (section 10), with D8 to D14 and Amendment 2 (section 11):

- both structures are fitted by output-error least squares on the recorded
  input and recover the model a response was simulated from;
- a fit is valid when the response clears 10 noise bands of 6 sigma (V1), the
  fit settles inside the window (V2), no parameter sits at an upper bound
  (V3), and its fit is at least 80 % (V4); sigma is the spread of the
  residual between the response and the fitted model (N1), and the fit is
  measured against the pre-step value, so a pure gain is well-posed (N2);
- a failed test is retested at most twice: a doubled step for V1, a doubled
  window for V2 or V3, a P-only stabilised retest after a trip;
- a pair-seed is valid, insignificant (SNR < 1, or its gain's 3 sigma
  interval holds zero) or invalid, and a CV takes the majority class.

Synthetic responses are written here in closed form, never with the module.
"""

import math

import numpy as np
import pytest

from target_gym.identification import (
    Plant,
    Run,
    class_of,
    evidence,
    fit,
    identify,
    insignificant,
    judge,
    nrmse_fit,
    quiet_fits,
)
from target_gym.registry import REGISTRY


def fopdt(t, k, tau, theta):
    return k * (1 - np.exp(-np.maximum(t - theta, 0.0) / tau))


def ramp_lag(t, kp, tau2, theta):
    x = np.maximum(t - theta, 0.0)
    return kp * (x - tau2 * (1 - np.exp(-x / tau2)))


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
    f = fit("fopdt", dt, du * fopdt(t, k, tau, theta), np.full(120, du))
    np.testing.assert_allclose([f["k"], f["tau"]], [k, tau], rtol=1e-6)
    assert f["theta"] == pytest.approx(theta, abs=1e-6)
    assert f["V2"] and f["V3"] and f["V4"]


def test_integrating_fit_recovers_slope_lag_and_delay():
    t = 0.5 * np.arange(1, 201)
    f = fit("integrating_lag", 0.5, 0.2 * ramp_lag(t, 0.3, 5.0, 2.0), np.full(200, 0.2))
    np.testing.assert_allclose(
        [f["k"], f["tau2"], f["theta"]], [0.3, 5.0, 2.0], rtol=1e-6
    )


def test_fit_is_driven_by_the_recorded_input_not_a_step():
    # The input steps 0.1 at t = 0 and 0.1 more at t = 20: by superposition
    # the output is 0.1 S(t) + 0.1 S(t - 20).
    t = np.arange(1, 101.0)
    u = np.where(t > 20, 0.2, 0.1)
    dy = 0.1 * fopdt(t, 2.0, 5.0, 1.0) + 0.1 * fopdt(t - 20, 2.0, 5.0, 1.0)
    f = fit("fopdt", 1.0, dy, u)
    np.testing.assert_allclose(
        [f["k"], f["tau"], f["theta"]], [2.0, 5.0, 1.0], rtol=1e-6
    )


def test_fopdt_that_outlasts_the_window_fails_v2_and_a_long_lag_pins_v3():
    t = np.arange(1, 101.0)
    slow = fit(
        "fopdt", 1.0, fopdt(t, 1.0, 40.0, 0.0), np.ones(100)
    )  # 5 tau = 200 > 100
    # least_squares stops at a 1e-8 relative change of the cost; on this
    # flat valley that leaves about sqrt(1e-8) = 1e-4 on tau.
    assert slow["tau"] == pytest.approx(40.0, rel=1e-3) and not slow["V2"]
    # tau2 = 10 windows cannot be reached: the estimate sits at its bound.
    long = fit("integrating_lag", 1.0, ramp_lag(t, 1.0, 1000.0, 0.0), np.ones(100))
    assert not long["V3"]


def test_fit_is_measured_against_the_pre_step_value():
    # |y - yhat| / |y - 0|: 1 / 2 on the alternating signal, 1 / 2 on the flat
    # one, whose mean-referenced NRMSE would be 0 / 0.
    y = np.array([1.0, -1.0, 1.0, -1.0])
    assert nrmse_fit(y, y) == 100.0
    assert nrmse_fit(y, 0.5 * y) == pytest.approx(50.0)
    assert nrmse_fit(np.ones(4), np.full(4, 0.5)) == pytest.approx(50.0)


def test_a_pure_gain_is_a_valid_fopdt_with_no_lag():
    # The output moves 2 x 0.1 at the first sample and stays: k 2, fit 100 %.
    # Sampled at dt = 1 s, any tau under a tenth of a step reaches 1 - e^-10
    # by the first sample, so the data only bound it (10.3: "< dt").
    a = judge(np.full(50, 0.2), np.full((50, 1), 0.1), 1.0)
    f = a["fits"]["fopdt"]
    assert f["k"] == pytest.approx(2.0) and f["tau"] < 0.1 and f["fit"] > 99.9
    assert evidence(a) == "fopdt" and f["V4"]


ALT = 0.01 * (-1.0) ** np.arange(200)  # no first-order model can follow it


def test_sigma_is_the_residual_noise_about_the_fitted_model():
    t = np.arange(1, 201.0)
    a = judge(fopdt(t, 1.0, 5.0, 0.0) + ALT, np.ones((200, 1)), 1.0)
    assert a["fits"]["fopdt"]["sigma"] == pytest.approx(0.01, rel=1e-2)
    clean = judge(fopdt(t, 1.0, 5.0, 0.0), np.ones((200, 1)), 1.0)
    assert clean["fits"]["fopdt"]["sigma"] < 1e-6  # nothing but the model


def test_v1_needs_ten_bands_of_six_sigma():
    # sigma 0.01: a band is 0.06, ten bands 0.6; the peaks are 0.51 and 0.71.
    t = np.arange(1, 201.0)
    weak = judge(0.5 * fopdt(t, 1.0, 5.0, 0.0) + ALT, np.ones((200, 1)), 1.0)
    strong = judge(0.7 * fopdt(t, 1.0, 5.0, 0.0) + ALT, np.ones((200, 1)), 1.0)
    assert (weak["fits"]["fopdt"]["V1"], strong["fits"]["fopdt"]["V1"]) == (False, True)


def test_v4_needs_a_fit_of_80_percent():
    # S = 1 - e^(-t/5) over 200 s has |S| = 13.9; the alternating a ALT / 0.01
    # is what the fit leaves, |a alt| = 14.1 a, so the fit is about
    # 100 (1 - 14.1 a / sqrt(193 + 200 a^2)): 84.9 % at a = 0.15, 75.3 % at 0.25.
    t, u = np.arange(1, 201.0), np.ones((200, 1))
    good = judge(fopdt(t, 1.0, 5.0, 0.0) + 15 * ALT, u, 1.0)["fits"]["fopdt"]
    poor = judge(fopdt(t, 1.0, 5.0, 0.0) + 25 * ALT, u, 1.0)["fits"]["fopdt"]
    assert (good["V4"], poor["V4"]) == (True, False)


def test_insignificant_when_inside_one_band_or_identically_zero():
    # Peak 0.03 + 0.01 against a band of 0.06: under one band.
    t = np.arange(1, 201.0)
    assert insignificant(judge(np.zeros(200), np.ones((200, 1)), 1.0))
    assert insignificant(judge(0.03 * fopdt(t, 1, 5, 0) + ALT, np.ones((200, 1)), 1.0))
    assert not insignificant(judge(fopdt(t, 1, 5, 0) + ALT, np.ones((200, 1)), 1.0))


def test_evidence_names_the_structure_that_fits_and_the_cv_takes_the_majority():
    t = np.arange(1, 201.0)
    u = np.ones((200, 1))
    settled = judge(fopdt(t, 1.0, 10.0, 2.0), u, 1.0)
    ramp = judge(ramp_lag(t, 0.1, 5.0, 2.0), u, 1.0)
    assert (evidence(settled), evidence(ramp)) == ("fopdt", "integrating_lag")
    assert class_of([settled, ramp, ramp])[0] == "integrating_lag"
    assert class_of([settled, settled, ramp])[0] == "fopdt"


class FakePlant:
    """One MV, one CV with a trip limit at 5, dt 1 s, N_ep 100 steps: the
    response is k S(t) per unit step plus seeded noise of ``sigma``.
    ``trip_at`` cuts the stepped run when no loop holds it."""

    dt, n_ep, n_cv, n_mv, n_seeds = 1.0, 100, 1, 1, 1
    limits, mv_range, expert_mv = [(None, 5.0)], (-1.0, 1.0), {}

    def __init__(self, k=1.0, tau=5.0, sigma=0.0, trip_at=None):
        self.k, self.tau, self.sigma, self.trip_at = k, tau, sigma, trip_at
        self.u0, self.calls = np.zeros((1, 1)), []

    def direction(self, u, size):
        return size

    def test(self, s, j, step, window, loops):
        self.calls.append((step, window, loops))
        m = window if self.trip_at is None or loops else self.trip_at
        t = np.arange(1, m + 1.0)
        noise = self.sigma * np.random.default_rng(0).standard_normal(m)
        dy = step * fopdt(t, self.k, self.tau, 0.0) + noise
        trips = {} if m == window else {"step": (m, 0)}
        return Run(
            step,
            window,
            loops,
            dy[:, None],
            np.zeros((m, 1)),
            np.full((m, 1), step),
            np.zeros(1),
            trips,
        )


def test_a_step_buried_in_noise_fails_v1_then_passes_once_doubled():
    # A settled move of 0.1 in noise of sigma 0.0022 (a band 0.013): about
    # 8 bands, then about 16 once the step doubles.
    plant = FakePlant(sigma=0.0022)
    pairs, _ = identify(plant, [0.1])
    p = pairs[0, 0, 0]
    assert [c[0] for c in plant.calls] == [0.1, 0.2]
    assert (p["label"], p["kept"]) == ("valid", 1)


def test_the_step_grows_only_to_the_room_below_a_trip_limit():
    # SEB eq. 20-75 with the trip limit at 0.15 and the CV at 0: the doubled
    # step 0.2 is cut to (0.15 - 3 sigma) / K, sigma and K (near 1) those of
    # the failed test's fit.
    plant = FakePlant(sigma=0.0022)
    plant.limits = [(None, 0.15)]
    pairs, _ = identify(plant, [0.1])
    f = pairs[0, 0, 0]["tests"][0][1]["fits"]["fopdt"]
    assert plant.calls[1][0] == pytest.approx((0.15 - 3 * f["sigma"]) / f["k"])


def test_a_response_slower_than_the_window_gets_twice_the_window():
    # tau 30 s: 5 tau = 150 s is past N_ep = 100 steps, inside 200.
    plant = FakePlant(tau=30.0)
    pairs, _ = identify(plant, [0.1])
    assert [c[1] for c in plant.calls] == [100, 200]
    assert pairs[0, 0, 0]["label"] == "valid"


def test_a_trip_brings_a_p_only_setpoint_test_on_the_tripped_cv():
    # The cut test fits k 2, tau 10, theta 0; theta_P = 0 + dt / 2 = 0.5 s.
    # COOPER eq. 5.4: Kc = (0.2 / 2)(10 / 0.5)^1.22 = 0.1 x 20^1.22 = 3.86;
    # the loop runs 5 sqrt(10) 0.5 = 7.9, so 8 steps before the step.
    plant = FakePlant(k=2.0, tau=10.0, trip_at=30)
    pairs, _ = identify(plant, [0.1])
    (loop,) = plant.calls[1][2]
    assert (loop.cv, loop.mv, loop.n_pre) == (0, 0, 8)
    assert loop.kc == pytest.approx(0.1 * 20**1.22, rel=1e-3)
    assert pairs[0, 0, 0]["label"] == "valid"


def test_a_zero_response_is_insignificant_and_never_retested():
    plant = FakePlant(k=0.0)
    pairs, classes = identify(plant, [0.1])
    assert len(plant.calls) == 1 and pairs[0, 0, 0]["label"] == "insignificant"
    assert classes[0]["class"] is None


def test_bump_test_on_first_order_returns_its_documented_lag():
    # first_order is dx/dt = (K u - x) / tau with K 1, tau 0.5 s, and u in
    # [-2, 2] for a raw action in [-1, 1]: 2 per raw unit, no delay.
    spec = REGISTRY["first_order"]
    plant = Plant(
        spec,
        spec.make_test_params(),
        [lambda o, s, p: s.x],
        [(-3.0, 3.0)],
        seeds=(100, 101),
    )
    for s in range(2):
        run = plant.test(s, 0, 0.1, 60, ())
        f = fit("fopdt", plant.dt, run.dy[:, 0], run.U[:, :1])
        np.testing.assert_allclose([f["k"], f["tau"]], [2.0, 0.5], rtol=1e-3)
        assert f["theta"] == pytest.approx(0.0, abs=1e-3) and math.isfinite(f["fit"])


def _analysis(fit_percent=96.0, **failed):
    """One response's analysis, its FOPDT fit failing the checks in ``failed``."""
    checks = {"V1": True, "V2": True, "V3": True, "V4": True, **failed}
    return {"zero": False, "fits": {"fopdt": {"fit": fit_percent, **checks}}}


def _pairs(analysis, label="invalid"):
    return {(0, 0, 0): {"tests": [(None, analysis)], "label": label}}


FOPDT = {0: {"class": "fopdt"}}


def test_a_fit_failing_v1_alone_at_95_percent_is_quiet():
    """Amendment 4: V1 failed, the rest passed, a fit of at least 95 %."""
    assert list(quiet_fits(_pairs(_analysis(95.0, V1=False)), FOPDT)) == [(0, 0, 0, 0)]


@pytest.mark.parametrize(
    "analysis",
    [
        _analysis(94.9, V1=False),  # under the bar
        _analysis(99.0, V1=False, V2=False),  # V1 not alone
        _analysis(99.0),  # valid, which needs no exception
    ],
)
def test_a_fit_that_is_not_quiet_stays_out(analysis):
    assert quiet_fits(_pairs(analysis), FOPDT) == {}


def test_a_zero_model_has_no_quiet_fit():
    """As H21: an insignificant pair is noise, whatever one test's fit read."""
    pairs = _pairs(_analysis(99.0, V1=False), label="insignificant")
    assert quiet_fits(pairs, FOPDT) == {}


def test_a_quiet_fit_is_read_under_its_cvs_class():
    """Only the structure the CV's class names counts (H17)."""
    pairs = _pairs(_analysis(99.0, V1=False))
    assert quiet_fits(pairs, {0: {"class": "integrating_lag"}}) == {}


def test_every_test_of_a_pair_is_read_not_only_the_kept_one():
    """A retest can keep a fast valid fit while an earlier test held the slow
    quiet one: hvac's seed 103 kept its 979 s fit over its 139,000 s one."""
    pairs = {
        (0, 0, 0): {
            "tests": [(None, _analysis(98.0, V1=False)), (None, _analysis(92.0))],
            "label": "valid",
        }
    }
    assert list(quiet_fits(pairs, FOPDT)) == [(0, 0, 0, 0)]
