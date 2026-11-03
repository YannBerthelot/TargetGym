"""Bump tests, fits, validity and retests (target_gym.identification).

Contract: docs/proposals/oracle-horizon-protocol.md, H8 to H11 as amended and
H14 to H19 (section 10), with D8 to D14:

- both structures are fitted by output-error least squares on the recorded
  input and recover the model a response was simulated from;
- a fit is valid when the response clears 10 noise bands of 6 sigma (V1), the
  fit settles inside the window (V2), no parameter sits at an upper bound
  (V3), and its NRMSE fit is at least 80 % (V4); sigma is the unstepped run's
  spread about its straight line;
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
    noise_sigma,
    nrmse_fit,
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


def test_nrmse_fit_and_noise_about_a_line():
    y = np.array([1.0, -1.0, 1.0, -1.0])
    assert nrmse_fit(y, y) == 100.0
    assert nrmse_fit(y, 0.5 * y) == pytest.approx(50.0)  # |y - yhat| = 1, |y| = 2
    drift = 3.0 + 0.5 * np.arange(400.0)
    assert noise_sigma(drift) == pytest.approx(0.0, abs=1e-9)
    assert noise_sigma(drift + 0.01 * (-1.0) ** np.arange(400)) == pytest.approx(
        0.01, rel=1e-3
    )


def test_v1_needs_ten_bands_of_six_sigma():
    # sigma 0.01: a band is 0.06, ten bands 0.6.
    t = np.arange(1, 201.0)
    base = 0.01 * (-1.0) ** np.arange(200)
    weak = judge(0.5 * fopdt(t, 1.0, 5.0, 0.0), base, np.ones((200, 1)), 1.0)
    strong = judge(0.7 * fopdt(t, 1.0, 5.0, 0.0), base, np.ones((200, 1)), 1.0)
    assert (weak["V1"], strong["V1"]) == (False, True)


def test_insignificant_when_inside_one_band_or_identically_zero():
    t = np.arange(1, 201.0)
    base = 0.01 * (-1.0) ** np.arange(200)
    assert insignificant(judge(np.zeros(200), base, np.ones((200, 1)), 1.0))
    assert insignificant(judge(0.05 * fopdt(t, 1, 5, 0), base, np.ones((200, 1)), 1.0))
    assert not insignificant(judge(fopdt(t, 1, 5, 0), base, np.ones((200, 1)), 1.0))


def test_evidence_names_the_structure_that_fits_and_the_cv_takes_the_majority():
    t = np.arange(1, 201.0)
    base, u = np.zeros(200), np.ones((200, 1))
    settled = judge(fopdt(t, 1.0, 10.0, 2.0), base, u, 1.0)
    ramp = judge(ramp_lag(t, 0.1, 5.0, 2.0), base, u, 1.0)
    assert (evidence(settled), evidence(ramp)) == ("fopdt", "integrating_lag")
    assert class_of([settled, ramp, ramp])[0] == "integrating_lag"
    assert class_of([settled, settled, ramp])[0] == "fopdt"


class FakePlant:
    """One MV, one CV with a trip limit at 5, dt 1 s, N_ep 100 steps: the
    response is k S(t) per unit step with seeded noise of ``sigma`` on the
    unstepped run. ``trip_at`` cuts the stepped run when no loop holds it."""

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
        dy = step * fopdt(t, self.k, self.tau, 0.0)
        trips = {} if m == window else {"step": (m, 0)}
        return Run(
            step,
            window,
            loops,
            dy[:, None],
            noise[:, None],
            np.full((m, 1), step),
            np.zeros(1),
            trips,
        )


def test_a_step_buried_in_noise_fails_v1_then_passes_once_doubled():
    # Settled move 0.1 against sigma 0.0022: 7.6 bands, then 15.2 doubled.
    plant = FakePlant(sigma=0.0022)
    pairs, _ = identify(plant, [0.1])
    p = pairs[0, 0, 0]
    assert [c[0] for c in plant.calls] == [0.1, 0.2]
    assert (p["label"], p["kept"]) == ("valid", 1)


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
    plant = FakePlant(k=0.0, sigma=0.01)
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
