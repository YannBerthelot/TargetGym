"""Single-run bump tests on noisy, late readings (target_gym.identification).

Contract: docs/proposals/expert-protocols-noise.md, N1 to N8 with its Decided
section, on C3 to C7 of docs/proposals/expert-protocols-process.md:

- N1: one run holds, records a baseline, steps and is fitted, with no twin, so
  the response keeps the sensor noise; a loop nobody holds is tested closed.
- N2: the noise band is 6 sd of the reading about a line, pooled over seeds and
  floored at one rounding step.
- N3: the step is the larger of the given one and what moves the reading 5 NB,
  rerun once; a limit that blocks it ships with the ratio printed.
- N4: the delay is read at the first exit from the band, with a persistence
  count, and held in the fit.
- N5 to N7: a PV filter keeps noise-driven output inside the dead band, capped
  at 10 % of Ti; the delay takes the filter, and past the cap Kc falls to
  DB / NB_f, never below Kc,min.
- N8: an analyser's counts are updates. Its reading is ``dead_time`` plus up to
  one update old, so a loop's delay is D + h / 2 at least, 1.5 cycles for a
  chromatograph, and a move cannot show for that long.
- C5: the median fit, unless the spread exceeds what the measured noise explains.

Synthetic responses are written here in closed form, and plants are real
environments, never the module.
"""

import math

import numpy as np
import pytest

from target_gym.base import FieldReading
from target_gym.identification import (
    TRAIN_SEEDS,
    Bump,
    Instrument,
    Loop,
    Plant,
    band_exit,
    fit,
    noise_band,
    noise_tune,
    pick_fit,
    simc,
    single_run,
    sized_step,
)
from target_gym.registry import REGISTRY

# first_order: dx/dt = (K u - x) / tau, K 1, tau 0.5 s, u in [-2, 2] for a raw
# action in [-1, 1]: 2 per raw unit, no delay of its own, dt 0.05 s.
K_RAW, TAU, DT = 2.0, 0.5, 0.05
SIGMA = 1.1e-3  # the reading's documented scatter at level 1
FAST = Instrument(DT)


def fopdt(t, k, tau, theta):
    return k * (1 - np.exp(-np.maximum(t - theta, 0.0) / tau))


def bumps(du, k, tau, theta, sigma=0.0, seeds=10, n=120, n_base=60, dt=1.0):
    """Ten single runs: Gaussian noise on a fopdt step, read every ``dt``."""
    rng = np.random.default_rng(0)
    t = dt * np.arange(1, n + 1)
    return [
        Bump(
            base=sigma * rng.standard_normal(n_base),
            dy=du * fopdt(t, k, tau, theta) + sigma * rng.standard_normal(n),
            U=np.full((n, 1), du),
            step=du,
        )
        for _ in range(seeds)
    ]


class Late:
    """A spec for first_order with its reading swapped, as an instrument PR
    gives a task: the real environment, on an instance of our own."""

    def __init__(self, reading):
        self.spec = REGISTRY["first_order"]
        self.env = self.spec.env_factory()
        self.env.readings = (reading,)

    def make_env(self):
        return self.env

    def make_pid_controller(self):
        return self.spec.make_pid_controller()


def _reading(dead=0, every=0, sigma=0.0, **kwargs):
    return FieldReading(
        "x",
        sigma=sigma,
        dead_time=dead * DT,
        interval=every * DT,
        sampled=every > 0,
        **kwargs,
    )


def _plant(spec, scale, seeds=(100, 101), **kwargs):
    params = REGISTRY["first_order"].make_test_params(sensor_noise_scale=scale)
    return Plant(spec, params, [lambda o, s, p: o[0]], [(None, None)], seeds=seeds)


# --- N2 ---------------------------------------------------------------------


def test_noise_band_is_six_sd_about_a_fitted_line():
    # [0, 1, 0] about its flat line: residuals -1/3, 2/3, -1/3, sum of squares
    # 2/3 on 1 degree of freedom, sd sqrt(2/3).
    assert noise_band([[0.0, 1.0, 0.0]]) == pytest.approx(6 * math.sqrt(2 / 3))


def test_a_drifting_baseline_keeps_its_noise_band():
    t = np.arange(40.0)
    y = np.resize([0.3, -0.1, -0.4, 0.2], 40)
    assert noise_band([y + 5.0 * t]) == pytest.approx(noise_band([y]), rel=1e-6)


def test_seeds_pool_their_degrees_of_freedom():
    # Sum of squares 2/3 over 1 + 1 degrees of freedom: sd sqrt(1/3).
    band = noise_band([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    assert band == pytest.approx(6 * math.sqrt(1 / 3))


def test_a_rounded_reading_has_at_least_one_step_of_band():
    assert noise_band([np.ones(8)], rounding=0.01) == 0.01


def test_noise_band_measures_gaussian_noise():
    # 580 degrees of freedom: the sd of an sd is 1 / sqrt(2 * 580), 3 of which
    # are 9 %.
    rng = np.random.default_rng(1)
    band = noise_band([0.2 * rng.standard_normal(60) for _ in range(10)])
    assert band == pytest.approx(6 * 0.2, rel=0.09)


# --- N4 ---------------------------------------------------------------------


def test_band_exit_needs_the_persistence_count_on_one_side():
    # Band +/- 0.5. One spike is noise, and so is a swing across the band.
    assert band_exit([0, 1, 0, 0, 1, 1, 1, 1], 1.0) == 4
    assert band_exit([1, -1, 1, -1, 1, -1], 1.0) is None
    assert band_exit([0, 0, -1, -1, -1], 1.0) == 2


def test_band_exit_is_none_for_a_response_inside_the_band():
    assert band_exit([0.1, 0.2, 0.1, 0.2], 1.0) is None


def test_a_held_delay_stays_in_the_fit():
    t = np.arange(1, 121.0)
    f = fit("fopdt", 1.0, 0.5 * fopdt(t, 2.0, 10.0, 4.0), np.full(120, 0.5), theta=4.0)
    np.testing.assert_allclose([f["k"], f["tau"]], [2.0, 10.0], rtol=1e-5)
    assert f["theta"] == pytest.approx(4.0)


def test_a_wrong_held_delay_is_not_refitted():
    t = np.arange(1, 121.0)
    f = fit("fopdt", 1.0, 0.5 * fopdt(t, 2.0, 10.0, 4.0), np.full(120, 0.5), theta=1.0)
    assert f["theta"] == pytest.approx(1.0) and f["tau"] > 10.0


def test_an_integrating_loop_takes_off_the_time_a_ramp_spends_in_the_band():
    # Slope 0.01 per s after 20 s, band 0.1: the ramp reaches 0.05 at 25 s and
    # the read delay would be 5 s late. The persistence count adds nothing, as
    # the first of the three exits is the one read.
    n, du = 200, 1.0
    t = np.arange(1, n + 1.0)
    ramp = 0.01 * np.maximum(t - 20.0, 0.0)
    run = Bump(base=np.zeros(60), dy=ramp, U=np.full((n, 1), du), step=du)
    read = single_run(
        lambda _: [run], du, 1.0, Instrument(1.0, rounding=0.1), "integrating_lag"
    )
    assert read.fits[0]["theta"] == pytest.approx(20.0, abs=1.0)
    assert read.fits[0]["k"] == pytest.approx(0.01, rel=0.02)


# --- N3 ---------------------------------------------------------------------


def test_step_grows_to_five_bands():
    # k 2, band 0.1: moving the reading 0.5 takes 0.25.
    assert sized_step(0.1, 2.0, 0.1, 1.0) == (pytest.approx(0.25), pytest.approx(5.0))


def test_a_step_that_clears_five_bands_stays():
    step, ratio = sized_step(0.5, 2.0, 0.1, 1.0)
    assert step == 0.5 and ratio == pytest.approx(10.0)


def test_a_limit_stops_the_step_and_the_ratio_says_so():
    step, ratio = sized_step(-0.1, 2.0, 0.1, 0.2)
    assert step == pytest.approx(-0.2) and ratio == pytest.approx(4.0)


def test_a_slope_model_sizes_the_step_on_its_record():
    # k' 0.01 per s over 100 s moves the reading 1 per unit: 0.5 / 1.
    assert sized_step(0.1, 0.01, 0.1, 1.0, record=100.0)[0] == pytest.approx(0.5)


def test_a_small_step_is_rerun_once_at_the_size_noise_asks():
    # sd 0.01, band 0.06, k 2: the first step moves the reading 0.04, under
    # 5 bands, so the rerun takes 5 x 0.06 / 2 = 0.15.
    calls = []

    def run(du):
        calls.append(du)
        return bumps(du, 2.0, 10.0, 3.0, sigma=0.01)

    read = single_run(run, 0.02, 1.0, Instrument(1.0))
    assert len(calls) == 2 and calls[0] == 0.02
    assert calls[1] == pytest.approx(0.15, rel=0.1)
    assert read.step == calls[1]
    assert read.ratio == pytest.approx(5.0, rel=0.1)


def test_a_step_that_reads_is_not_rerun():
    calls = []

    def run(du):
        calls.append(du)
        return bumps(du, 2.0, 10.0, 3.0, sigma=0.01)

    single_run(run, 0.5, 1.0, Instrument(1.0))
    assert calls == [0.5]


def test_a_test_too_small_to_read_ships_with_its_ratio():
    # Decided 3: sd 0.1, band 0.6, and the limit lets the reading move 0.02.
    read = single_run(
        lambda du: bumps(du, 2.0, 10.0, 0.0, sigma=0.1), 0.01, 0.01, Instrument(1.0)
    )
    assert read.step == 0.01 and read.ratio < 1.0
    assert len(read.fits) == 10


def test_a_test_cut_short_by_a_trip_is_left_out():
    good = bumps(0.5, 2.0, 10.0, 3.0, sigma=0.01)
    cut = Bump(base=good[0].base, dy=good[0].dy[:2], U=good[0].U[:2], step=0.5)
    read = single_run(lambda _: [*good, cut], 0.5, 1.0, Instrument(1.0))
    assert len(read.fits) == 10


# --- C5 ---------------------------------------------------------------------


def _fits(gains, half=0.0):
    return [
        {"model": "fopdt", "k": k, "tau": 10.0 * k, "theta": 1.0, "k_3sigma": half}
        for k in gains
    ]


def test_a_quiet_spread_under_the_limit_takes_the_median():
    chosen, spread, _ = pick_fit(_fits([1.0] * 9 + [1.4]))
    assert chosen["k"] == 1.0 and spread == pytest.approx(1.4)


def test_a_quiet_spread_over_the_limit_takes_the_highest_gain_with_its_own_model():
    chosen, spread, _ = pick_fit(_fits([1.0] * 9 + [2.0]))
    assert (chosen["k"], chosen["tau"]) == (2.0, 20.0) and spread == 2.0


def test_noise_keeps_the_median_where_it_explains_the_spread():
    # k 1 +/- 0.5 explains 1.5 / 0.5 = 3, and 1.5 x 3 covers a spread of 2.
    chosen, _, explained = pick_fit(_fits([1.0] * 9 + [2.0], half=0.5))
    assert chosen["k"] == 1.0 and explained == pytest.approx(3.0)


def test_a_spread_past_what_noise_explains_takes_the_highest_gain():
    # k 1 +/- 0.1 explains 1.1 / 0.9 = 1.22, and 1.5 x 1.22 = 1.83 is under 2.
    chosen, _, explained = pick_fit(_fits([1.0] * 9 + [2.0], half=0.1))
    assert chosen["k"] == 2.0 and explained == pytest.approx(1.1 / 0.9)


def test_gains_inside_their_own_noise_take_the_median():
    chosen, _, explained = pick_fit(_fits([0.1] * 9 + [5.0], half=1.0))
    assert chosen["k"] == 0.1 and explained == math.inf


# --- C7 and N5 to N7 --------------------------------------------------------


def test_simc_on_a_first_order_loop():
    # 1 / (2 (0.1 + 0.1)) and min(1, 4 x 0.2).
    assert simc(2.0, 1.0, 0.1, 0.1) == pytest.approx((2.5, 0.8))


def test_simc_on_an_integrating_loop():
    # 1 / (0.5 (1 + 1)) and 4 (1 + 1).
    assert simc(0.5, None, 1.0, 1.0) == pytest.approx((1.0, 8.0))


def test_simc_takes_a_third_of_the_delay_on_a_lag_shorter_than_it():
    # tau1 0.1 < theta 0.3 becomes 0.2: 0.2 / (1 x 0.6) and min(0.2, 2.4).
    assert simc(1.0, 0.1, 0.3, 0.3) == pytest.approx((1 / 3, 0.2))


def test_quiet_loops_take_no_filter():
    # Kc NB = 2.5 x 0.001 is under the dead band 0.02.
    out = noise_tune(2.0, 1.0, 0.1, 0.001, 0.05, 0.02)
    assert out["Tf"] == 0.0
    assert (out["Kc"], out["Ti"]) == pytest.approx((2.5, 0.8))


def test_non_valve_actuators_take_no_filter_and_no_detune():
    out = noise_tune(2.0, 1.0, 0.1, 1.0, 0.05, None)
    assert out["Tf"] == 0.0 and out["Kc"] == pytest.approx(2.5)


def test_the_filter_stays_under_a_tenth_of_the_first_ti():
    # First pass Kc 5, Ti 8, so the cap is 0.8 and any noise this size hits it.
    out = noise_tune(1.0, 10.0, 1.0, 1.0, 1.0, 0.02)
    assert out["Tf"] == pytest.approx(0.8)


def test_the_filter_enters_the_delay():
    # theta_eff = 1.8: Kc would be 10 / 3.6 before N7, Ti min(10, 14.4).
    out = noise_tune(1.0, 10.0, 1.0, 1e-9, 1.0, 1e9)
    assert out["theta"] == pytest.approx(1.0)
    capped = noise_tune(1.0, 10.0, 1.0, 1.0, 1.0, 2.0)
    assert capped["theta"] == pytest.approx(1.8)
    assert capped["Ti"] == pytest.approx(10.0)


def test_past_the_cap_kc_falls_to_what_the_dead_band_allows():
    # a = 1 - exp(-1 / 0.8) = 0.7135, sqrt(a / (2 - a)) = 0.7447, so NB_f is
    # 0.7447 and Kc = 0.02 / 0.7447 = 0.02686. tau_c + theta = 2 x 1.8 x
    # (10 / 3.6) / 0.02686 = 372.4, so Ti stays tau1.
    out = noise_tune(1.0, 10.0, 1.0, 1.0, 1.0, 0.02)
    assert out["Kc"] == pytest.approx(0.02686, rel=1e-3)
    assert out["Ti"] == pytest.approx(10.0)
    assert out["tau_c"] == pytest.approx(372.4 - 1.8, rel=1e-3)


def test_kc_scales_as_one_over_noise_once_it_binds():
    quiet = noise_tune(1.0, 10.0, 1.0, 1.0, 1.0, 0.02)["Kc"]
    loud = noise_tune(1.0, 10.0, 1.0, 2.0, 1.0, 0.02)["Kc"]
    assert loud == pytest.approx(quiet / 2)


def test_kc_never_falls_below_kc_min():
    out = noise_tune(1.0, 10.0, 1.0, 1.0, 1.0, 0.02, kc_min=0.05)
    assert out["Kc"] == pytest.approx(0.05)


def test_kc_min_never_raises_a_gain_simc_set_lower():
    out = noise_tune(1.0, 10.0, 1.0, 1.0, 1.0, 0.02, kc_min=100.0)
    assert out["Kc"] == pytest.approx(10 / 3.6)


def test_a_negative_gain_keeps_its_sign():
    assert noise_tune(-1.0, 10.0, 1.0, 1.0, 1.0, 0.02)["Kc"] == pytest.approx(
        -0.02686, rel=1e-3
    )


@pytest.mark.parametrize("nb", [0.01, 0.03, 0.1, 0.3])
def test_noise_driven_output_stays_inside_the_dead_band_or_the_filter_is_capped(nb):
    out = noise_tune(2.0, 1.0, 0.1, nb, 0.05, 0.02)
    a = 1 - math.exp(-0.05 / out["Tf"]) if out["Tf"] else 1.0
    output = abs(out["Kc"]) * nb * math.sqrt(a / (2 - a))
    assert output <= 0.02 * (1 + 1e-9)


# --- N8 ---------------------------------------------------------------------


def test_instrument_reads_a_chromatograph_from_its_reading():
    # distillation: every 5 min, 5 min old, one step of 1 min... in delta_t.
    spec = REGISTRY["distillation"]
    params = spec.make_test_params()
    late = next(r for r in spec.make_env().readings if r.dead_time > 0)
    got = Instrument.of(spec.make_env(), params, late.field)
    dt = float(params.delta_t)
    assert (got.every, got.dead_time) == (round(5.0 / dt), 5.0) and got.sampled


def test_a_reading_no_instrument_describes_is_fast():
    spec = REGISTRY["first_order"]
    params = spec.make_test_params()
    assert Instrument.of(spec.make_env(), params, "nothing") == FAST
    assert Instrument.of(spec.make_env(), params, "x") == FAST


def test_a_fast_reading_counts_steps():
    assert FAST.delay(3) == pytest.approx(0.15)
    assert FAST.half_sample == pytest.approx(0.025)


def test_a_chromatograph_loop_sees_one_and_a_half_cycles():
    # Cycle 5, dead time 5, step 1: moved at the first update after the dead
    # time, so the read says 5 and the half sample adds 2.5. That is McMillan's
    # 1.5 cycles for an analyser.
    gc = Instrument(1.0, every=5, dead_time=5.0, sampled=True)
    assert gc.delay(1) == 5.0 and gc.half_sample == 2.5


def test_a_read_on_a_coarse_grid_is_the_middle_of_what_it_can_tell():
    # Moved at update 2 of a 5 step cycle: the delay is 6 to 10 steps.
    gc = Instrument(1.0, every=5, dead_time=5.0, sampled=True)
    assert gc.delay(2) == 8.0


def test_no_exit_leaves_the_datasheet_delay():
    assert Instrument(1.0, every=5, dead_time=5.0, sampled=True).delay(None) == 5.0


def test_held_noise_on_a_current_value_keeps_the_half_step_rule():
    # The value is fresh at every step, only the noise is held.
    assert Instrument(1.0, every=5).half_sample == 0.5


def test_baseline_covers_enough_updates_for_a_line():
    assert FAST.baseline == 60
    assert Instrument(1.0, every=120).baseline == 960


def test_a_move_on_an_analyser_waits_for_the_update_that_can_show_it():
    # Acting on the update at step 0: the plant moves at step 1, and with a
    # dead time of 5 that is reported by the update at step 10, the second.
    gc = Instrument(1.0, every=5, dead_time=5.0, sampled=True)
    assert gc.gate(0.01, 5.0) == {"every": 5, "band": 0.01, "wait": 2}


def test_a_fast_loop_acts_at_every_step():
    assert FAST.gate(0.01, 0.0)["wait"] == 1


# --- N1 on real environments ------------------------------------------------


def test_the_response_keeps_the_sensor_noise_a_twin_run_would_cancel():
    # The fit's residual is the reading's noise. A twin on the same noise
    # would leave a residual of about zero.
    spec = REGISTRY["first_order"]
    plant = _plant(spec, 1.0)
    run = plant.bump(0, 0, 0, 0.1, 60, FAST)
    f = fit("fopdt", DT, run.dy, run.U)
    assert f["sigma"] == pytest.approx(SIGMA, rel=0.3)


def test_baseline_and_response_come_from_one_run_at_the_steady_point():
    plant = _plant(REGISTRY["first_order"], 0.0)
    run = plant.bump(0, 0, 0, 0.1, 60, FAST)
    assert len(run.base) == 60 and len(run.dy) == 60
    # One step of x' = (K u - x) / tau from rest, then the rest of the rise.
    expected = 0.1 * K_RAW * (1 - np.exp(-DT * np.arange(1, 61) / TAU))
    np.testing.assert_allclose(run.dy, expected * math.copysign(1, run.step), atol=1e-3)


@pytest.mark.parametrize("level", [0.5, 1.0, 2.0])
def test_first_order_returns_its_plant_at_every_level(level):
    spec = REGISTRY["first_order"]
    plant = _plant(spec, level, seeds=TRAIN_SEEDS)
    inst = Instrument.of(spec.make_env(), plant_params(level), "x")
    read = single_run(
        lambda du: [plant.bump(s, 0, 0, du, 60, inst) for s in range(10)],
        0.1,
        plant.room(0),
        inst,
    )
    chosen, spread, _ = pick_fit(read.fits)
    # Exact parameters; the fit's error is about sd / (du sqrt(60)) = 0.3 % at
    # 1x, and 3 % covers 2x and the held delay.
    np.testing.assert_allclose(
        [abs(chosen["k"]), chosen["tau"]], [K_RAW, TAU], rtol=0.03
    )
    assert read.nb == pytest.approx(6 * SIGMA * level, rel=0.1)
    assert read.ratio > 5.0


def plant_params(level):
    return REGISTRY["first_order"].make_test_params(sensor_noise_scale=level)


def test_a_late_sampled_reading_moves_where_its_datasheet_says():
    # Dead time 8 steps, an update every 4. The step lands one step before an
    # update, and the plant moves after that step. The update 8 steps after the
    # step's first effect reports it, the third on the grid, and the two
    # before report the plant at rest. The first effect is one step of the lag.
    late = Late(_reading(dead=8, every=4))
    inst = Instrument.of(late.env, plant_params(0.0), "x")
    plant = _plant(late, 0.0)
    run = plant.bump(0, 0, 0, 0.1, 40, inst)
    assert len(run.dy) == 10
    np.testing.assert_allclose(run.dy[:2], 0.0, atol=1e-5)
    assert abs(run.dy[2]) == pytest.approx(0.2 * (1 - math.exp(-0.1)), rel=2e-3)


def test_a_baseline_holds_one_value_per_update():
    # Sigma 0.01 held for 4 steps: repeated values would shrink the sd by the
    # degrees of freedom they claim. Decimated, one value per update.
    late = Late(_reading(every=4, sigma=0.01))
    inst = Instrument.of(late.env, plant_params(1.0), "x")
    run = _plant(late, 1.0).bump(0, 0, 0, 0.1, 40, inst)
    assert len(run.base) == 60 // 4
    assert len(set(run.base.tolist())) == len(run.base)


def test_a_synthetic_analyser_gives_its_delay_gain_and_lag():
    # Dead time 8 steps, cycle 4 steps, sd 2e-3 held per update, ten seeds.
    late = Late(_reading(dead=8, every=4, sigma=2e-3))
    params = plant_params(1.0)
    inst = Instrument.of(late.env, params, "x")
    plant = _plant(late, 1.0, seeds=TRAIN_SEEDS)
    read = single_run(
        lambda du: [plant.bump(s, 0, 0, du, 120, inst) for s in range(10)],
        0.1,
        plant.room(0),
        inst,
    )
    chosen, _, _ = pick_fit(read.fits)
    assert read.nb == pytest.approx(6 * 2e-3, rel=0.15)
    # The update grid can tell the delay to one cycle, 0.2 s, about the true 0.4.
    assert chosen["theta"] == pytest.approx(8 * DT, abs=4 * DT)
    np.testing.assert_allclose(
        [abs(chosen["k"]), chosen["tau"]], [K_RAW, TAU], rtol=0.1
    )


def test_sensor_lag_lands_in_the_fit():
    # Sensor lag 0.1 s on plant lag 0.5 s: their sum is what a first order fit
    # holds on to, whatever it gives the delay.
    late = Late(_reading(lag=0.1))
    params = plant_params(0.0)
    inst = Instrument.of(late.env, params, "x")
    run = _plant(late, 0.0).bump(0, 0, 0, 0.1, 80, inst)
    f = fit("fopdt", DT, run.dy, run.U)
    assert f["tau"] + f["theta"] == pytest.approx(TAU + 0.1, rel=0.05)


def test_a_closed_loop_test_recovers_the_open_loop_model():
    # N1's loop nobody holds: a P loop stays closed through the test, the step
    # goes on its setpoint and the fit reads the input it recorded.
    plant = _plant(REGISTRY["first_order"], 0.0)
    run = plant.bump(0, 0, 0, 0.1, 80, FAST, loops=(Loop(0, 0, 1.0, 60),))
    assert not np.allclose(run.U, run.U[0])  # the loop moved the input
    f = fit("fopdt", DT, run.dy, run.U)
    np.testing.assert_allclose([f["k"], f["tau"]], [K_RAW, TAU], rtol=0.02)
    assert len(run.base) == 60


def test_room_is_the_longest_step_every_seed_can_take_away_from_its_limit():
    plant = _plant(REGISTRY["first_order"], 0.0)
    u = plant.u0[:, 0]
    # Raw range [-1, 1], a step away from the nearer limit.
    assert plant.room(0) == pytest.approx(min(1.0 + np.abs(u)))
