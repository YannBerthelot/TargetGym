"""Physics validation for the nuclear reactor.

Enforces ``src/target_gym/reactor/PHYSICS.md``. Complements
``test_reactor_env.py``, which covers structure and behaviour; this file checks
the model against *reference* reactor physics -- the inhour equation, published
nuclear data, and the reactivity budget a real plant has to close.
"""

import numpy as np
import pytest
from scipy.optimize import brentq

from target_gym.reactor.env import (
    BETA_I,
    BETA_TOT,
    LAMBDA_I,
    LAMBDA_IODINE,
    LAMBDA_XENON,
    N_GROUPS,
    ReactorParams,
    steady_state_xenon,
)


@pytest.fixture(scope="module")
def params():
    return ReactorParams()


# ---------------------------------------------------------------------------
# 1. Nuclear data (PHYSICS.md section 2)
# ---------------------------------------------------------------------------


def test_delayed_neutron_fraction_matches_u235():
    """Keepin's six-group U-235 thermal data sums to beta = 0.0065."""
    assert BETA_TOT == pytest.approx(0.0065, rel=0.01)
    assert len(BETA_I) == N_GROUPS == 6
    assert len(LAMBDA_I) == N_GROUPS


def test_precursor_decay_constants_are_ordered_and_physical():
    """Groups run from the long-lived Br-87 (~55 s) to the shortest (~0.2 s)."""
    lam = np.asarray(LAMBDA_I)
    assert np.all(np.diff(lam) > 0), "groups should be ordered by decay constant"
    assert 1.0 / lam[0] == pytest.approx(80.6, rel=0.05)  # ~55 s half-life
    assert 1.0 / lam[-1] < 1.0


def test_iodine_and_xenon_half_lives_match_published_values():
    assert np.log(2) / LAMBDA_IODINE / 3600.0 == pytest.approx(6.57, rel=0.01)
    assert np.log(2) / LAMBDA_XENON / 3600.0 == pytest.approx(9.14, rel=0.01)


def test_xenon_decays_more_slowly_than_its_iodine_parent():
    """This ordering is what makes the pit form: iodine keeps feeding xenon."""
    assert LAMBDA_XENON < LAMBDA_IODINE


def test_generation_time_is_in_the_pwr_range(params):
    assert 1e-5 <= params.Lambda_gen <= 1e-4


def test_feedback_coefficients_are_negative_and_doppler_is_in_range(params):
    """Both negative is what makes a PWR passively self-regulating."""
    assert params.alpha_fuel < 0
    assert params.alpha_coolant < 0
    assert -4e-5 <= params.alpha_fuel <= -2e-5, "Doppler outside the PWR range"


def test_moderator_coefficient_is_weaker_than_a_real_pwr(params):
    """Documented deviation D1, asserted so it cannot drift unnoticed.

    A real PWR sits at -10 to -50 pcm/K. This model uses -5, which makes it
    less self-regulating against coolant swings than the real machine.
    """
    assert params.alpha_coolant == pytest.approx(-5e-5, rel=0.01)
    assert abs(params.alpha_coolant) < 10e-5


# ---------------------------------------------------------------------------
# 2. Kinetics -- the inhour equation
# ---------------------------------------------------------------------------


def _stable_period(rho, params):
    """Solve the inhour equation for the asymptotic reactor period."""
    b = np.asarray(BETA_I)
    lam = np.asarray(LAMBDA_I)

    def f(w):
        return params.Lambda_gen * w + np.sum(b * w / (w + lam)) - rho

    return 1.0 / brentq(f, 1e-9, 1e3)


def test_reactor_period_at_one_hundred_pcm(params):
    """+100 pcm on a PWR gives a period of tens of seconds.

    This is the load-bearing kinetics check: it ties the generation time and
    all twelve delayed-neutron constants together through the dispersion
    relation, and a period is what an operator would actually recognise.
    """
    assert 20.0 < _stable_period(100e-5, params) < 120.0


def test_period_shortens_as_reactivity_rises(params):
    periods = [_stable_period(p * 1e-5, params) for p in (10, 50, 100, 300)]
    assert periods[0] > periods[1] > periods[2] > periods[3]
    assert periods[0] > 300.0, "a small insertion should give a very long period"


def test_delayed_neutrons_dominate_below_prompt_critical(params):
    """Well below beta the period is set by the precursors, not by Lambda.

    Halving the generation time must barely move a sub-prompt period -- if it
    does, the delayed groups are not doing their job.
    """
    fast = ReactorParams(Lambda_gen=params.Lambda_gen / 2)
    assert _stable_period(100e-5, fast) == pytest.approx(
        _stable_period(100e-5, params), rel=0.05
    )


def test_rod_worth_never_reaches_prompt_critical(params):
    """Maximum rod withdrawal must stay safely below beta.

    At or above beta the chain reaction no longer needs delayed neutrons and
    the period collapses to milliseconds -- a regime this model has no business
    representing.
    """
    assert params.rho_ext_max < BETA_TOT
    assert params.rho_ext_max / BETA_TOT < 0.85


def test_rods_insert_faster_than_they_withdraw(params):
    """Insertion is the safety direction, so it is the fast one."""
    assert params.rod_speed_insert > params.rod_speed_withdraw
    assert params.rod_speed_insert / params.rod_speed_withdraw == pytest.approx(
        2.0, rel=0.1
    )


# ---------------------------------------------------------------------------
# 3. Thermal-hydraulics
# ---------------------------------------------------------------------------


def test_coolant_rise_across_the_core(params):
    """30-40 K between cold and hot leg for a PWR."""
    assert 28.0 < params.P_thermal_ref / params.m_dot_cp < 42.0


def test_fuel_runs_hot_but_below_the_trip(params):
    """Full-power fuel temperature must sit under the trip, far under melting."""
    T_c = params.T_inlet + params.P_thermal_ref / params.m_dot_cp
    T_f = T_c + params.P_thermal_ref / params.UA
    assert T_f < params.T_fuel_max
    assert T_f < 3120.0 * 0.5, "UO2 melts near 3120 K"
    assert T_f > T_c, "fuel must be hotter than the coolant it heats"


def test_fuel_time_constant_is_seconds_not_minutes(params):
    """Doppler feedback is prompt because the fuel responds fast."""
    assert 1.0 < params.C_fuel / params.UA < 30.0


def test_thermal_feedback_opposes_power(params):
    """A hotter core is a less reactive core, at every power level."""
    rho = []
    for n in (0.3, 0.6, 1.0):
        T_c = params.T_inlet + n * params.P_thermal_ref / params.m_dot_cp
        T_f = T_c + n * params.P_thermal_ref / params.UA
        rho.append(
            params.alpha_fuel * (T_f - params.T_fuel_ref)
            + params.alpha_coolant * (T_c - params.T_coolant_ref)
        )
    assert rho[0] > rho[1] > rho[2]


# ---------------------------------------------------------------------------
# 4. Reactivity budget -- can the rods actually hold each power level?
# ---------------------------------------------------------------------------


def _rod_worth_needed(n, params):
    T_c = params.T_inlet + n * params.P_thermal_ref / params.m_dot_cp
    T_f = T_c + n * params.P_thermal_ref / params.UA
    return -(
        params.alpha_fuel * (T_f - params.T_fuel_ref)
        + params.alpha_coolant * (T_c - params.T_coolant_ref)
    )


def test_rods_can_hold_every_power_in_the_target_range(params):
    """The required worth is an output of the model, not something tuned."""
    for n in np.linspace(params.target_n_range[0], params.target_n_range[1], 9):
        need = _rod_worth_needed(float(n), params)
        assert (
            params.rho_ext_min <= need <= params.rho_ext_max
        ), f"n={n:.2f} needs {need * 1e5:.0f} pcm, outside the rod range"


def test_full_power_leaves_almost_no_rod_margin(params):
    """Deliberate, and it is what gives the xenon pit its teeth.

    Holding full power consumes nearly all available worth, so a xenon
    overshoot makes returning to full power impossible for hours.
    """
    margin = params.rho_ext_max - _rod_worth_needed(1.0, params)
    assert 0.0 < margin < 100e-5, f"margin {margin * 1e5:.0f} pcm"


# ---------------------------------------------------------------------------
# 5. Xenon -- the environment's central mechanic
# ---------------------------------------------------------------------------


def _xenon_after_power_drop(n_after, params, hours=40.0, dt=4.0):
    """Integrate the normalised iodine/xenon chain from full-power equilibrium."""
    lam_I, lam_X, sph = LAMBDA_IODINE, LAMBDA_XENON, params.sigma_phi0
    g = params.gamma_ratio
    a = g * (lam_X + sph) / (1 + g)
    b = (lam_X + sph) / (1 + g)
    I = Xe = 1.0
    peak_t, peak_x = 0.0, 1.0
    for i in range(int(hours * 3600 / dt)):
        I += dt * lam_I * (n_after - I)
        Xe += dt * (a * n_after + b * I - (lam_X + sph * n_after) * Xe)
        if Xe > peak_x:
            peak_t, peak_x = (i + 1) * dt / 3600.0, Xe
    return peak_t, peak_x


def test_equilibrium_xenon_is_normalised_to_one(params):
    """At full power the poison contributes no reactivity by construction."""
    I_hat, Xe_hat = steady_state_xenon(1.0, params)
    assert float(I_hat) == pytest.approx(1.0, rel=1e-3)
    assert float(Xe_hat) == pytest.approx(1.0, rel=1e-3)


def test_xenon_builds_after_a_power_reduction(params):
    """The pit: iodine keeps decaying into xenon while burn-up collapses."""
    peak_t, peak_x = _xenon_after_power_drop(0.0, params)
    assert peak_x > 1.5, "no xenon overshoot after a scram"
    assert peak_t > 4.0, f"peak arrived after only {peak_t:.1f} h"


def test_xenon_peak_timing_is_early(params):
    """Documented deviation D2, asserted so it cannot drift further.

    A real scram peaks at 9-11 h. This model peaks around 8.3 h -- close
    enough that the reactivity consequence is right, but about an hour early.
    """
    peak_t, _ = _xenon_after_power_drop(0.0, params)
    assert 7.0 < peak_t < 9.5


def test_peak_xenon_worth_is_in_the_published_band(params):
    """The reactivity consequence, which is what actually matters."""
    _, peak_x = _xenon_after_power_drop(0.0, params)
    worth_pcm = params.rho_Xe_full * (peak_x - 1.0) * 1e5
    assert 1800.0 < worth_pcm < 3200.0


def test_deeper_power_cuts_make_bigger_and_later_pits(params):
    """Both the size and the delay scale with how far power was dropped."""
    results = [_xenon_after_power_drop(n, params) for n in (0.8, 0.4, 0.0)]
    peaks = [x for _, x in results]
    times = [t for t, _ in results]
    assert peaks[0] < peaks[1] < peaks[2]
    assert times[0] < times[1] < times[2]


def test_xenon_overshoot_exceeds_the_rod_margin(params):
    """Why the pit is a trap rather than an inconvenience.

    Even a modest power reduction builds more negative reactivity than the
    rods have spare at full power, so full power becomes unreachable.
    """
    margin = params.rho_ext_max - _rod_worth_needed(1.0, params)
    _, peak_x = _xenon_after_power_drop(0.8, params)
    overshoot = params.rho_Xe_full * (peak_x - 1.0)
    assert overshoot > margin


def test_xenon_worth_is_realistic_for_a_pwr(params):
    """Equilibrium xenon is worth roughly 2000-3500 pcm."""
    assert 0.020 <= params.rho_Xe_full <= 0.035
