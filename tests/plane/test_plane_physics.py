"""Physics validation for the 2D plane model.

These tests enforce ``src/target_gym/plane/PHYSICS.md``. They assert *emergent*
behaviour -- ISA table values, figures of merit, monotonicity, integrator
convergence -- rather than re-stating the formula under test. A test that
re-implements its subject validates transcription, not correctness: if the
formula is wrong, the test is wrong the same way. The previous version of this
module did exactly that (``test_newton_second_law`` recomputed the function's
own expression), which is why none of the deviations in §5 were caught.

Reference aircraft: Airbus A320-200, clean, cruise.
"""

import numpy as np
import pytest
from scipy.optimize import brentq

from target_gym.plane.dynamics import (
    aero_coefficients,
    compute_air_density_from_altitude,
    compute_drag,
    compute_speed_of_sound_from_altitude,
    compute_thrust_output,
    compute_weight,
)
from target_gym.plane.env import PlaneParams

KNOT = 0.514444  # m/s

# A320 geometry used to derive expectations (PHYSICS.md §3).
A320_SPAN_M = 34.1
OSWALD_E = 0.85

# Cruise reference condition: FL350, M0.78.
CRUISE_ALT_M = 10_668.0
CRUISE_MACH = 0.78


@pytest.fixture(scope="module")
def params():
    return PlaneParams()


def _CL(aoa_deg, mach, params):
    return float(aero_coefficients(float(aoa_deg), float(mach), params)[0])


def _CD(aoa_deg, mach, params):
    return float(aero_coefficients(float(aoa_deg), float(mach), params)[1])


def _peak_CL(params, mach, lo=-5.0, hi=25.0, n=3001):
    """Maximum lift coefficient the model can actually produce."""
    aoas = np.linspace(lo, hi, n)
    cls = np.array([_CL(a, mach, params) for a in aoas])
    i = int(cls.argmax())
    return float(cls[i]), float(aoas[i])


# ---------------------------------------------------------------------------
# 1. Atmosphere -- validated against the ISA table (ISO 2533 / ICAO Doc 7488)
# ---------------------------------------------------------------------------

# (altitude m, density kg/m^3, speed of sound m/s)
ISA_TABLE = [
    (0.0, 1.22500, 340.294),
    (5_000.0, 0.73643, 320.545),
    (11_000.0, 0.36392, 295.070),
]


@pytest.mark.parametrize("altitude,rho_ref,a_ref", ISA_TABLE)
def test_isa_density_matches_published_table(altitude, rho_ref, a_ref):
    """ISA density within 0.1% of the tabulated standard."""
    rho = float(compute_air_density_from_altitude(altitude))
    assert rho == pytest.approx(rho_ref, rel=1e-3)


@pytest.mark.parametrize("altitude,rho_ref,a_ref", ISA_TABLE)
def test_isa_speed_of_sound_matches_published_table(altitude, rho_ref, a_ref):
    """ISA speed of sound within 0.1% of the tabulated standard."""
    a = float(compute_speed_of_sound_from_altitude(altitude))
    assert a == pytest.approx(a_ref, rel=1e-3)


def test_density_decreases_monotonically_with_altitude():
    alts = np.linspace(0.0, 11_000.0, 50)
    rho = np.array([float(compute_air_density_from_altitude(h)) for h in alts])
    assert np.all(np.diff(rho) < 0.0)


# ---------------------------------------------------------------------------
# 2. Airframe figures of merit (PHYSICS.md §4)
# ---------------------------------------------------------------------------


def test_max_lift_to_drag_ratio_matches_a320(params):
    """L/D_max = 1/(2*sqrt(cd0*k)) should land near the A320's published ~17.

    This is the single strongest check that cd0 and k are jointly sane: it is
    independent of wing area, mass and altitude.
    """
    ld_max = 1.0 / (2.0 * np.sqrt(params.cd0 * params.k))
    assert 15.0 < ld_max < 19.0, f"L/D_max={ld_max:.1f} implausible for an airliner"


def test_induced_drag_factor_consistent_with_aspect_ratio(params):
    """k must correspond to the real wing geometry for a defensible Oswald e.

    k = 1/(pi*AR*e). With AR = 9.48 this pins e; anything outside 0.6-1.0 means
    k and the wing geometry disagree.
    """
    aspect_ratio = A320_SPAN_M**2 / params.wings_surface
    assert aspect_ratio == pytest.approx(9.48, abs=0.05)
    implied_e = 1.0 / (np.pi * aspect_ratio * params.k)
    assert 0.6 < implied_e < 1.0, f"k={params.k} implies Oswald e={implied_e:.2f}"


def test_cruise_lift_coefficient_in_typical_jet_range(params):
    """CL required to hold FL350 at M0.78 should be a typical jet cruise CL."""
    rho = float(compute_air_density_from_altitude(CRUISE_ALT_M))
    a = float(compute_speed_of_sound_from_altitude(CRUISE_ALT_M))
    v = CRUISE_MACH * a
    weight = compute_weight(params.initial_mass, params.gravity)
    cl_required = 2.0 * weight / (rho * v**2 * params.wings_surface)
    assert 0.35 < cl_required < 0.75, f"cruise CL={cl_required:.3f}"


def test_level_flight_trim_exists_at_cruise(params):
    """A level-flight equilibrium must exist at the cruise condition.

    If no AoA in the usable range generates exactly enough lift, the aircraft
    cannot cruise at all and the whole altitude-hold task is ill-posed.
    """
    rho = float(compute_air_density_from_altitude(CRUISE_ALT_M))
    a = float(compute_speed_of_sound_from_altitude(CRUISE_ALT_M))
    v = CRUISE_MACH * a
    weight = compute_weight(params.initial_mass, params.gravity)
    q_s = 0.5 * rho * v * v * params.wings_surface

    residual = lambda aoa: q_s * _CL(aoa, CRUISE_MACH, params) - weight  # noqa: E731
    assert residual(-5.0) < 0.0 < residual(12.0), "no sign change: trim bracket invalid"
    aoa_trim = brentq(residual, -5.0, 12.0)
    # Well inside the stall boundary, and positive (cambered wing at cruise).
    assert 0.0 < aoa_trim < params.aoa_stall - 2.0


def test_thrust_exceeds_drag_at_cruise(params):
    """Available thrust at FL350 must exceed cruise drag, or cruise is impossible."""
    rho = float(compute_air_density_from_altitude(CRUISE_ALT_M))
    a = float(compute_speed_of_sound_from_altitude(CRUISE_ALT_M))
    v = CRUISE_MACH * a
    weight = compute_weight(params.initial_mass, params.gravity)
    q_s = 0.5 * rho * v * v * params.wings_surface

    residual = lambda aoa: q_s * _CL(aoa, CRUISE_MACH, params) - weight  # noqa: E731
    aoa_trim = brentq(residual, -5.0, 12.0)
    drag = q_s * _CD(aoa_trim, CRUISE_MACH, params)

    thrust = float(
        compute_thrust_output(
            power=1.0,
            thrust_output_at_sea_level=params.thrust_output_at_sea_level,
            M=CRUISE_MACH,
            rho=rho,
        )
    )
    assert thrust > drag, f"thrust {thrust/1e3:.1f} kN < drag {drag/1e3:.1f} kN"


# ---------------------------------------------------------------------------
# 3. Structural / monotonicity properties (first principles, no source needed)
# ---------------------------------------------------------------------------


def test_drag_scales_quadratically_with_speed():
    """Doubling airspeed must quadruple dynamic-pressure drag."""
    base = float(compute_drag(S=122.6, C=0.03, V=100.0, rho=1.0))
    double = float(compute_drag(S=122.6, C=0.03, V=200.0, rho=1.0))
    assert double == pytest.approx(4.0 * base, rel=1e-6)


def test_drag_scales_linearly_with_density():
    base = float(compute_drag(S=122.6, C=0.03, V=100.0, rho=0.5))
    double = float(compute_drag(S=122.6, C=0.03, V=100.0, rho=1.0))
    assert double == pytest.approx(2.0 * base, rel=1e-6)


def test_lift_increases_with_angle_of_attack_below_stall(params):
    aoas = np.linspace(0.0, params.aoa_stall - 5.0, 20)
    cls = [_CL(a, 0.3, params) for a in aoas]
    assert np.all(np.diff(cls) > 0.0)


def test_lift_collapses_beyond_stall(params):
    """Past the stall AoA, lift collapses -- measured where the collapse is.

    This used to sample 10 deg past the peak and demand under 35 % of it, which
    the old model satisfied because its lift went to *zero* and stayed there.
    That was the same defect that left a separated wing with less drag than in
    cruise. A real wing sheds most of its lift immediately past the stall and
    then follows the flat-plate curve back up, so the collapse is asserted just
    past the peak, and the recovery is bounded rather than forbidden.
    """
    cl_peak, aoa_peak = _peak_CL(params, mach=0.3)
    # the depth of the break, not the value at an arbitrary angle: a point
    # assertion lands wherever the flat-plate recovery happens to be
    break_region = [
        _CL(a, 0.3, params) for a in np.linspace(aoa_peak, aoa_peak + 15.0, 40)
    ]
    assert (
        min(break_region) < 0.45 * cl_peak
    ), f"no stall break: lift only fell to {min(break_region)/cl_peak:.0%} of peak"
    # the flat-plate branch tops out at CL = 1 (sin 2a at 45 deg), so lift must
    # never return to anything like the attached-flow peak
    beyond = [_CL(a, 0.3, params) for a in np.linspace(aoa_peak + 5.0, 90.0, 30)]
    assert max(beyond) < 0.7 * cl_peak, "post-stall lift recovered too far"


def test_drag_rises_when_the_wing_separates(params):
    """The contract the old model was missing entirely.

    ``CD = cd0 + k*CL**2`` ties drag to lift, so a stall sigmoid that collapses
    CL collapsed CD with it: a fully separated wing came out at CD = cd0, less
    drag than in cruise. A departed aircraft then had no aerodynamic forces at
    all -- it fell at 300 m/s and tumbled undamped, because nothing opposed the
    rotation either.
    """
    cd_cruise = _CD(2.0, 0.3, params)
    cd_stalled = _CD(params.aoa_stall + 15.0, 0.3, params)
    cd_broadside = _CD(90.0, 0.3, params)

    assert cd_stalled > 3 * cd_cruise, "separation must cost drag"
    # Side-on, the wing develops its full separated normal force. That peak is
    # not assumed: Viterna & Corrigan (1981) give CD_max = 1.11 + 0.018 AR for
    # a finite wing, so it follows from the aspect ratio the geometry already
    # fixes. A 2D flat plate would reach 2.0; this wing, relieving around its
    # tips, reaches about 1.3.
    expected = 1.11 + 0.018 * params.aspect_ratio + params.cd0
    assert cd_broadside == pytest.approx(expected, rel=0.05), (
        f"broadside CD {cd_broadside:.2f} does not match the aspect ratio's "
        f"{expected:.2f}"
    )


def test_aerodynamics_are_defined_at_any_attitude(params):
    """``theta`` is not wrapped, so incidence can arrive here unbounded."""
    for aoa in (-720.0, -180.0, 0.0, 180.0, 540.0, 1331.0):
        cl, cd = _CL(aoa, 0.3, params), _CD(aoa, 0.3, params)
        assert np.isfinite(cl) and np.isfinite(cd), f"non-finite at {aoa} deg"
        assert abs(cl) <= 2.0 and 0.0 <= cd <= 2.5, f"out of range at {aoa} deg"
    # and it must be periodic: 720 deg of incidence is 0 deg of incidence
    assert _CL(720.0, 0.3, params) == pytest.approx(_CL(0.0, 0.3, params), rel=1e-6)


def test_drag_rises_beyond_critical_mach(params):
    """Transonic drag divergence: CD must climb once M exceeds M_crit."""
    aoa = 2.0
    cd_sub = _CD(aoa, params.M_crit - 0.10, params)
    cd_at = _CD(aoa, params.M_crit, params)
    cd_super = _CD(aoa, params.M_crit + 0.05, params)
    assert cd_at == pytest.approx(cd_sub, rel=0.35)
    assert cd_super > cd_at


def test_thrust_lapses_with_altitude(params):
    """Turbofan thrust must fall with density; roughly proportional to it."""
    thrusts = [
        float(
            compute_thrust_output(
                power=1.0,
                thrust_output_at_sea_level=params.thrust_output_at_sea_level,
                M=0.5,
                rho=float(compute_air_density_from_altitude(h)),
            )
        )
        for h in (0.0, 5_000.0, 11_000.0)
    ]
    assert thrusts[0] > thrusts[1] > thrusts[2]
    # At 11 km, density is 29.7% of sea level; thrust should be in that region.
    assert 0.15 < thrusts[2] / thrusts[0] < 0.45


def test_thrust_is_proportional_to_throttle(params):
    kwargs = dict(
        thrust_output_at_sea_level=params.thrust_output_at_sea_level, M=0.3, rho=1.225
    )
    half = float(compute_thrust_output(power=0.5, **kwargs))
    full = float(compute_thrust_output(power=1.0, **kwargs))
    assert half == pytest.approx(0.5 * full, rel=1e-6)


def test_zero_throttle_gives_zero_thrust(params):
    thrust = float(
        compute_thrust_output(
            power=0.0,
            thrust_output_at_sea_level=params.thrust_output_at_sea_level,
            M=0.3,
            rho=1.225,
        )
    )
    assert thrust == pytest.approx(0.0, abs=1e-9)


def test_weight_is_mass_times_gravity():
    assert float(compute_weight(1000.0, 9.81)) == pytest.approx(9810.0)


# ---------------------------------------------------------------------------
# 4. Mass bookkeeping (PHYSICS.md §3)
# ---------------------------------------------------------------------------


def test_reported_mass_is_physically_possible(params):
    """The modelled aircraft must not exceed the A320's 78 t MTOW.

    ``state.m`` used to report ``initial_mass + fuel`` = 92 588 kg, which both
    exceeded MTOW and disagreed with the mass the dynamics actually integrate.
    """
    A320_MTOW_KG = 78_000.0
    A320_OEW_KG = 42_600.0
    assert A320_OEW_KG < params.initial_mass <= A320_MTOW_KG
    # Fuel is a component of the all-up mass, not additional to it.
    assert params.initial_fuel_quantity < params.initial_mass - A320_OEW_KG + 1e-6


def test_mach_crit_declared_once(params):
    """Guard against the duplicate-field regression fixed in PHYSICS.md §4.

    ``M_crit`` was declared twice in ``PlaneParams`` (0.78 then 0.80); Python
    kept the last, silently discarding the first.
    """
    import re
    from pathlib import Path

    import target_gym.plane.env as env_mod

    source = Path(env_mod.__file__).read_text()
    declarations = re.findall(r"^\s+M_crit\s*:\s*float", source, flags=re.MULTILINE)
    assert len(declarations) == 1, f"M_crit declared {len(declarations)}x"
    assert 0.7 < params.M_crit < 0.9


# ---------------------------------------------------------------------------
# 5. Lift curve: slope, attainable CL_max, stall speed (PHYSICS.md §4)
#    (formerly deviations D1/D2 -- fixed)
# ---------------------------------------------------------------------------


def test_stall_speed_matches_published_value(params):
    """Clean stall speed at sea level should be ~145-150 kt."""
    cl_peak, _ = _peak_CL(params, mach=0.2)
    weight = compute_weight(params.initial_mass, params.gravity)
    v_stall = np.sqrt(2.0 * weight / (1.225 * params.wings_surface * cl_peak))
    assert 135.0 < v_stall / KNOT < 165.0


def test_cl_max_is_attainable(params):
    cl_peak, _ = _peak_CL(params, mach=0.2)
    assert cl_peak == pytest.approx(params.CL_max, rel=0.10)


def test_lift_slope_consistent_with_aspect_ratio(params):
    """Prandtl finite-wing lift slope: a = a0 / (1 + a0/(pi*AR*e))."""
    aspect_ratio = A320_SPAN_M**2 / params.wings_surface
    a0 = 2.0 * np.pi  # per radian, thin-airfoil theory
    slope_per_rad = a0 / (1.0 + a0 / (np.pi * aspect_ratio * OSWALD_E))
    slope_per_deg = slope_per_rad * np.pi / 180.0
    assert params.cl_alpha == pytest.approx(slope_per_deg, rel=0.25)


def test_max_lift_falls_above_the_critical_mach(params):
    """Shock-induced separation cuts the attainable lift past M_crit.

    This was deviation D3, and the old form of the test compared M 0.20 with
    M 0.78 -- both *below* the critical Mach number, where Prandtl-Glauert
    amplification is correct physics and peak lift genuinely does rise. The
    defect was narrower than that: the stall clamp was applied *before* the
    1/beta factor, so peak lift went on rising past M_crit as well, when lift
    divergence should make it collapse.

    The model now caps lift at the Prandtl-Glauert-scaled stall limit up to
    M_crit and lets it fall beyond, so the rise below and the fall above are
    both represented.
    """
    cl_crit, _ = _peak_CL(params, mach=params.M_crit)
    cl_beyond, _ = _peak_CL(params, mach=0.92)
    assert cl_beyond < cl_crit, "peak lift must fall past the critical Mach"
    assert cl_beyond < 0.85 * cl_crit, "the fall should be substantial, not marginal"


def test_prandtl_glauert_rise_is_kept_below_the_critical_mach(params):
    """The complement: fixing D3 must not remove the effect that is real.

    The lift *slope* does grow with Mach in attached flow, so peak lift rises
    with Mach up to M_crit. A fix that simply clamped lift to its low-speed
    limit would pass the test above and be wrong.
    """
    cl_low, _ = _peak_CL(params, mach=0.20)
    cl_crit, _ = _peak_CL(params, mach=params.M_crit)
    assert cl_crit > cl_low


# ---------------------------------------------------------------------------
# 6. Integrator convergence (PHYSICS.md §6.5)
# ---------------------------------------------------------------------------


def _cruise_state(params):
    """A trimmed-ish cruise state to propagate."""
    from target_gym.plane.env import PlaneState

    return PlaneState(
        x=0.0,
        x_dot=230.0,
        z=CRUISE_ALT_M,
        z_dot=0.0,
        theta=np.deg2rad(4.0),
        theta_dot=0.0,
        alpha=np.deg2rad(4.0),
        gamma=0.0,
        m=params.initial_mass,
        power=0.8,
        stick=0.0,
        fuel=params.initial_fuel_quantity,
        time=0,
        target_altitude=CRUISE_ALT_M,
    )


@pytest.mark.parametrize("method", ["rk4_2", "rk4_10", "euler_100"])
def test_trajectory_is_independent_of_integrator(params, method):
    """The dynamics must be physics, not step-size artefact.

    Propagating the same cruise state with a finer/different integrator must
    give the same trajectory to within a small tolerance. If it does not, what
    the environment simulates is the integrator's truncation error.
    """
    from target_gym.plane.env import compute_next_state

    def roll(integration_method, n=60):
        state = _cruise_state(params)
        for _ in range(n):
            state, _ = compute_next_state(
                0.8, 0.0, state, params, integration_method=integration_method
            )
        return float(state.z), float(state.x_dot)

    z_ref, v_ref = roll("rk4_1")
    z_alt, v_alt = roll(method)

    # Altitude drift over 60 s must agree to within 1 m, speed to within 0.1 m/s.
    assert abs(z_alt - z_ref) < 1.0, f"{method}: dz={z_alt - z_ref:.3f} m"
    assert abs(v_alt - v_ref) < 0.1, f"{method}: dv={v_alt - v_ref:.4f} m/s"


# ---------------------------------------------------------------------------
# 7. Lift-curve shape after the D1/D2 fix (PHYSICS.md §4)
# ---------------------------------------------------------------------------


def test_measured_lift_slope_matches_declared_cl_alpha(params):
    """In the linear region the model's dCL/dalpha must equal cl_alpha.

    Guards the stall sigmoid against creeping back down into the linear
    range, which is precisely the D1 failure mode: the cutoff silently ate
    the top of the lift curve.
    """
    aoas = np.linspace(0.0, 6.0, 25)
    # Undo the Prandtl-Glauert 1/beta boost so we compare like with like.
    mach = 0.2
    beta = np.sqrt(1.0 - mach**2)
    cls = np.array([_CL(a, mach, params) * beta for a in aoas])
    measured = np.polyfit(aoas, cls, 1)[0]
    assert measured == pytest.approx(params.cl_alpha, rel=0.02)


def test_peak_lift_occurs_at_stall_angle(params):
    """CL must be maximal at (not well before) the declared stall AoA."""
    _, aoa_peak = _peak_CL(params, mach=0.2)
    assert aoa_peak == pytest.approx(params.aoa_stall, abs=1.5)


def test_lift_parameters_are_mutually_consistent(params):
    """cl0 + cl_alpha*aoa_stall must reproduce CL_max.

    These four constants describe one wing; if the linear curve does not
    reach CL_max exactly at the stall angle they contradict each other.
    """
    implied = params.cl0 + params.cl_alpha * params.aoa_stall
    assert implied == pytest.approx(params.CL_max, rel=0.05)


def test_cruise_trim_angle_matches_a320(params):
    """The A320 cruises at roughly 2 deg AoA."""
    rho = float(compute_air_density_from_altitude(CRUISE_ALT_M))
    a = float(compute_speed_of_sound_from_altitude(CRUISE_ALT_M))
    v = CRUISE_MACH * a
    weight = compute_weight(params.initial_mass, params.gravity)
    q_s = 0.5 * rho * v * v * params.wings_surface
    residual = lambda aoa: q_s * _CL(aoa, CRUISE_MACH, params) - weight  # noqa: E731
    aoa_trim = brentq(residual, -5.0, 12.0)
    assert 1.0 < aoa_trim < 3.5, f"cruise trim AoA {aoa_trim:.2f} deg"


def test_lift_curve_is_finite_across_envelope(params):
    """No NaN/Inf anywhere in the usable AoA x Mach envelope."""
    for mach in np.linspace(0.05, 0.95, 19):
        for aoa in np.linspace(-15.0, 30.0, 46):
            cl, cd = _CL(aoa, mach, params), _CD(aoa, mach, params)
            assert np.isfinite(cl) and np.isfinite(cd), f"aoa={aoa} M={mach}"
            assert cd > 0.0


def test_drag_polar_is_convex_in_lift(params):
    """CD = cd0 + k*CL^2 -- drag must grow with |CL| away from zero lift."""
    mach = 0.3
    cds = [(_CL(a, mach, params), _CD(a, mach, params)) for a in np.linspace(0, 10, 11)]
    for (cl_a, cd_a), (cl_b, cd_b) in zip(cds, cds[1:]):
        assert cl_b > cl_a and cd_b > cd_a


# ---------------------------------------------------------------------------
# 8. Pitch static stability -- the airworthiness property most at risk from
#    a change in lift-curve slope (the tail is a lifting surface too).
# ---------------------------------------------------------------------------


def _pitch_ang_accel(params, theta_deg, stick=0.0, v=230.0, alt=CRUISE_ALT_M):
    """Angular acceleration about the pitch axis at a given body attitude."""
    import jax.numpy as jnp

    from target_gym.plane.dynamics import compute_acceleration

    # Velocity purely horizontal => gamma = 0 => alpha = theta.
    velocities = jnp.array([v, 0.0, 0.0])
    positions = jnp.array([0.0, alt, np.deg2rad(theta_deg)])
    accel, _ = compute_acceleration(
        velocities, positions, action=(100_000.0, stick), params=params
    )
    return float(accel[2])


def test_aircraft_is_statically_stable_in_pitch(params):
    """dM/d(alpha) < 0: a nose-up disturbance must produce a nose-down moment.

    This is the defining condition for longitudinal static stability. The
    horizontal stabiliser supplies the restoring moment, and it uses the same
    ``aero_coefficients`` as the wing -- so raising ``cl_alpha`` changes both
    the destabilising wing moment and the stabilising tail moment. This test
    is what proves the D2 fix did not make the aircraft divergent.
    """
    thetas = np.linspace(0.0, 8.0, 17)
    moments = np.array([_pitch_ang_accel(params, t) for t in thetas])
    slope = np.polyfit(thetas, moments, 1)[0]
    assert slope < 0.0, f"dM/dalpha = {slope:.3e} >= 0 -- statically unstable"


def test_elevator_produces_correct_sign_of_pitch_moment(params):
    """Positive stick must pitch one way and negative stick the other."""
    up = _pitch_ang_accel(params, theta_deg=2.0, stick=0.15)
    neutral = _pitch_ang_accel(params, theta_deg=2.0, stick=0.0)
    down = _pitch_ang_accel(params, theta_deg=2.0, stick=-0.15)
    assert (up - neutral) * (down - neutral) < 0.0
    assert abs(up - neutral) > 1e-6


def test_elevator_authority_is_bounded(params):
    """Full elevator must not produce an absurd angular acceleration."""
    for stick in (-0.3, 0.3):
        acc = _pitch_ang_accel(params, theta_deg=2.0, stick=stick)
        assert abs(acc) < 1.0, f"stick={stick}: {acc:.3f} rad/s^2"


def test_stall_transition_sharpness_is_set_by_the_lift_peak(params):
    """The attached-to-separated transition is derived, not chosen.

    The separated branch develops far less lift than the aerofoil, so any
    appreciable blend at the stall angle would eat CL_max and move the
    validated stall speed with it. Requiring the wing to be essentially
    attached at ``aoa_stall`` is what fixes the sharpness -- there is no free
    constant here, and this test is what stops one being reintroduced.
    """
    centre = params.aoa_stall + params.aoa_stall_width
    k = np.log(99.0) / params.aoa_stall_width

    separated_at_stall = 1.0 / (1.0 + np.exp((centre - params.aoa_stall) * k))
    assert (
        separated_at_stall < 0.02
    ), f"{separated_at_stall:.1%} separated at the stall angle would eat CL_max"

    # CL_max must survive the blend: the peak is the attached-flow limit
    peak = max(_CL(a, 0.3, params) for a in np.arange(0.0, 25.0, 0.05))
    attached_limit = params.CL_max / np.sqrt(1 - 0.3**2)
    assert peak == pytest.approx(
        attached_limit, rel=0.02
    ), f"blend moved the lift peak: {peak:.3f} vs attached limit {attached_limit:.3f}"


def test_stall_transition_width_is_physically_plausible(params):
    """A clean transport wing breaks over a few degrees -- not instantly."""
    k = np.log(99.0) / params.aoa_stall_width
    width_10_to_90 = 2 * np.log(9.0) / k
    assert (
        2.0 <= width_10_to_90 <= 5.0
    ), f"stall transition spans {width_10_to_90:.1f} deg; a real wing shows 2-5"


# ---------------------------------------------------------------------------
# Anchored checks: quantities the model does not get to define for itself
#
# Every defect found in this model was found by testing a quantity somewhere
# other than where it is derived. Drag was tested only in attached flow, where
# ``CD = cd0 + k*CL**2`` defines it from lift and is self-consistent by
# construction. The two below are anchored outside the model: one to a
# published aircraft's descent performance, one to the exact analytic result
# for a flat plate. Both fail loudly on the zero-drag stall defect.
# ---------------------------------------------------------------------------


def test_terminal_velocity_of_a_falling_aircraft_is_plausible(params):
    """A falling airframe must reach a survivable-to-state terminal velocity.

    v_term = sqrt(2 m g / (rho S CD)) with the airframe broadside. This is the
    single cheapest check that would have caught the missing post-stall drag:
    with a separated wing at CD = cd0 the implied terminal velocity was 693 m/s,
    past Mach 2 at sea level, and the model would happily integrate an aircraft
    to it. No hypothesis about *what* was missing is needed -- only that the
    number is absurd.
    """
    rho_sl = params.air_density_at_sea_level
    cd = _CD(90.0, 0.3, params)
    v_term = np.sqrt(
        2 * params.initial_mass * params.gravity / (rho_sl * params.wings_surface * cd)
    )
    assert 50.0 < v_term < 200.0, (
        f"a broadside airframe falls at {v_term:.0f} m/s (CD={cd:.3f}); "
        "a real one is 100-150"
    )


def test_glide_ratio_becomes_a_flat_plate_past_stall(params):
    """L/D spans lift and drag together, so it sees defects in either.

    At 45 deg a fully separated surface is a flat plate, and a flat plate's
    lift-to-drag ratio is exactly 1 -- CL = C_N cos a and CD = C_N sin a are
    equal there, whatever C_N is. That makes it an anchor the model cannot
    move: it does not depend on the aspect ratio, the stall angle, or any
    coefficient this model chose.

    The defect gave CL = 0 and CD = cd0, so L/D was 0. No single-coefficient
    test noticed, because each looked defensible alone.
    """
    ld_cruise = _CL(2.0, 0.3, params) / _CD(2.0, 0.3, params)
    ld_plate = _CL(45.0, 0.3, params) / _CD(45.0, 0.3, params)

    assert ld_cruise > 8.0, f"cruise L/D is {ld_cruise:.1f}; a jet is well above 8"
    assert ld_plate == pytest.approx(
        1.0, abs=0.25
    ), f"a separated wing at 45 deg has L/D 1 by construction, got {ld_plate:.2f}"


# ---------------------------------------------------------------------------
# Energy bookkeeping
#
# The weak form of this check is ``dE/dt == thrust_power - drag_power``, which
# closes by construction because both sides use the model's own forces: it
# tests the integrator, not the physics. The strong form never mentions drag.
# Energy can only *enter* the system through the engine, so whatever the
# aerodynamics do, the total cannot rise faster than the engine can supply it.
# That bounds the model from outside without assuming anything about it.
# ---------------------------------------------------------------------------


def _mechanical_energy(state, params):
    """Total mechanical energy in joules: kinetic plus gravitational potential."""
    v_sq = float(state.x_dot) ** 2 + float(state.z_dot) ** 2
    return (
        0.5 * params.initial_mass * v_sq
        + params.initial_mass * params.gravity * float(state.z)
    )


@pytest.mark.parametrize(
    "action",
    [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0), (0.0, 0.0)],
    ids=["full-up", "full-down", "idle-up", "idle-down", "neutral"],
)
def test_energy_never_exceeds_what_the_engine_can_supply(action):
    """The system may trade and dissipate energy, never create it.

    Kinetic and potential energy may exchange freely, and drag may remove as
    much as it likes. What must never happen is the total rising faster than
    the engine could raise it -- ``P = T·V`` at full thrust is the ceiling, and
    it holds whatever the aerodynamics are doing.

    This is the check that needs no anchor and no reference model: a sign error
    in a force, a bad integration step, or a regime switch that quietly injects
    energy all break it, and none of them need to be anticipated.
    """
    import jax
    import jax.numpy as jnp

    from target_gym.plane.env_jax import Airplane2D

    env = Airplane2D()
    params = PlaneParams(max_steps_in_episode=300, turbulence_sigma=0.0)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    step = jax.jit(env.step_env)

    dt = float(params.delta_t)
    for i in range(int(params.max_steps_in_episode)):
        before = _mechanical_energy(state, params)
        speed_before = np.hypot(float(state.x_dot), float(state.z_dot))
        _, state, _, terminated, _ = step(key, state, jnp.asarray(action), params)
        after = _mechanical_energy(state, params)

        # Ceiling: full thrust acting along the flight path, at the higher of
        # the two speeds, for one step. Generous on purpose -- it is a bound,
        # not a budget, and it should never be approached let alone crossed.
        speed = max(speed_before, np.hypot(float(state.x_dot), float(state.z_dot)))
        ceiling = params.thrust_output_at_sea_level * speed * dt
        gained = after - before

        assert gained <= ceiling + 1.0, (
            f"step {i}: mechanical energy rose by {gained:.3e} J, more than the "
            f"{ceiling:.3e} J the engine can deliver at {speed:.0f} m/s. "
            "Something is creating energy."
        )

        # And the floor. Energy removed has to be accounted for too: the only
        # sink is aerodynamic drag, and the most of it the airframe can possibly
        # produce is the broadside coefficient over its whole reference area.
        # A loss beyond that is energy going somewhere the model does not
        # describe -- a clip quietly discarding it, or an integration artefact.
        # Like the ceiling this is anchored outside the model: it uses the
        # largest CD the geometry admits, never the instantaneous one, so it
        # cannot close by construction the way ``dE/dt = T·V - D·V`` does.
        rho = float(compute_air_density_from_altitude(jnp.asarray(state.z)))
        cd_max = _CD(90.0, 0.3, params)
        max_dissipation = 0.5 * rho * params.wings_surface * cd_max * speed**3 * dt
        assert gained >= -(max_dissipation * 2.0) - 1.0, (
            f"step {i}: mechanical energy fell by {-gained:.3e} J, more than the "
            f"{max_dissipation:.3e} J that maximum drag could remove at "
            f"{speed:.0f} m/s. Energy is going somewhere unmodelled."
        )
        if bool(terminated):
            break


@pytest.mark.parametrize(
    "sweep", [(0.0, 45.0), (-45.0, 0.0)], ids=["positive", "negative"]
)
def test_energy_flow_is_continuous_across_the_stall(params, sweep):
    """Crossing between regimes must not step the energy budget.

    The model is two descriptions stitched together -- an aerofoil below the
    stall, a separated plate above it -- and the seam is the place a benchmark
    will sit, because that is where a controller pushed to its limits lives. If
    the two do not join, drag jumps at the boundary, and with it the rate the
    system dissipates energy: the aircraft loses or gains power for no reason
    that the physics accounts for.

    Continuity is measured as the largest step in dissipated power relative to
    the typical step. A rapid transition is expected and fine; a *jump* is not.
    The shipped blend gives 4.7x. A naive piecewise switch between the two
    branches gives 32x, and the original defect -- drag collapsing to cd0 past
    the stall -- gives infinity, so this separates a smooth handover from both
    ways of getting it wrong.
    """
    lo, hi = sweep
    alphas = np.arange(lo, hi, 0.05)
    # power dissipated by drag goes as CD at fixed airspeed and density
    cd = np.array([_CD(a, 0.3, params) for a in alphas])

    steps = np.abs(np.diff(cd)) / np.maximum(cd[:-1], 1e-9)
    worst, typical = steps.max(), np.median(steps)
    ratio = worst / max(typical, 1e-12)

    assert ratio < 20.0, (
        f"dissipated power steps {ratio:.0f}x its typical increment at "
        f"alpha = {alphas[steps.argmax()]:.2f} deg. The regimes do not join, so "
        "energy appears or vanishes at the seam."
    )


# ---------------------------------------------------------------------------
# Fuel burn (PHYSICS.md D4, previously "mass is constant")
# ---------------------------------------------------------------------------


def test_cruise_fuel_flow_matches_an_a320(params):
    """Anchored to a published figure, not to the model's own arithmetic.

    In level cruise thrust equals drag, and drag is the weight divided by the
    lift-to-drag ratio. That fixes the fuel flow from quantities validated
    elsewhere in this file, so the check tests ``specific_fuel_consumption``
    against the real aircraft rather than against itself.
    """
    l_over_d = 1.0 / (2.0 * np.sqrt(params.cd0 * params.k))
    thrust_cruise = params.initial_mass * params.gravity / l_over_d
    kg_per_hour = params.specific_fuel_consumption * 1e-3 * thrust_cruise * 3600.0

    assert (
        2000.0 < kg_per_hour < 3200.0
    ), f"cruise fuel flow {kg_per_hour:.0f} kg/h; an A320 burns 2400-2600"


def test_burning_fuel_lightens_the_aircraft():
    """Mass must follow the tanks, continuously and only downwards."""
    import jax
    import jax.numpy as jnp

    from target_gym.registry import REGISTRY

    spec = REGISTRY["plane"]
    env = spec.make_env()
    p = spec.params_cls(max_steps_in_episode=600)
    step = jax.jit(env.step_env)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)

    masses, fuels = [float(state.m)], [float(state.fuel)]
    for _ in range(600):
        _, state, _, terminated, _ = step(key, state, jnp.array([0.8, 0.0]), p)
        masses.append(float(state.m))
        fuels.append(float(state.fuel))
        if bool(terminated):
            break

    masses, fuels = np.array(masses), np.array(fuels)
    assert masses[0] == pytest.approx(p.initial_mass), (
        "the aircraft must start at its takeoff mass; fuel is a component of it, "
        "and adding the two counted 19 tonnes twice"
    )
    assert np.all(np.diff(fuels) <= 1e-6), "fuel went up"
    assert masses[-1] < masses[0], "burning fuel did not lighten the aircraft"
    # mass tracks the tanks -- no separate bookkeeping. The tolerance is set by
    # float32 at 73 500 kg, where the representable step is about 0.01 kg.
    np.testing.assert_allclose(masses[0] - masses, fuels[0] - fuels, atol=0.1)
    # and no step change, which a reset/dynamics disagreement would produce
    assert np.abs(np.diff(masses)).max() < 50.0, "mass jumped between steps"
