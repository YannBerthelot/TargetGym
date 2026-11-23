"""The 2D airliner against Airbus's published envelope of the A320-233 (charter B6).

Sources (src/target_gym/plane/REALITY.md):

- [SF21] Airbus, Safety first #21, January 2016, "Control your speed... in cruise",
  p. 6, figure "A320-233 Speed envelope for GW = 70T ISA". Its curves were read from
  the figure's vector paths, with the axes from its own grid lines (0.02 Mach by
  1 000 ft), so each number below is the chart's, not an eye's.
- [EASA.A.064] the A320-233 carries two IAE V2527E-A5; [EASA IM.E.069] each gives
  9 892 daN at its maximum continuous rating, sea level static. Its take-off
  rating, 11 031 daN, "shall not be used continuously more than 5 minutes" (note
  1), so the most the engines hold in flight is the maximum continuous one.
- [HAMMAMI] Hammami, HAW Hamburg, 2021, eq. 2.1 and 2.2: Mattingly's law for the
  fuel a high-bypass turbofan burns per unit of thrust, as Roux and Scholz adjusted
  it: ``3.38e-8 V + 1.04e-5 sqrt(T / T0)`` kg/(N s), V the true airspeed in m/s.

Contract, at 70 t in the standard atmosphere:

- Service ceiling. "Service curve, corresponding to the propulsion capacity of the
  aircraft's engines to maintain +300ft/minute at a constant Mach" [SF21 p. 5]. The
  plant's residual climb at full thrust falls to 300 ft/min within 1 000 ft of the
  chart's curve.
- ECON speed at cost index 0, the max-range Mach: the plant's Mach of least fuel
  per air distance is within 0.02 of the chart's curve, up to 33 000 ft.
- Green dot, "an estimate of the speed for best lift-to-drag ratio" [SF21 p. 3]:
  the plant's Mach of least drag in level flight is within 0.02 of the chart's.
- The polar is ``CD = cd0 + k CL^2`` on the lift the wing makes, the whole
  airframe's (PHYSICS.md, figures of merit): the wing's best lift over drag is
  ``1 / (2 sqrt(cd0 k))`` at any Mach below the drag rise, and the trimmed
  airframe's is within 5 % of it (ours: the tail's lift and its induced drag,
  which the polar leaves out).

Residual climb is ``V (T_max - D) / W`` with D the level-trim thrust: the
definition, as in test_plane_energy_reality.py. Tolerances are one division of the
chart's own grid, written before the plant was changed.

Outside the tests, each a known deviation with its numbers in PHYSICS.md
(Amendment 5), accepted by the owner on 2026-11-23:

- below green dot, where "the crew should not fly" [SF21 p. 4], and above MMO,
  0.82 [TCDS]: the parabolic polar carries no high-lift drag;
- the chart's top speed in level flight at max climb thrust, where the residual
  climb is zero: the plant's is 700 to 1 700 ft higher;
- the ECON curve above 33 000 ft, where the plant's max-range Mach is 0.024 to
  0.045 over the chart's.

The last two ask for drag that grows with lift near M 0.75 to 0.80, which the
plant does not have.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import target_gym as tg
from target_gym.plane.dynamics import (
    aero_coefficients,
    compute_air_density_from_altitude,
    compute_speed_of_sound_from_altitude,
    compute_thrust_output,
    trim_level,
)
from target_gym.plane.env import PlaneParams

FT, FPM = 0.3048, 0.3048 / 60.0
MASS = 70_000.0  # the chart's gross weight
P = PlaneParams()

# [SF21 p. 6], digitised: (Mach, ft). The cruise side of each curve, from green
# dot (M 0.737 at 36 987 ft) to MMO.
VZ_300 = [
    (0.7385, 36_991.0),
    (0.7558, 37_093.0),
    (0.7738, 37_093.0),
    (0.7943, 36_336.0),
    (0.7991, 36_016.0),
    (0.8128, 33_760.0),
]
# (ft, Mach) along the ECON speed curve at cost index 0, up to 33 000 ft.
ECON_CI_0 = [
    (25_484.0, 0.6604),
    (28_025.0, 0.6908),
    (29_946.0, 0.7130),
    (31_620.0, 0.7303),
    (32_610.0, 0.7408),
]
# (ft, Mach) along the green dot curve, up to where the buffet line still lets
# the plant trim either side of it.
GREEN_DOT = [
    (23_000.0, 0.5205),
    (25_472.0, 0.5534),
    (27_167.0, 0.5766),
    (29_782.0, 0.6153),
    (32_075.0, 0.6513),
    (35_447.0, 0.7091),
]
ONE_LEVEL = 1_000.0  # ft, the chart's vertical grid
ONE_MACH_LINE = 0.02  # the chart's horizontal grid


@jax.jit
def _level_flight(mach, altitude_ft):
    """``(trim thrust N, max thrust N, true airspeed m/s)`` at 70 t."""
    z = altitude_ft * FT
    speed = mach * compute_speed_of_sound_from_altitude(z)
    _, _, power = trim_level(speed, z, MASS, P)
    rho = compute_air_density_from_altitude(z)
    full = compute_thrust_output(1.0, P.thrust_output_at_sea_level, mach, rho)
    return power * full, full, speed


def _residual_climb(mach, altitude_ft):
    """Residual climb at full thrust, ft/min."""
    trim, full, speed = (float(x) for x in _level_flight(mach, altitude_ft))
    return speed * (full - trim) / (MASS * P.gravity) / FPM


def _altitude_where_climb_is(rate, mach, low=28_000.0, high=41_000.0):
    """The altitude, ft, where the residual climb at ``mach`` falls to ``rate``:
    it falls with height, so a bisection between two levels the buffet line
    still allows at this weight."""
    for _ in range(20):
        middle = 0.5 * (low + high)
        low, high = (
            (middle, high) if _residual_climb(mach, middle) > rate else (low, middle)
        )
    return 0.5 * (low + high)


@pytest.mark.parametrize("mach, chart_ft", VZ_300)
def test_residual_climb_falls_to_300_fpm_at_the_charts_service_ceiling(mach, chart_ft):
    assert _altitude_where_climb_is(300.0, mach) == pytest.approx(
        chart_ft, abs=ONE_LEVEL
    )


@pytest.mark.parametrize("altitude_ft, chart_mach", ECON_CI_0)
def test_the_max_range_mach_is_the_charts_econ_speed_at_cost_index_0(
    altitude_ft, chart_mach
):
    from target_gym.plane.dynamics import fuel_flow

    machs = np.arange(0.60, 0.8201, 0.005)
    per_metre = []
    for mach in machs:
        trim, _, speed = (float(x) for x in _level_flight(float(mach), altitude_ft))
        per_metre.append(float(fuel_flow(trim, speed, altitude_ft * FT, P)) / speed)
    best = float(machs[int(np.argmin(per_metre))])
    assert best == pytest.approx(chart_mach, abs=ONE_MACH_LINE)


@pytest.mark.parametrize("altitude_ft, chart_mach", GREEN_DOT)
def test_the_least_drag_mach_is_the_charts_green_dot(altitude_ft, chart_mach):
    machs = np.arange(chart_mach - 0.08, chart_mach + 0.0801, 0.005)
    drag = [float(_level_flight(float(mach), altitude_ft)[0]) for mach in machs]
    best = float(machs[int(np.argmin(drag))])
    assert best == pytest.approx(chart_mach, abs=ONE_MACH_LINE)


def test_the_trimmed_airframes_best_lift_over_drag_is_the_polars():
    """16.67 from the polar; at FL330 the trimmed airframe, tail and all,
    reaches it within 5 %."""
    machs = np.arange(0.55, 0.8001, 0.005)
    drag = min(float(_level_flight(float(mach), 33_000.0)[0]) for mach in machs)
    assert MASS * P.gravity / drag == pytest.approx(16.67, rel=0.05)


@pytest.mark.parametrize("mach", [0.30, 0.78])
def test_the_wings_best_lift_over_drag_does_not_grow_with_mach(mach):
    """``1 / (2 sqrt(cd0 k))`` = 1 / (2 sqrt(0.02 x 0.045)) = 16.67, below the
    drag rise at any Mach: induced drag follows the lift the wing makes."""
    incidence = jnp.arange(0.0, 10.0, 0.02)
    lift, drag = aero_coefficients(incidence, mach, P)
    assert float(jnp.max(lift / drag)) == pytest.approx(16.67, rel=0.01)


def test_full_thrust_is_two_v2527e_a5_at_their_maximum_continuous_rating():
    """2 x 9 892 daN [EASA IM.E.069, section 6] = 197 840 N, sea level static."""
    static = compute_thrust_output(1.0, P.thrust_output_at_sea_level, 0.0, 1.225)
    assert float(static) == pytest.approx(197_840.0, rel=1e-6)


def test_fuel_per_thrust_follows_the_published_law():
    """[HAMMAMI eq. 2.2]. Standing at sea level: 1.04e-5 kg/(N s), so 100 kN
    burns 1.04 kg/s. At 11 000 m (216.65 K) and 230 m/s: 3.38e-8 x 230
    + 1.04e-5 x sqrt(216.65 / 288.15) = 7.774e-6 + 9.018e-6 = 1.6792e-5
    kg/(N s), so 50 kN burns 0.8396 kg/s."""
    from target_gym.plane.dynamics import fuel_flow

    assert float(fuel_flow(100_000.0, 0.0, 0.0, P)) == pytest.approx(1.04, rel=1e-4)
    assert float(fuel_flow(50_000.0, 230.0, 11_000.0, P)) == pytest.approx(
        0.8396, rel=1e-4
    )


def test_the_plant_burns_what_the_law_gives_over_a_trimmed_second():
    """One second of established cruise in calm air burns the law's flow at the
    thrust the engines make (1 %: the thrust barely moves inside the step)."""
    from target_gym.plane.dynamics import fuel_flow

    env, params = tg.make("plane", turbulence_sigma=0.0, sensor_noise_scale=0.0)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key, params)
    _, after, *_ = env.step_env(key, state, env.trim_action(state, params), params)
    speed = float(obs[env.obs_index["airspeed"]])
    mach = speed / float(compute_speed_of_sound_from_altitude(state.z))
    thrust = compute_thrust_output(
        state.power,
        params.thrust_output_at_sea_level,
        mach,
        compute_air_density_from_altitude(state.z),
    )
    expected = float(fuel_flow(thrust, speed, state.z, params)) * params.delta_t
    assert float(state.fuel - after.fuel) == pytest.approx(expected, rel=0.01)
