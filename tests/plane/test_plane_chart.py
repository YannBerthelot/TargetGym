"""The 2D airliner against Airbus's published envelope of the A320-233 (charter B6).

Sources (src/target_gym/plane/REALITY.md):

- [SF21] Airbus, Safety first #21, January 2016, "Control your speed... in cruise",
  p. 6, figure "A320-233 Speed envelope for GW = 70T ISA". Its curves were read from
  the figure's vector paths, with the axes from its own grid lines (0.02 Mach by
  1 000 ft), so each number below is the chart's, not an eye's.
- [EASA.A.064] the A320-233 carries two IAE V2527E-A5; [EASA IM.E.069] each gives
  11 031 daN at take-off, sea level static.
- [HAMMAMI] Hammami, HAW Hamburg, 2021, eq. 2.1 and 2.2: Mattingly's law for the
  fuel a high-bypass turbofan burns per unit of thrust, as Roux and Scholz adjusted
  it: ``3.38e-8 V + 1.04e-5 sqrt(T / T0)`` kg/(N s), V the true airspeed in m/s.

Contract, at 70 t in the standard atmosphere:

- Service ceiling. "Service curve, corresponding to the propulsion capacity of the
  aircraft's engines to maintain +300ft/minute at a constant Mach" [SF21 p. 5]. The
  plant's residual climb at full thrust falls to 300 ft/min within 1 000 ft of the
  chart's curve.
- Top speed in level flight at max climb thrust: thrust equals drag there, so the
  residual climb is zero. The plant's zero-climb altitude is within 1 000 ft of the
  chart's curve.
- ECON speed at cost index 0, the max-range Mach: the plant's Mach of least fuel
  per air distance is within 0.02 of the chart's curve.
- The wing's polar is ``CD = cd0 + k CL^2`` on the lift the wing makes, so its best
  lift over drag is ``1 / (2 sqrt(cd0 k))`` at any Mach below the drag rise
  (PHYSICS.md, figures of merit).

Residual climb is ``V (T_max - D) / W`` with D the level-trim thrust: the
definition, as in test_plane_energy_reality.py. Tolerances are one division of the
chart's own grid, written before the plant was changed. Two stretches of the
chart are outside the tests: below green dot, where "the crew should not fly"
[SF21 p. 4] and the plant's parabolic polar carries no high-lift drag
(PHYSICS.md, known deviations), and above MMO, 0.82 [TCDS].
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
VMAX_AT_MAX_CLIMB = [
    (0.7681, 39_000.0),
    (0.7921, 38_043.0),
    (0.8067, 36_919.0),
    (0.8174, 35_128.0),
]
# (ft, Mach) along the ECON speed curve at cost index 0.
ECON_CI_0 = [
    (25_484.0, 0.6604),
    (28_025.0, 0.6908),
    (29_946.0, 0.7130),
    (31_620.0, 0.7303),
    (33_978.0, 0.7511),
    (36_467.0, 0.7600),
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


@pytest.mark.parametrize("mach, chart_ft", VMAX_AT_MAX_CLIMB)
def test_level_flight_at_full_thrust_tops_out_on_the_charts_curve(mach, chart_ft):
    assert _altitude_where_climb_is(0.0, mach) == pytest.approx(chart_ft, abs=ONE_LEVEL)


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


@pytest.mark.parametrize("mach", [0.30, 0.78])
def test_the_wings_best_lift_over_drag_does_not_grow_with_mach(mach):
    """``1 / (2 sqrt(cd0 k))`` = 1 / (2 sqrt(0.02 x 0.045)) = 16.67, below the
    drag rise at any Mach: induced drag follows the lift the wing makes."""
    incidence = jnp.arange(0.0, 10.0, 0.02)
    lift, drag = aero_coefficients(incidence, mach, P)
    assert float(jnp.max(lift / drag)) == pytest.approx(16.67, rel=0.01)


def test_static_thrust_at_sea_level_is_two_v2527e_a5():
    """2 x 11 031 daN [EASA IM.E.069, section 6] = 220 620 N."""
    static = compute_thrust_output(1.0, P.thrust_output_at_sea_level, 0.0, 1.225)
    assert float(static) == pytest.approx(220_620.0, rel=1e-6)


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
