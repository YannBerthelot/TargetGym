"""Physics validation for the CSTR.

Enforces ``src/target_gym/pc_gym/cstr/PHYSICS.md``. Complements
``test_cstr_env.py``, which covers the environment interface; this file checks
the reactor against reference behaviour -- steady-state multiplicity, branch
stability, and the reachability of what the environment asks for.
"""

import numpy as np
import pytest
from scipy.optimize import brentq

from target_gym.pc_gym.cstr.env import CSTRParams, compute_velocity


@pytest.fixture(scope="module")
def params():
    return CSTRParams()


def _rate(T, Ca, p):
    return p.k0 * np.exp(-p.EA_over_R / T) * Ca


def _energy_residual(T, Tc, p):
    """Energy balance with concentration eliminated through its own balance."""
    k = p.k0 * np.exp(-p.EA_over_R / T)
    Ca = (p.q / p.V * p.Caf) / (p.q / p.V + k)
    return (
        p.q / p.V * (p.Ti - T)
        + (-p.deltaHr) * k * Ca / (p.rho * p.C)
        + p.UA * (Tc - T) / (p.rho * p.C * p.V)
    )


def _steady_states(Tc, p, lo=280.0, hi=500.0, n=4000):
    Ts = np.linspace(lo, hi, n)
    r = np.array([_energy_residual(T, Tc, p) for T in Ts])
    roots = []
    for i in range(n - 1):
        if r[i] * r[i + 1] < 0:
            roots.append(brentq(_energy_residual, Ts[i], Ts[i + 1], args=(Tc, p)))
    out = []
    for T in roots:
        k = p.k0 * np.exp(-p.EA_over_R / T)
        out.append((T, (p.q / p.V * p.Caf) / (p.q / p.V + k)))
    return out


def _operating_point(Tc, p):
    """The low-conversion branch the environment actually lives on."""
    T = brentq(_energy_residual, 290.0, 340.0, args=(Tc, p))
    k = p.k0 * np.exp(-p.EA_over_R / T)
    return (p.q / p.V * p.Caf) / (p.q / p.V + k), T


def _jacobian(Ca, T, p):
    k = p.k0 * np.exp(-p.EA_over_R / T)
    dk = k * p.EA_over_R / T**2
    return np.array(
        [
            [-p.q / p.V - k, -dk * Ca],
            [
                (-p.deltaHr) * k / (p.rho * p.C),
                -p.q / p.V
                + (-p.deltaHr) * dk * Ca / (p.rho * p.C)
                - p.UA / (p.rho * p.C * p.V),
            ],
        ]
    )


# ---------------------------------------------------------------------------
# 1. Model constants
# ---------------------------------------------------------------------------


def test_residence_time_and_step_resolution(params):
    """delta_t must resolve the residence time."""
    tau = params.V / params.q
    assert tau == pytest.approx(1.0)
    assert params.delta_t <= tau / 3.0


def test_reaction_is_exothermic(params):
    """The sign of the enthalpy is what makes this reactor interesting."""
    assert params.deltaHr < 0


def test_heat_release_raises_temperature(params):
    """More reaction must mean more heating, at fixed coolant."""
    hot, _ = compute_velocity(np.array([0.9, 320.0]), 298.0, params)
    cold, _ = compute_velocity(np.array([0.1, 320.0]), 298.0, params)
    assert float(hot[1]) > float(cold[1])


def test_coolant_removes_heat(params):
    """A colder jacket must cool the reactor."""
    warm, _ = compute_velocity(np.array([0.9, 320.0]), 302.0, params)
    cool, _ = compute_velocity(np.array([0.9, 320.0]), 295.0, params)
    assert float(cool[1]) < float(warm[1])


# ---------------------------------------------------------------------------
# 2. Steady-state multiplicity -- the reactor's defining feature
# ---------------------------------------------------------------------------


def test_reactor_exhibits_steady_state_multiplicity(params):
    """Three steady states over a window of coolant temperature.

    This is emergent: it falls out of the Arrhenius feedback rather than being
    put in, and its presence is what separates a real reactor from a lag with
    a nonlinear gain.
    """
    assert len(_steady_states(300.0, params)) == 3
    assert len(_steady_states(302.0, params)) == 3


def test_multiplicity_disappears_outside_that_window(params):
    """Cold enough and only the extinguished branch survives; hot enough and
    only the ignited one does."""
    assert len(_steady_states(295.0, params)) == 1
    assert len(_steady_states(310.0, params)) == 1


def test_the_runaway_trip_sits_between_the_branches(params):
    """T_max is not an arbitrary safety number.

    At Tc = 300 K the unstable middle solution sits at 350 K, which is exactly
    the trip. So termination fires when the reactor leaves the extinguished
    branch and begins to ignite.
    """
    branches = sorted(T for T, _ in _steady_states(300.0, params))
    assert branches[0] < params.T_max <= branches[2]


# ---------------------------------------------------------------------------
# 3. The operating branch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("Tc", [295.0, 297.0, 298.5, 300.0, 302.0])
def test_operating_branch_is_stable(params, Tc):
    """Every coolant setting in range must give a stable operating point."""
    Ca, T = _operating_point(Tc, params)
    assert np.all(np.linalg.eigvals(_jacobian(Ca, T, params)).real < 0)


def test_operating_temperature_stays_in_range(params):
    for Tc in (params.T_c_min, params.T_c_max):
        _, T = _operating_point(Tc, params)
        assert 310.0 < T < 335.0
        assert T < params.T_max, "the operating branch must not trip on its own"


def test_conversion_is_low_on_this_branch(params):
    """The environment runs the extinguished branch, so conversion is modest."""
    for Tc in (params.T_c_min, params.T_c_max):
        Ca, _ = _operating_point(Tc, params)
        assert 0.03 < 1.0 - Ca < 0.30


def test_the_plant_slows_as_it_is_pushed(params):
    """Eigenvalues approach the imaginary axis toward the ignition boundary.

    A loop tuned at the cold end is running on a plant half as fast at the hot
    end -- a real nonlinearity, not a modelling artefact.
    """
    fast = np.linalg.eigvals(_jacobian(*_operating_point(295.0, params), params))
    slow = np.linalg.eigvals(_jacobian(*_operating_point(302.0, params), params))
    assert abs(slow.real).max() < abs(fast.real).max()


# ---------------------------------------------------------------------------
# 4. Reachability -- the check the four-tank environment failed
# ---------------------------------------------------------------------------


def test_every_target_concentration_is_reachable(params):
    """The whole sampled band must lie inside what the coolant can hold."""
    ca_hot, _ = _operating_point(params.T_c_max, params)  # lowest Ca
    ca_cold, _ = _operating_point(params.T_c_min, params)  # highest Ca
    lo, hi = params.target_CA_range
    assert ca_hot <= lo, f"target {lo} below the reachable minimum {ca_hot:.3f}"
    assert hi <= ca_cold, f"target {hi} above the reachable maximum {ca_cold:.3f}"


def test_the_bottom_of_the_target_band_has_thin_margin(params):
    """Documented deviation D1, asserted so it cannot quietly become worse.

    Reaching the lowest target needs the coolant within a fraction of a kelvin
    of its stop, leaving almost no authority for disturbance rejection.
    """
    ca_hot, _ = _operating_point(params.T_c_max, params)
    margin = params.target_CA_range[0] - ca_hot
    assert 0.0 < margin < 0.02


def test_initial_concentrations_are_inside_the_envelope(params):
    ca_hot, _ = _operating_point(params.T_c_max, params)
    ca_cold, _ = _operating_point(params.T_c_min, params)
    lo, hi = params.initial_CA_range
    assert ca_hot - 0.05 <= lo and hi <= ca_cold + 0.05
