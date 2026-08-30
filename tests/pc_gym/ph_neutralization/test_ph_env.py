"""Physics validation for the pH neutralisation CSTR.

Enforces ``src/target_gym/pc_gym/ph_neutralization/PHYSICS.md``. Assertions are
emergent: the benchmark's nominal operating point, titration-curve shape, the
gain variation that defines the task's difficulty, and invariant conservation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.pc_gym.ph_neutralization.env import (
    PHParams,
    PHState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_velocity,
    process_gain,
    solve_pH,
    steady_state_invariants,
    titration_residual,
)
from target_gym.pc_gym.ph_neutralization.env_jax import PHNeutralization

NOMINAL_Q3, NOMINAL_Q2 = 15.6, 0.55


@pytest.fixture(scope="module")
def params():
    return PHParams()


def _steady_pH(q3, q2, params):
    return float(solve_pH(*steady_state_invariants(q3, q2, params), params))


# ---------------------------------------------------------------------------
# 1. The pH solver
# ---------------------------------------------------------------------------


def test_titration_residual_is_monotone_in_ph(params):
    """Bisection is only safe because the residual increases with pH."""
    Wa, Wb = steady_state_invariants(NOMINAL_Q3, NOMINAL_Q2, params)
    grid = np.linspace(0.5, 13.5, 200)
    values = np.array([float(titration_residual(x, Wa, Wb, params)) for x in grid])
    assert np.all(np.diff(values) > 0.0)


def test_solved_ph_zeroes_the_charge_balance(params):
    """The returned pH must actually be a root, not merely close."""
    for q3 in (11.0, 15.6, 20.0):
        Wa, Wb = steady_state_invariants(q3, NOMINAL_Q2, params)
        pH = solve_pH(Wa, Wb, params)
        assert abs(float(titration_residual(pH, Wa, Wb, params))) < 1e-9


def test_ph_stays_within_the_physical_range(params):
    for q3 in np.linspace(params.q3_min, params.q3_max, 25):
        for q2 in (0.0, 0.55, 2.5):
            pH = _steady_pH(float(q3), q2, params)
            assert 0.0 <= pH <= 14.0


# ---------------------------------------------------------------------------
# 2. Validation against the benchmark (PHYSICS.md §2)
# ---------------------------------------------------------------------------


def test_nominal_operating_point_matches_the_benchmark(params):
    """At the published nominal flows the reactor sits at pH ~7.

    Load-bearing check: it pins feed concentrations and flows jointly.
    """
    assert _steady_pH(NOMINAL_Q3, NOMINAL_Q2, params) == pytest.approx(7.03, abs=0.1)


def test_residence_time_is_about_ninety_seconds(params):
    tau = params.V / (params.q1 + NOMINAL_Q2 + NOMINAL_Q3)
    assert 60.0 < tau < 120.0


def test_ph_increases_monotonically_with_base_flow(params):
    flows = np.linspace(params.q3_min, params.q3_max, 30)
    phs = [_steady_pH(float(q), NOMINAL_Q2, params) for q in flows]
    assert np.all(np.diff(phs) > 0.0)


def test_more_acid_lowers_ph(params):
    """Sanity on the acid stream's sign."""
    strong = params.replace(Wa1=params.Wa1 * 2.0)
    assert _steady_pH(NOMINAL_Q3, NOMINAL_Q2, strong) < _steady_pH(
        NOMINAL_Q3, NOMINAL_Q2, params
    )


def test_titration_curve_is_steepest_near_equivalence(params):
    """The S-shape: gain peaks in the middle, not at the extremes."""
    flows = np.linspace(params.q3_min + 0.5, params.q3_max - 0.5, 40)
    gains = np.array(
        [abs(float(process_gain(float(q), NOMINAL_Q2, params))) for q in flows]
    )
    peak = flows[int(gains.argmax())]
    # The peak is interior -- that is the S-shape -- and clearly above both
    # shoulders. Measured ratios are ~2.8x (acidic side) and ~15x (alkaline);
    # 2x is a floor that still fails a curve without a distinct equivalence
    # region, which is what this asserts.
    assert flows[0] < peak < flows[-1]
    assert gains.max() > 2.0 * gains[0]
    assert gains.max() > 2.0 * gains[-1]


def test_gain_varies_by_an_order_of_magnitude(params):
    """The defining difficulty: a fixed-gain controller cannot suit both ends."""
    flows = np.linspace(params.q3_min + 0.5, params.q3_max - 0.5, 60)
    gains = np.array(
        [abs(float(process_gain(float(q), NOMINAL_Q2, params))) for q in flows]
    )
    assert gains.max() / gains.min() > 10.0


def test_buffering_flattens_the_titration_curve(params):
    """More buffer, less gain variation -- and a different operating point."""

    def spread(q2):
        flows = np.linspace(params.q3_min + 0.5, params.q3_max - 0.5, 40)
        g = np.array([abs(float(process_gain(float(q), q2, params))) for q in flows])
        return g.max() / g.min()

    unbuffered, nominal, heavy = spread(0.0), spread(0.55), spread(4.0)
    assert unbuffered > nominal > heavy
    # Buffering also shifts where the reactor sits.
    assert _steady_pH(NOMINAL_Q3, 0.0, params) < _steady_pH(NOMINAL_Q3, 4.0, params)


# ---------------------------------------------------------------------------
# 3. Dynamics and conservation
# ---------------------------------------------------------------------------


def test_invariants_relax_toward_their_mixing_steady_state(params):
    """With flows held fixed the invariants must approach the mixed value."""
    Wa_ss, Wb_ss = steady_state_invariants(NOMINAL_Q3, NOMINAL_Q2, params)
    for Wa0, Wb0 in ((Wa_ss * 2.0, Wb_ss * 0.5), (Wa_ss * -1.0, Wb_ss * 3.0)):
        v, _ = compute_velocity(
            jnp.array([Wa0, Wb0]), action=NOMINAL_Q3, q2=NOMINAL_Q2, params=params
        )
        # Each invariant is driven toward the steady state, so the derivative
        # points that way.
        assert float(v[0]) * (Wa_ss - Wa0) > 0.0
        assert float(v[1]) * (Wb_ss - Wb0) > 0.0


def test_steady_state_is_a_fixed_point(params):
    """At the mixing steady state both derivatives vanish."""
    Wa, Wb = steady_state_invariants(NOMINAL_Q3, NOMINAL_Q2, params)
    v, _ = compute_velocity(
        jnp.array([Wa, Wb]), action=NOMINAL_Q3, q2=NOMINAL_Q2, params=params
    )
    # Compare against the invariant's own scale rather than an absolute
    # epsilon: Wa is ~1e-4, so a float32 residual of ~1e-12 is exactly zero to
    # working precision. The meaningful statement is that a full residence
    # time of this drift would not move the invariant appreciably.
    tau = params.V / (params.q1 + NOMINAL_Q2 + NOMINAL_Q3)
    assert abs(float(v[0])) * tau < 1e-4 * abs(float(Wa))
    assert abs(float(v[1])) * tau < 1e-4 * abs(float(Wb))


@pytest.mark.parametrize("method", ["rk2_2", "rk4_2", "rk4_4"])
def test_state_advances_under_each_integrator(method, params):
    env = PHNeutralization()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    new_state, _ = compute_next_state(
        0.0, state, params, jax.random.PRNGKey(0), integration_method=method
    )
    assert new_state.time == state.time + 1
    for leaf in jax.tree_util.tree_leaves(new_state):
        assert np.all(np.isfinite(np.asarray(leaf)))


# ---------------------------------------------------------------------------
# 4. Task interface
# ---------------------------------------------------------------------------


def test_action_maps_to_the_base_flow_range(params):
    env = PHNeutralization()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    for raw, expected in ((-1.0, params.q3_min), (1.0, params.q3_max)):
        new_state, _ = compute_next_state(raw, state, params, jax.random.PRNGKey(0))
        assert float(new_state.q3) == pytest.approx(expected, rel=1e-5)


def test_observation_hides_the_invariants_and_the_buffer(params):
    """A plant has a pH probe, not an assay of carbonate speciation."""
    env = PHNeutralization()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    state = state.replace(Wa=1234.5, Wb=6789.0, q2=4321.0)
    obs = env.get_obs(state, params)
    assert obs.shape == (3,)
    for hidden in (1234.5, 6789.0, 4321.0):
        assert not bool(jnp.any(jnp.isclose(obs, hidden)))


def test_reward_peaks_on_target(params):
    env = PHNeutralization()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    on = state.replace(pH=7.0, target_pH=7.0, q3=params.q3_min)
    off = state.replace(pH=9.0, target_pH=7.0, q3=params.q3_min)
    assert float(compute_reward(on, params)) == pytest.approx(1.0, abs=1e-6)
    assert float(compute_reward(on, params)) > float(compute_reward(off, params))


def test_reward_saturates_rather_than_inverting(params):
    """The clip must not let a larger error score better -- see PHYSICS.md §5."""
    env = PHNeutralization()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    rewards = [
        float(
            compute_reward(
                state.replace(pH=7.0 + e, target_pH=7.0, q3=params.q3_min), params
            )
        )
        for e in (0.0, 0.5, 1.0, 2.0, 4.0, 6.0)
    ]
    assert all(a >= b - 1e-9 for a, b in zip(rewards, rewards[1:])), rewards


def test_terminates_when_grossly_off_spec(params):
    env = PHNeutralization()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    assert bool(check_is_terminal(state.replace(pH=params.pH_min - 0.5), params)[0])
    assert bool(check_is_terminal(state.replace(pH=params.pH_max + 0.5), params)[0])
    assert not bool(check_is_terminal(state.replace(pH=7.0), params)[0])


def test_reset_starts_at_a_consistent_steady_state(params):
    """A running CSTR is at equilibrium for its flows."""
    env = PHNeutralization()
    for seed in range(5):
        _, state = env.reset_env(jax.random.PRNGKey(seed), params)
        assert isinstance(state, PHState)
        Wa, Wb = steady_state_invariants(state.q3, state.q2, params)
        assert float(state.Wa) == pytest.approx(float(Wa), rel=1e-6)
        assert float(state.pH) == pytest.approx(
            float(solve_pH(Wa, Wb, params)), abs=1e-6
        )
