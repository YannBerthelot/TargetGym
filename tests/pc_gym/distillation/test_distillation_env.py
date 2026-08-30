"""Physics validation for the binary distillation column (Column A).

Enforces ``src/target_gym/pc_gym/distillation/PHYSICS.md``. Assertions are the
published operating point, exact balance closure, the ill-conditioning that
defines the control problem, and the integrator stability requirement.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.pc_gym.distillation.env import (
    N_FEED,
    N_STAGES,
    DistillationParams,
    DistillationState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_velocity,
    flows,
    separation_factor,
    vle,
)
from target_gym.pc_gym.distillation.env_jax import (
    _NOMINAL_PROFILE,
    DistillationColumn,
)

NOMINAL_L, NOMINAL_V = 2.706, 3.206


@pytest.fixture(scope="module")
def params():
    return DistillationParams()


def _raw(L, V, params):
    return jnp.array(
        [
            2.0 * (L - params.L_min) / (params.L_max - params.L_min) - 1.0,
            2.0 * (V - params.V_min) / (params.V_max - params.V_min) - 1.0,
        ]
    )


def _settle(params, L=NOMINAL_L, V=NOMINAL_V, steps=4000, zF=None):
    p = params.replace(zF_noise_std=0.0)
    state = DistillationState(
        time=0,
        x=jnp.linspace(0.01, 0.99, N_STAGES),
        zF=p.zF_nominal if zF is None else zF,
        L=L,
        V=V,
        target_yD=0.99,
        target_xB=0.01,
    )
    raw = _raw(L, V, p)
    key = jax.random.PRNGKey(0)
    for _ in range(steps):
        state, _ = compute_next_state(raw, state, p, key)
    return state


# ---------------------------------------------------------------------------
# 1. The published operating point (PHYSICS.md §2)
# ---------------------------------------------------------------------------


def test_nominal_operating_point_matches_column_a(params):
    """yD = 0.99 and xB = 0.01 together at the published flows.

    Load-bearing: pins relative volatility, stage count, feed location and
    flows jointly against the reference.
    """
    state = _settle(params)
    assert float(state.x[-1]) == pytest.approx(0.99, abs=0.005)
    assert float(state.x[0]) == pytest.approx(0.01, abs=0.005)


def test_distillate_to_feed_ratio_is_one_half(params):
    D, B = flows(NOMINAL_L, NOMINAL_V, params)
    assert float(D) / params.F == pytest.approx(0.5, abs=1e-6)
    assert float(B) / params.F == pytest.approx(0.5, abs=1e-6)


def test_overall_component_balance_closes(params):
    """D*yD + B*xB must equal F*zF at steady state."""
    state = _settle(params)
    D, B = flows(NOMINAL_L, NOMINAL_V, params)
    lhs = float(D) * float(state.x[-1]) + float(B) * float(state.x[0])
    assert lhs == pytest.approx(params.F * params.zF_nominal, abs=1e-3)


def test_composition_increases_up_the_column(params):
    """The light component must be progressively enriched toward the top."""
    profile = np.asarray(_settle(params).x)
    assert np.all(np.diff(profile) > 0.0)
    assert profile[0] < profile[N_FEED - 1] < profile[-1]


def test_separation_factor_is_large_at_high_purity(params):
    state = _settle(params)
    assert float(separation_factor(state)) > 1000.0


# ---------------------------------------------------------------------------
# 2. Ill-conditioning -- what makes the task hard
# ---------------------------------------------------------------------------


def _gain_matrix(params, d=0.01):
    """Steady-state dG = d[yD, xB] / d[L, V] by central differences."""
    yp = _settle(params, L=NOMINAL_L + d)
    ym = _settle(params, L=NOMINAL_L - d)
    g11 = (float(yp.x[-1]) - float(ym.x[-1])) / (2 * d)
    g21 = (float(yp.x[0]) - float(ym.x[0])) / (2 * d)
    vp = _settle(params, V=NOMINAL_V + d)
    vm = _settle(params, V=NOMINAL_V - d)
    g12 = (float(vp.x[-1]) - float(vm.x[-1])) / (2 * d)
    g22 = (float(vp.x[0]) - float(vm.x[0])) / (2 * d)
    return np.array([[g11, g12], [g21, g22]])


def test_gains_satisfy_the_mass_balance_identity(params):
    """dyD/dL + dxB/dL = (yD - xB)/D = 1.96 at the nominal point.

    Independent cross-check on the gains: it caught a first, badly-converged
    computation that reported 0 and a spuriously singular gain matrix.
    """
    G = _gain_matrix(params)
    assert G[0, 0] + G[1, 0] == pytest.approx(1.96, rel=0.15)


def test_plant_is_ill_conditioned(params):
    """RGA >> 1 and a large condition number -- the defining difficulty."""
    G = _gain_matrix(params)
    rga = G * np.linalg.inv(G).T
    assert abs(rga[0, 0]) > 10.0
    singular = np.linalg.svd(G, compute_uv=False)
    assert singular[0] / singular[1] > 30.0


def test_reflux_and_boilup_push_compositions_oppositely(params):
    """More reflux purifies the top; more boilup purifies the bottom."""
    G = _gain_matrix(params)
    assert G[0, 0] > 0.0  # dyD/dL > 0
    assert G[1, 1] < 0.0  # dxB/dV < 0


# ---------------------------------------------------------------------------
# 3. Numerics (PHYSICS.md §4)
# ---------------------------------------------------------------------------


def test_default_integrator_is_stable_and_fewer_substeps_are_not(params):
    """16 substeps is a stability requirement, not accuracy tuning.

    Tray time constant ~0.17 min gives |lambda| ~ 11.8 /min, and RK4 needs
    h*|lambda| < 2.78. At dt = 1 min that means at least ~5 substeps; in
    practice the profile only survives at 16.
    """
    p = params.replace(zF_noise_std=0.0)
    raw = _raw(NOMINAL_L, NOMINAL_V, p)
    key = jax.random.PRNGKey(0)

    def run(method, steps=400):
        state = DistillationState(
            time=0,
            x=_NOMINAL_PROFILE,
            zF=p.zF_nominal,
            L=NOMINAL_L,
            V=NOMINAL_V,
            target_yD=0.99,
            target_xB=0.01,
        )
        for _ in range(steps):
            state, _ = compute_next_state(raw, state, p, key, integration_method=method)
        return np.asarray(state.x)

    stable = run("rk4_16")
    assert np.all(np.diff(stable) > 0.0)
    assert 0.95 < stable[-1] < 1.0

    unstable = run("rk4_2")
    assert not np.all(np.diff(unstable) > 0.0)


def test_vle_is_monotone_and_enriching(params):
    """Vapour is always richer in the light component than the liquid."""
    x = np.linspace(0.01, 0.99, 50)
    y = np.array([float(vle(float(xi), params)) for xi in x])
    assert np.all(np.diff(y) > 0.0)
    assert np.all(y > x)


def test_state_stays_finite_over_an_episode(params):
    env = DistillationColumn()
    p = params.replace(max_steps_in_episode=100)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    for _ in range(100):
        key, sub = jax.random.split(key)
        _, state, reward, terminated, _ = env.step_env(sub, state, jnp.zeros(2), p)
        assert np.all(np.isfinite(np.asarray(state.x)))
        assert np.isfinite(float(reward))
        if bool(terminated):
            break


# ---------------------------------------------------------------------------
# 4. Task interface
# ---------------------------------------------------------------------------


def test_observation_hides_the_interior_profile(params):
    """Only the two products are measured; 39 stage compositions are hidden."""
    env = DistillationColumn()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    marked = state.x.at[10].set(0.4242).at[30].set(0.7373)
    state = state.replace(x=marked, zF=0.4321)
    obs = env.get_obs(state, params)
    assert obs.shape == (6,)
    for hidden in (0.4242, 0.7373, 0.4321):
        assert not bool(jnp.any(jnp.isclose(obs, hidden)))


def test_reward_requires_both_specifications(params):
    """Multiplicative: meeting one spec while losing the other scores ~0."""
    env = DistillationColumn()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    x = state.x
    both = state.replace(
        x=x.at[-1].set(0.99).at[0].set(0.01),
        target_yD=0.99,
        target_xB=0.01,
        V=params.V_min,
    )
    top_only = both.replace(x=x.at[-1].set(0.99).at[0].set(0.09))
    assert float(compute_reward(both, params)) > 0.9
    assert float(compute_reward(top_only, params)) < 0.1


def test_terminates_when_purity_is_lost(params):
    env = DistillationColumn()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    lost_top = state.replace(x=state.x.at[-1].set(params.yD_floor - 0.05))
    lost_bottom = state.replace(x=state.x.at[0].set(params.xB_ceiling + 0.05))
    assert bool(check_is_terminal(lost_top, params)[0])
    assert bool(check_is_terminal(lost_bottom, params)[0])
    assert not bool(check_is_terminal(state, params)[0])


def test_reset_starts_from_the_converged_profile(params):
    env = DistillationColumn()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    assert isinstance(state, DistillationState)
    assert state.x.shape == (N_STAGES,)
    assert np.all(np.diff(np.asarray(state.x)) > 0.0)
    assert float(state.x[-1]) == pytest.approx(0.99, abs=0.01)
