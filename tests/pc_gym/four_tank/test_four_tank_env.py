import jax.numpy as jnp
import numpy as _np
import pytest

from target_gym.pc_gym.four_tank.env import (
    FourTankParams,
    FourTankState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_velocity,
)


@pytest.fixture
def params():
    return FourTankParams()


@pytest.fixture
def state():
    return FourTankState(
        time=0,
        h1=0.5,
        h2=0.5,
        h3=0.3,
        h4=0.3,
        target_h1=0.7,
        target_h2=0.7,
        v1=5.0,
        v2=5.0,
    )


# -----------------------------------------------------------------------
# Velocity / ODE structure
# -----------------------------------------------------------------------


def test_velocity_shape(params, state):
    v, _ = compute_velocity(
        jnp.array([state.h1, state.h2, state.h3, state.h4]),
        action=jnp.array([5.0, 5.0]),
        params=params,
    )
    assert v.shape == (4,)


def test_gravity_draining_with_no_pumps(params, state):
    """With zero pump input, all non-empty tanks drain under gravity."""
    v, _ = compute_velocity(
        jnp.array([state.h1, state.h2, state.h3, state.h4]),
        action=jnp.array([0.0, 0.0]),
        params=params,
    )
    # Upper tanks (h3, h4) have no pump inflow when v=0, only outflow
    assert v[2] < 0.0  # dh3/dt < 0
    assert v[3] < 0.0  # dh4/dt < 0


def test_pump_v1_fills_lower_tank_1(params):
    """Pump v1 drives inflow to tank 1 (gamma1 fraction) and tank 4 (1-gamma1 fraction)."""
    # With h3=0 there's no cascade inflow from tank 3 to tank 1
    h = jnp.array([0.01, 0.5, 0.0, 0.5])
    v_low, _ = compute_velocity(h, action=jnp.array([1.0, 0.0]), params=params)
    v_high, _ = compute_velocity(h, action=jnp.array([9.0, 0.0]), params=params)
    # Higher v1 -> more inflow to tank 1
    assert v_high[0] > v_low[0]


def test_pump_v2_fills_lower_tank_2(params):
    h = jnp.array([0.5, 0.01, 0.5, 0.0])
    v_low, _ = compute_velocity(h, action=jnp.array([0.0, 1.0]), params=params)
    v_high, _ = compute_velocity(h, action=jnp.array([0.0, 9.0]), params=params)
    assert v_high[1] > v_low[1]


def test_split_ratio_v1_distributes_to_h1_and_h4(params):
    """Pump v1 splits: gamma1 fraction goes to h1, (1-gamma1) to h4.
    With h=0 for all tanks, sqrt(max(h,0))=0 so only pump inflow terms contribute.
    """
    h = jnp.array([0.0, 0.0, 0.0, 0.0])
    v, _ = compute_velocity(h, action=jnp.array([5.0, 0.0]), params=params)

    inflow_h1 = params.gamma1 * params.k1 * 5.0 / params.A1
    inflow_h4 = (1 - params.gamma1) * params.k1 * 5.0 / params.A4

    assert v[0] == pytest.approx(inflow_h1, rel=1e-5)
    assert v[3] == pytest.approx(inflow_h4, rel=1e-5)


def test_split_ratio_v2_distributes_to_h2_and_h3(params):
    """Pump v2 splits: gamma2 fraction goes to h2, (1-gamma2) to h3."""
    h = jnp.array([0.0, 0.0, 0.0, 0.0])
    v, _ = compute_velocity(h, action=jnp.array([0.0, 5.0]), params=params)

    inflow_h2 = params.gamma2 * params.k2 * 5.0 / params.A2
    inflow_h3 = (1 - params.gamma2) * params.k2 * 5.0 / params.A3

    assert v[1] == pytest.approx(inflow_h2, rel=1e-5)
    assert v[2] == pytest.approx(inflow_h3, rel=1e-5)


def test_outflow_proportional_to_sqrt_level(params):
    """Tank outflow rate scales with sqrt(h) (Torricelli's law)."""
    # 4x level -> 2x outflow (sqrt(4h) = 2*sqrt(h))
    h_base = 0.25
    h_quad = 1.0  # 4x

    v_base, _ = compute_velocity(
        jnp.array([h_base, h_base, 0.0, 0.0]),
        action=jnp.array([0.0, 0.0]),
        params=params,
    )
    v_quad, _ = compute_velocity(
        jnp.array([h_quad, h_quad, 0.0, 0.0]),
        action=jnp.array([0.0, 0.0]),
        params=params,
    )
    # Outflow term: -(a/A)*sqrt(2g)*sqrt(h); so ratio should be sqrt(4) = 2
    assert abs(v_quad[0]) == pytest.approx(2 * abs(v_base[0]), rel=1e-4)


def test_no_outflow_at_zero_level(params):
    """sqrt(max(h, 0)) ensures zero outflow when tank is empty."""
    h = jnp.array([0.0, 0.0, 0.0, 0.0])
    v, _ = compute_velocity(h, action=jnp.array([0.0, 0.0]), params=params)
    # With h=0 and v=0, all velocities should be 0
    assert jnp.allclose(v, 0.0, atol=1e-10)


# -----------------------------------------------------------------------
# Integration
# -----------------------------------------------------------------------


@pytest.mark.parametrize("method", ["euler_1", "rk2_1", "rk4_1"])
def test_next_state_advances_time(method, params, state):
    action = jnp.array([0.5, 0.5])
    new_state, _ = compute_next_state(action, state, params, integration_method=method)
    assert new_state.time == state.time + 1


def test_integration_methods_agree_for_small_dt(state):
    params = FourTankParams(delta_t=0.01)
    action = jnp.array([0.3, 0.3])
    s_euler, _ = compute_next_state(action, state, params, integration_method="euler_1")
    s_rk4, _ = compute_next_state(action, state, params, integration_method="rk4_1")
    assert jnp.allclose(s_euler.h1, s_rk4.h1, atol=1e-4)
    assert jnp.allclose(s_euler.h2, s_rk4.h2, atol=1e-4)


# -----------------------------------------------------------------------
# Action scaling
# -----------------------------------------------------------------------


def test_action_scaled_to_pump_range(params, state):
    action_min = jnp.array([-1.0, -1.0])
    action_max = jnp.array([1.0, 1.0])
    s_min, _ = compute_next_state(
        action_min, state, params, integration_method="euler_1"
    )
    s_max, _ = compute_next_state(
        action_max, state, params, integration_method="euler_1"
    )
    assert s_min.v1 == pytest.approx(params.v_min)
    assert s_min.v2 == pytest.approx(params.v_min)
    assert s_max.v1 == pytest.approx(params.v_max)
    assert s_max.v2 == pytest.approx(params.v_max)


# -----------------------------------------------------------------------
# Terminal / Reward
# -----------------------------------------------------------------------


def test_not_terminal_in_bounds(params, state):
    term, trunc = check_is_terminal(state, params)
    assert not term
    assert not trunc


def test_terminal_when_tank_overflows(params, state):
    term, _ = check_is_terminal(state.replace(h1=params.h_max + 0.1), params)
    assert term


def test_terminal_when_tank_underflows(params, state):
    term, _ = check_is_terminal(state.replace(h3=params.h_min - 0.01), params)
    assert term


def test_reward_is_1_at_target(params):
    state = FourTankState(
        time=0,
        h1=0.7,
        h2=0.7,
        h3=0.3,
        h4=0.3,
        target_h1=0.7,
        target_h2=0.7,
        v1=5.0,
        v2=5.0,
    )
    assert compute_reward(state, params) == pytest.approx(1.0)


def test_reward_decreases_with_error(params):
    """Errors are compared inside the tracking band.

    Both offsets used to be tenths of a metre, which the band-scaled reward
    now clips to zero -- indistinguishable, and on a plant whose setpoints
    live between 0.11 and 0.19 m they were never realistic errors anyway.
    """
    common = dict(
        time=0, h3=0.3, h4=0.12, target_h1=0.15, target_h2=0.20, v1=5.0, v2=7.0
    )
    r_close = compute_reward(
        FourTankState(h1=0.15 - 0.005, h2=0.20 - 0.005, **common), params
    )
    r_far = compute_reward(
        FourTankState(h1=0.15 - 0.030, h2=0.20 - 0.030, **common), params
    )
    assert r_close > r_far > 0.0


def test_reward_is_finite(params, state):
    assert jnp.isfinite(compute_reward(state, params))


# ---------------------------------------------------------------------------
# Reachability and loop pairing
#
# Added after the target range was found to sit entirely above what the plant
# can produce: with both pumps saturated the steady state tops out at
# h1 = 0.360, h2 = 0.429, while targets were sampled from (0.5, 1.0). Every
# episode was unwinnable, and the shared effectiveness contract could not see
# it because the PID still beat every constant action -- both simply sat far
# from setpoint. These tests make that class of defect impossible to reintroduce.
# ---------------------------------------------------------------------------


def _steady_levels(v1, v2, p):
    """Steady state of the four tanks at constant pump voltages."""
    g = p.g
    h3 = ((1 - p.gamma2) * p.k2 * v2 / p.a3) ** 2 / (2 * g)
    h4 = ((1 - p.gamma1) * p.k1 * v1 / p.a4) ** 2 / (2 * g)
    h1 = ((p.gamma1 * p.k1 * v1 + p.a3 * _np.sqrt(2 * g * h3)) / p.a1) ** 2 / (2 * g)
    h2 = ((p.gamma2 * p.k2 * v2 + p.a4 * _np.sqrt(2 * g * h4)) / p.a2) ** 2 / (2 * g)
    return h1, h2


def test_every_target_is_individually_reachable():
    """No sampled setpoint may lie above what saturated pumps can hold."""
    p = FourTankParams()
    h1_max, h2_max = _steady_levels(p.v_max, p.v_max, p)
    assert p.target_h1_range[1] < h1_max, (
        f"target h1 up to {p.target_h1_range[1]} exceeds the maximum "
        f"sustainable level {h1_max:.3f}"
    )
    assert p.target_h2_range[1] < h2_max, (
        f"target h2 up to {p.target_h2_range[1]} exceeds the maximum "
        f"sustainable level {h2_max:.3f}"
    )


def test_every_target_pair_is_jointly_reachable():
    """The pumps are cross-coupled, so h1 and h2 must be attainable *together*.

    Targets are sampled independently, so individual reachability is not
    enough -- the whole box has to lie inside the image of the steady-state
    map over the admissible voltages.
    """
    p = FourTankParams()
    V = _np.linspace(max(p.v_min, 1e-3), p.v_max, 220)
    reach = _np.array([_steady_levels(a, b, p) for a in V for b in V])
    for t1 in _np.linspace(*p.target_h1_range, 6):
        for t2 in _np.linspace(*p.target_h2_range, 6):
            d = _np.min((reach[:, 0] - t1) ** 2 + (reach[:, 1] - t2) ** 2)
            assert (
                _np.sqrt(d) < 0.006
            ), f"target pair ({t1:.3f}, {t2:.3f}) is not jointly reachable"


def test_targets_leave_voltage_headroom():
    """Holding the top of the range must not need a saturated pump."""
    p = FourTankParams()
    need = None
    for v in _np.linspace(max(p.v_min, 1e-3), p.v_max, 400):
        if _steady_levels(v, v, p)[0] >= p.target_h1_range[1]:
            need = v
            break
    assert need is not None
    assert need < 0.9 * p.v_max, f"top target needs {need:.2f} V of {p.v_max}"


def test_initial_levels_start_inside_the_operating_envelope():
    """An episode must not begin above a level the plant can never hold."""
    p = FourTankParams()
    h1_max, h2_max = _steady_levels(p.v_max, p.v_max, p)
    assert p.initial_h1_range[1] <= h1_max
    assert p.initial_h2_range[1] <= h2_max
    for rng in (
        p.initial_h1_range,
        p.initial_h2_range,
        p.initial_h3_range,
        p.initial_h4_range,
    ):
        assert rng[0] > p.h_min, "an episode may not start already tripped"


def test_relative_gain_array_demands_the_cross_pairing():
    """gamma1 + gamma2 < 1 puts the plant in the non-minimum-phase regime.

    Johansson (2000) gives lambda11 = g1 g2 / (g1 + g2 - 1). A negative value
    means closing one diagonal loop reverses the sign of the other, so integral
    action on the diagonal pairing is unstable -- which is why the shipped PID
    drives v1 from h2 and v2 from h1.
    """
    p = FourTankParams()
    assert p.gamma1 + p.gamma2 < 1.0, "expected the non-minimum-phase configuration"
    lam = p.gamma1 * p.gamma2 / (p.gamma1 + p.gamma2 - 1.0)
    assert lam < 0.0

    # The same conclusion from the measured gain matrix: the off-diagonal
    # terms dominate.
    v0, eps = 7.0, 1e-4
    G = _np.zeros((2, 2))
    for j, (d1, d2) in enumerate(((eps, 0.0), (0.0, eps))):
        plus = _np.array(_steady_levels(v0 + d1, v0 + d2, p))
        minus = _np.array(_steady_levels(v0 - d1, v0 - d2, p))
        G[:, j] = (plus - minus) / (2 * eps)
    assert abs(G[1, 0]) > 2 * abs(G[0, 0]), "v1 should move h2 far more than h1"
    assert abs(G[0, 1]) > 2 * abs(G[1, 1]), "v2 should move h1 far more than h2"


def _voltages_for(t1, t2, p):
    """Pump voltages that hold a target pair at steady state."""
    A = _np.array(
        [
            [p.gamma1 * p.k1, (1 - p.gamma2) * p.k2],
            [(1 - p.gamma1) * p.k1, p.gamma2 * p.k2],
        ]
    )
    b = _np.array([p.a1 * _np.sqrt(2 * p.g * t1), p.a2 * _np.sqrt(2 * p.g * t2)])
    return _np.linalg.solve(A, b)


def test_upper_tanks_keep_margin_above_the_low_level_trip():
    """The subtler reachability constraint, and the one that actually bit.

    A high h1 with a low h2 is held by a *low* v1 and a high v2 -- and v1 is
    what feeds tank 4. With a wider target box the steady h4 at that corner sat
    only 16 mm above the trip, so a transient dip ended the episode and one
    seed in twenty failed regardless of gains. Every corner must leave real
    margin.
    """
    p = FourTankParams()
    worst = _np.inf
    for t1 in _np.linspace(*p.target_h1_range, 6):
        for t2 in _np.linspace(*p.target_h2_range, 6):
            v1, v2 = _voltages_for(t1, t2, p)
            assert 0.0 <= v1 <= p.v_max and 0.0 <= v2 <= p.v_max
            h4 = ((1 - p.gamma1) * p.k1 * v1 / p.a4) ** 2 / (2 * p.g)
            h3 = ((1 - p.gamma2) * p.k2 * v2 / p.a3) ** 2 / (2 * p.g)
            assert h3 < p.h_max, "upper tank 3 would overflow"
            worst = min(worst, h4)
    assert worst - p.h_min >= 0.030, (
        f"worst-case steady h4 is {worst:.3f} m, only "
        f"{worst - p.h_min:.3f} m above the {p.h_min} m trip"
    )


def test_tracking_band_is_scaled_to_the_operating_range():
    """The reward must discriminate over the levels this plant actually reaches.

    It was previously scaled by the whole tank span (h_max - h_min = 1.45 m),
    about three times the reachable range, so a half-metre miss still scored
    0.43 and a saturated controller looked much like a working one.
    """
    p = FourTankParams()
    span = p.target_h1_range[1] - p.target_h1_range[0]
    assert p.tracking_band < p.h_max - p.h_min
    assert p.tracking_band <= span, "band is wider than the whole setpoint range"


def test_reward_falls_to_zero_outside_the_band():
    """Clipped, so a larger error can never score better than a smaller one."""
    p = FourTankParams()
    base = dict(time=0, h3=0.3, h4=0.12, target_h1=0.15, target_h2=0.20, v1=5.0, v2=7.0)
    prev = None
    for err in (0.0, 0.01, 0.03, 0.05, 0.20, 1.0):
        st = FourTankState(h1=0.15 + err, h2=0.20 + err, **base)
        r = float(compute_reward(st, p))
        assert 0.0 <= r <= 1.0
        if prev is not None:
            assert r <= prev + 1e-9, "reward must never increase with error"
        prev = r
    assert prev == pytest.approx(0.0)
