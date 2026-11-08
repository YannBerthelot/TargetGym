"""The LDPE autoclave against its plant family (REALITY.md, Amendment 3).

Contract, from the family's sources and the audit's proposed tests, not from
the code (tolerances fixed before any run):
- the operating point is open-loop unstable: the industrial reactor "normally
  operates at an unstable steady state" [L1, p. 2];
- unstable at low temperatures, less so at high ones [L4]: the unstable mode's
  growth rate falls as the setpoint rises across 480 to 537 K;
- the family's numbers: a residence time of 20 to 90 s, and an initiator
  half-life of 0.02 to 1 s at the reactor's temperature [L2, pp. 21-22];
- held, the reactor leaves its point: from 0.1 K off, it leaves +-2 K within
  10 min with the initiator held (audit U-T4);
- a gain floor exists and the point is controllable (audit U-T1): a P loop on
  temperature to initiator holds +-2 K for 10 min at some gain, and a gain a
  tenth of that does not;
- the temperature either climbs or falls until polymerisation stops [L1]:
  with no initiator the reactor cools toward its feed; with full initiator it
  trips at its high temperature [L2; L3];
- the episode starts at rest on its setpoint (rule 17).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.pc_gym.unstable_cstr.env import (
    UnstableCSTRParams,
    compute_next_state,
    compute_velocity,
    rate_constant,
    residence,
    steady_state,
)
from target_gym.pc_gym.unstable_cstr.env_jax import UnstableCSTR

ENV = UnstableCSTR()


def quiet(**kw):
    return UnstableCSTRParams(sensor_noise_scale=0.0, **kw)


def rest(p):
    _, s = ENV.reset_env(jax.random.PRNGKey(0), p)
    return s.replace(event_at=jnp.asarray(1e9))


def growth_rate(T, p):
    """The largest real part of the balances' Jacobian at the steady state of
    ``T`` with the initiator held."""
    I, M, I_f = steady_state(T, p)
    x = jnp.array([I, M, T])
    J = jax.jacfwd(lambda z: compute_velocity(z, I_f, p)[0])(x)
    return float(np.max(np.linalg.eigvals(np.asarray(J)).real))


def test_the_operating_point_is_open_loop_unstable():
    assert growth_rate(UnstableCSTRParams().T_setpoint, quiet()) > 0.0


def test_the_instability_weakens_as_the_temperature_rises():
    rates = [growth_rate(T, quiet()) for T in (480.0, 500.0, 513.0, 537.0)]
    assert all(a > b for a, b in zip(rates, rates[1:])), rates


def test_residence_and_initiator_half_life_are_the_familys():
    p = quiet()
    assert 20.0 <= float(residence(p)) <= 90.0
    half_life = np.log(2.0) / float(rate_constant(p.kd0, p.Ed, p.T_setpoint, p))
    assert 0.02 <= half_life <= 1.0, half_life


def _run(p, s, policy, seconds):
    """True temperatures under ``policy(T, s) -> raw`` by ``compute_next_state``."""
    Ts = []
    for _ in range(int(seconds / p.delta_t)):
        s = compute_next_state(jnp.asarray(policy(float(s.T))), s, p)[0]
        Ts.append(float(s.T))
    return np.asarray(Ts)


def test_held_it_leaves_its_point_within_ten_minutes():
    p = quiet()
    for offset in (0.1, -0.1):
        s = rest(p)
        raw = float(ENV.initial_action(s, p)[0])
        s = s.replace(T=s.T + offset)
        Ts = _run(p, s, lambda T: raw, 600)
        assert np.abs(Ts - p.T_setpoint).max() > 2.0, offset


def _p_loop_holds(gain, p):
    s = rest(p)
    raw = float(ENV.initial_action(s, p)[0])
    s = s.replace(T=s.T + 0.2)
    Ts = _run(
        p, s, lambda T: float(np.clip(raw - gain * (T - p.T_setpoint), -1, 1)), 600
    )
    return np.abs(Ts - p.T_setpoint).max() <= 2.0


def test_a_gain_floor_exists_and_the_point_is_controllable():
    p = quiet()
    gains = [0.003, 0.01, 0.03, 0.1]
    held = [_p_loop_holds(g, p) for g in gains]
    assert any(held), "no P gain held the point"
    first = held.index(True)
    assert not _p_loop_holds(gains[first] / 10.0, p)


def test_with_no_initiator_polymerisation_stops_and_it_cools():
    p = quiet()
    Ts = _run(p, rest(p), lambda T: -1.0, 600)
    assert Ts[-1] < p.T_setpoint - 100.0


def test_with_full_initiator_it_trips_at_its_high_temperature():
    p = quiet(continuing=False)
    s = rest(p)
    step = jax.jit(ENV.step_env)
    tripped = False
    for k in range(600):
        _, s, _, _, info = step(jax.random.PRNGKey(k), s, jnp.array([1.0]), p)
        if bool(info["tripped"]):
            tripped = True
            break
    assert tripped and float(s.T) >= p.T_trip


def test_an_episode_starts_at_rest_on_its_setpoint():
    p = quiet()
    s = rest(p)
    raw = float(ENV.initial_action(s, p)[0])
    Ts = _run(p, s, lambda T: raw, 30)
    assert float(s.T) == pytest.approx(p.T_setpoint)
    assert np.abs(Ts - p.T_setpoint).max() < 0.05
