"""The compressor's hidden surge line and its transmitters (owner, 2026-11-23).

Contract, from ``REALITY.md`` ("Why the margin is kept") and the owner's choice,
not from the code:

1. The plant's true surge line sits a hidden fraction off the map the controller
   reads: field and shop curves differ, the line drifts with the machine's
   condition, and invariant coordinates leave 2.6 to 7 % at low suction pressure
   [C1; C10; C14]. The model computes its characteristic with the cubic's width
   scaled by ``1 + shift``: the head peaks at Phi = 2W (1 + shift), at the same
   peak head.
2. The shift is the machine's: drawn at the reset from the episode's key,
   uniformly within ``line_shift_max`` either side (ours, inside [C14]'s range),
   and kept through a trip's restart.
3. It is hidden: no observation channel changes with it.
4. The plant is established on the map's control line, the line its control
   system knows: at rest at the header setpoint, the compressor passing the
   mapped line's flow where the consumers take less, and right of its true line
   on every draw.
5. The flow and pressure transmitters lag: first order, 0.15 s on flow (dP
   transmitters at 100 to 150 ms, 200 ms at most) and 0.3 s on pressure (300 to
   400 ms acceptable) [C10, pp. 29-30].
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.compressor_surge.env import (
    CompressorSurgeParams,
    characteristic,
    compute_next_state,
    consumer_flow,
    recycle_flow,
)
from target_gym.compressor_surge.env_jax import CompressorSurge

#: The pressure transmitter's own sigma is 9.17 Pa (PHYSICS.md, Readings).
AT_SETPOINT = 10.0  # Pa
#: The map's 10 % line passes 11.711 kg/s at 24 kPa (hand-derived in
#: test_compressor_surge_reality.py).
MAP_LINE_FLOW = 11.711


def test_the_head_peaks_on_the_true_line_at_the_same_height():
    """W = 0.25: the map's line is Phi = 0.5 and the peak head 0.30 + 2 x 0.18 =
    0.66. A shift of +5 % puts the peak at 0.525 and of -5 % at 0.475, at 0.66."""
    p = CompressorSurgeParams()
    phi = np.linspace(0.3, 0.7, 4001)
    for shift, peak in ((0.0, 0.5), (0.05, 0.525), (-0.05, 0.475)):
        psi = np.asarray(characteristic(phi, p, np, shift=shift))
        assert phi[np.argmax(psi)] == pytest.approx(peak, abs=2e-4)
        assert psi.max() == pytest.approx(0.66, abs=1e-6)


def test_the_machine_is_drawn_once_per_episode_either_side_of_its_map():
    """512 resets: every shift within 5 % either side, both sides drawn, mean
    within three standard errors of 0 (uniform: 0.05 / sqrt(3) / sqrt(512) =
    1.3e-3), and one key gives one machine."""
    p = CompressorSurgeParams()
    env = CompressorSurge()
    keys = jax.random.split(jax.random.PRNGKey(0), 512)
    shift = np.asarray(jax.vmap(lambda k: env.reset_env(k, p)[1].line_shift)(keys))
    assert np.abs(shift).max() <= p.line_shift_max
    assert shift.min() < -0.04 and shift.max() > 0.04
    assert abs(shift.mean()) < 3 * 1.3e-3
    again = env.reset_env(keys[7], p)[1].line_shift
    assert float(again) == float(shift[7])


def test_no_observation_channel_carries_the_shift():
    p = CompressorSurgeParams(sensor_noise_scale=0.0)
    env = CompressorSurge()
    _, s = env.reset_env(jax.random.PRNGKey(0), p)
    one = env.get_obs(s.replace(line_shift=jnp.float32(0.05)), p)
    other = env.get_obs(s.replace(line_shift=jnp.float32(-0.05)), p)
    assert one.shape == (8,)
    np.testing.assert_array_equal(np.asarray(one), np.asarray(other))


def test_the_duct_feels_the_true_characteristic():
    """Two machines in one state, 10 % right of the map's line (Phi 0.55). By
    hand, the cubic there gives Psi = 0.6575 with the line 5 % right of the map
    and 0.6302 with it 5 % left, against the map's 0.6485: after one step the
    first machine's flow is the higher."""
    p = CompressorSurgeParams(demand_range=(0.4, 0.4), demand_sigma=0.0)
    env = CompressorSurge()
    _, s = env.reset_env(jax.random.PRNGKey(0), p)
    hold = env.initial_action(s, p)

    def flow_after(shift):
        at = s.replace(line_shift=jnp.float32(shift))
        return float(compute_next_state(hold, at, p, jax.random.PRNGKey(0))[0].m_c)

    assert flow_after(0.05) > flow_after(0.0) > flow_after(-0.05)


def test_the_plant_is_established_on_the_map_line_whatever_its_machine():
    """64 machines at opening 0.4, where the consumers take less than the map
    line's flow: each starts at the header setpoint, passes the mapped line's
    11.711 kg/s, sits right of its own true line, and is still at its setpoint
    after 10 s of holding."""
    p = CompressorSurgeParams(demand_range=(0.4, 0.4), demand_sigma=0.0)
    env = CompressorSurge()
    keys = jax.random.split(jax.random.PRNGKey(1), 64)
    s = jax.vmap(lambda k: env.reset_env(k, p)[1])(keys)
    np.testing.assert_allclose(np.asarray(s.dp), p.header_setpoint, atol=AT_SETPOINT)
    np.testing.assert_allclose(np.asarray(s.m_c), MAP_LINE_FLOW, atol=0.01)
    true_line = 2.0 * p.W * (1.0 + np.asarray(s.line_shift))
    assert np.all(np.asarray(s.phi_min) > true_line)

    def held(k):
        s0 = env.reset_env(k, p)[1]
        hold = env.initial_action(s0, p)

        def body(st, _):
            return compute_next_state(hold, st, p, k)[0], None

        return jax.lax.scan(body, s0, None, length=100)[0].dp

    np.testing.assert_allclose(
        np.asarray(jax.vmap(held)(keys[:8])), p.header_setpoint, atol=AT_SETPOINT
    )


def test_a_restart_keeps_the_machine():
    """A trip does not change the machine [C10: the line moves with its
    condition]: the restarted plant has the shift it had, and is at rest at its
    setpoint with its flows balanced on that machine."""
    env = CompressorSurge()
    p = CompressorSurgeParams(demand_range=(0.2, 0.6), demand_sigma=0.0)
    _, s = env.reset_env(jax.random.PRNGKey(1), p)
    machine = float(s.line_shift)
    low = jnp.array([-1.0, -1.0])
    step = jax.jit(env.step_env)
    for k in range(1200):
        _, s, _, _, info = step(jax.random.PRNGKey(k), s, low, p)
        if bool(info["tripped"]):
            break
    assert bool(info["tripped"])
    assert float(s.line_shift) == machine
    assert float(s.dp) == pytest.approx(p.header_setpoint, abs=AT_SETPOINT)
    hold = env.initial_action(s, p)
    after = compute_next_state(
        hold, s, p.replace(demand_sigma=0.0), jax.random.PRNGKey(0)
    )[0]
    assert float(after.dp) == pytest.approx(float(s.dp), abs=AT_SETPOINT)


def test_the_flow_and_pressure_readings_lag_first_order():
    """[C10, pp. 29-30]. With the noise off, over 5 s after the recycle is asked
    full open, each reading is the first-order lag of the true value at its time
    constant: per 0.1 s step it closes 1 - exp(-0.1 / tau) of the gap, 0.487 at
    0.15 s (flow) and 0.283 at 0.3 s (pressure). Float32: 1e-4 relative."""
    p = CompressorSurgeParams(sensor_noise_scale=0.0, demand_sigma=0.0)
    env = CompressorSurge()
    obs, s = env.reset_env(jax.random.PRNGKey(0), p)
    action = env.initial_action(s, p).at[1].set(1.0)
    step = jax.jit(env.step_env)
    read = {"dp": float(obs[0]) * 1e3, "m_c": float(obs[1])}
    np.testing.assert_allclose(read["dp"], float(s.dp), rtol=1e-6)
    seen, expected, true = [], [], []
    for k in range(50):
        obs, s, *_ = step(jax.random.PRNGKey(k), s, action, p)
        for name, tau, value in (("dp", 0.3, float(s.dp)), ("m_c", 0.15, float(s.m_c))):
            read[name] += (1.0 - np.exp(-0.1 / tau)) * (value - read[name])
        seen.append([float(obs[0]) * 1e3, float(obs[1])])
        expected.append([read["dp"], read["m_c"]])
        true.append([float(s.dp), float(s.m_c)])
    np.testing.assert_allclose(seen, expected, rtol=1e-4)
    # The readings do trail: the flow moved, and its reading was behind it.
    gap = np.abs(np.asarray(seen) - np.asarray(true))[:10, 1]
    assert gap.max() > 0.05


def test_the_flows_balance_at_the_reset_on_every_machine():
    """At rest the compressor's flow is the consumers' plus the recycle's, to
    0.01 kg/s of about 12, whatever the machine."""
    p = CompressorSurgeParams(demand_sigma=0.0)
    env = CompressorSurge()
    keys = jax.random.split(jax.random.PRNGKey(2), 32)
    s = jax.vmap(lambda k: env.reset_env(k, p)[1])(keys)
    opening = np.asarray(s.demand_levels)[:, 0]
    out = consumer_flow(s.dp, opening, p) + recycle_flow(s.dp, s.x, p)
    np.testing.assert_allclose(np.asarray(s.m_c), np.asarray(out), atol=0.01)
