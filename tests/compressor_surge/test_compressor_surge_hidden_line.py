"""The compressor's hidden surge line and its transmitters (owner, 2026-11-23).

Contract, from ``REALITY.md`` ("Why the margin is kept") and the owner's choice,
not from the code:

1. The plant's true surge line sits a hidden fraction off the map the controller
   reads: field and shop curves differ, the line drifts with the machine's
   condition, and invariant coordinates leave 2.6 to 7 % at low suction pressure
   [C1; C10; C14]. The head peaks at Phi = 2W (1 + shift), at the map's peak
   head.
   The difference is local to the line (owner, 2026-11-23): a plant cannot
   measure where its line has drifted [C10], so in normal operation the machine
   must read as its map. From the map's control line on (10 % right of the map's
   surge line, the line the plant is established on and holds [C6; C1]) the
   machine's head is the map's. Between its true line and that control line the
   head falls smoothly, with no second hump and no jump in slope.
   The property: two machines with opposite shifts read alike, to within their
   instruments' resolution, in normal operation, and part as the line nears.
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


def test_right_of_the_control_line_every_machine_is_its_map():
    """From the map's 10 % control line, Phi = 0.5 x 1.10 = 0.55, each machine's
    head is the map's, exactly."""
    p = CompressorSurgeParams()
    phi = np.linspace(0.55, 0.85, 301)
    on_map = np.asarray(characteristic(phi, p, np))
    for shift in (0.05, -0.05, 0.02):
        np.testing.assert_array_equal(
            np.asarray(characteristic(phi, p, np, shift=shift)), on_map
        )


@pytest.mark.parametrize("shift", [0.05, -0.05])
def test_a_machines_head_falls_smoothly_from_its_line_to_the_control_line(shift):
    """One hump: from its true line to the control line the head only falls.
    And no jump in slope where the stretch starts and ends: one-sided slopes
    over 1e-5 of Phi agree to 1e-3 (the head's curvature is of order 10, so each
    one-sided slope is good to 1e-4)."""
    p = CompressorSurgeParams()
    line, control = 0.5 * (1.0 + shift), 0.55

    def head(phi):
        return np.asarray(characteristic(np.asarray(phi, float), p, np, shift=shift))

    assert np.all(np.diff(head(np.linspace(line, control, 2001))) <= 1e-12)
    for at in (line, control):
        left = (head(at) - head(at - 1e-5)) / 1e-5
        right = (head(at + 1e-5) - head(at)) / 1e-5
        assert float(left) == pytest.approx(float(right), abs=1e-3)


def test_two_machines_read_alike_in_normal_operation_and_part_near_the_line():
    """The owner's property. Two machines in one episode's world, their lines 5 %
    right and 5 % left of the map, the recycle shut and the speed held.

    Normal operation, the consumers at 0.92 and the plant right of its control
    line: over 60 s every reading of the two machines agrees to within its
    instrument's resolution (PHYSICS.md, Readings: pressure 9.17 Pa, flows
    0.021 kg/s, speed 1e-4 of rated, valve 1.5e-3).

    Then the demand falls to 0.725 over 5 s, which carries the map's own machine
    onto its surge line. By hand: at 0.92 the consumers take 0.12 x 0.92 x
    sqrt(14000) = 13.06 kg/s, which 24 kPa needs 89.3 % speed for; at that speed
    the line passes 24.5 x 0.893 x 0.5 = 10.94 kg/s under a peak head of
    25.79 kPa, where consumers at u take 0.12 u sqrt(15790) = 15.08 u, so u =
    0.725. There the first machine is 4.8 % left of its own line and the second
    5.3 % right of its own: the first surges, the second does not, and their
    pressure readings part by more than ten of the transmitter's sigma."""
    p = CompressorSurgeParams(demand_range=(0.92, 0.92), demand_sigma=0.0)
    env = CompressorSurge()
    _, s = env.reset_env(jax.random.PRNGKey(0), p)
    hold = env.initial_action(s, p)
    assert float(s.x) == 0.0

    def run(state, steps, first):
        def body(st, k):
            obs, st, _, _, _ = env.step_env(jax.random.PRNGKey(k), st, hold, p)
            return st, (obs, st.surges)

        return jax.lax.scan(body, state, first + jnp.arange(steps))

    normal, approach = {}, {}
    for name, shift in (("right", 0.05), ("left", -0.05)):
        machine = s.replace(line_shift=jnp.float32(shift))
        at_60s, (obs, _) = run(machine, 600, 0)
        normal[name] = np.asarray(obs)
        falling = at_60s.replace(demand_levels=at_60s.demand_levels.at[3:].set(0.725))
        _, (obs, surges) = run(falling, 300, 600)
        approach[name] = (np.asarray(obs), float(np.sum(np.asarray(surges))))

    # [dp kPa, m_c, m_d, N %, N_ramp %, x %, setpoint kPa, surge count]
    resolution = np.array([9.17e-3, 0.021, 0.021, 0.01, 0.01, 0.15, 1e-6, 0.5])
    assert np.all(np.abs(normal["right"] - normal["left"]) <= resolution)

    assert approach["right"][1] >= 1.0
    assert approach["left"][1] == 0.0
    parted = np.abs(approach["right"][0][:, 0] - approach["left"][0][:, 0])
    assert parted.max() > 10 * 9.17e-3


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
    # The reset does not give the machine away: on the control line every
    # machine is its map, so all start at one speed.
    assert np.ptp(np.asarray(s.N)) <= 1e-6
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
