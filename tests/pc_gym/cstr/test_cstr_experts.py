"""The shipped expert for cstr: valve position control.

Contract, from its source and the task's record, not from the law's code:

- The structure is Shinskey's valve position control as Reyes-Lua, Zotica and
  Skogestad lay it out [S13, section 2.3]: a PI holds the reactor temperature by
  the cooling water, and a second, slow PI holds the water valve at a set point
  short of its limit by the feed. The feed never passes the throughput asked for.
- It takes over a running plant bumplessly (charter B5): with the temperature
  on its target and the water valve on its set point, both valves stay where
  the observation shows them.
- The temperature is held ``under_top`` K under the top of the band the product
  is on spec in, ``T_band`` above the setpoint (PHYSICS.md, "Expert").
- The observation is ``[T (K), T_j (K), water valve (%), feed (%), setpoint
  (K)]``; an action is ``[water, feed]`` in ``[-1, 1]``, and a valve at ``x`` %
  of its span is the raw ``x / 50 - 1``.
- It is the reference's law: with the same gains it moves the valves as the
  ladder's valve position rung does (``experts.ladder``, tested on its own).
- Its set points are placed so that no deciding training seed of the normal
  day leaves the band (owner's rule, 2026-11-23); the gains' record names the
  seeds that did not decide.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.experts.ladder import Pi, Rung, ValvePosition, ladder_controller
from target_gym.experts.pid import load_gains
from target_gym.pc_gym.cstr.env import CSTRParams
from target_gym.pc_gym.cstr.experts import cstr_valve_position
from target_gym.registry import REGISTRY

P = CSTRParams()
DT = float(P.delta_t)
SP = float(P.T_setpoint)
BAND = float(P.T_band)
# Gains picked for round arithmetic, not the shipped ones.
KP, TI, KV, TIV, UNDER, VALVE = 0.2, 10.0, 0.5, 100.0, 0.4, 0.95
TARGET = SP + BAND - UNDER


def expert(**changes):
    settings = dict(Kp=KP, Ti=TI, Kv=KV, Tiv=TIV, under_top=UNDER, valve=VALVE)
    return cstr_valve_position(**{**settings, **changes}, params=P)


def first_action(law, obs):
    action, _ = law.step(law.params, law.init_state, np.asarray(obs, float), np)
    return np.asarray(action, float)


def test_on_target_the_valves_stay_where_the_plant_had_them():
    # Water at its 95 % set point is the raw 0.9; the feed at 70 % is 0.4.
    action = first_action(expert(), [TARGET, 310.0, 95.0, 70.0, SP])
    np.testing.assert_allclose(action, [0.9, 0.4], atol=1e-12)


def test_a_reactor_above_its_target_gets_more_water():
    # 0.5 K hot from a valve at 60 % (raw 0.2): the PI's first move is
    # Kp (e + e dt / Ti) = 0.2 (0.5 + 0.5 * dt / 10).
    action = first_action(expert(), [TARGET + 0.5, 310.0, 60.0, 70.0, SP])
    assert action[0] == pytest.approx(0.2 + 0.2 * (0.5 + 0.5 * DT / 10.0), abs=1e-12)


def test_the_target_sits_under_the_bands_top_by_the_placed_margin():
    # At the band's top less the margin the temperature loop asks for nothing,
    # whatever the margin: the water stays at the 60 % it was found at.
    for under in (0.1, 0.7):
        law = expert(under_top=under)
        action = first_action(law, [SP + BAND - under, 310.0, 60.0, 70.0, SP])
        assert action[0] == pytest.approx(0.2, abs=1e-12)


def test_a_water_valve_past_its_set_point_cuts_the_feed():
    # The valve reads 99 % (raw 0.98) against 95 % (0.9): e = -0.08, and the
    # feed leaves its 70 % (0.4) by Kv (e + e dt / Tiv).
    action = first_action(expert(), [TARGET, 310.0, 99.0, 70.0, SP])
    cut = 0.5 * (-0.08 + -0.08 * DT / 100.0)
    assert action[1] == pytest.approx(0.4 + cut, abs=1e-12)


def test_the_feed_stops_at_the_throughput_asked_for_and_does_not_wind_up():
    law = expert()
    memory = law.init_state
    # A valve at 60 % has room, so the loop asks for more feed than there is.
    for _ in range(200):
        action, memory = law.step(
            law.params, memory, np.array([TARGET, 310.0, 60.0, 100.0, SP]), np
        )
        assert action[1] == 1.0
    # The valve then passes its set point. A loop that had wound up would hold
    # the feed at its top; this one cuts it by its first move from there.
    action, _ = law.step(
        law.params, memory, np.array([TARGET, 310.0, 99.0, 100.0, SP]), np
    )
    assert action[1] == pytest.approx(1.0 + 0.5 * (-0.08 - 0.08 * DT / 100.0), abs=1e-9)


def test_a_filtered_reading_moves_part_of_the_way_each_step():
    # A filter of dt / ln 2 moves half way a step. The first reading is taken
    # whole; a reading 1 K hot on the second step is then seen as 0.5 K.
    law = expert(Tf=DT / np.log(2.0))
    _, memory = law.step(
        law.params, law.init_state, np.array([TARGET, 310.0, 60.0, 70.0, SP]), np
    )
    action, _ = law.step(
        law.params, memory, np.array([TARGET + 1.0, 310.0, 60.0, 70.0, SP]), np
    )
    assert action[0] == pytest.approx(0.2 + 0.2 * (0.5 + 0.5 * DT / 10.0), abs=1e-9)


def test_it_moves_the_valves_as_the_ladders_valve_position_rung_does():
    def raw(k):
        return lambda o, xp: o[k] / 50.0 - 1.0

    # The ladder's error is target less reading, so its water gain is -Kp.
    rung = Rung(
        "valve position",
        (Pi(0, -KP, TI),),
        np.array([[1.0], [0.0]]),
        (),
        (),
        (ValvePosition(raw(2), 2 * VALVE - 1, 1, KV, TIV),),
    )
    reference = ladder_controller(
        rung,
        lambda o, xp: xp.stack([o[0]]),
        lambda o, xp: xp.stack([o[4] + BAND - UNDER]),
        lambda o, xp: xp.zeros(2),
        DT,
        lambda o, xp: xp.stack([raw(2)(o, xp), raw(3)(o, xp)]),
    )
    law = expert()
    rng = np.random.default_rng(0)
    mine, theirs = law.init_state, reference.init_state
    for _ in range(300):
        # Temperatures around the target and valves across their spans, so the
        # run meets both limits.
        obs = np.array(
            [
                TARGET + rng.normal(0.0, 2.0),
                310.0,
                rng.uniform(0.0, 100.0),
                rng.uniform(0.0, 100.0),
                SP,
            ]
        )
        a, mine = law.step(law.params, mine, obs, np)
        b, theirs = reference.step(reference.params, theirs, obs, np)
        np.testing.assert_allclose(a, b, atol=1e-9)


def test_the_shipped_expert_keeps_the_band_on_the_seeds_that_placed_it():
    spec = REGISTRY["cstr"]
    env, params = spec.make_env(), spec.make_test_params()
    law = spec.make_pid_controller()
    left_out = load_gains()["cstr"]["model"]["placement"]["left_out"]
    seeds = [s for s in (100, 101, 102) if s not in left_out]
    assert seeds

    def episode(key):
        def body(carry, t):
            obs, state, memory = carry
            action, memory = law.step(law.params, memory, obs, jnp)
            obs, state, _, _, info = env.step_env(
                jax.random.fold_in(key, t), state, action, params
            )
            off = jnp.abs(state.T - state.target_T) > params.T_band
            return (obs, state, memory), (off, info["tripped"])

        start = (*env.reset_env(key, params), law.init_state)
        steps = jnp.arange(int(params.max_steps_in_episode))
        return jax.lax.scan(body, start, steps)[1]

    keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds))
    off, tripped = jax.jit(jax.vmap(episode))(keys)
    assert not np.asarray(tripped).any()
    assert not np.asarray(off).any()
