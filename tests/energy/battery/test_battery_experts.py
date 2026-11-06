"""The battery's oracle, on the shared solver.

Where the charge window does not bind, its claim is that it leaves only the
regulation innovation. With the noise switched off it tracks the target to
float32 rounding, which pins the block it reads (the one the step is scored
against, ``dispatch_block(t + 1)``). With the noise on, its mean error is the
closed-form floor ``e_floor``. Levels of 0.1 MW keep the window out of the way.

Its half-cycles, from the env's ``half_cycle``: a half-cycle turns where a
reference following the planned signal reverses, once it has swung past the
hysteresis, and the heading after a turn points where the charge then goes. A
swing closes where the reference runs back past it. A plan that turns only there
costs what the env charges, to rounding.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import registry
from target_gym.energy.battery.env import N_TURNS
from target_gym.energy.battery.oracle import half_cycles
from target_gym.experts.mpc import plan_params
from target_gym.oracles.controller import Oracle, enable_x64, float64


def _errors(params, seed):
    """|target - power| on every step of one episode under the oracle."""
    spec = registry.get("battery")
    env = spec.make_env()
    oracle = spec.make_mpc(env, plan_params(spec, params))
    oracle.reset()
    key = jax.random.PRNGKey(seed)
    _, state = env.reset_env(key, params)
    step = jax.jit(env.step_env)
    errors = []
    for _ in range(int(params.max_steps_in_episode)):
        action = np.atleast_1d(oracle.step(None, state))
        _, state, _, _, info = step(key, state, action, params)
        assert not bool(info["tripped"])
        errors.append(abs(float(state.target_power) - float(state.power)))
    return np.array(errors)


def test_the_registry_builds_the_shared_oracle():
    spec = registry.get("battery")
    oracle = spec.make_mpc(spec.make_env(), spec.make_test_params())
    assert isinstance(oracle, Oracle)


def _reference_signal(mw):
    """A signal at each block's level, ``mw`` MW for 720 steps a block."""
    return jnp.concatenate([jnp.full(720, level * 1e6) for level in mw])


def _cycles(mw, soc=0.5, anchor=0.5, peak=0.5, turns=()):
    p = registry.get("battery").make_test_params()
    with enable_x64():
        signal = _reference_signal(mw).astype(jnp.float64)
        stack = jnp.zeros(N_TURNS).at[: len(turns)].set(jnp.asarray(turns, float))
        cycles = half_cycles(
            float64(p), *(jnp.float64(v) for v in (soc, anchor, peak)),
            stack, len(turns), signal,
        )  # fmt: skip
    return cycles


def _turns(mw, **kw):
    cycles = _cycles(mw, **kw)
    return np.asarray(cycles.turned), np.asarray(cycles.heading)


def test_a_half_cycle_from_rest_starts_at_once_and_heads_with_the_charge():
    """Discharging at 0.5 MW from rest: charge falls, so the first half-cycle
    heads down from step 0 and never turns."""
    turns, heading = _turns([0.5, 0.5])
    assert np.flatnonzero(turns).tolist() == [0]
    assert (heading == -1.0).all()


def test_a_half_cycle_turns_where_the_signal_reverses():
    """Down for an hour, up for the next: 0.515 MW x 1 h on 1692 kWh moves the
    charge 0.30, far past the hysteresis, so the turn is at step 720, leaving
    the first at the new level, which leaves the peak that 720 steps at the old
    one reached, and the heading is up from there."""
    turns, heading = _turns([0.5, -0.5])
    assert np.flatnonzero(turns).tolist() == [0, 720]
    assert (heading[:720] == -1.0).all() and (heading[720:] == 1.0).all()


def test_a_swing_inside_the_hysteresis_is_not_a_turn():
    """Down, up by 0.0002 MW-h short of 5 % of the charge, down again: the
    hysteresis is 0.05, 84.6 kWh. An hour at -0.07 MW with the 15 kW load is
    55 kWh, 0.033 of charge, so the half-cycle runs on through it."""
    turns, heading = _turns([0.3, -0.07, 0.3])
    assert np.flatnonzero(turns).tolist() == [0]
    assert (heading == -1.0).all()


def test_a_reference_resting_at_the_window_turns_as_it_leaves():
    """Down at 0.8 MW from 0.2: the charge reaches 0.05 in under 16 min and rests
    there until the signal reverses at step 720. The turn is there, not where it
    first reached the window."""
    turns, _ = _turns([0.8, -0.8], soc=0.2, anchor=0.2, peak=0.2)
    assert np.flatnonzero(turns).tolist() == [0, 720]


def test_a_retreat_under_way_turns_at_the_first_step():
    """Peak 0.6, anchor 0.3, charge already 0.02 back from the peak and still
    falling: the half-cycle that rose to 0.6 is over, heading down from step 0."""
    turns, heading = _turns([0.5], soc=0.58, anchor=0.3, peak=0.6)
    assert np.flatnonzero(turns).tolist() == [0]
    assert heading[0] == -1.0


def test_a_swing_the_signal_runs_back_past_closes():
    """Down 0.3 MW, up 0.1 MW, down 0.3 MW from 0.8, 720 steps each, 1692 kWh, the
    15 kW load always on: 0.186, then 0.05 up at 85 kW net, so a swing of 0.0502
    that just turns, then down again. It passes the peak it left about 0.0502 of
    charge later, at 0.315 MW, or 0.27 of an hour, 194 steps into the last block:
    that is the close. The heading stays down through it."""
    cycles = _cycles([0.3, -0.1, 0.3], soc=0.8, anchor=0.8, peak=0.8)
    turned = np.flatnonzero(np.asarray(cycles.turned)).tolist()
    assert turned == [0, 720, 1440]
    folds = np.flatnonzero(np.asarray(cycles.folded)).tolist()
    assert len(folds) == 1 and folds[0] == pytest.approx(1440 + 194, abs=3)
    assert np.asarray(cycles.heading)[folds[0]] == -1.0


@pytest.mark.medium
@pytest.mark.parametrize(
    "levels, soc",
    [
        pytest.param((0.5, -0.5, 0.0), 0.6, id="a turn"),
        pytest.param((0.3, -0.1, 0.3), 0.8, id="a swing that closes"),
    ],
)
def test_a_plan_that_turns_where_the_reference_does_costs_what_the_env_charges(
    levels, soc
):
    """The signal followed with the noise off, the peak grown exactly to the charge.
    Env and model then run the same half-cycles, so K8's replay matches to float64
    rounding, a turn and a closed swing included. The gap is relative to a reward of
    order 10: 1e-10 is many decades under what one missed half-cycle would leave,
    about 133 d^2.03 dollars."""
    spec = registry.get("battery")
    env = spec.make_env()
    params = spec.make_test_params().replace(max_steps_in_episode=2160)
    plan = plan_params(spec, params)
    oracle = spec.make_mpc(env, plan)
    with enable_x64():
        _, state = env.reset_env(jax.random.PRNGKey(3), plan)
        st = float64(state).replace(
            soc=jnp.float64(soc),
            soc_anchor=jnp.float64(soc),
            soc_peak=jnp.float64(soc),
            regulation=jnp.float64(0.0),
            dispatch_schedule=jnp.array([v * 1e6 for v in levels] + [0.0, 0.0]),
        )
        m, p = oracle.model, oracle.params
        d = {
            "task": m.data(st),
            "end": 2160,
            "widths": jnp.zeros_like(jnp.asarray(oracle.widths)),
        }
        signal, cycles = d["task"]
        grows = 2.0 * p.power_max * p.delta_t / (p.pack_kWh * 3.6e6)
        ahead = jax.jit(lambda x, u, t: m.step(d, x, u, t)[0])
        x, moves = m.state_x(st), []
        for t in range(2160):
            u = jnp.array([(signal[t] + p.aux_load) / p.power_max, -1.0])
            turned, folded = cycles.turned[t] > 0, cycles.folded[t] > 0
            base = x[0] if bool(turned) else x[5] if bool(folded) else x[4]
            growth = max(float(cycles.heading[t] * (ahead(x, u, t)[0] - base)), 0.0)
            u = u.at[1].set(2.0 * growth / float(grows) - 1.0)
            x, _ = ahead(x, u, t), moves.append(u)
        cert = oracle._certify(d, st, m.state_x(st), jnp.stack(moves), 2160)
    assert not bool(cert["replay_trip"])
    assert float(cert["replay_gap"]) <= 1e-10


def _unbound(**fields):
    return (
        registry.get("battery")
        .make_test_params()
        .replace(dispatch_range=1e5, max_steps_in_episode=720, **fields)
    )


@pytest.mark.large
def test_it_never_trips_where_following_exactly_would():
    """The default window, on seeds where the pack runs out under exact following."""
    params = registry.get("battery").make_test_params()
    for seed in range(100, 104):
        _errors(params, seed)  # asserts no trip


@pytest.mark.large
def test_without_noise_it_tracks_every_step_to_rounding():
    params = _unbound(dispatch_noise_std=0.0)
    # float32 rounding leaves a few W; reading the wrong block misses by the jump
    # between levels, typically tens of kW here.
    tolerance = 1e-4 * float(params.power_max)  # 100 W
    for seed in range(3):
        assert _errors(params, seed).max() < tolerance


@pytest.mark.large
def test_with_noise_it_holds_the_closed_form_floor():
    params = _unbound()
    mean_error = np.mean([_errors(params, seed).mean() for seed in range(3)])
    # 2160 steps of |N(0, 10 kW)| have a standard error of about 130 W.
    assert mean_error == pytest.approx(float(params.e_floor), rel=0.05)
