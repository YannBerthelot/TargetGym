"""The rebuilt reactor: the turbine sets the load, the rods set Tavg.

Contract, from the plan (docs/proposals/industrial-fixes.md, PR 6) and the sources
in PHYSICS.md section 3, not from the code:

* Energy balance: with a steam generator, steady reactor power equals the turbine
  load whatever the rods do.
* What the rods set is Tavg. With power fixed, the fuel-to-coolant difference is
  fixed, so a rod move ``d rho`` shifts both temperatures by ``d rho / |alpha_f +
  alpha_m|``: 100 pcm over 33 pcm/K is 3.03 K.
* The turbine drives the cold leg, through the steam generator.
* Tavg follows the NRC program, 557 F at no load to 584.7 F at full load.
* A day is 100-50-100 at 5 %/min over 12 h, at 30 s steps, because at 60 s the
  program's slowest speed overshoots its dead band.
* Rods move at 72 steps/min both ways, over a 228-step stroke of the 1500 pcm bank:
  7.89 pcm/s. The jacks have one speed.
* Wear charges rods that moved, not rods that were asked to.
* Tavg off program costs lost output, 0.23 %/K of 1 GWe at $100/MWh: $230/h per K.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.reactor.env import (
    ReactorParams,
    compute_next_state,
    compute_reward_terms,
    rod_balance,
    scheduled_load,
    steady_state_precursors,
    steady_state_temperatures,
    steady_state_xenon,
    tavg_program,
)
from target_gym.reactor.env_jax import CONTROL_PERIOD, Reactor

P = ReactorParams()
ENV = Reactor()
QUIET = P.replace(demand_sigma=0.0)


def _lined_out(load, rods_offset=0.0):
    """The plant at rest at ``load``, ``rods_offset`` of reactivity off balance on
    its boron, with the rods parked and the turbine held there."""
    params = QUIET.replace(load_min=load)
    _, state = ENV.reset_env(jax.random.PRNGKey(0), params)
    T_fuel, T_hot, T_cold, T_sg = steady_state_temperatures(load, params)
    I_hat, Xe_hat = steady_state_xenon(load, params)
    rods = rod_balance(T_fuel, 0.5 * (T_hot + T_cold), Xe_hat, params)
    state = state.replace(
        n=jnp.float32(load),
        C=steady_state_precursors(load, params),
        T_fuel=jnp.float32(T_fuel),
        T_coolant=jnp.float32(T_hot),
        T_cold=jnp.float32(T_cold),
        T_sg=jnp.float32(T_sg),
        I_hat=jnp.float32(I_hat),
        Xe_hat=jnp.float32(Xe_hat),
        target_n=jnp.float32(load),
        T_ref=jnp.float32(tavg_program(load, params)),
        boron=jnp.float32(params.boron_ref + (rods + rods_offset) / params.boron_worth),
        ramp_down_at=jnp.float32(-1e6),
        ramp_up_at=jnp.float32(1e9),
    )
    return state, params


def _action(state):
    """The rod demand that leaves the rods where they are."""
    return 1.0 - 2.0 * state.rod_in


def _hold(state, params, seconds, raw=None):
    """Run the plant with the rods at ``raw``, by default where they are."""
    raw = _action(state) if raw is None else raw

    def body(s, _):
        return compute_next_state(jnp.array([raw, 0.0]), s, params)[0], None

    steps = int(seconds / float(params.delta_t))
    return jax.lax.scan(body, state, None, length=steps)[0]


def _tavg(state):
    return 0.5 * (float(state.T_coolant) + float(state.T_cold))


# ---------------------------------------------------------------------------
# What the secondary side changes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [-100e-5, 100e-5])
def test_power_settles_at_the_load_whatever_the_rods_do(offset):
    state, params = _lined_out(0.8, offset)
    assert float(_hold(state, params, 2400.0).n) == pytest.approx(0.8, abs=2e-3)


def test_rods_set_tavg_by_the_feedback_sum():
    """200 pcm between two rod positions over 33 pcm/K: 6.06 K."""
    low = _hold(*_lined_out(0.8, -100e-5), 2400.0)
    high = _hold(*_lined_out(0.8, 100e-5), 2400.0)
    assert _tavg(high) - _tavg(low) == pytest.approx(200.0 / 33.0, rel=0.03)


def test_a_turbine_cut_heats_the_cold_leg():
    """Less steam leaves, so the secondary warms, and the cold leg with it."""
    state, params = _lined_out(0.8)
    cold_before, steam_before = float(state.T_cold), float(state.T_sg)
    cut = params.replace(load_min=0.6)
    state = state.replace(target_n=jnp.float32(0.6))
    held = _hold(state, cut, 60.0)
    assert float(held.T_sg) > steam_before + 0.1
    assert float(held.T_cold) > cold_before + 0.1


def test_a_lined_out_plant_stays_lined_out():
    state, params = _lined_out(0.8)
    held = _hold(state, params, 600.0)
    assert _tavg(held) == pytest.approx(_tavg(state), abs=0.05)
    assert float(held.n) == pytest.approx(0.8, abs=1e-4)


def test_the_full_power_heat_balance():
    """33.3 K across the core at 9e7 W/K, 22 K from Tavg to the steam at 1.36e8 W/K,
    and Tavg on its program."""
    _, T_hot, T_cold, T_sg = steady_state_temperatures(1.0, P)
    assert float(T_hot) - float(T_cold) == pytest.approx(3e9 / 9e7, rel=1e-3)
    assert 0.5 * (float(T_hot) + float(T_cold)) == pytest.approx(580.2, abs=0.05)
    assert 0.5 * (float(T_hot) + float(T_cold)) - float(T_sg) == pytest.approx(
        22.0, rel=0.01
    )


def test_the_program_is_the_nrc_one():
    """557 F is 564.8 K and 584.7 F is 580.2 K."""
    assert float(tavg_program(0.0, P)) == pytest.approx(564.8, abs=0.1)
    assert float(tavg_program(1.0, P)) == pytest.approx(580.2, abs=0.1)
    assert float(tavg_program(0.5, P)) == pytest.approx(572.5, abs=0.1)


# ---------------------------------------------------------------------------
# The day
# ---------------------------------------------------------------------------


def test_inserting_rods_lowers_power_on_the_scored_step():
    """Past an inner solve of about 1.25 s, TR-BDF2 misses the prompt jump and this
    goes the wrong way: 40 pcm in, then a 5 s sub-step, reads 1.04 at two solves."""
    state, params = _lined_out(0.8)
    new, _ = compute_next_state(jnp.array([-1.0, 0.0]), state, params)
    assert float(new.n) < 0.8


def test_a_day_is_24_hours_of_30_s_steps():
    assert CONTROL_PERIOD * float(P.delta_t) == pytest.approx(30.0)
    assert int(P.max_steps_in_episode) == 2880


def test_the_load_ramps_at_two_percent_a_minute():
    down, up = 3600.0, 5.0 * 3600.0
    load = lambda t: float(scheduled_load(t, down, up, P))  # noqa: E731
    assert load(0.0) == pytest.approx(1.0)
    assert load(down + 750.0) == pytest.approx(0.75, abs=1e-3)
    assert load(down + 1500.0) == pytest.approx(0.5, abs=1e-3)
    assert load(up - 1.0) == pytest.approx(0.5, abs=1e-3)
    assert load(up + 750.0) == pytest.approx(0.75, abs=1e-3)
    assert load(up + 1500.0) == pytest.approx(1.0, abs=1e-3)


# test_a_day_has_two_moves_for_the_scorer (seeds 100, 101, 103) was retired by the
# owner on 2026-11-08 (TESTING.md section 10): it checked the old scorer's move
# detector, which the weighted-saving headline does not use.


def test_reset_starts_on_program_at_full_load():
    """With no dilution carried over (reactor-v7 carries 0 to 1 day of it)."""
    params = P.replace(sensor_noise_scale=0.0, carried_days_range=(0.0, 0.0))
    obs, state = ENV.reset_env(jax.random.PRNGKey(0), params)
    assert float(state.n) == pytest.approx(1.0)
    assert float(obs[4]) == pytest.approx(float(obs[5]), abs=1e-3)
    assert float(obs[5]) == pytest.approx(580.2, abs=0.1)


# ---------------------------------------------------------------------------
# The rods
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw, sign", [(1.0, 1.0), (-1.0, -1.0)])
def test_rods_move_at_72_steps_a_minute_both_ways(raw, sign):
    """72 / 60 / 228 of the stroke a second, over the 5 s sub-step: 6 steps."""
    state, params = _lined_out(0.8)
    state = state.replace(rod_in=jnp.float32(0.5))
    new, _ = compute_next_state(jnp.array([raw, 0.0]), state, params)
    moved = (float(state.rod_in) - float(new.rod_in)) * 228.0
    assert moved == pytest.approx(sign * 6.0, rel=1e-3)


def test_wear_charges_travel_and_not_a_command_the_rods_cannot_reach():
    """Asked for the far end of the bank, rods still move 72 steps/min: 6 steps in
    the 5 s sub-step, $6, however far off the command is. Past their stop they
    move nothing, so it costs nothing."""
    state, params = _lined_out(0.8)
    state = state.replace(rod_in=jnp.float32(0.5))
    new, _ = compute_next_state(jnp.array([1.0, 0.0]), state, params)
    assert float(compute_reward_terms(new, params)["running"]) == pytest.approx(
        6.0, rel=1e-3
    )
    stuck = state.replace(rod_in=jnp.float32(0.0))
    new, _ = compute_next_state(jnp.array([1.0, 0.0]), stuck, params)
    assert float(compute_reward_terms(new, params)["running"]) == pytest.approx(0.0)


def test_a_rod_step_costs_its_price():
    """One step of the 228-step stroke at $1 a step."""
    state, params = _lined_out(0.8)
    moved = state.replace(rod_moved=jnp.float32(1.0 / 228.0))
    assert float(compute_reward_terms(moved, params)["running"]) == pytest.approx(
        1.0, rel=1e-3
    )


def test_one_kelvin_past_the_dead_band_costs_its_lost_output():
    """$230/h per K past the rods' 1.6 K dead band, over the 5 s sub-step: $0.319.
    The owner moved the charge outside the dead band on 2026-11-08 (EDF practice),
    and to Tavg under its program only (rule 15, Amendment 3)."""
    state, params = _lined_out(0.8)
    off = state.replace(T_coolant=state.T_coolant - 5.2)  # Tavg down 2.6 K
    cost = float(compute_reward_terms(off, params)["tracking"])
    on = float(compute_reward_terms(state, params)["tracking"])
    assert cost - on == pytest.approx(230.0 * 5.0 / 3600.0, rel=0.01)


def test_a_step_sums_the_sub_steps_of_a_moving_rod():
    """A full-speed step moves 72 / 2 = 36 rod steps in 30 s, so $36 of wear."""
    state, params = _lined_out(0.8)
    state = state.replace(rod_in=jnp.float32(0.5))
    _, _, _, _, info = ENV.step_env(
        jax.random.PRNGKey(0), state, jnp.array([1.0, 0.0]), params
    )
    assert float(info["reward_terms"]["running"]) == pytest.approx(36.0, rel=0.01)


def test_the_expert_keeps_the_offset_in_band_on_a_steady_load():
    """No use if it limit-cycles: over 4 h at a steady load the rods barely move, and
    delta I and Tavg stay inside their bands."""
    import target_gym as tg

    expert = tg.make_expert("reactor").host()
    expert.reset()
    params = QUIET.replace(max_steps_in_episode=480)
    obs, state = ENV.reset_env(jax.random.PRNGKey(100), params)
    far = jnp.float32(1e8)
    state = state.replace(ramp_down_at=far, ramp_up_at=far)
    obs = ENV.get_obs(state, params)  # the plan the expert reads: no ramp today
    step = jax.jit(ENV.step_env)
    key = jax.random.PRNGKey(1)
    rods, errors = [float(state.rod_in) * 228.0], []
    for _ in range(480):
        obs, state, _, _, _ = step(key, state, expert(np.asarray(obs)), params)
        rods.append(float(state.rod_in) * 228.0)
        errors.append(abs(float(obs[4]) - float(obs[5])))
    assert np.abs(np.diff(rods)).sum() < 500.0
    assert np.median(errors) < params.e_tol  # inside the rods' dead band
    offset = float(state.delta_I) / float(state.n) - params.ao_ref
    assert abs(offset) < params.ao_band
