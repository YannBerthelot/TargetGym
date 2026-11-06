"""The rebuilt battery: following a regulation signal from a finite, ageing pack.

Contract, from the plan (docs/proposals/industrial-fixes.md, PR 6) and the sources
in PHYSICS.md section 3, not from the code:

* Energy binds: over the 4 h window, following the signal exactly leaves the 5 to
  95 % charge window on at least half of the seeds.
* The meter reads what reaches the grid: the inverter's output less the auxiliary
  load, 15 kW (MESA-1 measured 30 kW on a 2 MW system, nearly constant up to 1C).
* The pack supplies the inverter's loss: 3 % at rated power.
* A cycle of depth ``u`` costs ``Phi(u) = 5.24e-4 u^2.03`` of the pack's life, half
  on each half-cycle (Xu et al. 2017, 2018). The life ends at 20 % fade and the
  pack costs $300/kWh.
* The regulation signal is AR(1): ``r' = phi r + std eps`` with ``phi =
  exp(-dt / tau)``, drawn after the action, so no causal policy beats
  ``E|eps| = std sqrt(2 / pi)``.
* Shelf life is 10 years to end of life: 2 % fade a year at rest.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.energy.battery.env import (
    BatteryParams,
    compute_next_state,
    compute_reward_terms,
    current_for_power,
    cycle_stress,
    degradation_rate,
    half_cycle,
    terminal_voltage,
)
from target_gym.energy.battery.env_jax import GridBattery

P = BatteryParams()
ENV = GridBattery()


def _rest(soc=0.5, target=0.0):
    _, state = ENV.reset_env(jax.random.PRNGKey(0), P)
    return state.replace(
        soc=jnp.float32(soc),
        soc_anchor=jnp.float32(soc),
        soc_peak=jnp.float32(soc),
        target_power=jnp.float32(target),
    )


def _step(state, command_W):
    new, _ = compute_next_state(
        command_W / float(P.power_max), state, P, jax.random.PRNGKey(1)
    )
    return new


def _follow(soc_path):
    """Life used by moving through ``soc_path``, half-cycle by half-cycle."""
    anchor = peak = soc_path[0]
    life = 0.0
    for soc in soc_path[1:]:
        anchor, peak, used = half_cycle(anchor, peak, soc, P)
        life += float(used)
    return life


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


def test_the_window_is_four_hours_of_hourly_blocks():
    assert float(P.delta_t) * int(P.max_steps_in_episode) == pytest.approx(4 * 3600)
    assert float(P.dispatch_block_seconds) == pytest.approx(3600.0)


@pytest.mark.medium
def test_following_exactly_leaves_the_charge_window_on_most_seeds():
    """Half the seeds is the plan's bar: state of charge has to be the problem."""
    seeds = 20
    keys = jax.random.split(jax.random.PRNGKey(100), seeds)

    def episode(key):
        _, state = ENV.reset_env(key, P)

        def body(carry, k):
            state, out = carry
            # The level the next step is scored against, plus the signal now.
            action = (state.target_power + float(P.aux_load)) / float(P.power_max)
            new, _ = compute_next_state(action, state, P, k)
            out = out | ENV.is_terminated(new, P)
            return (new, out), None

        ks = jax.random.split(key, int(P.max_steps_in_episode))
        (_, left), _ = jax.lax.scan(body, (state, jnp.bool_(False)), ks)
        return left

    left = np.asarray(jax.jit(jax.vmap(episode))(keys))
    assert left.mean() >= 0.5


# ---------------------------------------------------------------------------
# What the meter reads
# ---------------------------------------------------------------------------


def test_a_resting_inverter_shows_the_auxiliary_load_at_the_meter():
    state = _step(_rest(), 0.0)
    assert float(state.power) == pytest.approx(-15e3, abs=5.0)


def test_the_meter_reads_the_command_less_the_auxiliary_load():
    state = _step(_rest(), 500e3)
    assert float(state.power) == pytest.approx(500e3 - 15e3, rel=1e-4)


def test_the_pack_supplies_the_inverters_loss_at_rated_power():
    """3 % of 1 MW on top of 1 MW: V I at the pack terminals is 1.03 MW."""
    state = _step(_rest(), float(P.power_max))
    dc = float(terminal_voltage(state.current, state.soc, state.v_rc, P)) * float(
        state.current
    )
    assert dc == pytest.approx(1.03e6, rel=2e-3)


def test_charging_stores_less_than_the_grid_gives():
    """The same loss, now taken from what reaches the pack: 0.97 MW."""
    state = _step(_rest(), -float(P.power_max))
    dc = float(terminal_voltage(state.current, state.soc, state.v_rc, P)) * float(
        state.current
    )
    assert dc == pytest.approx(-0.97e6, rel=2e-3)


# ---------------------------------------------------------------------------
# The price of a cycle
# ---------------------------------------------------------------------------


def test_the_stress_function_gives_the_published_cycle_life():
    """3000 cycles at 80 % depth (Xu et al. 2017)."""
    assert 1.0 / float(cycle_stress(0.8, P)) == pytest.approx(3000.0, rel=0.01)


def test_a_full_cycle_costs_one_stress_and_a_half_cycle_half():
    """0.2 to 0.8 and back is one cycle of depth 0.6: Phi(0.6) = 1.872e-4.
    The first leg alone is half of it."""
    phi = 5.24e-4 * 0.6**2.03
    assert _follow(np.linspace(0.2, 0.8, 60)) == pytest.approx(0.5 * phi, rel=1e-3)
    there_and_back = np.concatenate(
        [np.linspace(0.2, 0.8, 60), np.linspace(0.8, 0.2, 60)[1:]]
    )
    assert _follow(there_and_back) == pytest.approx(phi, rel=0.02)


def test_ripple_inside_the_hysteresis_adds_no_deep_cycle():
    """A swing back of less than 5 % is no turning point, so the whole rise is one
    half-cycle, of the path's full range."""
    ripple = np.linspace(0.2, 0.8, 200) + 0.02 * np.sin(np.arange(200.0))
    depth = ripple.max() - ripple[0]
    assert _follow(ripple) == pytest.approx(0.5 * 5.24e-4 * depth**2.03, rel=1e-3)


def test_splitting_a_depth_costs_less_than_going_deep():
    """Convexity: 0.3 down and back twice is cheaper than 0.6 once."""
    shallow = np.concatenate([np.linspace(0.5, 0.2, 30), np.linspace(0.2, 0.5, 30)[1:]])
    deep = np.concatenate([np.linspace(0.8, 0.2, 60), np.linspace(0.2, 0.8, 60)[1:]])
    assert 2.0 * _follow(shallow) < 0.5 * _follow(deep)


def test_past_the_threshold_the_next_mwh_wears_more_than_its_imbalance():
    """The trade-off itself. Half of ``R dPhi/du`` with R = $300,000/MWh is
    $159.6/MWh at u^1.03 = 1, so wear equals the $100/MWh imbalance at
    u = (100 / 159.6)^(1 / 1.03) = 0.635. The price is ``fade_price`` on the fade
    that half-cycle adds."""
    u_hat = (100.0 / 159.6) ** (1.0 / 1.03)
    assert u_hat == pytest.approx(0.635, abs=1e-3)

    def wear_per_MWh(depth):
        du = 1e-3
        start = 0.05
        _, _, before = half_cycle(start, start + depth, start + depth + du, P)
        fade = float(before) * float(P.life_fade)
        return float(P.fade_price) * float(P.pack_kWh) * fade / (du * P.pack_kWh / 1e3)

    assert wear_per_MWh(u_hat - 0.05) < 100.0 < wear_per_MWh(u_hat + 0.05)


def test_the_reward_charges_a_step_its_deep_fade():
    """Moving 1 % of charge at depth 0.5 costs what the price says, whatever the
    calendar does: ``fade_price * pack_kWh * life_fade * life``."""
    state = _rest(soc=0.3)
    state = state.replace(soc_anchor=jnp.float32(0.8), soc_peak=jnp.float32(0.3))
    _, _, life = half_cycle(0.8, 0.3, 0.29, P)
    fade = float(P.life_fade) * float(life)
    moved = state.replace(soc=jnp.float32(0.29), deep_fade=jnp.float32(fade))
    extra = float(compute_reward_terms(moved, P)["running"]) - float(
        compute_reward_terms(state, P)["running"]
    )
    assert extra == pytest.approx(float(P.fade_price) * P.pack_kWh * fade, rel=1e-3)


def test_a_year_at_rest_fades_two_percent():
    """10 years of shelf life to the 20 % that ends it."""
    year = 365.25 * 86400.0
    assert float(degradation_rate(0.0, 25.0, P)) * year == pytest.approx(0.02, rel=0.01)


def test_a_rested_pack_pays_no_wear():
    state = _rest()
    assert float(compute_reward_terms(state, P)["running"]) == pytest.approx(
        0.0, abs=1e-9
    )


# ---------------------------------------------------------------------------
# The regulation signal and the next block
# ---------------------------------------------------------------------------


def test_the_signal_decays_by_phi_on_its_mean_path():
    """With no noise, ``exp(-5 / 900)`` of it is left after a step."""
    params = P.replace(dispatch_noise_std=0.0)
    state = _rest().replace(regulation=jnp.float32(100e3))
    new, _ = compute_next_state(0.0, state, params, jax.random.PRNGKey(0))
    assert float(new.regulation) == pytest.approx(
        100e3 * np.exp(-5.0 / 900.0), rel=1e-5
    )


def test_the_floor_is_the_innovations_mean_absolute_value():
    """``std sqrt(2 / pi)`` with std = 10 kW: 7979 W."""
    assert float(P.e_floor) == pytest.approx(1e4 * np.sqrt(2.0 / np.pi), rel=1e-3)


def test_reset_draws_the_signal_at_its_stationary_spread():
    """``std / sqrt(1 - phi^2)``: 95.0 kW at 10 kW and 900 s. 400 draws give a
    standard error of about 4 %."""
    keys = jax.random.split(jax.random.PRNGKey(3), 400)
    reset = jax.vmap(lambda k: ENV.reset_env(k, P)[1].regulation)
    phi = np.exp(-5.0 / 900.0)
    expected = 1e4 / np.sqrt(1.0 - phi**2)
    assert float(np.std(np.asarray(reset(keys)))) == pytest.approx(expected, rel=0.12)


def test_the_observation_carries_the_next_block_and_the_time_to_it():
    """Hourly blocks at 5 s: at step 0 the next block is 60 min off, and at step
    719 it is 5 s off. From the last block the next stays the last."""
    _, state = ENV.reset_env(jax.random.PRNGKey(0), P)
    schedule = np.asarray(state.dispatch_schedule) / 1e6
    params = P.replace(sensor_noise_scale=0.0)
    obs = np.asarray(ENV.get_obs(state, params))
    assert obs[5] == pytest.approx(schedule[1], rel=1e-5)
    assert obs[6] == pytest.approx(60.0)
    late = np.asarray(ENV.get_obs(state.replace(time=719), params))
    assert late[5] == pytest.approx(schedule[1], rel=1e-5)
    assert late[6] == pytest.approx(5.0 / 60.0, rel=1e-5)
    end = np.asarray(ENV.get_obs(state.replace(time=2879), params))
    assert end[5] == pytest.approx(schedule[-1], rel=1e-5)


def test_the_pack_traverses_ten_to_ninety_percent_in_75_minutes():
    """At rated power the pack gives 1.03 MW at about 665 V under load, so 1.55 kA:
    0.8 * 2500 Ah / 1.59 kA = 75.5 min. The old docs said 96."""
    current = float(current_for_power(1.03e6, 0.5, 0.0, P))
    minutes = 0.8 * float(P.capacity_As) / current / 60.0
    assert minutes == pytest.approx(75.0, abs=1.5)
