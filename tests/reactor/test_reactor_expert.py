"""The reactor expert: Westinghouse PWR rod control on average temperature.

Contract, from the NRC's Westinghouse Technology training manual (HRTD 8.1,
rod control; HRTD 5.0, the Tavg program and the 40 s rate/lag), not from the
code:

* Tavg is programmed linearly in load from no-load to full power. Here the
  cold leg is the coolant inlet and the hot leg the observed coolant node, so
  ``Tref = T_inlet + P / (2 m cp) * load``; the load is the demand.
* The total error, in deg F, is ``(Tref - Tavg)`` plus the power-mismatch
  term: a rate/lag ``40 s / (1 + 40 s)`` on (demand - power) in % power,
  through a nonlinear gain of 0.3 F per % up to 1 % and 1.5 F per % beyond,
  times a variable gain of 2 up to 50 % load and 1 / load above.
* Rod speed: none inside +-1.5 F from rest; once moving, the rods stop only
  at 1.0 F (lockup); 8 steps/min to 3 F, then 32 steps/min per F, 72 steps/min
  from 5 F. A positive error withdraws.
* Ours: the 228-step stroke spans the rods' range, so a speed of ``s``
  steps/min moves the command by ``2 s * 10 / (60 * 228)`` of the [-1, 1]
  action per 10 s step (72 steps/min: 0.10526).
"""

import jax
import numpy as np
import pytest

import target_gym as tg
from target_gym.registry import REGISTRY

P = REGISTRY["reactor"].make_test_params()
PER_SPM = 2.0 * 10.0 / (60.0 * 228.0)  # action per step per step/min
DT_HALF = float(P.P_thermal_ref) / (2.0 * float(P.m_dot_cp))  # 16.67 K


def _obs(error_F=0.0, n=0.8, target=0.8, rod=0.0):
    """An observation whose temperature error (Tref - Tavg) is ``error_F``."""
    T_ref = float(P.T_inlet) + DT_HALF * target
    T_avg = T_ref - error_F / 1.8
    T_hot = 2.0 * T_avg - float(P.T_inlet)
    return np.array([n, T_hot, rod, target], np.float32)


def _speed(expert, obs):
    """The rod speed commanded, in steps/min, signed (withdrawal positive)."""
    return (float(np.asarray(expert(obs))[0]) - float(obs[2])) / PER_SPM


def _expert():
    expert = tg.make_expert("reactor").host()
    expert.reset()
    return expert


@pytest.mark.parametrize(
    "error_F, spm",
    [(0.0, 0.0), (1.2, 0.0), (2.0, 8.0), (4.0, 40.0), (6.0, 72.0), (-6.0, -72.0)],
)
def test_the_rod_speed_program(error_F, spm):
    """From rest, on the temperature error alone. 4 F: 8 + 32 * (4 - 3) = 40."""
    assert _speed(_expert(), _obs(error_F)) == pytest.approx(spm, abs=1e-2)


def test_moving_rods_stop_only_inside_the_lockup():
    expert = _expert()
    assert _speed(expert, _obs(2.0)) == pytest.approx(8.0, abs=1e-2)
    assert _speed(expert, _obs(1.2)) == pytest.approx(8.0, abs=1e-2)
    assert _speed(expert, _obs(0.9)) == pytest.approx(0.0, abs=1e-2)
    assert _speed(expert, _obs(1.2)) == pytest.approx(0.0, abs=1e-2)


def test_a_demand_step_moves_the_rods_through_the_rate_lag():
    """Demand (the load, 80 %) 5 % above power, Tavg on its program. The rate/lag
    leaves 5 e^(-k/4) % after k steps of 10 s: 3.894, 3.033, 2.362, 1.839,
    1.433, 1.116. Through the gains (0.3 + 1.5 (m - 1)) * 1.25 F: 5.80, 4.19,
    2.93, 1.95, 1.19, 0.59. So 72, 8 + 32 * 1.19 = 45.96, 8, 8, 8 (still
    above the 1.0 F lockup), then stop."""
    expert = _expert()
    _speed(expert, _obs(n=0.8, target=0.8))
    speeds = [_speed(expert, _obs(n=0.75, target=0.80)) for _ in range(6)]
    np.testing.assert_allclose(speeds, [72.0, 45.96, 8.0, 8.0, 8.0, 0.0], atol=0.05)


def test_the_mismatch_gain_doubles_below_half_load():
    """2 % of mismatch leaves 1.558 % after one step, 1.136 F through the
    nonlinear gain: times 2 at 40 % load it moves the rods (2.27 F), times
    1 / 0.8 at 80 % it does not (1.42 F)."""
    low, high = _expert(), _expert()
    _speed(low, _obs(n=0.40, target=0.40))
    _speed(high, _obs(n=0.80, target=0.80))
    assert _speed(low, _obs(n=0.38, target=0.40)) == pytest.approx(8.0, abs=1e-2)
    assert _speed(high, _obs(n=0.78, target=0.80)) == pytest.approx(0.0, abs=1e-2)


def test_power_below_demand_withdraws_and_above_inserts():
    up, down = _expert(), _expert()
    _speed(up, _obs())
    _speed(down, _obs())
    assert _speed(up, _obs(n=0.75, target=0.80)) > 0.0
    assert _speed(down, _obs(n=0.85, target=0.80)) < 0.0


# Spec, stated before measuring: Westinghouse rod control holds Tavg inside its
# deadband. On held-out seeds, over the scored hold, Tavg is within 1.5 F of
# Tref on at least 90 % of steps, and nothing trips.
HELD_OUT = range(200, 210)


@pytest.mark.slow
def test_tavg_stays_inside_the_deadband_on_held_out_seeds():
    from target_gym.eval import scored_burn_in

    spec = REGISTRY["reactor"]
    env, p = spec.make_env(), spec.make_test_params()
    step = jax.jit(env.step_env)
    burn = scored_burn_in("reactor", p)
    for seed in HELD_OUT:
        expert = tg.make_expert("reactor").host()
        expert.reset()
        key = jax.random.PRNGKey(seed)
        obs, state = env.reset_env(key, p)
        inside = []
        for t in range(int(p.max_steps_in_episode)):
            key, k = jax.random.split(key)
            obs, state, _, _, info = step(k, state, expert(np.asarray(obs)), p)
            assert not bool(info["tripped"])
            if t >= burn:
                o = np.asarray(obs)
                T_avg = 0.5 * (float(p.T_inlet) + o[1])
                T_ref = float(p.T_inlet) + DT_HALF * o[3]
                inside.append(abs(T_ref - T_avg) * 1.8 <= 1.5)
        assert np.mean(inside) >= 0.9, f"seed {seed}: {np.mean(inside):.2f}"
