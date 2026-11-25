"""An optional wrapper that scales a task's observations and rewards for
training, the score untouched (owner, picker 2026-11-25).

Observations come in the plant's own units (kelvin, kilopascals, percent), and
rewards in its own money or product, orders of magnitude apart between tasks.
``ScaledEnv`` maps each observation's documented range to [-1, 1] and divides
the reward by one documented per-task constant: one step of the task's main
loss at its own price. Both are fixed numbers (``SCALES``), never fitted to
data, so a scaled run means the same on every seed and every machine.

The score never reads it: ``score_policy`` and the charter records run the raw
env, and ``info["raw_reward"]`` carries the raw reward through the wrapper.
A policy trained on scaled observations is scored through ``raw_policy``.

Pure JAX, gymnax's API: ``reset(key, params)`` and ``step(key, state, action,
params)``, so it jits, vmaps and nests in other gymnax wrappers.
Outside every fingerprint (``provenance``): a change here stales no record.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax.numpy as jnp
from gymnax.wrappers.purerl import GymnaxWrapper

__all__ = ["SCALES", "Scale", "ScaledEnv", "raw_policy", "scale_obs", "unscale_obs"]


class Scale(NamedTuple):
    """Each observation's documented range, mapped to [-1, 1], and the reward's
    divisor, in the task's reward units."""

    low: tuple
    high: tuple
    reward: float


#: Ranges from each task's parameters and docs page; the reward constant is one
#: step of its main loss at its own price.
SCALES = {
    # Temperatures: the feed at 23 C (296.15 K) to the interlock's 75 C
    # (T_trip 348.15 K); flows in percent of their ranges. Costs are fractions
    # of the nameplate's production per step: a step that makes nothing is 1.
    "cstr": Scale(
        low=(296.15, 296.15, 0.0, 0.0, 296.15),
        high=(348.15, 348.15, 100.0, 100.0, 348.15),
        reward=1.0,
    ),
    # Header and target on the pressure transmitter's 0 to 50 kPa span
    # (reward-shaping.md); flows up to 20 kg/s, above the 18.70 kg/s the
    # machine delivers at 105 % (PHYSICS.md); speed and its setpoint in
    # percent, N_min 70 to N_max 105; valve 0 to 100 %; surges 0 to the trip's
    # count of 3. Reward: a step outside the header zone, zone_price 1000 kJ/s
    # over delta_t 0.1 s.
    "compressor_surge": Scale(
        low=(0.0, 0.0, 0.0, 70.0, 70.0, 0.0, 0.0, 0.0),
        high=(50.0, 20.0, 20.0, 105.0, 105.0, 100.0, 50.0, 3.0),
        reward=100.0,
    ),
    # pH on its pH_min 2 to pH_max 12 range; valves in percent; minutes in an
    # excursion up to the 60 min limit; the influent flow 0 to twice its
    # nominal 16.6 mL/s; the basin 0 to 100 %. Reward: a step outside the
    # permit, excursion_price 1.6667 L/s over delta_t 5 s.
    "ph_neutralization": Scale(
        low=(2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 0.0),
        high=(12.0, 100.0, 12.0, 60.0, 12.0, 100.0, 33.2, 100.0),
        reward=8.3333,
    ),
    # Indoor temperatures (room, slab, water, band centre) on the supply
    # water's T_supply_min 15 to T_supply_max 49 C, which bounds a slab heated
    # by it; outdoor temperatures -20 to 30 C, the -10 C design point less
    # three spreads of 3 K and the mild fortnight's 12 C mean, 5 K swing and
    # three spreads; the clock's sine and cosine; the tariff 0 to its peak,
    # price_punta 0.263 EUR/kWh; hours to its change 0 to 24. Reward: an
    # occupied step 1 K outside the band, comfort_price 9.54 EUR/(K h) over
    # 0.25 h.
    "hvac": Scale(
        low=(15.0, -20.0, -20.0, 15.0, 15.0, 15.0, -1.0, -1.0, 15.0, -20.0, 0.0, 0.0),
        high=(49.0, 30.0, 30.0, 49.0, 49.0, 49.0, 1.0, 1.0, 49.0, 30.0, 0.263, 24.0),
        reward=2.385,
    ),
    # Power and the turbine's load, which rides the frequency wobble above the
    # plan, 0 to n_max 1.09; coolant temperatures (hot leg, Tavg and its
    # program, on one scale so their difference keeps its meaning) from
    # T_avg_min 556 K to T_coolant_max 610 K; rods -1 to +1; the dispatch's
    # loads, the power history, the burnt fraction and the plan 0 to 1; delta I within
    # the lower acceptable-operation limit, ao_limit_low 0.31, either side;
    # boron 0 to 2000 ppm, above boron_ref's 1690 at the cycle's start; hours
    # 0 to 24; the ramp rate 0 to 5 % a minute, the EUR load-follow rate
    # (REALITY.md section 6). Reward: the rated 1 GW for one 30 s step at the
    # spot price, $80/MWh, as the trip's failure_cost.
    "reactor": Scale(
        low=(0.0, 556.0, -1.0, 0.0, 556.0, 556.0, -0.31, 0.0)
        + (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        high=(1.09, 610.0, 1.0, 1.09, 610.0, 610.0, 0.31, 2000.0)
        + (24.0, 24.0, 1.0, 5.0, 1.0, 1.0, 1.0, 24.0),
        reward=666.667,
    ),
}


def _affine(name: str):
    s = SCALES[name]
    low, high = jnp.asarray(s.low, jnp.float32), jnp.asarray(s.high, jnp.float32)
    return (high + low) / 2.0, (high - low) / 2.0


def scale_obs(name: str, obs):
    """``obs`` with each entry's documented range mapped to [-1, 1]."""
    mid, half = _affine(name)
    return (obs - mid) / half


def unscale_obs(name: str, z):
    """The plant's observation from a scaled one."""
    mid, half = _affine(name)
    return z * half + mid


def raw_policy(name: str, policy: Callable) -> Callable:
    """A policy on the plant's observation from one trained on scaled ones,
    for ``score_policy`` and any loop that runs the raw env."""
    return lambda obs: policy(scale_obs(name, obs))


class ScaledEnv(GymnaxWrapper):
    """Task ``name``'s env with scaled observations and rewards; the raw
    reward is ``info["raw_reward"]``. Everything else is the env's."""

    def __init__(self, env, name: str):
        super().__init__(env)
        self.name = name
        self.reward_scale = float(SCALES[name].reward)

    def _out(self, obs, state, reward, *rest):
        *flags, info = rest
        info = {**info, "raw_reward": reward}
        z = scale_obs(self.name, obs)
        return (z, state, reward / self.reward_scale, *flags, info)

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        return scale_obs(self.name, obs), state

    def reset_env(self, key, params):
        obs, state = self._env.reset_env(key, params)
        return scale_obs(self.name, obs), state

    def step(self, key, state, action, params=None):
        return self._out(*self._env.step(key, state, action, params))

    def step_env(self, key, state, action, params):
        return self._out(*self._env.step_env(key, state, action, params))
