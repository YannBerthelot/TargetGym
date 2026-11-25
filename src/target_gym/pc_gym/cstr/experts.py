"""The expert (the PID slot) for the CSTR task: valve position control.

Version 8 ships no oracle (``oracle.py`` keeps the ceiling's pieces), and the
expert has a fingerprint of its own (``provenance.expert_fingerprint``).

The structure is Shinskey's valve position control as Reyes-Lua, Zotica and
Skogestad lay it out [S13, section 2.3], the reference ladder's rung of that
name (``experts.ladder.ValvePosition``) written out so that its gains are
leaves: a PI from the reactor's temperature to the fresh cooling water, as
PARSEX's usual SISO scheme runs it [S1, p. 4], and a second, slow PI that
holds the water valve at a set point short of its limit by moving the feed,
never past the throughput asked for. It reads ``obs = [T, T_j, F_w_pct, q_pct,
T_setpoint]`` and takes both valves over where the reset left them. Its gains
and its two set points are the recipe's (``scripts/tune_process_experts.py``;
PHYSICS.md, "Expert").
"""

import math

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class ValvePositionParams:
    """``Kp`` in raw water per K (positive: a hotter reactor opens the water),
    ``Kv`` in raw feed per raw of valve position, ``Ti``, ``Tiv`` and ``dt`` in
    minutes, ``alpha`` the reading filter's step, ``above`` the K over the
    setpoint the temperature is held at and ``valve`` the water valve's set
    point, raw."""

    Kp: float
    Ti: float
    Kv: float
    Tiv: float
    alpha: float
    dt: float
    above: float
    valve: float


@struct.dataclass
class ValvePositionState:
    T: float
    integral: float
    valve_integral: float
    #: The two valves as the first observation showed them, raw.
    water: float
    feed: float
    started: float


def valve_position_step(params: ValvePositionParams, state, obs, xp=jnp):
    """One update. Each integral holds while its valve is limited."""
    running = state.started > 0
    seen = obs[2] / 50.0 - 1.0
    T = xp.where(running, state.T + params.alpha * (obs[0] - state.T), obs[0])
    water_0 = xp.where(running, state.water, seen)
    feed_0 = xp.where(running, state.feed, obs[3] / 50.0 - 1.0)

    e = T - (obs[4] + params.above)
    integral = state.integral + e * params.dt
    asked = water_0 + params.Kp * (e + integral / params.Ti)
    water = xp.clip(asked, -1.0, 1.0)

    e_v = params.valve - seen
    valve_integral = state.valve_integral + e_v * params.dt
    wanted = feed_0 + params.Kv * (e_v + valve_integral / params.Tiv)
    feed = xp.clip(wanted, -1.0, 1.0)

    return xp.stack([water, feed]), ValvePositionState(
        T=T,
        integral=xp.where(water == asked, integral, state.integral),
        valve_integral=xp.where(feed == wanted, valve_integral, state.valve_integral),
        water=water_0,
        feed=feed_0,
        started=xp.ones_like(T),
    )


def cstr_valve_position(
    Kp: float,
    Ti: float,
    Kv: float,
    Tiv: float,
    under_top: float,
    valve: float,
    params=None,
    Tf: float = 0.0,
):
    """The expert with these gains on ``params``' reactor (the scored
    configuration's by default), as a pure
    :class:`~target_gym.experts.pid.Controller`. The temperature is held
    ``under_top`` K under the top of the on-spec band and the water valve at
    ``valve`` of its span; ``Tf`` is the temperature reading's filter, in
    minutes."""
    from target_gym.experts.pid import Controller
    from target_gym.pc_gym.cstr.env import CSTRParams

    p = params if params is not None else CSTRParams()
    dt = float(p.delta_t)
    return Controller(
        ValvePositionParams(
            Kp=Kp,
            Ti=Ti,
            Kv=Kv,
            Tiv=Tiv,
            alpha=-math.expm1(-dt / Tf) if Tf else 1.0,
            dt=dt,
            above=float(p.T_band) - under_top,
            valve=2.0 * valve - 1.0,
        ),
        ValvePositionState(
            T=0.0, integral=0.0, valve_integral=0.0, water=0.0, feed=0.0, started=0.0
        ),
        valve_position_step,
    )


def make_cstr_pid_controller():
    """The scored expert as a pure :class:`~target_gym.experts.pid.Controller`;
    ``.host()`` is what ``make_pid()`` returns. Gains from
    src/target_gym/data/pid_gains.json."""
    from target_gym.experts.pid import gain

    names = ("Kp", "Ti", "Kv", "Tiv", "under_top", "valve", "Tf")
    return cstr_valve_position(**{name: gain("cstr", name) for name in names})
