"""The expert (the PID slot) for the CSTR task: a temperature PI on the cooling
water.

The oracle (the MPC slot) is in ``oracle.py``, so that the expert and
the oracle each have a fingerprint of their own
(``provenance.expert_fingerprint``, ``provenance.oracle_fingerprint``).

The expert is the loop reactor practice closes on the measured temperature
[S5; S6]: a PI from the reactor's temperature to the fresh cooling water, as
PARSEX's usual SISO scheme runs it [S1, p. 4], biased at the water the
setpoint needs at the design feed. It reads ``obs = [T, T_j, F_w_pct,
T_setpoint]``. Its gains are SIMC on a bump test (PHYSICS.md, "Expert");
the reference ladder replaces them.
"""

import math

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class TemperaturePIParams:
    """``Kp`` in raw action per K (positive: a hotter reactor opens the water),
    ``Ti`` and ``dt`` in minutes, ``alpha`` the reading filter's step, and the
    design feed's ``bias`` in raw action."""

    Kp: float
    Ti: float
    alpha: float
    dt: float
    bias: float


@struct.dataclass
class TemperaturePIState:
    integral: float
    T: float
    started: float


def temperature_pi_step(params: TemperaturePIParams, state, obs, xp=jnp):
    """One update; the integral holds while the valve is saturated."""
    T = xp.where(state.started > 0, state.T + params.alpha * (obs[0] - state.T), obs[0])
    e = T - obs[3]
    integral = state.integral + e * params.dt
    u = params.bias + params.Kp * (e + integral / params.Ti)
    u_clipped = xp.clip(u, -1.0, 1.0)
    return xp.reshape(u_clipped, (1,)), TemperaturePIState(
        integral=xp.where(u == u_clipped, integral, state.integral),
        T=T,
        started=xp.ones_like(T),
    )


def cstr_temperature_pi(Kp: float, Ti: float, params=None, Tf: float = 0.0):
    """The expert with these gains on ``params``' reactor (the scored
    configuration's by default), as a pure
    :class:`~target_gym.experts.pid.Controller`. ``Tf`` is the reading's
    filter, in minutes."""
    from target_gym.experts.pid import Controller
    from target_gym.pc_gym.cstr.env import CSTRParams, steady_state

    p = params if params is not None else CSTRParams()
    dt = float(p.delta_t)
    T_f = 0.5 * (p.feed_T_range[0] + p.feed_T_range[1])
    _, _, F_w = steady_state(p.T_setpoint, T_f, p)
    bias = 2.0 * (float(F_w) - p.Fw_min) / (p.Fw_max - p.Fw_min) - 1.0
    return Controller(
        TemperaturePIParams(
            Kp=Kp,
            Ti=Ti,
            alpha=-math.expm1(-dt / Tf) if Tf else 1.0,
            dt=dt,
            bias=bias,
        ),
        TemperaturePIState(integral=0.0, T=0.0, started=0.0),
        temperature_pi_step,
    )


def make_cstr_pid_controller():
    """The scored expert as a pure :class:`~target_gym.experts.pid.Controller`;
    ``.host()`` is what ``make_pid()`` returns. Gains from
    src/target_gym/data/pid_gains.json."""
    from target_gym.experts.pid import gain

    return cstr_temperature_pi(
        Kp=gain("cstr", "Kp"), Ti=gain("cstr", "Ti"), Tf=gain("cstr", "Tf")
    )
