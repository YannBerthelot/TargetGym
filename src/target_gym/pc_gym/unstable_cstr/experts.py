"""The expert (the PID slot) for the LDPE autoclave: a temperature PID on the
initiator feed, as the plant runs it [L1, p. 2: "Temperature control is done
by manipulating the initiator feed based on the actual temperature"].

The oracle (the MPC slot) is in ``oracle.py``. The expert reads ``obs = [T,
I_f_pct, T_setpoint]`` and adds its PI(D) to the initiator the setpoint needs
at the nominal feed. Gains from ``src/target_gym/data/pid_gains.json``, by a
published rule for an unstable point (PHYSICS.md, "Expert").
"""

import math

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class AutoclavePIDParams:
    """``Kp`` in raw action per K (negative: a hotter reactor gets less
    initiator), ``Ti`` and ``Td`` in s, the reading filter's ``alpha``, the
    step ``dt`` and the setpoint's ``bias`` in raw action."""

    Kp: float
    Ti: float
    Td: float
    alpha: float
    dt: float
    bias: float


@struct.dataclass
class AutoclavePIDState:
    integral: float
    T: float
    started: float


def autoclave_pid_step(params: AutoclavePIDParams, state, obs, xp=jnp):
    """One update; the integral holds while the pump is saturated."""
    T = xp.where(state.started > 0, state.T + params.alpha * (obs[0] - state.T), obs[0])
    e = T - obs[2]
    slope = xp.where(state.started > 0, (T - state.T) / params.dt, 0.0)
    integral = state.integral + e * params.dt
    u = params.bias + params.Kp * (e + integral / params.Ti + params.Td * slope)
    u_clipped = xp.clip(u, -1.0, 1.0)
    return xp.reshape(u_clipped, (1,)), AutoclavePIDState(
        integral=xp.where(u == u_clipped, integral, state.integral),
        T=T,
        started=xp.ones_like(T),
    )


def autoclave_pid(Kp: float, Ti: float, Td: float, params=None, Tf: float = 0.0):
    """The expert with these gains on ``params``' reactor, as a pure
    :class:`~target_gym.experts.pid.Controller`."""
    from target_gym.experts.pid import Controller
    from target_gym.pc_gym.unstable_cstr.env import UnstableCSTRParams, steady_state

    p = params if params is not None else UnstableCSTRParams()
    dt = float(p.delta_t)
    _, _, I_f = steady_state(p.T_setpoint, p)
    bias = 2.0 * (float(I_f) - p.I_f_min) / (p.I_f_max - p.I_f_min) - 1.0
    return Controller(
        AutoclavePIDParams(
            Kp=Kp,
            Ti=Ti,
            Td=Td,
            alpha=-math.expm1(-dt / Tf) if Tf else 1.0,
            dt=dt,
            bias=bias,
        ),
        AutoclavePIDState(integral=0.0, T=0.0, started=0.0),
        autoclave_pid_step,
    )


def make_unstable_cstr_pid_controller():
    """The scored expert; ``.host()`` is what ``make_pid()`` returns."""
    from target_gym.experts.pid import gain

    return autoclave_pid(
        Kp=gain("unstable_cstr", "Kp"),
        Ti=gain("unstable_cstr", "Ti"),
        Td=gain("unstable_cstr", "Td"),
        Tf=gain("unstable_cstr", "Tf"),
    )
