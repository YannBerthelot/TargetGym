"""The expert (the PID slot) for the first-order task.

The oracle (the MPC slot) is in ``oracle.py``, so that the expert and
the oracle each have a fingerprint of their own
(``provenance.expert_fingerprint``, ``provenance.oracle_fingerprint``).
"""

import math

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class FilteredPIParams:
    """A PI on a filtered reading: ``alpha`` is how far the filter moves per step."""

    Kp: float
    Ki: float
    Kd: float
    alpha: float
    dt: float


@struct.dataclass
class FilteredPIState:
    integral: float
    x: float
    started: float


def filtered_pi_step(params: FilteredPIParams, state, obs, xp=jnp):
    """One update on ``obs = [x, target]``. The filter starts at the first
    reading, and the integral holds while the output is saturated."""
    x = xp.where(state.started > 0, state.x + params.alpha * (obs[0] - state.x), obs[0])
    e = obs[1] - x
    integral = state.integral + e * params.dt
    u = params.Kp * e + params.Ki * integral
    u_clipped = xp.clip(u, -1.0, 1.0)
    return xp.array([u_clipped]), FilteredPIState(
        integral=xp.where(u == u_clipped, integral, state.integral),
        x=x,
        started=xp.ones_like(x),
    )


def first_order_pi(Kp: float, Ki: float, Tf: float = 0.0, Kd: float = 0.0, dt=0.05):
    """The PI with these gains and a filter of time ``Tf`` on the reading, as a
    pure :class:`~target_gym.experts.pid.Controller`. ``Kd`` is stored for the
    gains' record; the recipe gives no derivative action."""
    from target_gym.experts.pid import Controller

    alpha = -math.expm1(-dt / Tf) if Tf else 1.0
    return Controller(
        FilteredPIParams(Kp=Kp, Ki=Ki, Kd=Kd, alpha=alpha, dt=dt),
        FilteredPIState(integral=0.0, x=0.0, started=0.0),
        filtered_pi_step,
    )


def make_first_order_pid_controller():
    """The scored PI as a pure :class:`~target_gym.experts.pid.Controller`;
    ``.host()`` is what ``make_pid()`` returns. Gains and filter from step tests,
    stored in src/target_gym/data/pid_gains.json by
    ``scripts/tune_process_experts.py`` (PHYSICS.md section 6)."""
    from target_gym.experts.pid import gain

    return first_order_pi(
        Kp=gain("first_order", "Kp"),
        Ki=gain("first_order", "Ki"),
        Tf=gain("first_order", "Tf"),
        Kd=gain("first_order", "Kd"),
    )
