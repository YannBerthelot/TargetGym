"""The expert (the PID slot) for the glass furnace task: a PI on the fining share read
at the cold end, held two sigma above its pass mark (charter, the two-sigma back-off).

Gains by SIMC from the valve's step test on the settled plant, with the closed-loop time
equal to the reading's first move (``scripts/glass_slow_props.py``, step and walks); the
values and the model are in src/target_gym/data/pid_gains.json. The reference ladder
(``scripts/reference_ladder.py``) is the task's reference; this is its first rung, kept
so the registry's expert slot runs the plant that is scored.
"""

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class FiningPIParams:
    """The fining PI in raw action units: ``u0`` the design valve, ``backoff`` the
    setpoint's distance above the pass mark."""

    Kp: float
    Ki: float
    u0: float
    backoff: float
    Kd: float = 0.0  # kept for the store's record: the recipe gives none
    dt: float = 120.0


@struct.dataclass
class FiningPIState:
    integral: float = 0.0


def fining_pi_step(params, state, obs, xp=jnp):
    """A PI on the cold-end fining reading (obs 4) against its pass mark (obs 5) plus the
    back-off, the integral stopped at the valve's limits."""
    p = params
    e = obs[..., 5] + p.backoff - obs[..., 4]
    integral = state.integral + e * p.dt
    u = p.u0 + p.Kp * e + p.Ki * integral
    u_clipped = xp.clip(u, -1.0, 1.0)
    return xp.stack([u_clipped], axis=-1), FiningPIState(
        integral=xp.where(u == u_clipped, integral, state.integral)
    )


def make_glass_furnace_pid_controller():
    """The scored PI as a pure :class:`~target_gym.experts.pid.Controller`;
    ``.host()`` is what ``make_pid()`` returns."""
    from target_gym.experts.pid import Controller, gain

    params = FiningPIParams(
        Kp=gain("glass_furnace", "Kp"),
        Ki=gain("glass_furnace", "Ki"),
        u0=gain("glass_furnace", "u0"),
        backoff=gain("glass_furnace", "backoff"),
    )
    return Controller(params, FiningPIState(), fining_pi_step)
