"""The expert (the PID slot) for the glass furnace task.

The oracle (the MPC slot) is in ``oracle.py``, so that the expert and
the oracle each have a fingerprint of their own
(``provenance.expert_fingerprint``, ``provenance.oracle_fingerprint``).
"""

import math

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class CrownPIParams:
    """The crown PI, ``alpha`` what its reading's filter passes in one sample."""

    Kp: float
    Ki: float
    alpha: float
    Kd: float = 0.0  # kept for the store's record: the recipe gives none
    dt: float = 30.0


@struct.dataclass
class CrownPIState:
    """The integral, the filtered crown, and whether it has seen a reading."""

    integral: float = 0.0
    crown: float = 0.0
    started: float = 0.0


def crown_pi_step(params, state, obs, xp=jnp):
    """A PI on the filtered crown reading, the integral stopped at the fuel's
    limits. obs: [T_crown_meas, T_air_preheat, fuel_pct, reversal_phase,
    target_T_crown, pull]."""
    p = params
    reading = obs[..., 0]
    crown = xp.where(
        state.started > 0.5, state.crown + p.alpha * (reading - state.crown), reading
    )
    e = obs[..., 4] - crown
    integral = state.integral + e * p.dt
    u = p.Kp * e + p.Ki * integral
    u_clipped = xp.clip(u, -1.0, 1.0)
    return xp.stack([u_clipped], axis=-1), CrownPIState(
        integral=xp.where(u == u_clipped, integral, state.integral),
        crown=crown,
        started=1.0 + 0.0 * reading,
    )


def make_glass_furnace_pid_controller():
    """The scored PI as a pure :class:`~target_gym.experts.pid.Controller`;
    ``.host()`` is what ``make_pid()`` returns. Gains and filter from the recipe
    in ``scripts/tune_industrial_experts.py``, stored in
    src/target_gym/data/pid_gains.json."""
    from target_gym.experts.pid import Controller, gain

    tf, dt = gain("glass_furnace", "Tf"), 30.0
    params = CrownPIParams(
        Kp=gain("glass_furnace", "Kp"),
        Ki=gain("glass_furnace", "Ki"),
        alpha=-math.expm1(-dt / tf) if tf else 1.0,
        dt=dt,
    )
    return Controller(params, CrownPIState(), crown_pi_step)
