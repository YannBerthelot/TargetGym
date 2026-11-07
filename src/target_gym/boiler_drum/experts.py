"""The expert (the PID slot) for the boiler drum task.

The oracle (the MPC slot) is in ``oracle.py``, so that the expert and
the oracle each have a fingerprint of their own
(``provenance.expert_fingerprint``, ``provenance.oracle_fingerprint``).
"""

import math

import jax.numpy as jnp
from flax import struct

from target_gym.experts.pid import Controller, gain

# ---------------------------------------------------------------------------
# Three-element PID as one pure step (host NumPy and jit/vmap alike)
# ---------------------------------------------------------------------------


@struct.dataclass
class BoilerDrumPIDParams:
    """Gains and plant constants of the three-element controller.

    Every field is a Python float, so NumPy treats it as weakly typed and the
    host arithmetic stays in the observation's dtype, as it did when
    the baseline was recorded.
    """

    Kp_level: float
    Ki_level: float
    Kp_pressure: float
    Ki_pressure: float
    Kd_pressure: float
    q_feed_max: float
    Q_max: float
    h_feedwater: float
    dt: float = 2.0
    #: What each reading's filter passes of a step in one sample, 1 for none.
    alpha_level: float = 1.0
    alpha_pressure: float = 1.0


@struct.dataclass
class BoilerDrumPIDState:
    """Level and pressure integrators, the last pressure error, the filtered
    readings, and whether the filters have seen a first reading."""

    level_int: float = 0.0
    press_int: float = 0.0
    prev_press_err: float = 0.0
    level_f: float = 0.0
    press_f: float = 0.0
    started: float = 0.0


def boiler_drum_pid_step(params, state, obs, xp=jnp):
    """One step of the boiler drum's level and pressure PIDs, against ``xp``.

    obs: [level, pressure, q_steam, fuel_pct, feed_pct, target_level, target_pressure]
    """
    p = params
    first = state.started < 0.5
    level = xp.where(
        first,
        obs[..., 0],
        state.level_f + p.alpha_level * (obs[..., 0] - state.level_f),
    )
    pressure = xp.where(
        first,
        obs[..., 1],
        state.press_f + p.alpha_pressure * (obs[..., 1] - state.press_f),
    )
    q_steam = obs[..., 2]
    target_level = obs[..., 5]
    target_pressure = obs[..., 6]

    # --- Feedwater: steam-flow feedforward, trimmed by a level PI. The integral
    # stops only at the actuator's limits, while the error pushes past them
    # (conditional integration, Astrom and Hagglund 2006, 3.5). ---
    err_l = target_level - level
    grown = state.level_int + err_l * p.dt
    q_raw = q_steam + p.Kp_level * err_l + p.Ki_level * grown
    q_feed = xp.clip(q_raw, 0.0, p.q_feed_max)
    pushing = ((q_raw > p.q_feed_max) & (err_l > 0.0)) | ((q_raw < 0.0) & (err_l < 0.0))
    level_int = xp.where(pushing, state.level_int, grown)
    feed_raw = 2.0 * (q_feed / p.q_feed_max) - 1.0

    # --- Firing: enthalpy feedforward plus a PI on pressure ---
    Q_ff = q_steam * (2.75e6 - p.h_feedwater)
    err_p = target_pressure - pressure
    press_int = state.press_int + err_p * p.dt
    deriv = (err_p - state.prev_press_err) / p.dt
    Q = Q_ff + p.Q_max * (
        p.Kp_pressure * err_p + p.Ki_pressure * press_int + p.Kd_pressure * deriv
    )
    Q_clipped = xp.clip(Q, 0.0, p.Q_max)
    press_int = xp.where(Q != Q_clipped, press_int - err_p * p.dt, press_int)
    fuel_raw = 2.0 * (Q_clipped / p.Q_max) - 1.0

    new_state = BoilerDrumPIDState(
        level_int=level_int,
        press_int=press_int,
        prev_press_err=err_p,
        level_f=level,
        press_f=pressure,
        started=1.0 + 0.0 * q_steam,
    )
    return xp.stack([fuel_raw, feed_raw], axis=-1), new_state


def _passed(tf: float, dt: float) -> float:
    """What a first-order filter of ``tf`` passes of a step in one sample."""
    return -math.expm1(-dt / tf) if tf else 1.0


def make_boiler_drum_pid_controller() -> Controller:
    """The three-element controller as a :class:`Controller`. Gains and filters
    from the recipe in ``scripts/tune_industrial_experts.py``."""
    from target_gym.boiler_drum.env import BoilerDrumParams

    pr = BoilerDrumParams()
    params = BoilerDrumPIDParams(
        Kp_level=gain("boiler_drum", "Kp_level"),
        Ki_level=gain("boiler_drum", "Ki_level"),
        Kp_pressure=gain("boiler_drum", "Kp_pressure"),
        Ki_pressure=gain("boiler_drum", "Ki_pressure"),
        Kd_pressure=gain("boiler_drum", "Kd_pressure"),
        q_feed_max=pr.q_feed_max,
        Q_max=pr.Q_max,
        h_feedwater=pr.h_feedwater,
        dt=pr.delta_t,
        alpha_level=_passed(gain("boiler_drum", "Tf_level"), pr.delta_t),
        alpha_pressure=_passed(gain("boiler_drum", "Tf_pressure"), pr.delta_t),
    )
    return Controller(params, BoilerDrumPIDState(), boiler_drum_pid_step)
