"""The expert (the PID slot) for the reactor task.

The oracle (the MPC slot) is in ``oracle.py``, so that the expert and
the oracle each have a fingerprint of their own
(``provenance.expert_fingerprint``, ``provenance.oracle_fingerprint``).
"""

import jax.numpy as jnp
import numpy as np
from flax import struct

from target_gym.experts.pid import Controller

# ---------------------------------------------------------------------------
# The published program: Westinghouse rod control on average temperature
# ---------------------------------------------------------------------------


@struct.dataclass
class RodControlParams:
    """The Westinghouse rod control program.

    The program is the NRC's (Westinghouse Technology training manual,
    HRTD 8.1 and 5.0): a +-1.5 F deadband with a 1.0 F lockup, 8 steps/min to
    3 F, 32 steps/min per F to 72 steps/min at 5 F; the power mismatch through
    a 40 s rate/lag, a nonlinear gain (0.3 F per % to 1 %, 1.5 F per %
    beyond) and a variable gain (2 to 50 % load, 1 / load above). A speed in
    steps/min moves the rods over the plant's 228-step stroke.
    """

    dt: float = 30.0
    deadband_F: float = 1.5
    lockup_F: float = 1.0
    tau_rate: float = 40.0
    stroke_steps: float = 228.0


@struct.dataclass
class RodControlState:
    """The rate/lag's lagged mismatch (% power), whether the rods are moving,
    and whether the lag has seen a first observation."""

    mismatch_lag: float = 0.0
    moving: float = 0.0
    started: float = 0.0


def _mismatch_gain(m, xp):
    """The nonlinear gain unit: deg F for a mismatch rate of ``m`` % power."""
    a = xp.abs(m)
    return xp.sign(m) * xp.where(a <= 1.0, 0.3 * a, 0.3 + 1.5 * (a - 1.0))


def rod_control_step(params, state, obs, xp=jnp):
    """One 30 s step of Westinghouse rod control.

    obs: ``[n, T_hot, rho_ext_norm, load, T_avg, T_ref]``; the action is the
    desired rod position on the same [-1, 1] scale.
    """
    p = params
    n, rod, load = obs[..., 0], obs[..., 2], obs[..., 3]

    # Temperature channel: Tref (programmed on load) minus Tavg, in deg F.
    error = 1.8 * (obs[..., 5] - obs[..., 4])

    # Power-mismatch channel: tau s / (1 + tau s) on (turbine - nuclear), the
    # lag sampled exactly at the control step and started on the first reading.
    x = 100.0 * (load - n)
    alpha = np.exp(-p.dt / p.tau_rate)
    lagged = xp.where(
        state.started > 0.5, alpha * state.mismatch_lag + (1 - alpha) * x, x
    )
    variable = xp.where(load <= 0.5, 2.0, 1.0 / xp.maximum(load, 1e-6))
    error = error + variable * _mismatch_gain(x - lagged, xp)

    # Rod speed program, with the lockup holding moving rods to 1.0 F.
    size = xp.abs(error)
    moving = xp.where(state.moving > 0.5, size > p.lockup_F, size >= p.deadband_F)
    spm = xp.where(moving, xp.clip(8.0 + 32.0 * (size - 3.0), 8.0, 72.0), 0.0)
    move = xp.sign(error) * spm / 60.0 * p.dt / p.stroke_steps * 2.0
    action = xp.clip(rod + move, -1.0, 1.0)
    new_state = RodControlState(
        mismatch_lag=lagged, moving=xp.where(moving, 1.0, 0.0), started=1.0 + 0.0 * x
    )
    return xp.stack([action], axis=-1), new_state


# ---------------------------------------------------------------------------
# The expert: the ladder's best rung
# ---------------------------------------------------------------------------


@struct.dataclass
class ReferenceParams:
    """Rung 5 of ``scripts/reactor_ladder.py``: rods on delta I, boron on Tavg, by
    SIMC from bump tests on seeds 100 to 109. The rods' loop is a pure integral, a
    quarter of the inverse a step, on the largest gain of delta I per unit of rod
    action read at the day's two holds: 0.777 at full power with the rods out, 1.711
    at half power with the rods where delta I sits at 0, so the half-power one. The
    boron's is the integrating rule on -0.0170 K/s per unit of flow, over a median
    delay of 525 s. Reading the gain at full power only left the rods' loop past its
    margin in the half-power hold, where it cycled (docs/proposals/reactor-axial.md).
    Nothing here was set from a cost."""

    dt: float = 30.0
    rod_Kc: float = 0.1461203
    rod_Ti: float = 30.0
    boron_Kc: float = -0.0560285
    boron_Ti: float = 4200.0
    park: float = 1.0 - 2.0 * 3.0 / 228.0
    reach: float = 30.0 * 72.0 / 60.0 / 228.0 * 2.0


@struct.dataclass
class ReferenceState:
    """The two loops' outputs and last errors, and whether they have started."""

    rods: float = 0.0
    boron: float = 0.0
    last_e: float = 0.0
    last_di: float = 0.0
    started: float = 0.0


def reference_step(params, state, obs, xp=jnp):
    """One 30 s step: ``obs = [n, T_hot, rod, load, T_avg, T_ref, delta_I, boron]``,
    the action is ``[rod demand, boron flow]``."""
    p = params
    rod, T_avg, T_ref, di = obs[..., 2], obs[..., 4], obs[..., 5], obs[..., 6]
    rods = xp.where(state.started > 0.5, state.rods, p.park + 0.0 * rod)
    e_di = -di
    rods = rods + p.rod_Kc * ((e_di + state.last_di) + p.dt / p.rod_Ti * e_di)
    rods = xp.clip(xp.clip(rods, rod - p.reach, rod + p.reach), -1.0, 1.0)
    e = T_ref - T_avg
    boron = state.boron + p.boron_Kc * ((e - state.last_e) + p.dt / p.boron_Ti * e)
    boron = xp.clip(boron, -1.0, 1.0)
    new = ReferenceState(
        rods=rods, boron=boron, last_e=e, last_di=di, started=1.0 + 0.0 * e
    )
    return xp.stack([rods, boron], axis=-1), new


def make_reactor_pid_controller() -> Controller:
    """The expert: the ladder's best rung as a
    :class:`~target_gym.experts.pid.Controller`. :func:`rod_control_step`, the
    published rod program, is a row in the ladder's table beside it."""
    return Controller(ReferenceParams(), ReferenceState(), reference_step)
