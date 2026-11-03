"""The expert (the PID slot) for the four-tank task.

The oracle (the MPC slot) is in ``oracle.py``, so that the expert and
the oracle each have a fingerprint of their own
(``provenance.expert_fingerprint``, ``provenance.oracle_fingerprint``).

The expert is the quadruple-tank rig's multi-loop PI, on the crossed
pairing the non-minimum-phase setting needs, through a static decoupler
built from the measured steady-state gains, SIMC-tuned from step tests (``scripts/tune_process_experts.py``, PHYSICS.md section 6).
"""

import jax.numpy as jnp
import numpy as np
from flax import struct


@struct.dataclass
class DecoupledPIParams:
    """The decoupled pair, in the env's units (m, V, s).

    ``Kp_1``, ``Ti_1`` and ``Kp_2``, ``Ti_2`` are the PI loops on h1 and h2. Their
    outputs ``w`` reach the pumps through the decoupler ``v = v* + D w``,
    ``D = G(0)^-1 diag(G12, G21)``, so at steady state ``w1`` moves h1 alone
    as pump 2 alone would and ``w2`` moves h2 as pump 1 would. ``v*`` is the
    pair of voltages that holds the targets, from the steady-state balances
    and the rig's datasheet constants (orifice areas, pump constants, valve
    splits).
    """

    Kp_1: float
    Ti_1: float
    Kp_2: float
    Ti_2: float
    D11: float
    D12: float
    D21: float
    D22: float
    dt: float
    a1: float
    a2: float
    k1: float
    k2: float
    gamma1: float
    gamma2: float
    g: float
    v_min: float
    v_max: float


@struct.dataclass
class DecoupledPIState:
    integral1: float
    integral2: float


def steady_voltages(t1, t2, prm: DecoupledPIParams, xp=jnp):
    """The pump voltages at which the lower tanks rest at ``t1`` and ``t2``.

    At rest each upper tank drains what its pump sends it, so the lower
    balances are linear in the voltages:
    ``a1 sqrt(2 g h1) = gamma1 k1 v1 + (1 - gamma2) k2 v2`` and
    ``a2 sqrt(2 g h2) = (1 - gamma1) k1 v1 + gamma2 k2 v2``.
    """
    q1 = prm.a1 * xp.sqrt(2.0 * prm.g * t1)
    q2 = prm.a2 * xp.sqrt(2.0 * prm.g * t2)
    m11, m12 = prm.gamma1 * prm.k1, (1.0 - prm.gamma2) * prm.k2
    m21, m22 = (1.0 - prm.gamma1) * prm.k1, prm.gamma2 * prm.k2
    det = m11 * m22 - m12 * m21
    return (m22 * q1 - m12 * q2) / det, (m11 * q2 - m21 * q1) / det


def decoupled_pi_step(prm: DecoupledPIParams, state, obs, xp=jnp):
    """One update on ``obs = [h1, h2, h3, h4, target_h1, target_h2]``.

    Returns ``(action [shape (2,)], new_state)``. Both integrals hold while
    either pump is saturated, so neither winds up a term it cannot spend.
    """
    t1, t2 = obs[4], obs[5]
    e1, e2 = t1 - obs[0], t2 - obs[1]
    integral1 = state.integral1 + e1 * prm.dt
    integral2 = state.integral2 + e2 * prm.dt
    w1 = prm.Kp_1 * (e1 + integral1 / prm.Ti_1)
    w2 = prm.Kp_2 * (e2 + integral2 / prm.Ti_2)
    v1_ss, v2_ss = steady_voltages(t1, t2, prm, xp)
    v = xp.stack(
        [v1_ss + prm.D11 * w1 + prm.D12 * w2, v2_ss + prm.D21 * w1 + prm.D22 * w2]
    )
    u = 2.0 * (v - prm.v_min) / (prm.v_max - prm.v_min) - 1.0
    u_clipped = xp.clip(u, -1.0, 1.0)
    free = xp.all(u == u_clipped)
    return u_clipped, DecoupledPIState(
        integral1=xp.where(free, integral1, state.integral1),
        integral2=xp.where(free, integral2, state.integral2),
    )


def four_tank_decoupled_pi(Kp_1, Ti_1, Kp_2, Ti_2, D, params=None):
    """The expert with these gains and decoupler ``D`` (2 x 2, V per V) on
    ``params``' rig (the scored configuration's by default), as a pure
    :class:`~target_gym.experts.pid.Controller`."""
    from target_gym.experts.pid import Controller
    from target_gym.pc_gym.four_tank.env import FourTankParams

    p = params if params is not None else FourTankParams()
    D = np.asarray(D, float)
    return Controller(
        DecoupledPIParams(
            Kp_1=float(Kp_1),
            Ti_1=float(Ti_1),
            Kp_2=float(Kp_2),
            Ti_2=float(Ti_2),
            D11=float(D[0, 0]),
            D12=float(D[0, 1]),
            D21=float(D[1, 0]),
            D22=float(D[1, 1]),
            **{
                f: float(getattr(p, "delta_t" if f == "dt" else f))
                for f in ("dt", "a1", "a2", "k1", "k2", "gamma1", "gamma2", "g")
            },
            v_min=float(p.v_min),
            v_max=float(p.v_max),
        ),
        DecoupledPIState(integral1=0.0, integral2=0.0),
        decoupled_pi_step,
    )


def make_four_tank_pid_controller():
    """The scored expert as a pure :class:`~target_gym.experts.pid.Controller`;
    ``.host()`` is what ``make_pid()`` returns. Gains and decoupler from
    src/target_gym/data/pid_gains.json, written by
    ``scripts/tune_process_experts.py``."""
    from target_gym.experts.pid import gain

    D = [
        [gain("four_tank", "D11", -0.294), gain("four_tank", "D12", 1.065)],
        [gain("four_tank", "D21", 1.065), gain("four_tank", "D22", -0.235)],
    ]
    return four_tank_decoupled_pi(
        Kp_1=gain("four_tank", "Kp_1", 45.1),
        Ti_1=gain("four_tank", "Ti_1", 146.4),
        Kp_2=gain("four_tank", "Kp_2", 32.4),
        Ti_2=gain("four_tank", "Ti_2", 118.1),
        D=D,
    )
