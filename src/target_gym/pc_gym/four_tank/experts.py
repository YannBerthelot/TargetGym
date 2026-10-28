"""Controllers for the four-tank task: the expert and the oracle.

Each task keeps its oracle in its own package, so changing one re-records
only the tasks of that package. The shared machinery (GradientMPC,
CasadiMPC, ShrinkingHorizonNLP, SamplingMPC and the v2 objective helpers)
stays in ``target_gym.experts.mpc``, which is in every task's baseline
fingerprint.

The expert is the quadruple-tank rig's multi-loop PI, on the crossed
pairing the non-minimum-phase setting needs, through a static decoupler
built from the measured steady-state gains, SIMC-tuned from step tests (``scripts/tune_process_experts.py``, PHYSICS.md section 6).
"""

import jax.numpy as jnp
import numpy as np
from flax import struct

from target_gym.experts.mpc import (
    ShrinkingHorizonNLP,
    casadi_step_map,
    protocol_burn_in,
)

try:
    import casadi
except ImportError:  # ShrinkingHorizonNLP refuses to build without it
    pass


# ---------------------------------------------------------------------------
# FourTank
# ---------------------------------------------------------------------------

#: Typical level of each tank, m, which scales the NLP's state variables.
_X_SCALE = (0.25, 0.25, 0.25, 0.25)
#: How far inside the trip limits ``h_min`` and ``h_max`` the plan keeps every
#: level, m.
_LEVEL_MARGIN = 0.005


def _safe_sqrt(h):
    """``env._safe_sqrt`` in CasADi: ``sqrt(h)`` above zero, 0 at or below,
    with the inner guard that keeps the untaken branch's derivative finite."""
    positive = h > 0.0
    return casadi.if_else(positive, casadi.sqrt(casadi.if_else(positive, h, 1.0)), 0.0)


def four_tank_step_map(env, params):
    """The four-tank's discrete step, ``F(x, u) -> x_next``, as a CasADi
    Function.

    ``x = [h1, h2, h3, h4]`` and ``u`` the two raw pump commands in
    ``[-1, 1]``. The right-hand side is ``env.compute_velocity`` term for
    term, and the integration is the environment's own
    (``env.integration_method``, ``rk4_1`` as registered), so on the action
    box this is ``step_env`` in float64. The env clips the commands to the
    box first; the NLP bounds them there, so the clip is left out to keep the
    map smooth at the bounds.
    """
    p = params
    g2 = np.sqrt(2 * p.g)

    def velocity(x, u):
        h1, h2, h3, h4 = x[0], x[1], x[2], x[3]
        v1 = p.v_min + 0.5 * (u[0] + 1.0) * (p.v_max - p.v_min)
        v2 = p.v_min + 0.5 * (u[1] + 1.0) * (p.v_max - p.v_min)
        return casadi.vertcat(
            -(p.a1 / p.A1) * g2 * _safe_sqrt(h1)
            + (p.a3 / p.A1) * g2 * _safe_sqrt(h3)
            + (p.gamma1 * p.k1 / p.A1) * v1,
            -(p.a2 / p.A2) * g2 * _safe_sqrt(h2)
            + (p.a4 / p.A2) * g2 * _safe_sqrt(h4)
            + (p.gamma2 * p.k2 / p.A2) * v2,
            -(p.a3 / p.A3) * g2 * _safe_sqrt(h3) + ((1 - p.gamma2) * p.k2 / p.A3) * v2,
            -(p.a4 / p.A4) * g2 * _safe_sqrt(h4) + ((1 - p.gamma1) * p.k1 / p.A4) * v1,
        )

    method = getattr(env, "integration_method", "rk4_1")
    return casadi_step_map(velocity, 4, 2, float(p.delta_t), method)


def make_four_tank_mpc(
    env,
    params,
    window_start: int | None = None,
    resolve_every: int = 25,
    pre_weight: float = 1e-3,
):
    """The four-tank's oracle: a shrinking-horizon NLP over the rest of the
    episode.

    It re-solves every 25 steps on the env's own step
    (``four_tank_step_map``), with both lower tanks' tracking cost in floor
    units and every level held 5 mm inside the trip limits. ``window_start``
    is where the protocol starts scoring, read by default from its burn-in
    (``protocol_burn_in``: 270 steps of the registered 1200), and the steps
    before it weigh ``pre_weight``, so the window weighs 1000 times more.
    That weighting is what matters here: the plant is non-minimum-phase
    (``gamma1 + gamma2`` = 0.4), and the exact optimum of an unweighted
    all-step cost leaves a tail in the scored window larger than the old
    oracle's.

    It replaced a do-mpc controller on a 10-step horizon of 20 s steps, whose
    stage cost charged every step alike. That one
    gave 0.0074 per step on the protocol seeds and this one 3.1e-8, with zero
    trips (``tests/pc_gym/four_tank``).
    """
    if window_start is None:
        window_start = protocol_burn_in("four_tank", params)
    return ShrinkingHorizonNLP(
        env,
        params,
        step_map=four_tank_step_map(env, params),
        state_fields=("h1", "h2", "h3", "h4"),
        target_fields=("target_h1", "target_h2"),
        tracked=(0, 1),
        window_start=window_start,
        x_scale=_X_SCALE,
        x_lb=np.full(4, float(params.h_min) + _LEVEL_MARGIN),
        x_ub=np.full(4, float(params.h_max) - _LEVEL_MARGIN),
        pre_weight=pre_weight,
        resolve_every=resolve_every,
    )


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
