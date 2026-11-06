"""The expert (the PID slot) for the pH neutralisation task.

The oracle (the MPC slot) is in ``oracle.py``, so that the expert and
the oracle each have a fingerprint of their own
(``provenance.expert_fingerprint``, ``provenance.oracle_fingerprint``).

The expert is the pH loop practice runs: the measured pH is linearised
through the titration curve, a PI acts on the result, and a static bias puts
the base valve where the curve says the target needs it. The curve is the
one a lab titration of the streams gives, at the nominal buffer flow, so the
expert uses datasheet values only; the buffer flow itself is unmeasured.
SIMC-tuned from step tests (``scripts/tune_process_experts.py``, PHYSICS.md
section 7).
"""

import math

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class TitrationCurve:
    """What a lab titration of the streams tells the commissioning engineer:
    their compositions (M), the acid flow and the nominal buffer flow (mL/s)
    and the carbonate pKs. Copied from the plant's datasheet values."""

    q1: float
    q2: float
    Wa1: float
    Wb1: float
    Wa2: float
    Wb2: float
    Wa3: float
    Wb3: float
    pK1: float
    pK2: float

    @classmethod
    def of(cls, params) -> "TitrationCurve":
        return cls(
            **{
                f: float(getattr(params, f if f != "q2" else "q2_nominal"))
                for f in cls.__dataclass_fields__
            }
        )


def reagent_demand(pH, curve: TitrationCurve, xp=jnp):
    """The steady base flow q3 (mL/s) that leaves the effluent at ``pH``.

    At steady state the invariants are the flow-weighted means of the
    streams', and the charge balance (``env.titration_residual``) is then
    linear in q3, so the inverse titration curve is closed-form::

        q3 = -(q1 r1 + q2 r2) / r3,   ri = Wai + f(pH) + Wbi c(pH)

    with ``f = 10^(pH-14) - 10^-pH`` and ``c`` the carbonate term. ``r3``
    vanishes at the base stream's own pH, about 11.5, which a mix with the
    acid stream never reaches.
    """
    f = 10.0 ** (pH - 14.0) - 10.0 ** (-pH)
    c = (1.0 + 2.0 * 10.0 ** (pH - curve.pK2)) / (
        1.0 + 10.0 ** (curve.pK1 - pH) + 10.0 ** (pH - curve.pK2)
    )
    r1 = curve.Wa1 + f + curve.Wb1 * c
    r2 = curve.Wa2 + f + curve.Wb2 * c
    r3 = curve.Wa3 + f + curve.Wb3 * c
    return -(curve.q1 * r1 + curve.q2 * r2) / r3


@struct.dataclass
class LinearisedPIParams:
    """The PI on reagent demand: ``Kp`` in mL/s per mL/s, ``Ti`` and ``dt`` in
    s, ``alpha`` how far the reading's filter moves per step, and the base
    valve's range in mL/s."""

    Kp: float
    Ti: float
    alpha: float
    dt: float
    q3_min: float
    q3_max: float
    curve: TitrationCurve


@struct.dataclass
class LinearisedPIState:
    integral: float
    demand: float
    started: float


def linearised_pi_step(params: LinearisedPIParams, state, obs, xp=jnp):
    """One update on ``obs = [pH, q3_pct, target_pH]``.

    The error is in reagent demand, the target's minus the measured pH's, so
    the loop gain is about 1 across the titration curve instead of varying
    45 times. The bias is the target's demand. The integral holds while the
    valve is saturated. The filter runs on the linearised reading, which starts
    it at the first.
    """
    demand_sp = reagent_demand(obs[2], params.curve, xp)
    read = xp.clip(reagent_demand(obs[0], params.curve, xp), 0.0, 2 * params.q3_max)
    demand = xp.where(
        state.started > 0, state.demand + params.alpha * (read - state.demand), read
    )
    e = demand_sp - demand
    integral = state.integral + e * params.dt
    q3 = demand_sp + params.Kp * (e + integral / params.Ti)
    u = 2.0 * (q3 - params.q3_min) / (params.q3_max - params.q3_min) - 1.0
    u_clipped = xp.clip(u, -1.0, 1.0)
    return xp.array([u_clipped]), LinearisedPIState(
        integral=xp.where(u == u_clipped, integral, state.integral),
        demand=demand,
        started=xp.ones_like(demand),
    )


def ph_linearised_pi(Kp: float, Ti: float, params=None, Tf: float = 0.0):
    """The expert with these gains on ``params``' streams (the scored
    configuration's by default), as a pure
    :class:`~target_gym.experts.pid.Controller`. ``Tf`` is the filter on the
    linearised reading, in seconds."""
    from target_gym.experts.pid import Controller
    from target_gym.pc_gym.ph_neutralization.env import PHParams

    p = params if params is not None else PHParams()
    dt = float(p.delta_t)
    return Controller(
        LinearisedPIParams(
            Kp=Kp,
            Ti=Ti,
            alpha=-math.expm1(-dt / Tf) if Tf else 1.0,
            dt=dt,
            q3_min=float(p.q3_min),
            q3_max=float(p.q3_max),
            curve=TitrationCurve.of(p),
        ),
        LinearisedPIState(integral=0.0, demand=0.0, started=0.0),
        linearised_pi_step,
    )


def make_ph_pid_controller():
    """The scored expert as a pure :class:`~target_gym.experts.pid.Controller`;
    ``.host()`` is what ``make_pid()`` returns. Gains from
    src/target_gym/data/pid_gains.json, written by
    ``scripts/tune_process_experts.py``."""
    from target_gym.experts.pid import gain

    return ph_linearised_pi(
        Kp=gain("ph_neutralization", "Kp"),
        Ti=gain("ph_neutralization", "Ti"),
        Tf=gain("ph_neutralization", "Tf"),
    )
