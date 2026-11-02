"""Controllers for the pH neutralisation task: the expert and the oracle.

Each task keeps its oracle in its own package, so changing one re-records
only the tasks of that package. The shared machinery (GradientMPC,
NewtonMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every task's baseline fingerprint.

The expert is the pH loop practice runs: the measured pH is linearised
through the titration curve, a PI acts on the result, and a static bias puts
the base valve where the curve says the target needs it. The curve is the
one a lab titration of the streams gives, at the nominal buffer flow, so the
expert uses datasheet values only; the buffer flow itself is unmeasured.
SIMC-tuned from step tests (``scripts/tune_process_experts.py``, PHYSICS.md
section 7).
"""

import jax
import jax.numpy as jnp
from flax import struct

from target_gym import reward as R
from target_gym.experts.mpc import NewtonMPC
from target_gym.pc_gym.ph_neutralization.env import solve_pH, titration_residual

# ---------------------------------------------------------------------------
# pH neutralisation
# ---------------------------------------------------------------------------


@jax.custom_jvp
def planner_pH(Wa, Wb, params):
    """``env.solve_pH``, with the derivative of the curve it solves.

    The env finds pH by bisection, whose derivative is zero: a planner
    differentiating the env's pH sees no slope anywhere. The value here is
    the env's; the derivative is the implicit function theorem's on the
    charge balance, ``dpH = -(r_Wa dWa + r_Wb dWb) / r_pH``.
    """
    return solve_pH(Wa, Wb, params)


@planner_pH.defjvp
def _planner_pH_jvp(primals, tangents):
    Wa, Wb, params = primals
    dWa, dWb, _ = tangents
    pH = planner_pH(Wa, Wb, params)
    r_pH, r_Wa, r_Wb = jax.grad(titration_residual, argnums=(0, 1, 2))(
        pH, Wa, Wb, params
    )
    return pH, -(r_Wa * dWa + r_Wb * dWb) / r_pH


def ph_stage(env, state, action, params):
    """``NewtonMPC``'s stage: the env's step, and its reward in floor units
    with the pH recomputed by ``planner_pH``, so the cost has its slope."""
    _, new, _, _, _ = env.step_env(jax.random.PRNGKey(0), state, action, params)
    p = params
    pH = planner_pH(new.Wa, new.Wb, p)
    cost = R.tracking_cost(
        new.target_pH - pH, p.e_floor, p.e_tol, p.tracking_exponent
    ) + R.running_cost(new.q3, p.c_hold, p.running_weight)
    return new, cost / p.rho_floor_tracking


def make_ph_mpc(env, params, horizon: int = 20, n_iter: int = 10):
    """The oracle: ``NewtonMPC`` on the plant's own step, ``ph_stage``.

    With delta_t = 5 s, horizon = 20 gives 100 s of lookahead -- slightly more
    than one residence time (V/q_total ~ 88 s), so the controller can see a
    change work through the tank. It reads the true invariants and the buffer
    flow q2, which the observation withholds, and plans q2's mean path, the
    env's own OU update with the noise zeroed (``plan_params``). Planning on
    the nominal q2 instead, as an earlier oracle did, made almost all of its
    hold error (0.0146 pH against 0.0022 on the protocol seeds); what remains
    is the OU innovation itself, which no causal controller can predict.
    """
    return NewtonMPC(env, params, horizon=horizon, n_iter=n_iter, stage_fn=ph_stage)


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
    s, and the base valve's range in mL/s."""

    Kp: float
    Ti: float
    dt: float
    q3_min: float
    q3_max: float
    curve: TitrationCurve


@struct.dataclass
class LinearisedPIState:
    integral: float


def linearised_pi_step(params: LinearisedPIParams, state, obs, xp=jnp):
    """One update on ``obs = [pH, q3_pct, target_pH]``.

    The error is in reagent demand, the target's minus the measured pH's, so
    the loop gain is about 1 across the titration curve instead of varying
    45 times. The bias is the target's demand. The integral holds while the
    valve is saturated.
    """
    demand_sp = reagent_demand(obs[2], params.curve, xp)
    demand = xp.clip(reagent_demand(obs[0], params.curve, xp), 0.0, 2 * params.q3_max)
    e = demand_sp - demand
    integral = state.integral + e * params.dt
    q3 = demand_sp + params.Kp * (e + integral / params.Ti)
    u = 2.0 * (q3 - params.q3_min) / (params.q3_max - params.q3_min) - 1.0
    u_clipped = xp.clip(u, -1.0, 1.0)
    return xp.array([u_clipped]), LinearisedPIState(
        integral=xp.where(u == u_clipped, integral, state.integral)
    )


def ph_linearised_pi(Kp: float, Ti: float, params=None):
    """The expert with these gains on ``params``' streams (the scored
    configuration's by default), as a pure
    :class:`~target_gym.experts.pid.Controller`."""
    from target_gym.experts.pid import Controller
    from target_gym.pc_gym.ph_neutralization.env import PHParams

    p = params if params is not None else PHParams()
    return Controller(
        LinearisedPIParams(
            Kp=Kp,
            Ti=Ti,
            dt=float(p.delta_t),
            q3_min=float(p.q3_min),
            q3_max=float(p.q3_max),
            curve=TitrationCurve.of(p),
        ),
        LinearisedPIState(integral=0.0),
        linearised_pi_step,
    )


def make_ph_pid_controller():
    """The scored expert as a pure :class:`~target_gym.experts.pid.Controller`;
    ``.host()`` is what ``make_pid()`` returns. Gains from
    src/target_gym/data/pid_gains.json, written by
    ``scripts/tune_process_experts.py``."""
    from target_gym.experts.pid import gain

    return ph_linearised_pi(
        Kp=gain("ph_neutralization", "Kp", 7.14),
        Ti=gain("ph_neutralization", "Ti", 48.9),
    )
