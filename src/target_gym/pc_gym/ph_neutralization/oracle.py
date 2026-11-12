"""The oracle (the MPC slot) for the pH neutralisation task, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem (version 5,
charter Amendment 3, the job score), run by
``target_gym.oracles.controller.Oracle``:

- model: the env's own ``staged_velocity`` and integrator on both stages, the
  diversion basin and its pump-back, both valves' clips removed and the moves
  bounded instead (T1); the buffer flow, which the observation hides, on its
  noise-free mean path from the value the state carries (the OU update with the
  innovation zeroed: ``plan_params``); the influent's flow and acid strength as
  the state carries them at the solve, held over the plan (certainty equivalence,
  as the CSTR holds its event: a later upset or swing is not seen); the pH by
  ``planner_pH``, the env's value with the titration curve's slope;
- objective: the reagent both valves use, in litres, the reward's only term with
  a slope (T2);
- limit: the discharge inside the permit on every visited state, elastic at the
  excursion's price per hundredth of a pH (T3), as the ceiling's margin
  (``ceiling_margins``). Inside it neither the excursion, the diversion (the
  sewer floor is below the permit) nor the permit's clock charges, so the plan
  pays only reagent; K8's replay charges the rest if a plan leaves;
- resting target (G12): two valves, and only the reagent priced at a steady state,
  where it is their sum, so the split is free. The trim valve rests shut, its
  design point: stage 1 neutralises, the trim takes what it leaves (as the
  expert's stage-2 loop does, with no bias); 1 fully open;
- dropped guards (T4): the diversion's switch at the sewer floor and the
  pump-back's cap, which the permit keeps off, and the buffer's clip, which its
  mean path never meets.

Its horizon is the protocol's on the two-stage plant
(``data/oracle_horizons/ph_neutralization.json``).
"""

import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial

from target_gym import reward as R
from target_gym.integration import integrate_dynamics
from target_gym.oracles.controller import Oracle, TaskModel, raw_of
from target_gym.oracles.shooting import Limit
from target_gym.pc_gym.ph_neutralization.env import (
    BUFFER_OU_THETA,
    acid_scale,
    influent_flow,
    solve_pH,
    staged_velocity,
    steady_base_flow,
    titration_residual,
)

# ---------------------------------------------------------------------------
# pH neutralisation
# ---------------------------------------------------------------------------


@jax.custom_jvp
def planner_pH(Wa, Wb, params):
    """``env.solve_pH``, with the derivative of the curve it solves.

    The env finds pH by bisection, whose derivative is zero: a planner
    differentiating the env's pH sees no slope anywhere. The value here is
    the env's; the derivative is the implicit function theorem's on the
    charge balance r(pH, Wa, Wb, params) = 0: dpH = -dr / r_pH, with dr the
    residual's change at fixed pH along every input's tangent.
    """
    return solve_pH(Wa, Wb, params)


@planner_pH.defjvp
def _planner_pH_jvp(primals, tangents):
    Wa, Wb, params = primals
    pH = planner_pH(Wa, Wb, params)
    _, dr = jax.jvp(lambda a, b, p: titration_residual(pH, a, b, p), primals, tangents)
    return pH, -dr / jax.grad(titration_residual)(pH, Wa, Wb, params)


def ph_step(env, p, x, u, t, task):
    """One step of the planner's model from ``x`` = (stage 1's and stage 2's
    invariants, the basin's litres and invariants, the trim's flow) under the raw
    moves ``u``:
    ``(x_next, q3 + q3_trim, the discharge's pH)``, as ``compute_next_state``
    steps the plant with the noise zeroed. ``t`` steps into the plan, the buffer
    is on its mean path from the solve's ``q2``."""
    q1, Wa1, q2_0 = task
    q3 = p.q3_min + 0.5 * (u[0] + 1.0) * (p.q3_max - p.q3_min)  # T1
    q3t = 0.5 * (u[1] + 1.0) * p.q3_trim_max
    decay = 1.0 - BUFFER_OU_THETA * p.delta_t
    q2 = p.q2_nominal + (q2_0 - p.q2_nominal) * decay ** (t + 1)
    Wa_1, Wb_1, Wa, Wb, basin, Wa_b, Wb_b = x[:7]
    # T4: the interlock reads the discharge at the step's start.
    diverting = planner_pH(Wa, Wb, p) < p.sewer_floor
    qp = jnp.where(
        diverting, 0.0, jnp.minimum(p.pumpback_flow, 1000.0 * basin / p.delta_t)
    )
    flows = (q1, Wa1, q2, q3, q3t, qp, Wa_b, Wb_b)
    tanks, _ = integrate_dynamics(
        positions=x[:4], delta_t=p.delta_t, method=env.integration_method,
        compute_velocity=partial(staged_velocity, flows=flows, params=p),
    )  # fmt: skip
    outflow_L = (q1 + q2 + q3 + qp + q3t) * p.delta_t / 1000.0
    into = jnp.where(diverting, outflow_L, 0.0)
    held = basin + into
    weight = jnp.where(held > 0.0, into / jnp.maximum(held, 1e-9), 0.0)
    Wa_b = Wa_b + weight * (Wa - Wa_b)
    Wb_b = Wb_b + weight * (Wb - Wb_b)
    basin = jnp.maximum(held - qp * p.delta_t / 1000.0, 0.0)
    x1 = jnp.concatenate([tanks, jnp.stack([basin, Wa_b, Wb_b, q3t])])
    return x1, q3 + q3t, planner_pH(tanks[2], tanks[3], p)


def ph_model(env, p) -> TaskModel:
    per_mL_s = float(p.reagent_price * p.delta_t / 1000.0)  # L of reagent a step
    # G10: what half the electrode's resolution costs in one step, in reagent at
    # the operator's set point on the nominal buffer.
    sp, half = float(p.setpoint_pH), 0.5 * float(p.precision_floor)
    dq3 = steady_base_flow(sp + half, p.q2_nominal, p)
    dq3 = dq3 - steady_base_flow(sp, p.q2_nominal, p)

    def step(d, x, u, t):
        x1, reagent, pH = ph_step(env, p, x, u, t, d["task"])
        cost = R.priced_cost(per_mL_s * reagent, 1.0)
        return x1, cost, pH[None] / PH_UNIT, pH[None], ()

    return TaskModel(
        nx=8, nu=2, step=step,
        state_x=lambda s: jnp.stack([
            s.Wa_s1, s.Wb_s1, s.Wa, s.Wb, s.basin, s.Wa_basin, s.Wb_basin, s.q3_trim,
        ]),
        data=lambda s: (influent_flow(s, p), p.Wa1 * acid_scale(s, p), s.q2),
        hold=lambda s: jnp.stack(
            [raw_of(s.q3, p.q3_min, p.q3_max), raw_of(s.q3_trim, 0.0, p.q3_trim_max)]
        ),
        tracked=lambda s: jnp.stack([s.pH]),
        resolution=(float(p.precision_floor),),
        limits=(
            Limit(
                float(p.permit_low) / PH_UNIT, float(p.permit_high) / PH_UNIT,
                float(p.excursion_price * p.delta_t),
            ),
        ),
        tie_budget=float(abs(dq3)) * per_mL_s,
        rest=lambda x: x[7] / p.q3_trim_max,
    )  # fmt: skip


def make_ph_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``ph_model``."""
    return Oracle(env, params, "ph_neutralization", ph_model)


#: The reward terms the ceiling's solver keeps (``pc_gym.foresight``):
#: the reagent; ``excursion``, ``diversion`` and the trip are the margins.
CEILING_TERMS = ("reagent",)

#: The permit margins' unit, pH (a probe's 0.01).
PH_UNIT = 0.01


def ceiling_margins(state, p):
    """The ceiling's margins (``pc_gym.foresight``): the discharge inside the
    permit, where neither ``excursion`` nor ``diversion`` charges (the sewer
    floor is below the permit), on the pH ``planner_pH`` gives with its slope,
    in hundredths of a pH, backed off a tenth of one; the price is a step
    outside the permit."""
    pH = planner_pH(state.Wa, state.Wb, p)
    m = jnp.stack([pH - p.permit_low, p.permit_high - pH]) / PH_UNIT - 0.1
    return m, p.excursion_price * p.delta_t
