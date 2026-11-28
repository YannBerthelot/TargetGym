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
  a slope (T2), on the steps the evaluation scores (all but the first), and the
  evaluation's charge for the tanks' capacity (``env.stored_value``): its change
  over each scored step at the reagent's price, which sums to what the window
  lost, charged, or gained, credited (owner, 2026-11-28, v1 task pages, item 2).
  The charge is the evaluation's, not the reward's, so K8's replay adds it to the
  reward's cost (``replay_cost``);
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

The task's ceiling is not a solve (owner, 2026-11-23): it is ``perfect_hold``, the
discharge held exactly on the permit's low edge against the episode's own influent
and buffer, in closed form.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
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
    stored_capacity,
    stored_value,
    titration_residual,
)
from target_gym.pc_gym.ph_neutralization.scenarios import (
    ph_neutralization_evaluation,
)
from target_gym.utils import ou_decay

# ---------------------------------------------------------------------------
# pH neutralisation
# ---------------------------------------------------------------------------


#: Newton steps that polish the bisection's root (quadratic from 1e-5 pH).
NEWTON_POLISH = 3


@jax.custom_jvp
def planner_pH(Wa, Wb, params):
    """``env.solve_pH``, with the derivative of the curve it solves.

    The env finds pH by bisection, whose derivative is zero: a planner
    differentiating the env's pH sees no slope anywhere. The value here is
    the env's; the derivative is the implicit function theorem's on the
    charge balance r(pH, Wa, Wb, params) = 0: dpH = -dr / r_pH, with dr the
    residual's change at fixed pH along every input's tangent.

    The value is the env's bisection polished by Newton steps on the same
    residual: 20 halvings of [0, 14] leave a staircase of 1.3e-5 pH, and an
    exact slope on a staircase stalls the solver at that floor. The polished
    root is within the bisection's half-width of the env's.
    """
    pH = solve_pH(Wa, Wb, params)
    slope = jax.grad(titration_residual)
    for _ in range(NEWTON_POLISH):
        pH = pH - titration_residual(pH, Wa, Wb, params) / slope(pH, Wa, Wb, params)
    return pH


@planner_pH.defjvp
def _planner_pH_jvp(primals, tangents):
    Wa, Wb, params = primals
    pH = planner_pH(Wa, Wb, params)
    _, dr = jax.jvp(lambda a, b, p: titration_residual(pH, a, b, p), primals, tangents)
    return pH, -dr / jax.grad(titration_residual)(pH, Wa, Wb, params)


#: The plan states the invariants in mM: the solver scales a state by its size
#: floored at 1, and in M they sit near 1e-3 with the pH steep in them.
MM = 1000.0
#: Where the invariants sit in ``x``: both stages', then the basin's.
INVARIANTS = (0, 1, 2, 3, 5, 6)


#: The first step the evaluation scores (``scenarios.ph_neutralization_evaluation``).
FIRST_SCORED = ph_neutralization_evaluation()["unscored_steps"]


class _Tanks:
    """The two stages' invariants in M, as ``env.stored_value`` reads a state."""

    def __init__(self, x):
        self.Wa_s1, self.Wb_s1, self.Wa, self.Wb = (x[i] / MM for i in range(4))


def ph_step(env, p, x, u, t, task):
    """One step of the planner's model from ``x`` = (stage 1's and stage 2's
    invariants, the basin's litres and invariants, the trim's flow; invariants in
    mM) under the raw moves ``u``:
    ``(x_next, q3 + q3_trim, the discharge's pH)``, as ``compute_next_state``
    steps the plant with the noise zeroed. ``t`` steps into the plan, the buffer
    is on its mean path from the solve's ``q2``."""
    q1, Wa1, q2_0 = task[:3]
    q3 = p.q3_min + 0.5 * (u[0] + 1.0) * (p.q3_max - p.q3_min)  # T1
    q3t = 0.5 * (u[1] + 1.0) * p.q3_trim_max
    decay = ou_decay(BUFFER_OU_THETA, float(p.delta_t), xp=math)
    q2 = p.q2_nominal + (q2_0 - p.q2_nominal) * decay ** (t + 1)
    x = x.at[INVARIANTS,].divide(MM)
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
    return x1.at[INVARIANTS,].multiply(MM), q3 + q3t, planner_pH(tanks[2], tanks[3], p)


def ph_model(env, p) -> TaskModel:
    per_mL_s = float(p.reagent_price * p.delta_t / 1000.0)  # L of reagent a step
    # G10: what half the electrode's resolution costs in one step, in reagent at
    # the operator's set point on the nominal buffer.
    sp, half = float(p.setpoint_pH), 0.5 * float(p.precision_floor)
    dq3 = steady_base_flow(sp + half, p.q2_nominal, p)
    dq3 = dq3 - steady_base_flow(sp, p.q2_nominal, p)

    def step(d, x, u, t):
        x1, reagent, pH = ph_step(env, p, x, u, t, d["task"])
        scored = d["task"][3] + t >= FIRST_SCORED
        spent = stored_value(_Tanks(x), p) - stored_value(_Tanks(x1), p)
        cost = jnp.where(scored, R.priced_cost(per_mL_s * reagent, 1.0) + spent, 0.0)
        # Tracked: what the plant's probe reports, its own bisection (K8 compares
        # it with the replay). The limit keeps the polished root, whose slope the
        # solver needs; the two differ by at most the bisection's half-width,
        # 7e-6 pH, under the margin's 1e-3 back-off.
        read = solve_pH(x1[2] / MM, x1[3] / MM, p)
        return x1, cost, pH[None] / PH_UNIT, read[None], ()

    return TaskModel(
        nx=8, nu=2, step=step,
        state_x=lambda s: jnp.stack([
            s.Wa_s1, s.Wb_s1, s.Wa, s.Wb, s.basin, s.Wa_basin, s.Wb_basin, s.q3_trim,
        ]).at[INVARIANTS,].multiply(MM),
        data=lambda s: (influent_flow(s, p), p.Wa1 * acid_scale(s, p), s.q2, s.time),
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
        replay_cost=lambda s, s1, r: jnp.where(
            s.time >= FIRST_SCORED,
            -r + stored_value(s, p) - stored_value(s1, p), 0.0,
        ),
    )  # fmt: skip


def make_ph_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``ph_model``."""
    return Oracle(env, params, "ph_neutralization", ph_model)


#: What the observation-only MPC's estimator tracks: both stages' invariants, the
#: buffer flow, the probes' true values and their history (the readings are late
#: and lagged), and the influent's acid strength.
OBSERVED_FIELDS = (
    "Wa", "Wb", "Wa_s1", "Wb_s1", "q2", "pH", "pH_s1", "sensor_history",
    "upset_levels[0]",
)  # fmt: skip
#: The discharge's and stage 1's probes (``env.get_obs``).
OBSERVED_CHANNELS = (0, 4)
#: A step beyond any episode: the belief's schedules never move.
_NEVER = 10**6
#: A walk a step on each invariant, about 5 % of their prior spread (Wa 1.9e-4,
#: Wb 3.4e-4): an allowance for the Gaussian filter's own error on fields the plant
#: leaves noise-free. Without it the filter trusted its model over the probes in
#: transients (a pH error of 0.15 at a stated 0.004); with it and the cubature
#: update the error stays within its stated spread (the shared estimator's
#: ``test_on_ph_a_valve_step_keeps_the_error_within_the_stated_spread``, 2026-11-28).
INVARIANT_ALLOWANCE = 1e-5


def observed(params) -> dict:
    """The estimator's options for pH (``estimation.Estimator``), at the
    scenario's ``params``: ``known`` writes the metered influent (obs 6) into the
    belief and holds every schedule the reset drew still, so the acid strength is
    the walking ``upset_levels[0]``; ``random_walk`` gives that walk the variance
    the scenario's upsets add over an episode, two of them, each uniform within
    ``acid_upset_size``, spread over its steps, and the four invariants the
    allowance for the filter's own error (``INVARIANT_ALLOWANCE``)."""
    p = params

    def known(state, obs):
        flow = obs[6] / p.q1
        return state.replace(
            flow_levels=jnp.full_like(state.flow_levels, flow),
            flow_times=jnp.full_like(state.flow_times, _NEVER),
            upset_times=jnp.full_like(state.upset_times, _NEVER),
        )

    steps = float(p.max_steps_in_episode)
    walk = math.sqrt(2.0 * float(p.acid_upset_size) ** 2 / 3.0 / steps)
    allowance = dict.fromkeys(("Wa", "Wb", "Wa_s1", "Wb_s1"), INVARIANT_ALLOWANCE)
    return {"known": known, "random_walk": {"upset_levels[0]": walk, **allowance}}


def observation_only_mpc(env, params, plan):
    """The MPC on the true model that reads the observation only (v1 item 5), the
    line ``scripts/record_v1.py`` records: the registry's planner (``ph_model``)
    fed the shared estimator's belief (``estimation.ObservationOnly``), read
    through the two probes.
    ``params`` is the scenario's configuration, noise included, whose noise the
    estimator reads; ``plan`` the planner's (its noise zeroed, ``plan_params``),
    on which it plans and the estimator predicts. The update is the cubature
    one: the pH solve is a bisection, whose JAX derivative is zero, and secants
    along each axis leave the titration curve in transients. A host policy:
    ``policy.reset()``, then ``policy(obs)``."""
    from target_gym.estimation import Estimator, ObservationOnly

    estimator = Estimator(
        env, params, OBSERVED_FIELDS, channels=OBSERVED_CHANNELS, model=plan,
        linearize="cubature", **observed(params),
    )  # fmt: skip
    return ObservationOnly(make_ph_mpc(env, plan), estimator)


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


# ---------------------------------------------------------------------------
# The ceiling: the perfect hold on the permit's low edge
# ---------------------------------------------------------------------------


def hold_flow(pH, q1, q2, acid, params):
    """The base flow (mL/s) whose steady mix with ``q1`` of influent at ``acid`` times
    its nominal strength and ``q2`` of buffer sits at ``pH``: the env's closed-form
    inverse titration curve (``steady_base_flow``) at that influent flow."""
    return steady_base_flow(pH, q2, params.replace(q1=q1), Wa1=params.Wa1 * acid)


@functools.lru_cache(maxsize=None)
def _hold_run(env):
    """The jitted episode that reads what enters the plant at each step."""

    def run(params, state, keys):
        centre = 0.5 * (params.permit_low + params.permit_high)

        def step(state, key):
            # The influent as this step reads it. The buffer flow is drawn inside the
            # step, from its last value and the key alone, so no dose moves it: the
            # plant is held near the permit's centre meanwhile, where nothing trips.
            q1, acid = influent_flow(state, params), acid_scale(state, params)
            q3 = hold_flow(centre, q1, state.q2, acid, params)
            raw = 2.0 * (q3 - params.q3_min) / (params.q3_max - params.q3_min) - 1.0
            action = jnp.stack([jnp.clip(raw, -1.0, 1.0), -1.0])
            _, after, _, _, _ = env.step_env(key, state, action, params)
            return after, hold_flow(params.permit_low, q1, after.q2, acid, params)

        return jax.lax.scan(step, state, keys)[1]

    return jax.jit(run)


def start_inventory(state, params):
    """The base the two tanks hold above the permit's low edge, in litres of the
    base stream: what a policy that never leaves the permit can spend once.

    At a fixed pH the charge balance is affine in the invariants, and invariants mix
    by volume, so a tank whose content has residual ``g`` at the edge holds
    ``V g / g3`` of base beyond what its water needs there, ``g3`` the base stream's
    residual. Negative where a tank starts below the edge."""
    return float(stored_capacity(state, params, params.permit_low))


def perfect_hold(env, params, reset_key, step_keys) -> dict:
    """The task's ceiling (owner, 2026-11-23 and 2026-11-24): the discharge held
    exactly on the permit's low edge, the cheap one, through one episode, less the
    base the tanks hold above that edge at its start.

    The episode is the one its keys name: ``reset_key`` and ``step_keys``, one per
    step, which the caller draws as the scoring runner does (``reset_keys[0]`` and
    ``step_keys[:, 0]`` of ``benchmark.episode_keys(PRNGKey(seed), 1, T)``). This
    module draws nothing itself, so it imports no scoring code.

    At every step the base flow is the one whose mix with what enters stage 1 during
    that step (the influent's flow and acid strength, the buffer flow the step
    draws) sits on the edge. Mixing is linear in the invariants and so is the charge
    balance at a fixed pH, so tanks that start on the edge stay on it under these
    flows (tested on the plant). Nothing is planned.

    An episode starts on the reference's set point, above the edge, so the tanks
    hold base the edge does not need (``start_inventory``). The reagent a policy
    adds, plus what the tanks hold at the start, less what they hold at the end, is
    what the outflow carries; outflow inside the permit carries at least its water's
    need on the edge, and tanks inside the permit hold none below it. So no policy
    that never leaves the permit costs less than the hold less that inventory (owner,
    2026-11-24: counted in the ceiling).

    Returns ``flow``, the hold's base flow per step (mL/s); ``hold``, its mean cost
    per step (the ``reagent`` term; the hold is charged no other); ``inventory``,
    the start's base above the edge in litres; ``cost``, the ceiling: ``hold`` less
    the inventory's price spread over the episode; and ``feasible``, whether stage
    1's valve passes every flow.
    """
    _, state = env.reset_env(reset_key, params)
    flow = np.asarray(_hold_run(env)(params, state, step_keys), float)
    price = float(params.reagent_price)
    hold = float(flow.mean()) * price * float(params.delta_t) / 1000.0
    inventory = start_inventory(state, params)
    return {
        "flow": flow,
        "hold": hold,
        "inventory": inventory,
        "cost": hold - price * inventory / len(flow),
        "feasible": bool(
            np.all((flow >= float(params.q3_min)) & (flow <= float(params.q3_max)))
        ),
    }
