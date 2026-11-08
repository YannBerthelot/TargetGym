"""The oracle (the MPC slot) for the pH neutralisation task.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``): changing the oracle re-records
only the oracle rows of this package's tasks. The shared machinery
(GradientMPC, NewtonMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every oracle fingerprint.
"""

import jax

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


def ph_stage(env, state, action, params):
    """``NewtonMPC``'s stage: the env's step and its job cost (reagent,
    excursion, diversion), on the pH ``planner_pH`` gives.

    Only the reagent term has a slope: the excursion and diversion terms are
    steps in pH, so a gradient planner cannot see the permit edge. This oracle
    has not been rebuilt for the job score (charter rule 15) and is not to be
    run until it is; rule 9 reports its room as "not measured" meanwhile.
    """
    from target_gym.pc_gym.ph_neutralization.env import compute_reward_terms

    _, new, _, _, _ = env.step_env(jax.random.PRNGKey(0), state, action, params)
    planned = new.replace(pH=planner_pH(new.Wa, new.Wb, params))
    return new, -R.total(compute_reward_terms(planned, params))


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
