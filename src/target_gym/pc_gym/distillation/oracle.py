"""The oracle (the MPC slot) for the distillation task, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem, by the
smoothing protocol's inventory (docs/proposals/oracle-smoothing-protocol.md),
run by ``target_gym.oracles.controller.Oracle``:

- model: the env's own ``compute_velocity`` and integrator on all 41 stage
  compositions and the 39 tray flows, with the raw moves' clips removed and
  the moves bounded instead (T1); the feed's rate and composition as the
  state's schedule has them in force at the solve, held over the plan
  (certainty equivalence, as the four-tank holds its outlet): a feed step
  not yet come, or its return, is not in the plan;
- objective: the reward's two smooth terms on the state reached, the boilup
  and the light key lost to the bottoms, in kmol of boilup (T2);
- limits: each product inside its zone, ``yD >= yD_spec`` and ``xB <=
  xB_spec``, on every visited state, in the analysers' resolution, elastic at
  the reprocessing of one kmol a step (T3), as the ceiling's margins
  (``ceiling_margins``). Inside them ``off_spec`` charges nothing, so the
  plan pays only boilup and light key; K8's replay charges the rest if a plan
  leaves. And the distillate inside the range ``flows`` clips it to, D_min <=
  V - L <= D_max (T3, not a dropped guard): unclipped, a plan drew D = 1.39
  kmol/min, a negative bottoms that the light-key term counted as a gain.
  Elastic at a step's worst cost per kmol/min, which no breach recovers;
- dropped guard (T4): the compositions' clip to [0, 1], which guards the
  integrator only; the bottoms' clip follows D's range.

No resting target: two moves and two priced outputs at a steady state (G12).
Its horizon is the protocol's on the current column
(``data/oracle_horizons/distillation.json``). No noise drives the column, so
its limits carry no K11 back-off (``data/oracle_widths/distillation.json``).
"""

import jax.numpy as jnp
from jax.tree_util import Partial as partial

from target_gym.integration import integrate_dynamics
from target_gym.oracles.controller import Oracle, TaskModel, raw_of
from target_gym.oracles.shooting import Limit
from target_gym.pc_gym.distillation.env import N_STAGES, compute_velocity


def feed_in_force(s, p):
    """The feed's rate and composition the plant's next step runs on, as
    ``compute_next_state`` reads them off the state's schedule."""
    t_next = s.time + 1
    feed_on = (t_next > s.feed_step_at) & (t_next <= s.feed_until)
    zF_on = (t_next > s.zF_step_at) & (t_next <= s.zF_until)
    F = p.F * (1.0 + s.feed_size * s.feed_sign * feed_on)
    zF = p.zF_nominal + s.zF_size * s.zF_sign * zF_on
    return jnp.stack([F, zF])


def distillation_model(env, p) -> TaskModel:
    # T1: no clip on D, so none on the bottoms either; the guards check both.
    free = p.replace(D_min=-jnp.inf, D_max=jnp.inf)

    def step(d, x, u, t):
        F, zF = d["task"]
        L = p.L_min + 0.5 * (u[0] + 1.0) * (p.L_max - p.L_min)
        V = p.V_min + 0.5 * (u[1] + 1.0) * (p.V_max - p.V_min)
        x1, _ = integrate_dynamics(
            positions=x, delta_t=p.delta_t, method=env.integration_method,
            compute_velocity=partial(
                compute_velocity, action=(L, V), zF=zF, params=free, F=F
            ),
        )  # fmt: skip
        D = V - L
        B = F - D
        xB, yD = x1[0], x1[N_STAGES - 1]
        cost = (
            p.boilup_price * V + p.recovery_price * B * xB
        ) * p.delta_t  # fmt: skip
        y = jnp.stack([yD / p.e_floor_top, xB / p.e_floor_bottom, D])
        comps = x1[:N_STAGES]
        guards = jnp.concatenate([comps, 1.0 - comps])
        return x1, cost, y, jnp.stack([yD, xB]), guards

    price = float(p.reprocess_price * p.delta_t)
    # A step's worst cost: full boilup, and the whole feed at its highest rate
    # both off spec and lost as light key. No breach of D's range gains more.
    F_max = p.F * (1.0 + p.feed_step)
    worst = (
        p.boilup_price * p.V_max + (p.reprocess_price + p.recovery_price) * F_max
    ) * p.delta_t
    return TaskModel(
        nx=2 * N_STAGES - 2, nu=2, step=step,
        state_x=lambda s: jnp.concatenate([s.x, s.liquid]),
        data=lambda s: feed_in_force(s, p),
        hold=lambda s: jnp.stack(
            [raw_of(s.L, p.L_min, p.L_max), raw_of(s.V, p.V_min, p.V_max)]
        ),
        tracked=lambda s: jnp.stack([s.x[-1], s.x[0]]),
        resolution=(float(p.e_floor_top), float(p.e_floor_bottom)),
        limits=(
            Limit(float(p.yD_spec / p.e_floor_top), jnp.inf, price),
            Limit(-jnp.inf, float(p.xB_spec / p.e_floor_bottom), price),
            Limit(float(p.D_min), float(p.D_max), float(worst)),
        ),
        # G10: half the bottoms analyser's resolution of light key lost for one
        # step at the nominal bottoms, F - D = 0.5 kmol/min, in kmol of boilup.
        tie_budget=float(
            p.recovery_price * 0.5 * p.F * 0.5 * p.e_floor_bottom * p.delta_t
        ),
    )  # fmt: skip


def make_distillation_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``distillation_model``."""
    return Oracle(env, params, "distillation", distillation_model)


#: The reward terms the ceiling's solver keeps (``pc_gym.foresight``):
#: the boilup and the light key lost; ``off_spec`` is the margins.
CEILING_TERMS = ("boilup", "light_lost")


def ceiling_margins(state, p):
    """The ceiling's margins (``pc_gym.foresight``): each product inside its
    zone, where ``off_spec`` charges nothing (``yD >= yD_spec``, ``xB <=
    xB_spec``), in the analyser's resolution (``e_floor_top``,
    ``e_floor_bottom``), backed off a hundredth of it; the price is the
    reprocessing of one kmol."""
    top = (state.x[-1] - p.yD_spec) / p.e_floor_top
    bottom = (p.xB_spec - state.x[0]) / p.e_floor_bottom
    return jnp.stack([top, bottom]) - 0.01, p.reprocess_price * p.delta_t
