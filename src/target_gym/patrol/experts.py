"""Reference controller (the MPC slot) for the patrol tasks.

Each task keeps its oracle in its own package, so changing one re-records
only the tasks of that package. The shared machinery (GradientMPC,
CasadiMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every task's baseline fingerprint.

The oracle is a feedback law with a short planner on top
(:class:`PatrolTwinOracle`). Both read the true :class:`PatrolState`, as
every MPC slot does (``runners.baseline_policy``), and neither reads the step
key: the planner's model has the turbulence zeroed, so it predicts the decay
of the current gust and no innovation (certainty equivalence).

The same controller fills ``patrol_bearing_only``'s slot, labelled a
full-state bound (:func:`make_patrol_bearing_only_mpc`).
"""

import jax
import jax.numpy as jnp
import numpy as np

from target_gym.experts.mpc import GradientMPC, _is_v1
from target_gym.experts.pid import Plane3DPIDState, plane3d_heading_pid_step
from target_gym.patrol.env import (
    _relative_velocity_lead_frame,
    get_obs_heading,
    slot_error_vector,
    wrap_angle,
)

# ---------------------------------------------------------------------------
# The twin-autopilot law
# ---------------------------------------------------------------------------

#: Correction gains of the twin-autopilot law. Tuned in the oracle audit
#: (2026-10) by cross-entropy search on training seeds 100-115 (never the
#: protocol seeds 0-2), on the protocol window's mean cost of the law alone
#: under the version-2 reward with the earlier 18.6 m slot floor. Held-out
#: seeds 200-263 under that floor: mean 0.095, p90 0.197, worst 0.385, no
#: trip. Under the 3 m slot floor (patrol-v3) the same gains score a held-out
#: mean of 2.175 (p90 5.23, worst 11.4, no trip). Re-searched under that
#: floor (the slot weight 38x higher), the training mean fell 7% (1.916 to
#: 1.781) but the held-out seeds got worse (mean 2.312, worst 25.1), and so
#: did the protocol seeds (2.82 to 3.54), so these gains stand under either
#: floor. The planner on top re-weighs slot against heading by itself, since
#: it descends the reward.
TWIN_GAINS = dict(
    k_y=2.3974102432667543e-4,  # rad of heading offset per m of lateral error
    k_yd=5.869187831606593e-4,  # rad per (m/s) of lateral relative velocity
    k_y_max=0.08,  # clip on the total heading offset (rad)
    k_ff=1.0,  # turn-geometry feedforward, -slot_back * omega_lead / V
    k_b=2.137092119622214e-2,  # power per m of along-track error (behind > 0)
    k_bd=0.5872022560432463,  # power per (m/s) of forward relative velocity
    k_bi=3.0137139445569706e-6,  # power per (m s) of integrated along-track error
    k_z=7.392630787713048e-4,  # stick per m of vertical error
    k_zd=1.2955115290770135e-2,  # stick per (m/s) of vertical relative velocity
)


def _zero_pid_state():
    z = jnp.zeros(())
    return Plane3DPIDState(
        alt_integral=z,
        alt_prev=z,
        track_integral=z,
        track_prev=z,
        power_integral=z,
        power_prev=z,
        lead_psi_prev=z,
    )


def twin_law_init():
    """The law's memory at the start of an episode: the follower's copy of the
    lead autopilot starts from zero, as the lead's own does at reset; the
    along-track integral, the previous lead heading and the started flag are
    zero."""
    z = jnp.zeros(())
    return dict(pid=_zero_pid_state(), ib=z, psi_prev=z, started=z)


def twin_law_step(lead_pid_params, gains, carry, state):
    """One step of the twin-autopilot law. Pure, jittable, reads only the
    current state and the law's own memory.

    The follower flies the lead's own heading autopilot (same gains, its own
    memory) on its own state, aimed at

      heading   the lead's commanded heading this step (its target heading
                advanced by this leg's turn rate, as ``step_lead`` does),
                plus clip(k_y e_right + k_yd rv_right
                          - k_ff slot_back omega_lead / V, +-k_y_max);
      altitude  the lead's target altitude plus the slot's height offset;

    and adds power k_b e_back - k_bd rv_fwd + k_bi sum(e_back) and stick
    -(k_z e_up + k_zd rv_up). omega_lead is the lead's heading change over
    the last step (0 on the first). With the follower in the slot and no
    correction this is the lead's action exactly, so the shared gust moves
    both aircraft alike and drops out of the relative position the reward
    scores. Returns ``(action in [-1, 1]^3, carry)``.
    """
    g = gains
    eb, er, eu = slot_error_vector(state)
    rvf, rvr, rvu = _relative_velocity_lead_frame(state)
    lead = state.lead
    speed = jnp.sqrt(lead.x_dot**2 + lead.y_dot**2) + 1e-6
    omega = carry["started"] * wrap_angle(lead.psi - carry["psi_prev"])
    commanded = wrap_angle(lead.target_heading + state.lead_turn_rate)
    dpsi = jnp.clip(
        g["k_y"] * er + g["k_yd"] * rvr - g["k_ff"] * state.slot_back * omega / speed,
        -g["k_y_max"],
        g["k_y_max"],
    )
    cmd = state.follower.replace(
        target_heading=wrap_angle(commanded + dpsi),
        target_altitude=lead.target_altitude + state.slot_up,
    )
    a, pid = plane3d_heading_pid_step(
        lead_pid_params, carry["pid"], get_obs_heading(cmd)
    )
    ib = carry["ib"] + eb
    dp = g["k_b"] * eb - g["k_bd"] * rvf + g["k_bi"] * ib
    ds = -(g["k_z"] * eu + g["k_zd"] * rvu)
    a = jnp.clip(a + jnp.stack([dp, ds, 0.0]), -1.0, 1.0)
    # The heading PID rebuilds its state with lead_psi_prev at the dataclass
    # default, a weakly typed 0.0, where twin_law_init holds a float32 array.
    # Carrying the field through (the heading PID never reads it) keeps the
    # carry's types fixed, so the oracle's jitted step compiles once, not twice.
    pid = pid.replace(lead_psi_prev=carry["pid"].lead_psi_prev)
    return a, dict(pid=pid, ib=ib, psi_prev=lead.psi, started=jnp.ones(()))


# ---------------------------------------------------------------------------
# The oracle: the law, with a receding-horizon residual planner on top
# ---------------------------------------------------------------------------


class PatrolTwinOracle:
    """The twin-autopilot law plus a receding-horizon residual planner.

    At every control step, from the true state and the law's memory, the
    planner optimises a residual ``delta[0:H]`` (raw action units, each entry
    within +-``max_residual``) that is added to the law *inside* the rollout,
    ``u_t = clip(law(x_t) + delta_t, -1, 1)``, on the summed step cost
    (``-reward`` of the environment's own ``step_env``, so a trip on a
    proposal is charged its trip cost). Adam, ``n_iter`` iterations at step
    ``lr``; non-finite gradients are zeroed and the best iterate is kept. It
    applies ``law(x) + delta_0`` and shifts the plan one step as the next
    warm start. The model is the plant with ``turbulence_sigma`` zeroed: the
    current gust decays at its mean rate and no innovation is predicted.

    ``horizon=0`` or ``n_iter=0`` gives the law alone.

    Measured in the oracle audit (2026-10) on protocol seeds 0-2, the way
    ``eval.evaluate_controller`` runs it. Under the earlier 18.6 m slot
    floor: gain 0.0316 (seeds 0.0339 / 0.0437 / 0.0172), no trip, against
    3.0176 for the 20-step GradientMPC it replaced and 0.1094 for the law
    alone (0.0911 / 0.1548 / 0.0822). Under the 3 m slot floor of version 3:
    gain 0.0371 (seeds 0.0407 / 0.0459 / 0.0248), no trip
    (scripts/evaluate_baselines.py), and a hold of 0.124 / 0.102 m mean slot
    error and 1.63e-3 / 1.75e-3 rad of heading (scripts/measure_hold.py,
    seeds 0-1). About 90 s per 200-step episode on a loaded machine, about
    40 s of it the one compile and 0.2 s a step after it (the law alone,
    2 s). What is left is mostly
    heading: in a turn a slot behind the lead moves sideways at
    omega x slot_back, so holding it exactly needs a heading offset of about
    omega slot_back / V, 4.5e-3 rad at the largest turn rate and slot
    (0.003 rad/s, 300 m, 200 m/s).

    It reads ``state``, never ``obs``, so it runs unchanged on
    ``patrol_bearing_only``, where it is a full-state bound.
    """

    action_dim = 3

    def __init__(
        self,
        env,
        params,
        horizon: int = 30,
        n_iter: int = 40,
        lr: float = 0.02,
        max_residual: float = 0.3,
        gains: dict | None = None,
    ):
        self.env = env
        self.params = params
        self.horizon = int(horizon)
        self.n_iter = int(n_iter)
        self.lr = float(lr)
        self.max_residual = float(max_residual)
        self.gains = dict(TWIN_GAINS, **(gains or {}))
        self.plans = self.horizon > 0 and self.n_iter > 0
        # The planner's model: the turbulence zeroed here as well, so the
        # oracle predicts the mean gust even when it is built on raw params.
        # plan_params (which baseline_policy uses) does the same, so this is a
        # no-op on the protocol path.
        self._model = params.replace(turbulence_sigma=0.0)
        # With the turbulence zeroed the model's step key only seeds the
        # restart draw after a trip in a proposal, which is charged the trip
        # cost whatever the draw.
        self._key = jax.random.PRNGKey(0)
        lead_pid_params = env._lead_pid_params
        gains_ = self.gains

        def law(carry, state):
            return twin_law_step(lead_pid_params, gains_, carry, state)

        self._law = law
        self._jit_step = jax.jit(self._step_fn)
        self.reset()

    # The planner

    def _plan_cost(self, delta, state, carry):
        env, model, key, law = self.env, self._model, self._key, self._law

        def body(c, d):
            s, cc = c
            a, cc = law(cc, s)
            a = jnp.clip(a + d, -1.0, 1.0)
            _, s2, r, _, _ = env.step_env(key, s, a, model)
            return (s2, cc), -r

        _, costs = jax.lax.scan(body, (state, carry), delta)
        return costs.sum()

    def _solve(self, state, carry, delta0):
        vg = jax.value_and_grad(self._plan_cost)
        lr, dmax = self.lr, self.max_residual

        def it(c, _):
            d, m, v, k, best_d, best_f = c
            f, g = vg(d, state, carry)
            g = jnp.where(jnp.isfinite(g), g, 0.0)
            better = jnp.isfinite(f) & (f < best_f)
            best_d = jnp.where(better, d, best_d)
            best_f = jnp.where(better, f, best_f)
            k = k + 1
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            mh = m / (1 - 0.9**k)
            vh = v / (1 - 0.999**k)
            d = jnp.clip(d - lr * mh / (jnp.sqrt(vh) + 1e-8), -dmax, dmax)
            return (d, m, v, k, best_d, best_f), None

        z = jnp.zeros_like(delta0)
        c0 = (delta0, z, z, 0.0, delta0, jnp.inf)
        (d, _, _, _, best_d, best_f), _ = jax.lax.scan(it, c0, None, length=self.n_iter)
        # The last iterate is never scored inside the loop.
        f_last = self._plan_cost(d, state, carry)
        return jnp.where(f_last < best_f, d, best_d)

    def _step_fn(self, carry, plan, state):
        if self.plans:
            warm = jnp.concatenate([plan[1:], plan[-1:]], axis=0)
            delta = self._solve(state, carry, warm)
            first = delta[0]
        else:
            delta = plan
            first = jnp.zeros((self.action_dim,))
        a, carry = self._law(carry, state)
        a = jnp.clip(a + first, -1.0, 1.0)
        return a, carry, delta

    # The controller interface

    def step(self, _obs, state):
        """Return the next action. ``_obs`` is ignored (kept for API symmetry):
        the oracle reads the true state."""
        a, self._carry, self._plan = self._jit_step(self._carry, self._plan, state)
        return np.asarray(a)

    def reset(self):
        """Clear the law's memory and the warm start (a zero residual)."""
        self._carry = twin_law_init()
        self._plan = jnp.zeros((max(self.horizon, 1), self.action_dim), jnp.float32)

    def solver_report(self) -> dict:
        """No external solver, so no convergence to report (as GradientMPC)."""
        return {}


# ---------------------------------------------------------------------------
# Version 1 (legacy): the GradientMPC on the surrogate objective
# ---------------------------------------------------------------------------

# The follower's stall barrier, with the 2D aircraft planner's values
# (target_gym.plane.experts). Used only by the version-1 planner below.
_PATROL_STALL_MARGIN = 1.3  # multiples of stall speed at which the barrier starts
_PATROL_BARRIER_WEIGHT = 10.0

#: Fraction of ``slot_tolerance`` at which the patrol surrogate puts its
#: curvature. See :func:`_patrol_objective`.
_PATROL_ERROR_SCALE = 0.25


def _patrol_objective(state, params):
    """Version-1 surrogate: slot tracking ``1 / (1 + (e / scale)**2)`` times
    heading alignment, minus a stall barrier.

    ``scale`` is a quarter of ``slot_tolerance``. Measured over two seeds at
    300 iterations: a quarter of the tolerance scores 175.7, the raw log
    reward 144.6, the full tolerance 138.2 and the 3 m precision floor 19.4
    (at 50 iterations). The barrier is the aircraft objective's: the slot can
    be several hundred metres away at reset, so a planner is free to buy
    position with airspeed, and patrol terminates on the altitude envelope
    rather than on stall.
    """
    from target_gym.patrol.env import heading_alignment, slot_error

    err = slot_error(state) / (_PATROL_ERROR_SCALE * params.slot_tolerance)
    track = 1.0 / (1.0 + err**2)
    align = heading_alignment(state, params)

    f = state.follower
    speed = jnp.sqrt(f.x_dot**2 + f.y_dot**2 + f.z_dot**2)
    v_stall = jnp.sqrt(
        2.0 * f.m * params.gravity / (f.rho * params.wings_surface * params.CL_max)
    )
    margin = speed / (_PATROL_STALL_MARGIN * v_stall)
    penalty = _PATROL_BARRIER_WEIGHT * jnp.maximum(1.0 - margin, 0.0) ** 2
    return track * align - penalty


def _make_patrol_mpc_v1(env, params, horizon, n_iter, lr, n_tail):
    """The version-1 GradientMPC, unchanged: 30 s horizon, 300 iterations,
    a 60-step open-loop tail. Measured over two seeds against a PID scoring
    about 105: 106.1 at 100 iterations, 130.3 at 150, 153.8 at 300, 171.5 at
    600 with no tail, 175.7 at 300 with the tail."""
    return GradientMPC(
        env,
        params,
        action_dim=3,
        action_lb=-1.0,
        action_ub=1.0,
        horizon=horizon,
        n_iter=n_iter,
        lr=lr,
        n_tail=n_tail,
        objective_fn=_patrol_objective,
        done_value=-(_PATROL_BARRIER_WEIGHT + 1.0),
    )


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def make_patrol_mpc(
    env,
    params,
    horizon: int | None = None,
    n_iter: int | None = None,
    lr: float | None = None,
    n_tail: int | None = None,
    max_residual: float = 0.3,
    gains: dict | None = None,
):
    """The patrol oracle.

    Under the version-2 reward: :class:`PatrolTwinOracle`, the twin-autopilot
    law with a residual planner on top (defaults: 30-step horizon, 40 Adam
    iterations at 0.02, residual within +-0.3). ``horizon=0`` or
    ``n_iter=0`` gives the law alone. ``n_tail`` is accepted for
    compatibility and ignored: the residual planner has no open-loop tail.

    It replaced a 20-step GradientMPC started from and guided by the PID's
    rollout (300 iterations of fixed-step descent at 0.05) in the oracle
    audit (2026-10). Under the earlier 18.6 m slot floor that planner held
    29-32 m RMS slot error on the protocol seeds (gain 3.0176), where the law
    alone holds 4-6 m and the law with the planner 0.4-1.1 m (gain 0.0316):
    the shared gust is common-mode, so what it lost was solver weakness, not
    disturbance.

    Under the version-1 reward the version-1 GradientMPC is returned
    unchanged (defaults: horizon 30, 300 iterations, lr 0.05, tail 60).
    """
    if _is_v1(params):
        return _make_patrol_mpc_v1(
            env,
            params,
            horizon=30 if horizon is None else horizon,
            n_iter=300 if n_iter is None else n_iter,
            lr=0.05 if lr is None else lr,
            n_tail=60 if n_tail is None else n_tail,
        )
    return PatrolTwinOracle(
        env,
        params,
        horizon=30 if horizon is None else horizon,
        n_iter=40 if n_iter is None else n_iter,
        lr=0.02 if lr is None else lr,
        max_residual=max_residual,
        gains=gains,
    )


def make_patrol_bearing_only_mpc(env, params, **kwargs):
    """``patrol_bearing_only``'s slot: the patrol oracle, reading the TRUE
    state. A full-state bound, not a bearing-only controller.

    Every MPC slot reads the true state (``runners.baseline_policy``) and is
    presented as a full-state upper bound. This one does so on a task defined
    by what its observation withholds (the lead's heading, its autopilot and
    turn schedule), so its number bounds information and control together:
    the gap between it and a bearing-only policy is what the hidden lead
    state is worth plus what the policy leaves on the table. The controller
    never reads ``obs``, so its per-step costs equal patrol's seed for seed
    (both tasks share PatrolParams, step_env and the reward); the protocol
    scores both after the same burn-in of 100 steps, so their protocol rows
    agree (scripts/evaluate_baselines.py).
    """
    return make_patrol_mpc(env, params, **kwargs)
