"""The oracle (the MPC slot) for the compressor.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``): changing the oracle re-records
only the oracle rows of this package's tasks. The shared machinery
(GradientMPC, NewtonMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every oracle fingerprint.

The reference controller is an NMPC on the env's own step
(``CompressorSurgeMPC``), solved by IPOPT's algorithm written in JAX
(``experts_ipm``), as the CasADi/IPOPT build of the same NLP was solved
until 0.7. It reads the true state, both schedules and the deviation the
coming step uses, keeps the trip variable of every planned step
``MPC_SURGE_MARGIN`` right of the surge line through a soft constraint
priced at a trip's cost, and hands any step whose solve fails to the pair,
whose memory it keeps tracked to the actions it applies.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from target_gym.compressor_surge.env import (
    KPA,
    N_SETPOINT_BLOCKS,
    CompressorSurgeParams,
    characteristic,
    check_valve_sqrt,
    consumer_flow,
    demand_opening,
    sound_speed,
    suction_density,
)
from target_gym.compressor_surge.experts import (
    CompressorSurgePair,
    PairState,
    check_pair_gains,
    commands_from_raw,
    load_gains,
    pair_init,
    pair_step,
    raw_from_commands,
)
from target_gym.compressor_surge.experts_ipm import (
    NLP,
    Result,
    interior_point,
    stage_kkt_solve,
)

try:
    # Newer JAX; 0.10 removed the experimental one. The declared range starts
    # at JAX 0.6, which has only that.
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64

# ---------------------------------------------------------------------------
# The reference controller: an NMPC solved by IPOPT's algorithm in JAX
# ---------------------------------------------------------------------------

#: Horizon in steps of ``delta_t`` (ours), 6 s. A full 20 to 28 kPa move on
#: the 15 % control line needs 14.7 % of rated speed, 4.9 s of drive travel
#: at 3 %/s (derived, scripts/compressor_surge_numbers.py --section steady)
#: plus the 1 s lag, and closing the recycle from its reset opening takes
#: 3.5 s, so 6 s covers the slowest actuator move a block asks for.
MPC_HORIZON = 60
#: The flow-coefficient margin over the surge line that every planned step's
#: ``phi_min`` keeps (TUNED, with the CasADi build of this NLP); ``phi_min``
#: is the env's trip variable, the soft minimum of Phi over the step's
#: substeps. The gate sized the smallest candidate from the margin a seen
#: deviation innovation costs and the model gap, 2 x (1.019 + 0.020) = 2.08
#: points, so 0.03, and accepted 0.03 and 0.05 on 128 episodes and the stress
#: battery at the same cost, 1.373e5 per episode (measured at the provisional
#: ``c_hold``, 49.6 kW), so the wider one ships.
MPC_SURGE_MARGIN = 0.05
#: Header-pressure error that costs 1 per stage, kPa (ours). It only sets the
#: objective's units, so the solver sees numbers of order one.
MPC_ERROR_SCALE_KPA = 1.0
#: Weight on the squared move of each command between stages (ours).
MPC_MOVE_WEIGHT = 1e-3
#: Terminal cost, in multiples of the last stage's tracking term (ours).
MPC_TERMINAL_WEIGHT = 20.0
#: Recycle power above which the objective charges energy, W (ours, frozen):
#: the provisional ``c_hold`` the NMPC was built and accepted with. The NMPC
#: never reads ``params.c_hold``.
MPC_POWER_REF = 49.6e3
#: Energy weight per stage (derived, frozen): v2's ``running_weight`` 1 in
#: tracking units of (0.0275 kPa / ``MPC_ERROR_SCALE_KPA``)^2, 0.0275 kPa being
#: the precision floor ``e_floor`` is clamped at. The NMPC never reads
#: ``params.e_floor``.
MPC_ENERGY_WEIGHT = 1.0 * (0.0275 / MPC_ERROR_SCALE_KPA) ** 2
#: Width of the smoothed max(., 0) on the power excess, as a fraction of
#: ``MPC_POWER_REF`` (ours): (y + sqrt(y^2 + k^2)) / 2.
MPC_POWER_SMOOTHING = 0.05
#: Width of the smoothed substep rate limits, as a fraction of one substep's
#: actuator travel (ours). The env's rate limits make the next state a
#: piecewise smooth function of the command, with a kink wherever the command
#: is a whole number of substeps' travel away. With the CasADi build, over
#: the first 12 steps of four episodes (seeds 1000 to 1003), IPOPT's
#: 150-iteration cap stopped 4 of the 48 solves with the exact clip and none
#: with the clip rounded at 0.01 or 0.1, and a solve took 10.2 iterations on
#: average at 0.1, against 18.5 at 0.01 and 29.4 exact. Against the exact
#: clip, the rounding at 0.1 moves the next header pressure by at most
#: 0.555 Pa and a substep's Phi by at most 6.7e-6 (measured then).
#: test_mpc_predicts_one_step_like_the_plant holds the one-step error under
#: 0.05 ``e_floor``; test_mpc_model_is_the_env_step checks the restatement
#: with the exact clip.
MPC_CLIP_SMOOTHING = 0.1
#: A surge slack above this, in Phi, counts as an active constraint (ours).
#: The solver leaves an unused slack at about -1e-8 (``MPC_BOUND_RELAX``).
MPC_SLACK_ACTIVE = 1e-6
#: Substeps of the restated step: the env's ``"rk4_10"``.
MPC_SUBSTEPS = 10
#: Typical magnitude of each state (ours), (m_c, dp, N, N_ramp, x, phi_min):
#: the solver works on the states divided by these, as do-mpc scaled them.
MPC_STATE_SCALE = (10.0, 2.0e4, 1.0, 1.0, 0.3, 0.5)
#: IPOPT's iteration cap per solve, which the CasADi build set (ours).
MPC_MAX_ITER = 150
#: IPOPT's default tolerance on its scaled error (read: IPOPT's ``tol``).
MPC_TOL = 1e-8
#: IPOPT's barrier start and bound push: its defaults for a cold start, and
#: the CasADi build's warm-start options (ours) for a warm one, which start
#: from the shifted plan and multipliers without pushing them off their
#: bounds. Without them IPOPT's bound push moves each surge slack off zero,
#: where the trip-sized penalty makes the start expensive.
MPC_COLD_MU, MPC_COLD_PUSH = 0.1, 1e-2
MPC_WARM_MU, MPC_WARM_PUSH = 1e-4, 1e-9
#: IPOPT's relaxation of every bound, relative (read: ``bound_relax_factor``).
#: An unused surge slack sits at about -1e-8 rather than 0.
MPC_BOUND_RELAX = 1e-8
#: IPOPT's gradient-based scaling: the objective is scaled down until its
#: largest gradient entry at the start is at most this (read:
#: ``nlp_scaling_max_gradient``).
MPC_MAX_GRADIENT = 100.0

_N_X, _N_U = 6, 2  # states (with phi_min) and commands per stage


def surge_penalty(params: CompressorSurgeParams) -> float:
    """The NLP's price per unit of surge slack in Phi: a trip's cost in the
    objective's units, restart_steps x failure_cost x (e_floor / error
    scale)^2, in which e_floor cancels."""
    return params.restart_steps * 2.0 * (params.dp_error_max / MPC_ERROR_SCALE_KPA) ** 2


def soft_min(phis, temperature, xp=np):
    """The env's trip variable over substep values, -T logsumexp(-Phi / T),
    shifted by the minimum. The last axis is the substeps."""
    low = phis.min(axis=-1)
    return low - temperature * xp.log(
        xp.exp(-(phis - low[..., None]) / temperature).sum(axis=-1)
    )


def _smooth_clip(z, lo, hi, width):
    """clip(z, lo, hi) for lo < 0 < hi, each edge rounded by a softplus whose
    width is ``width`` times that edge's limit, so the map is twice
    differentiable. At a distance d inside an edge of width w it moves z by
    w exp(-d / w) or less, and on the edge by w ln 2."""
    w_hi, w_lo = width * hi, -width * lo
    return (
        z
        - w_hi * jax.nn.softplus((z - hi) / w_hi)
        + w_lo * jax.nn.softplus((lo - z) / w_lo)
    )


def mpc_step(
    params: CompressorSurgeParams,
    state,
    command,
    opening,
    smooth_width=MPC_CLIP_SMOOTHING,
):
    """One env step restated for the NLP: ``(next state, the substep Phis)``,
    for a command within one step's reach of the actuators.

    ``state`` is (m_c, dp, N, N_ramp, x), ``command`` (N_cmd, x_cmd) in
    physical units and ``opening`` the consumers' opening at the RK4 stage
    times (``preview``). The env's own functions give the forces; what is
    restated is the env's order (per substep, the drive setpoint and the
    valve move toward their commands, then one RK4 step of the duct, plenum
    and speed equations, then Phi) with the rate limits in closed form.

    The env moves each actuator by at most one substep's travel a per
    substep, which puts it at start + clip(d, -j a, j a) after j substeps,
    with d the command less the start. After the last substep it is the
    command itself, since the NLP bounds every command to one step's travel,
    so no clip edge sits where an actuator runs at full rate, which is where
    the NLP's solution often lies. Substeps 1 to n - 1 round the clip over
    ``smooth_width`` of one substep's travel (``MPC_CLIP_SMOOTHING``; 0 is
    exact). test_mpc_model_is_the_env_step and
    test_mpc_predicts_one_step_like_the_plant hold it to the env.
    """
    p = params
    m_c, dp, N, N_ramp, x = (state[i] for i in range(5))
    N_cmd, x_cmd = command[0], command[1]
    h = p.delta_t / MPC_SUBSTEPS
    rho = suction_density(p)
    plenum = sound_speed(p, jnp) ** 2 / p.V_p

    def velocity(y, o, N_ramp, x):
        m, P, n = y
        U = n * p.U_r
        dp_c = rho * U**2 * characteristic(m / (rho * p.A_c * U), p)
        outflow = p.k_d * o * check_valve_sqrt(
            P - p.dp_out, p
        ) + p.k_r * x * check_valve_sqrt(P, p)
        return jnp.stack(
            [
                (p.A_c / p.L_c) * (dp_c - P),
                plenum * (m - outflow),
                (N_ramp - n) / p.tau_N,
            ]
        )

    d_N, d_x = N_cmd - N_ramp, x_cmd - x

    def clip(z, lo, hi, n):
        if smooth_width == 0.0:
            return jnp.clip(z, lo, hi)
        return _smooth_clip(z, lo, hi, smooth_width / n)

    def substep(y, j):
        # A scan rather than a Python loop: the solver differentiates this
        # twice, and an unrolled step compiled several times slower.
        n = j + 1
        last = n == MPC_SUBSTEPS
        r_N = n * p.drive_rate * h
        N_j = jnp.where(last, N_cmd, N_ramp + clip(d_N, -r_N, r_N, n))
        x_j = jnp.where(
            last,
            x_cmd,
            x + clip(d_x, -n * p.valve_close_rate * h, n * p.valve_open_rate * h, n),
        )
        o = jax.lax.dynamic_slice_in_dim(opening, 2 * j, 3)
        k1 = velocity(y, o[0], N_j, x_j)
        k2 = velocity(y + 0.5 * h * k1, o[1], N_j, x_j)
        k3 = velocity(y + 0.5 * h * k2, o[1], N_j, x_j)
        k4 = velocity(y + h * k3, o[2], N_j, x_j)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return y, y[0] / (rho * p.A_c * p.U_r * y[2])

    y, phis = jax.lax.scan(substep, jnp.stack([m_c, dp, N]), jnp.arange(MPC_SUBSTEPS))
    # After the last substep both actuators are at their commands.
    return jnp.stack([y[0], y[1], y[2], N_cmd, x_cmd]), phis


def _stage_next(params, x6, command, opening):
    """The NLP's state transition, ``phi_min`` included."""
    nxt, phis = mpc_step(params, x6[:5], command, opening)
    return jnp.concatenate(
        [nxt, soft_min(phis, params.soft_min_temperature, jnp)[None]]
    )


def _stage_cost(params, x6, p_ref, w_track, w_energy):
    """Tracking, ((p_ref - dp) / error scale)^2 in kPa, and the smoothed
    recycle power above ``MPC_POWER_REF``, relative to it."""
    dp, x = x6[1], x6[4]
    tracking = ((p_ref - dp) / KPA / MPC_ERROR_SCALE_KPA) ** 2
    power = params.k_r * x * check_valve_sqrt(dp, params) * dp / suction_density(params)
    excess = (power - MPC_POWER_REF) / MPC_POWER_REF
    k = MPC_POWER_SMOOTHING
    energy = MPC_ENERGY_WEIGHT * 0.5 * (excess + jnp.sqrt(excess**2 + k * k))
    return w_track * tracking + w_energy * energy


class _Problem(NamedTuple):
    """One solve's NLP, as do-mpc laid it out for ``horizon`` + 1 stages S.

    Variables z: states x_1..x_S (x_0 is measured), commands u_0..u_{S-1},
    surge slacks e_0..e_{S-1}. Stage k's cost is on x_k, its constraints on
    x_k and u_k (do-mpc evaluates them on the state a stage starts from), so
    the stage past the horizon only lets the last planned step's ``phi_min``
    be constrained.
    """

    params: CompressorSurgeParams
    x0: jnp.ndarray
    u_prev: jnp.ndarray
    opening: jnp.ndarray  # (S + 1, 2 MPC_SUBSTEPS + 1)
    p_ref: jnp.ndarray
    w_track: jnp.ndarray
    w_energy: jnp.ndarray

    @property
    def S(self) -> int:
        return self.opening.shape[0] - 1

    def unpack(self, z):
        S = self.S
        return (
            z[: _N_X * S].reshape(S, _N_X),
            z[_N_X * S : 8 * S].reshape(S, _N_U),
            z[8 * S :],
        )

    def pack(self, xs, us, eps):
        return jnp.concatenate([xs.ravel(), us.ravel(), eps.ravel()])

    def starts(self, xs):
        """x_0..x_{S-1}: the state each stage starts from."""
        return jnp.concatenate([self.x0[None], xs[:-1]])

    def objective(self, z):
        xs, us, eps = self.unpack(z)
        S, p = self.S, self.params
        stage = jax.vmap(lambda x, r, wt, we: _stage_cost(p, x, r, wt, we))(
            self.starts(xs), self.p_ref[:S], self.w_track[:S], self.w_energy[:S]
        )
        moves = jnp.diff(jnp.concatenate([self.u_prev[None], us]), axis=0)
        return (
            jnp.sum(stage)
            + MPC_MOVE_WEIGHT * jnp.sum(moves**2)
            + surge_penalty(p) * jnp.sum(eps)
        )

    def dynamics(self, z):
        """x_{k+1} - F(x_k, u_k), stage by stage."""
        xs, us, _ = self.unpack(z)
        F = jax.vmap(lambda x, u, o: _stage_next(self.params, x, u, o))(
            self.starts(xs), us, self.opening[: self.S]
        )
        return (xs - F).ravel()

    def inequalities(self, z):
        """Every row >= 0: the rate reach (four per stage), the surge margin
        with its slack, the slack, and the command bounds."""
        xs, us, eps = self.unpack(z)
        p = self.params
        X = self.starts(xs)
        dt = p.delta_t
        N_ramp, x, N_cmd, x_cmd = X[:, 3], X[:, 4], us[:, 0], us[:, 1]
        reach = jnp.stack(
            [
                dt * p.drive_rate - (N_cmd - N_ramp),
                dt * p.drive_rate - (N_ramp - N_cmd),
                dt * p.valve_open_rate - (x_cmd - x),
                dt * p.valve_close_rate - (x - x_cmd),
            ],
            axis=1,
        ).ravel()
        surge = eps - (2.0 * p.W * (1.0 + MPC_SURGE_MARGIN) - X[:, 5])
        bounds = jnp.concatenate(
            [N_cmd - p.N_min, p.N_max - N_cmd, x_cmd, 1.0 - x_cmd, eps]
        )
        return jnp.concatenate([reach, surge, bounds])

    def rollout(self, us):
        """The states the model reaches under commands ``us``."""

        def advance(x, k):
            nxt = _stage_next(self.params, x, us[k], self.opening[k])
            return nxt, nxt

        return jax.lax.scan(advance, self.x0, jnp.arange(self.S))[1]


def _solve(pr: _Problem, z0, lam0, w0, *, cold: bool, max_iter: int, tol) -> Result:
    """Solve ``pr`` from ``z0`` as IPOPT did: its scaling, its bound
    relaxation and push, and, cold, its least-squares multipliers and
    monotone barrier; warm, the given multipliers and the adaptive one.
    Returns the solution in the problem's units."""
    S, p = pr.S, pr.params
    n = 9 * S
    D = jnp.concatenate([jnp.tile(jnp.asarray(MPC_STATE_SCALE), S), jnp.ones(3 * S)])
    rows = jnp.tile(jnp.asarray(MPC_STATE_SCALE), S)  # each row in its state's scale
    xi = jnp.arange(_N_X * S).reshape(S, _N_X)
    ui = _N_X * S + jnp.arange(_N_U * S).reshape(S, _N_U)
    # Stage k's own variables, x_k and u_k; x_0 is not one, so index n.
    local = jnp.concatenate(
        [jnp.concatenate([jnp.full((1, _N_X), n), xi[:-1]]), ui], axis=1
    )

    def stage(v, k):
        return _stage_next(
            p, jnp.where(k == 0, pr.x0, v[:_N_X]), v[_N_X:], pr.opening[k]
        )

    def derivs(zh, lam):
        z = D * zh
        xs, us, _ = pr.unpack(z)
        V = jnp.concatenate([pr.starts(xs), us], axis=1)
        ks = jnp.arange(S)
        Jk = jax.vmap(jax.jacfwd(stage))(V, ks)
        L = (lam / rows).reshape(S, _N_X)
        Hk = jax.vmap(lambda v, k, l: jax.hessian(lambda u: l @ stage(u, k))(v))(
            V, ks, L
        )
        r = jnp.arange(_N_X * S).reshape(S, _N_X)
        J = (
            jnp.zeros((_N_X * S, n + 1))
            .at[r[:, :, None], xi[:, None, :]]
            .add(jnp.eye(_N_X)[None])
        )
        J = J.at[r[:, :, None], local[:, None, :]].add(-Jk)[:, :n]
        H = (
            jnp.zeros((n + 1, n + 1))
            .at[local[:, :, None], local[:, None, :]]
            .add(Hk)[:n, :n]
        )
        H = H + jax.hessian(pr.objective)(z)
        gf = jax.grad(pr.objective)(z)
        return (
            D * gf,
            D[:, None] * H * D[None, :],
            J / rows[:, None] * D[None, :],
            pr.dynamics(z) / rows,
        )

    zh = z0 / D
    osc = jnp.minimum(
        1.0, MPC_MAX_GRADIENT / jnp.max(jnp.abs(derivs(zh, jnp.zeros(_N_X * S))[0]))
    )

    def scaled_derivs(zh, lam):
        gf, H, J, c = derivs(zh, lam / osc)
        return osc * gf, osc * H, J, c

    G = jax.jacfwd(pr.inequalities)(jnp.zeros(n)) * D[None, :]
    g0 = pr.inequalities(jnp.zeros(n))
    g0 = g0 + MPC_BOUND_RELAX * jnp.maximum(1.0, jnp.abs(g0))

    def restore(zh):
        """The commands kept, the states re-simulated from them, each slack
        raised to what its surge row needs."""
        xs, us, eps = pr.unpack(D * zh)
        xs = pr.rollout(us)
        need = 2.0 * p.W * (1.0 + MPC_SURGE_MARGIN) - pr.starts(xs)[:, 5]
        return pr.pack(xs, us, jnp.maximum(eps, need)) / D

    nlp = NLP(
        f=lambda zh: osc * pr.objective(D * zh),
        c=lambda zh: pr.dynamics(D * zh) / rows,
        derivs=scaled_derivs,
        G=G,
        g0=g0,
        kkt=functools.partial(stage_kkt_solve, n_stages=S, nx=_N_X, nu=_N_U),
        restore=restore,
    )
    push = MPC_COLD_PUSH if cold else MPC_WARM_PUSH
    zh = zh.at[8 * S :].set(jnp.maximum(zh[8 * S :], push))  # eps is a bounded variable
    s = jnp.maximum(G @ zh + g0, push)
    if cold:
        w = jnp.ones_like(s)
        gf, _, J, _ = scaled_derivs(zh, jnp.zeros(_N_X * S))
        lam = jnp.linalg.solve(J @ J.T, J @ (gf - G.T @ w))
        lam = jnp.where(jnp.max(jnp.abs(lam)) > 1e3, 0.0, lam)  # IPOPT's cap
    else:
        w = jnp.maximum(w0 * osc, push)
        lam = lam0 * osc
    r = interior_point(
        nlp, zh, s, lam, w, mu0=MPC_COLD_MU if cold else MPC_WARM_MU, max_iter=max_iter,
        tol=tol, adaptive=not cold,
    )  # fmt: skip
    return r._replace(z=D * r.z, lam=r.lam / osc, w=r.w / osc)


def preview(state, params: CompressorSurgeParams, horizon: int = MPC_HORIZON) -> dict:
    """The NLP's data at ``state``, over its stages k = 0 to ``horizon`` + 1:

    - ``opening``, the consumers' opening at the RK4 stage times of the step
      from stage k, from the env's clock and the deviation a^k demand_dev,
      a = exp(-demand_theta delta_t), its conditional mean;
    - ``p_ref`` (Pa), the level the env scores the state k steps ahead
      against;
    - ``w_track`` and ``w_energy``, the stage's weights: 1 and 1 before the
      horizon, ``MPC_TERMINAL_WEIGHT`` and 0 at it, 0 past it.
    """
    p = params
    k = jnp.arange(horizon + 2)
    clock = state.block_clock + k
    block = jnp.minimum(clock // p.setpoint_block_steps, N_SETPOINT_BLOCKS - 1)
    dev = state.demand_dev * jnp.exp(-p.demand_theta * p.delta_t) ** k
    h = p.delta_t / MPC_SUBSTEPS
    tau = clock[:, None] * p.delta_t + 0.5 * h * jnp.arange(2 * MPC_SUBSTEPS + 1)[None]
    return {
        "opening": demand_opening(state.demand_levels, tau, dev[:, None], p),
        "p_ref": state.setpoint_levels[block],
        "w_track": jnp.where(k < horizon, 1.0, 0.0)
        .at[horizon]
        .set(MPC_TERMINAL_WEIGHT),
        "w_energy": jnp.where(k < horizon, 1.0, 0.0),
    }


def _problem(state, u_prev, params, horizon) -> _Problem:
    x0 = jnp.stack([state.m_c, state.dp, state.N, state.N_ramp, state.x, state.phi_min])
    return _Problem(params, x0, u_prev, **preview(state, params, horizon))


def _cold_guess(pr: _Problem, gains):
    """The cold start: the model rolled out with both commands holding the
    actuators where they are, those commands and zero surge slacks. Where
    that rollout takes some planned step's ``phi_min`` below the planned
    margin, the rollout under a fresh pair, its commands clipped to one
    step's reach of the actuators (where the env acts on them exactly as on
    the unclipped ones), instead.

    The hold rollout runs into surge when a demand drop lies inside the
    horizon and the valve is shut, as in the stress battery's demand drops.
    IPOPT started there converged, however many iterations it was given, to
    a plan that surges, and the warm start carried that plan forward until
    the plant surged on it: with the hold rollout as the only cold guess, the
    stress battery tripped in 12 of its 16 runs at both candidate margins,
    and in none from this guess (measured with the CasADi build).
    test_mpc_cold_start_avoids_a_surging_guess checks it.
    """
    p, S = pr.params, pr.S
    hold = pr.x0[3:5]
    held = pr.rollout(jnp.tile(hold, (S, 1)))
    dt = p.delta_t

    def pair(carry, k):
        x, ps = carry
        m_d = consumer_flow(x[1], pr.opening[k][0], p)
        obs = jnp.stack(
            [
                x[1] / KPA,
                x[0],
                m_d,
                100 * x[2],
                100 * x[3],
                100 * x[4],
                pr.p_ref[k] / KPA,
            ]
        )
        ps = jax.tree_util.tree_map(
            lambda a, b: jnp.where(k == 0, a, b), pair_init(obs, jnp), ps
        )
        raw, ps = pair_step(gains, ps, obs, p, jnp)
        N_cmd, x_cmd = commands_from_raw(raw, p, jnp)
        u = jnp.stack(
            [
                jnp.clip(N_cmd, x[3] - p.drive_rate * dt, x[3] + p.drive_rate * dt),
                jnp.clip(
                    x_cmd, x[4] - p.valve_close_rate * dt, x[4] + p.valve_open_rate * dt
                ),
            ]
        )
        nxt = _stage_next(p, x, u, pr.opening[k])
        return (nxt, ps), (nxt, u)

    ps0 = PairState(I_N=jnp.zeros(()), I_x=jnp.zeros(()), timer=jnp.zeros(()))
    _, (paired, us) = jax.lax.scan(pair, (pr.x0, ps0), jnp.arange(S))
    use_hold = jnp.all(held[:, 5] >= 2.0 * p.W * (1.0 + MPC_SURGE_MARGIN))
    xs = jnp.where(use_hold, held, paired)
    us = jnp.where(use_hold, jnp.tile(hold, (S, 1)), us)
    return pr.pack(xs, us, jnp.zeros(S)), hold


@functools.partial(
    jax.jit, static_argnames=("params", "gains", "horizon", "cold", "max_iter")
)
def _plan(state, u_prev, z0, lam0, w0, tol, *, params, gains, horizon, cold, max_iter):
    """One step's solve: cold from ``_cold_guess``, or warm from (z0, lam0,
    w0). Returns the result and the previous commands the NLP charged moves
    from (a cold start's are the held actuators). ``params`` and ``gains``
    (as items) are static, so their values enter as float64 constants."""
    gains = dict(gains)
    if cold:
        z0, u_prev = _cold_guess(_problem(state, u_prev, params, horizon), gains)
    pr = _problem(state, u_prev, params, horizon)
    return _solve(pr, z0, lam0, w0, cold=cold, max_iter=max_iter, tol=tol), u_prev


def _shift(z, lam, w, S: int):
    """The warm start: a plan and its multipliers moved one stage earlier,
    the last stage repeated."""

    def shifted(v, per):
        v = v.reshape(S, per)
        return jnp.concatenate([v[1:], v[-1:]]).ravel()

    xs, us, eps = z[: _N_X * S], z[_N_X * S : 8 * S], z[8 * S :]
    reach, surge, rest = w[: 4 * S], w[4 * S : 5 * S], w[5 * S :]
    return (
        jnp.concatenate([shifted(xs, _N_X), shifted(us, _N_U), shifted(eps, 1)]),
        shifted(lam, _N_X),
        jnp.concatenate(
            [shifted(reach, 4), shifted(surge, 1)]
            + [shifted(b, 1) for b in rest.reshape(5, S)]
        ),
    )


def _float64(tree):
    """Floats to float64 and int32 to int64, for a solve under x64."""

    def cast(a):
        a = jnp.asarray(a)
        if jnp.issubdtype(a.dtype, jnp.floating):
            return a.astype(jnp.float64)
        return a.astype(jnp.int64) if a.dtype == jnp.int32 else a

    return jax.tree_util.tree_map(cast, tree)


class CompressorSurgeMPC:
    """NMPC for the compressor, solved as the CasADi/IPOPT build solved it,
    falling back to the PID pair.

    States : [m_c, dp, N, N_ramp, x, phi_min]
    Inputs : [N_cmd, x_cmd], the speed and recycle commands in physical units

    The model is the env's discrete step (``mpc_step``): ten substeps, each
    moving the drive setpoint and the valve toward their commands at their
    rate limits, then one RK4 step. ``phi_min`` is the env's own trip
    variable, carried as a state as the env carries it. Multiple shooting
    over ``horizon`` steps, plus the stage past it (``_Problem``).

    Objective: per stage k < horizon, ((p_ref - dp) / error scale)^2 in kPa
    plus ``MPC_ENERGY_WEIGHT`` times the smoothed excess of the recycle power
    over ``MPC_POWER_REF``; ``MPC_TERMINAL_WEIGHT`` times the tracking term
    of the last state; ``MPC_MOVE_WEIGHT`` on each command's move. Surge:
    every planned step's ``phi_min`` at least 2W (1 + ``MPC_SURGE_MARGIN``),
    soft, at ``surge_penalty`` per unit of slack. Rate reach: each command
    within one step's actuator travel of the actuator, which removes
    redundant freedom and costs no authority.

    The solver is IPOPT's algorithm (``experts_ipm``) in float64, with the
    CasADi build's two instances: IPOPT's defaults for a cold start (the
    first step, the first after a trip, and the one after a failed cold
    solve), from ``_cold_guess``; the warm-start options and the adaptive
    barrier otherwise, from the last plan and its multipliers shifted one
    stage (``_shift``).

    Like every MPC in the suite it is an oracle ceiling. It reads the true
    state, both schedules with the clock, and the deviation the coming step
    integrates with; stage k plans with the deviation's conditional mean.

    A step whose solve fails is counted in ``solve_failures`` and handed to
    the pair, whose memory is tracked to every action applied; the warm start
    stays the one the failed solve began from, and the next solve's move
    penalty reads the commands applied. A solve the iteration cap stopped is
    applied and counted in ``solve_capped``.
    """

    def __init__(
        self,
        env,
        params: CompressorSurgeParams,
        horizon: int = MPC_HORIZON,
        max_iter: int = MPC_MAX_ITER,
        fallback: CompressorSurgePair | None = None,
    ):
        self.env, self.params = env, params
        self.horizon, self.max_iter = int(horizon), int(max_iter)
        self._pid = (
            CompressorSurgePair(load_gains(), params) if fallback is None else fallback
        )
        self.solve_calls = self.solve_iters = 0
        self.solve_failures = self.solve_capped = 0
        self.reset()

    def reset(self):
        """Forget the plan and the pair's memory: the next step starts cold."""
        self._plan = None  # (z, lam, w) of the last solve
        self._u = None  # the commands last applied
        self._last_clock = None
        self._pid.reset()

    @property
    def plan(self) -> dict | None:
        """The plan the next warm solve starts from, in physical units:
        ``states`` x_1..x_S (m_c, dp, N, N_ramp, x, phi_min), ``commands``
        u_0..u_{S-1} and the surge ``slacks``. The last solve's plan, or after
        a failed warm solve the warm start that solve began from; ``None``
        when the next step starts cold."""
        if self._plan is None:
            return None
        S = self.horizon + 1
        z = np.asarray(self._plan[0], float)
        return {
            "states": z[: _N_X * S].reshape(S, _N_X),
            "commands": z[_N_X * S : 8 * S].reshape(S, _N_U),
            "slacks": z[8 * S :],
        }

    def solver_report(self) -> dict:
        calls = max(self.solve_calls, 1)
        return {
            "solver_calls": self.solve_calls,
            "solver_failures": self.solve_failures,
            "solver_capped": self.solve_capped,
            "solver_mean_iters": round(self.solve_iters / calls, 1),
        }

    def step(self, obs, state):
        """The raw action for ``state``; ``obs`` is what the pair reads."""
        clock = int(state.block_clock)
        cold = self._plan is None or (
            self._last_clock is not None and clock < self._last_clock
        )
        self._last_clock = clock
        S = self.horizon + 1
        with enable_x64():
            if cold:
                start = (jnp.zeros(9 * S), jnp.zeros(_N_X * S), jnp.zeros(10 * S))
                u_prev = jnp.zeros(_N_U)
            else:
                start = _shift(*self._plan, S)
                u_prev = self._u
            result, u_prev = _plan(
                _float64(state), u_prev, *start, MPC_TOL, params=self.params,
                gains=tuple(sorted(self._pid.gains.items())), horizon=self.horizon,
                cold=cold, max_iter=self.max_iter,
            )  # fmt: skip
            iters, err = int(result.iterations), float(result.error)
            self.solve_calls += 1
            self.solve_iters += iters
            ok = err >= 0.0 and bool(jnp.all(jnp.isfinite(result.z)))
            if not ok:
                # The pair takes the step. A failed warm solve leaves the warm
                # start it began from, a failed cold one the next step cold.
                self.solve_failures += 1
                self._plan = None if cold else start
                raw = np.asarray(self._pid.step(obs), dtype=float)
                self._u = jnp.asarray(commands_from_raw(raw, self.params), jnp.float64)
                return np.clip(raw, -1.0, 1.0)
            self.solve_capped += int(iters >= self.max_iter and err > MPC_TOL)
            self._plan = (result.z, result.lam, result.w)
            self._u = result.z[_N_X * S : _N_X * S + _N_U]
            raw = np.asarray(
                raw_from_commands(self._u[0], self._u[1], self.params, np), float
            )
        raw = np.clip(raw, -1.0, 1.0)
        self._pid.track(obs, raw)
        return raw


def make_compressor_surge_mpc(env, params, **kwargs) -> CompressorSurgeMPC:
    """Registry factory: the NMPC on the env's own step, falling back to the
    PID pair.

    The fallback is built from the stored gains and held to the PID factory's
    guard, so gains the guard refuses stop this factory too. ``kwargs`` go to
    ``CompressorSurgeMPC`` (``horizon``, ``max_iter``).
    """
    gains = load_gains()
    check_pair_gains(gains)
    return CompressorSurgeMPC(
        env, params, fallback=CompressorSurgePair(gains, params), **kwargs
    )
