"""The glass furnace's oracle (the MPC slot): the valve a planner with foresight fires.

The oracle is the problem definition solved (charter J3): the true plant, its true state
and the episode's future, here every walk innovation and event the episode's keys draw,
planned on the task's own reward through the env's own ``step_env`` and
``compute_reward_terms``, so no cost is written twice.

The reward is the fuel bill, plus the plates the cold end rejects for their defects,
plus the step's whole charged pull at the lost glass price when the canal is under its
target or the fining share under its pass mark. That last term is a step: losing a step
costs 416.88 EUR, nine times the step's whole fuel bill. The planner therefore states
it as what it is at any plan worth having, a limit on every step, fining >= pass and
canal >= target, and minimises the fuel and the rejected plates under them. It also
asks no more than the crown override's cap: a valve asked over the cap gets the cap,
and asking that cap's path itself gets the same valve, the same override integral and
so the same episode, so on the env's step this loses nothing, and it keeps the
override's min out of the problem's derivatives. A plan that
meets them pays exactly the reward's cost, and it is a local minimiser of the reward
itself: a small move either keeps every step and costs more by the KKT conditions, or
loses a step and pays its 416.88 EUR jump for a saving that vanishes with the move. The
limits are planned ``FINING_BACKOFF`` and ``CANAL_BACKOFF`` inside the marks, so a plan
still passes them when the runner replays it step by step.

The furnace runs on after the episode: the tasks are continuing, ended by a time limit
that no agent sees (docs/rl-protocol.md). So the plan does not end at the cut. Its last
knot is held for ``TAIL_HOURS`` more on the walks' mean path, under the same limits, and
the tail's fuel is charged in the plan's objective, though not in the cost the plan
reports, which is the episode's. Without a tail the plan closes the valve on its last
knot, whose fuel first moves the canal's fining 5.4 h later, and spends the fining the
furnace needs after the cut. With a tail whose fuel is free, the plan drops the knot
before the last and buys its fining back with that fuel (about 0.6 point of the room,
measured 2026-11-09). Charged for it, the plan pays for the fining it leaves: over the
episode's last day it spends 0.26 point of the room more than the cascade, which leaves
its furnace as it is (measured 2026-11-10).

The planner runs in the env's float32: under x64, ``jax.random.normal`` draws other walk
innovations from the same keys, so a float64 plan would be planned on another episode.

The move is the valve, held over knots of ``knot_hours`` (a control parametrisation;
the env's step is 120 s, so the plan is the best one held over its knots). The solver
is sequential quadratic programming (scipy's SLSQP) on exact Jacobians by forward
differentiation through the episode. Convergence is shown by a certificate the
solver does not grade itself: the KKT stationarity of the plan, from non-negative
multipliers on its active limits and bounds (``kkt``), and its worst violation.

On the registered plant it does not converge by that certificate (measured 2026-11-15 on
the plant with its defects, the five scenarios on seeds 100 to 104,
``data/oracle_ceilings/glass_furnace.json``): none of 51 solves did, a start from the
design valve and a restart from its plan on each seed, and a third start on one. On 24
of the 25 kept plans it reads 1 for want of an active limit: ``ACTIVE`` is under one
float32 step of the canal and headroom rows. Read with a band of 1e-4, the normal plans'
stationarity is 0.003 to 0.04 (``scripts/glass_room/plan_certificate.py``), still over
``KKT_TOL``. The plant is not smooth in the valve: the fining band has two ends
(``physical.fining_done``), the refiner's cooling loop clamps at zero and the blocks
release on a rise only, so the Jacobians jump. SLSQP stops on its own tolerance (46
solves, after 1 to 233 iterations) or at its cap of 300 (5). Past its best plan its
iterates wander, up to 8 EUR per step dearer and outside the limits, and what it returns
is, on every solve, no dearer than the cheapest iterate its log shows inside the limits.
A last iterate that loses glass is repaired: the knots cheapest to first order raised,
or every knot alike, whichever costs less, and the cheaper stands against the best
iterate that lost none. Four last iterates lost glass and were raised alike, by under
1e-6 kg/s. Of 25 restarts, 23 moved the replayed cost by 0.003 EUR per step or less and
two found 0.5 and 1.9. The plan is replayed and its replayed cost reported, an upper
bound on the ceiling, so the room from it is a lower bound. On every scenario and seed
it costs less than the reference ladder's kept rung, by 3.8 to 7.1 %, and it holds the
valve between 0.40 and 0.54 kg/s. On the plant without defects (measured 2026-11-10) the
iterates dropped the valve for a knot every 12 to 20 h, since a temperature move cost
nothing, and the last one mostly lost glass.

With an episode's ``seed`` (``benchmark.run_policy``'s keys for it), ``step`` plays the
plan made with full foresight.

Without one the oracle is shown no future draw. It plans on the walks' mean path from
the true state (the innovations unseen, the events known, since the state holds their
times), a new plan each knot, started from the rest of the plan in force. On that
forecast a limit on the fining is met in the plan and lost on the plant: with a plan a
day resting ``FINING_BACKOFF`` over the mark, the fining came out up to 0.023 under its
forecast and a tenth of the steps were lost (measured 2026-11-16,
docs/proposals/glass-furnace-room.md). So this planner prices the mark as the reward
does, in expectation: each planned step is charged, over its fuel and plates, the bill
of a lost step times the chance that the fining comes out under the mark, the
forecast's error taken as normal (ours) with the spread the model's own walks give at
that lead (``forecast_spread``: sampled paths from the state of the episode's first
plan, the valve held, on keys of the planner's own). The margin is then where one more
step toward the mark saves as much fuel as the glass it risks. The spread is read over
the fining's dead time and a knot and held past them (ours): what comes out later, a
later plan can still answer. The canal and the override's cap stay limits; when no
valve meets them, the plan in force goes on.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm

from target_gym.glass_furnace import env_jax as E
from target_gym.glass_furnace import physical as P

#: The suite was solved on 4 h knots: 2 h knots cost twice and ran no better.
KNOT_HOURS = 4.0
#: Inside the pass mark (fining share) and the canal target (K): the planner's scan and
#: the runner's step-by-step replay round apart in float32, by up to 7e-5 of share and
#: 0.028 K of canal over an episode (measured 2026-11-10 on four kept plans, where a
#: 1e-3 K backoff let the runner lose 1 to 4 steps), so about three and two times that
#: (ours).
FINING_BACKOFF, CANAL_BACKOFF = 2e-4, 0.05
#: Limit rows' scale for the solver: fining shares to tenths, the canal in K, the
#: valve's headroom under the override in hundredths of a kg/s.
FINING_SCALE, CANAL_SCALE, HEADROOM_SCALE = 10.0, 0.1, 100.0
#: The objective's scale for the solver: SLSQP's first quasi-Newton step is the scaled
#: gradient itself, so the objective is scaled at each solve's start to move no knot by
#: more than this, kg/s (ours). On the fuel alone the gradient is 2.4 EUR per step per
#: kg/s on its largest knot and the scale comes to the hundredth the first ceilings ran
#: with. With the rejected plates it is 88 on a knot whose blocks are rising (normal,
#: seed 100, measured 2026-11-15, scripts/glass_room/oracle_scale.py): at a hundredth
#: the first steps threw the knots between the valve's minimum and 0.54 kg/s, and the
#: solve took three times the iterations to come back to the same plan.
FIRST_STEP = 0.025
#: A limit or bound within this of its edge counts as active in the certificate.
ACTIVE = 1e-6
#: SLSQP's cap, and its stop on the change in the objective, EUR per step, about
#: float32's resolution of a mean near 46 EUR; convergence is judged by the KKT
#: certificate, not by this.
MAX_ITER, FTOL = 400, 1e-6
#: Stationarity a converged float32 plan reaches (ours, float32 Jacobians).
KKT_TOL = 1e-3
#: The first-order repair of a last iterate that loses glass: at most this many rounds,
#: each aiming a lost row this far (scaled) past its limit, past the linearisation's
#: error (ours: one round sufficed on gas_lean 104, measured 2026-11-10).
REPAIR_ROUNDS, REPAIR_INSIDE = 5, 1e-3
#: Hours the last knot is held past the episode, under the limits, its fuel charged: the
#: fining's dead time (11.8 h, scripts/ladder_rows/glass_furnace.py) plus a knot, with a
#: margin (ours). Without a tail, the plans' end effect spans the last 24 h (measured
#: 2026-11-09). Industrial MPC predicts past its last move until the moves' effects have
#: settled (Qin and Badgwell 2003, p. 747); decision D4 lines out at the horizon's end.
TAIL_HOURS = 24.0
#: The fining's dead time, h (scripts/ladder_rows/glass_furnace.py, its step test).
DEAD_HOURS = 11.8
#: Sampled paths of the walks the forecast's spread is read from (ours): its estimate
#: is then within 6 % at one sigma, a quarter of a point of the fining's margin.
SPREAD_PATHS = 128


class Plan(NamedTuple):
    valve: np.ndarray  # kg/s asked, one per step
    knots: np.ndarray  # kg/s, one per knot
    cost: float  # EUR per step, the reward's mean cost on the planned rollout
    running: float  # EUR per step, its fuel and rejected plates
    tail_running: float  # EUR per tail step, the tail's fuel and plates as planned
    lost_steps: int  # steps the planned rollout loses
    status: int  # SLSQP's exit mode, 0 converged
    message: str
    iterations: int
    capped: bool  # the iteration cap was hit
    kkt: float  # stationarity, relative to the gradient
    violation: float  # worst limit violation, in the scaled rows
    active: int  # active limit rows
    how: str  # "final": SLSQP's last iterate; else how the plan played was had


def rollout(env, params, state, keys, valve):
    """The env's episode from ``state`` under ``valve`` (kg/s per step), on the
    episode's step ``keys``: per step ``(cost, running, fining, canal, headroom)``, EUR
    (``running`` the fuel and the rejected plates, what a step that keeps both marks
    costs), the reward's own readings, and kg/s between the override's cap and the
    valve."""
    return _scan(env, params, state, keys, valve)[1][:5]


def _scan(env, params, state, keys, valve):
    """``rollout``'s readings, what losing each step would add to its bill (EUR), and
    the state the rollout ends in."""

    def f(s, x):
        k, u = x
        hot = P.sensors(s.plant, params.grid, params.plant)["hot"]
        cap = s.limit + params.override_gain * (
            params.crown_high - params.override_margin - hot
        )  # the override's cap on this step (env_jax.override)
        _, s1, r, _, _ = env.step_env(k, s, E.to_action(u, params), params)
        read = P.sensors(s1.plant, params.grid, params.plant)
        # The bill of a step that keeps both marks: its fuel, and the plates its
        # defects reject, by the env's own ``costs`` at the marks.
        plates = E.costs(
            0.0,
            params.canal_target,
            params.fining_pass,
            s1.pull,
            params,
            read["defects"],
        )["tracking"]
        run = E.compute_reward_terms(s1, params)["running"] + plates
        # The same step lost: its whole charged pull (a fining share of 0 is under
        # the mark), by the same ``costs``.
        whole = E.costs(0.0, params.canal_target, 0.0, s1.pull, params)["tracking"]
        headroom = jnp.maximum(cap, params.fuel_min) - u
        return s1, (-r, run, read["fining"], read["canal"], headroom, whole - plates)

    return jax.lax.scan(f, state, (keys, valve))


def episode_keys(seed: int, n_steps: int):
    """``(reset key, step keys (n_steps, 2))`` that ``benchmark.run_policy`` runs
    seed ``seed`` on: ``benchmark.episode_keys`` for one episode, restated here
    so the oracle's fingerprint covers it (a test holds the two equal)."""
    k_reset, k_steps = jax.random.split(jax.random.PRNGKey(seed))
    return jax.random.split(k_reset, 1)[0], jax.random.split(k_steps, n_steps)


def chance_under(fining, mark, spread):
    """The chance that a fining share forecast at ``fining`` comes out under ``mark``,
    the forecast's error taken as normal with ``spread``."""
    return norm.cdf((mark - fining) / spread)


def forecast_spread(env, params, state, n_steps: int, paths: int = SPREAD_PATHS):
    """How far the fining share comes out from its forecast, at each of the next
    ``n_steps`` steps. The forecast is the walks' mean path from ``state`` with the
    valve held; the spread is the root mean square, over ``paths`` sampled paths of
    the walks, of what came out against it. The paths run on keys of the planner's
    own, which no episode is run on."""
    mean_p = params.replace(walk_sd=(0.0,) * len(E.WALKS))
    told_keys = jax.random.split(jax.random.PRNGKey(0), n_steps)
    own = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(0), 1), paths)
    path_keys = jax.vmap(lambda k: jax.random.split(k, n_steps))(own)

    def spread(s):
        valve = jnp.full(n_steps, s.fuel)
        told = _scan(env, mean_p, s, told_keys, valve)[1][2]
        came = jax.vmap(lambda ks: _scan(env, params, s, ks, valve)[1][2])(path_keys)
        return jnp.sqrt(jnp.mean((came - told) ** 2, axis=0))

    return jax.jit(spread)(state)


def kkt(grad, rows, jac, lo_gap, hi_gap):
    """Relative stationarity ``|grad - J_a' lam - mu_lo + mu_hi|_inf / |grad|_inf``
    with ``lam, mu >= 0`` on the active rows and bounds, by non-negative least
    squares, and the number of active rows."""
    from scipy.optimize import nnls

    a = rows <= ACTIVE
    n = grad.size
    cols = [jac[a].T, np.eye(n)[:, lo_gap <= ACTIVE], -np.eye(n)[:, hi_gap <= ACTIVE]]
    M = np.concatenate(cols, axis=1)
    if M.shape[1] == 0:
        res = grad
    else:
        x, _ = nnls(M, grad, maxiter=50 * M.shape[1])
        res = grad - M @ x
    scale = max(float(np.max(np.abs(grad))), 1e-12)
    return float(np.max(np.abs(res)) / scale), int(a.sum())


def solve(
    env, params, state, keys, knot_steps: int, start=None, max_iter=MAX_ITER, log=None,
    tail_hours=TAIL_HOURS, charge_tail=True, fining_spread=None,
):  # fmt: skip
    """The plan from ``state`` over the steps ``keys`` cover, one valve per
    ``knot_steps``, its last knot held ``tail_hours`` past them on the walks' mean path,
    under the limits, the tail's fuel charged in the objective unless ``charge_tail`` is
    False; ``start`` the knots to start from (the state's valve if None);
    ``log(iteration, knots, objective, worst row)`` called after each iteration if
    given. The objective is the fuel bill over the episode and the tail, per episode
    step; the plan's ``cost`` is the episode's alone.

    With ``fining_spread`` (share, one per step of lead, its last value held past its
    end and through the tail, ``FINING_BACKOFF`` at least) the fining mark is priced and
    not limited: each step is charged, over its fuel and plates, what losing it would
    add times its ``chance_under`` the mark. The plan's ``cost`` and ``running`` stay
    the env's, without that charge.
    """
    from scipy.optimize import linprog, minimize

    n = int(keys.shape[0])
    k = -(-n // knot_steps)
    p, st = params, state
    lo, hi = float(p.fuel_min), float(p.fuel_max)
    n_tail = int(round(tail_hours * 3600.0 / float(p.delta_t)))
    tail_p = p.replace(walk_sd=(0.0,) * len(E.WALKS))
    tail_keys = jax.random.split(jax.random.PRNGKey(0), n_tail)
    priced = fining_spread is not None
    if priced:
        lead = np.minimum(np.arange(n + n_tail), len(fining_spread) - 1)
        sd = np.maximum(np.asarray(fining_spread, np.float32)[lead], FINING_BACKOFF)
    m = (1 if priced else 2) * (n + n_tail)  # the rows a step losing glass breaks

    def valve(z):
        return jnp.repeat(z, knot_steps)[:n]

    def outs(z):
        end, (cost, run, fining, canal, headroom, at_risk) = _scan(
            env, p, st, keys, valve(z)
        )
        run_t = jnp.zeros(1)
        if n_tail:
            _, (_, run_t, f_t, c_t, h_t, r_t) = _scan(
                env, tail_p, end, tail_keys, jnp.full(n_tail, z[-1])
            )
            fining, canal, headroom, at_risk = (jnp.concatenate(v) for v in (
                (fining, f_t), (canal, c_t), (headroom, h_t), (at_risk, r_t)))  # fmt: skip
        rows = [
            CANAL_SCALE * (canal - p.canal_target - CANAL_BACKOFF),
            HEADROOM_SCALE * headroom,
        ]
        spent, spent_t = jnp.sum(run), jnp.sum(run_t)
        if priced:
            risk = at_risk * chance_under(fining, p.fining_pass, sd)
            spent, spent_t = spent + jnp.sum(risk[:n]), spent_t + jnp.sum(risk[n:])
        else:
            rows.insert(0, FINING_SCALE * (fining - p.fining_pass - FINING_BACKOFF))
        lost = (fining[:n] < p.fining_pass) | (canal[:n] < p.canal_target)
        return (
            (spent + (spent_t if charge_tail else 0.0)) / n,
            jnp.concatenate(rows),
            jnp.mean(cost),
            jnp.sum(lost),
            jnp.mean(run),
            jnp.mean(run_t),
        )

    value = jax.jit(outs)
    jac = jax.jit(jax.jacfwd(lambda z: outs(z)[:2]))
    cache: dict = {}

    def at(z, deriv):
        key = (z.tobytes(), deriv)
        if key not in cache:
            zz = jnp.asarray(z)
            cache[key] = [np.asarray(v, float) for v in
                          (jac(zz) if deriv else value(zz)[:2])]  # fmt: skip
        return cache[key]

    z0 = np.full(k, float(state.fuel)) if start is None else np.asarray(start)
    z0 = np.clip(z0, lo, hi)
    scale = FIRST_STEP / max(float(np.max(np.abs(at(z0, True)[0]))), 1e-12)
    count = [0]

    best = {}

    def keeps(rows):  # no step of the episode or the tail loses glass
        return bool(np.all(rows[:m] >= 0.0))

    def progress(z):
        count[0] += 1
        f, rows = at(z, False)
        if keeps(rows) and f < best.get("f", np.inf):
            best.update(f=f, z=np.clip(z, lo, hi))
        if log is not None:
            log(count[0], z, float(f), float(rows.min()))

    res = minimize(
        lambda z: scale * at(z, False)[0], z0,
        method="SLSQP", jac=lambda z: scale * at(z, True)[0],
        bounds=[(lo, hi)] * k,
        constraints=[{"type": "ineq", "fun": lambda z: at(z, False)[1],
                      "jac": lambda z: at(z, True)[1]}],
        options={"maxiter": max_iter, "ftol": FTOL * scale}, callback=progress,
    )  # fmt: skip
    z, how = np.clip(res.x, lo, hi), "final"
    if not keeps(at(z, False)[1]):
        # The last iterate loses glass. Two repairs, and the cheaper kept: the raise of the
        # knots cheapest to first order that lifts every limit row a little past its
        # limit, from the Jacobians, repeated from there; and the least raise of every
        # knot alike, by bisection. The cheaper stands against the best iterate that
        # lost none.
        lifted = z
        for _ in range(REPAIR_ROUNDS):
            rows = at(lifted, False)[1][:m]
            if np.all(rows >= 0.0):
                break
            g, J = at(lifted, True)
            moved = np.abs(J[:m]).max(axis=1) > 0.0  # rows no knot reaches stay as are
            need = np.where(rows < 0.0, REPAIR_INSIDE - rows, -rows)[moved]
            lp = linprog(g, A_ub=-J[:m][moved], b_ub=-need, method="highs",
                         bounds=[(0.0, hi - v) for v in lifted])  # fmt: skip
            if lp.status != 0:
                break
            lifted = np.clip(lifted + lp.x, lo, hi)
        repairs = [(lifted, "raised at least cost to first order")]

        def raised(d):
            return np.clip(z + d, lo, hi)

        a, b = 0.0, hi - lo
        if keeps(at(raised(b), False)[1]):
            for _ in range(30):
                mid = 0.5 * (a + b)
                a, b = (a, mid) if keeps(at(raised(mid), False)[1]) else (mid, b)
            repairs.append((raised(b), f"raised alike by {b:.2e} kg/s"))
        for fixed, step in repairs:
            f, rows = at(fixed, False)
            if keeps(rows) and f < best.get("f", np.inf):
                best.update(f=f, z=fixed)
                how = f"final iterate {step}"
        if how == "final" and "z" in best:
            how = "best iterate that lost no glass"
        z = best.get("z", z)
    _, rows, cost, lost, run, run_t = (
        np.asarray(v, float) for v in value(jnp.asarray(z))
    )
    g, J = at(z, True)
    stat, active = kkt(g, rows, J, z - lo, hi - z)
    return Plan(
        valve=np.asarray(valve(jnp.asarray(z))), knots=z, cost=float(cost),
        running=float(run), tail_running=float(run_t), lost_steps=int(lost), status=int(res.status),
        message=str(res.message), iterations=int(res.nit),
        capped=int(res.nit) >= max_iter, kkt=stat,
        violation=float(max(0.0, -rows.min())), active=active, how=how,
    )  # fmt: skip


class GlassOracle:
    """The registry's oracle: ``step(obs, state)`` reads the true state.

    With an episode's ``seed`` it plays the plan made with full foresight of it.
    Without one it is shown no future draw: it plans on the walks' mean path, a new
    plan each knot, the fining mark priced (the module's docstring). ``spread`` hands
    it the fining forecast's spread per step of lead; left out, it is read from the
    model at each episode's first plan, on ``paths`` sampled paths. ``kept`` counts
    the plans that went on because no valve met the limits.
    """

    oracle = True

    def __init__(
        self, env, params, seed=None, knot_hours=KNOT_HOURS, spread=None,
        paths=SPREAD_PATHS,
    ):  # fmt: skip
        self.env, self.params, self.seed = env, params, seed
        hour = 3600.0 / float(params.delta_t)
        self.knot = max(1, int(round(knot_hours * hour)))
        #: Steps of lead the spread is read over: the fining's dead time and a knot.
        self.lead = int(round(DEAD_HOURS * hour)) + self.knot
        self.spread, self.paths = spread, paths
        self.plans: list[Plan] = []
        self.kept = 0
        self.reset()

    def reset(self):
        self._valve, self._t0, self._spread = None, 0, self.spread

    def _plan(self, state, t: int):
        n = int(self.params.max_steps_in_episode)
        if self.seed is not None:
            keys = episode_keys(self.seed, n)[1][t:]
            plan = solve(self.env, self.params, state, keys, self.knot)
        else:
            plan = self._without_foresight(state, t, n - t)
        self.plans.append(plan)
        self._valve, self._t0 = plan.valve, t

    def _without_foresight(self, state, t: int, n: int):
        """The plan from ``state`` over the ``n`` steps left, on the walks' mean path:
        no innovation, the events where the state has them."""
        if self._spread is None:
            self._spread = np.asarray(
                forecast_spread(self.env, self.params, state, self.lead, self.paths)
            )
        p = self.params.replace(walk_sd=(0.0,) * len(E.WALKS))
        keys = jax.random.split(jax.random.PRNGKey(0), n)
        again = self._valve is not None and t - self._t0 == self.knot
        # The plan in force, less the knot just played: where the solve starts.
        rest = self.plans[-1].knots[1:] if again else None
        plan = solve(
            self.env, p, state, keys, self.knot, start=rest, fining_spread=self._spread
        )
        if again and plan.violation > 0.0:  # no valve meets the canal and the cap
            self.kept += 1
            plan = plan._replace(
                knots=np.asarray(rest), valve=np.repeat(rest, self.knot)[:n],
                how="the plan in force kept",
            )  # fmt: skip
        return plan

    def step(self, _obs, state):
        t = int(state.time)
        stale = self._valve is None or t < self._t0 or t - self._t0 >= len(self._valve)
        if stale or (self.seed is None and t - self._t0 >= self.knot):
            self._plan(state, t)
        u = self._valve[t - self._t0]
        return float(np.clip(np.asarray(E.to_action(u, self.params))[0], -1.0, 1.0))


def make_glass_furnace_mpc(env, params, seed=None, **kwargs):
    """The oracle for one episode: with ``seed``, full foresight of it."""
    return GlassOracle(env, params, seed=seed, **kwargs)
