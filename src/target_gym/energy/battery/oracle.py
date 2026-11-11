"""The oracle (the MPC slot) for the battery task.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``).

The unit's only lever is the baseline of each settlement period, declared
until gate closure (``env.submitted_period``): over a 4 h block, the periods
from the first still open at the reset to the block's last, five numbers. The
problem is the env itself: a plan is those baselines, sent on every step as
the open period's raw action, and its cost is the env's own reward summed over
the block (``block_cost``), so nothing of the score is restated here. The true
state of charge, the BMS's error and drift, the prices and the deviation are
read from the true state.

The cost is piecewise: availability is lost all at once when the reported
energy misses a period's requirement, and the losses and the BMS's power
limits are kinks. So the solver is derivative free (``search``): a grid over
the baselines, then compass search from the best grid points, on the
coordinates and on the transfers between two periods (energy moved from one
period to another, which keeps the later state of charge), halving
its step until it is below ``STEP_TOL`` with no improving direction among
the compass's and random ones (``compass``). That is its convergence test,
with the starts' agreement reported beside it.

- the ceiling (``ceiling_search``) knows the regulation signal ahead: the
  env's own draws on the episode's step keys, as a ceiling may;
- the registry's slot (``BaselineOracle``) cannot know the keys: it plans the
  open periods on the signal's mean path (the env at ``signal_std`` 0, which
  zeroes its innovations, as ``plan_params`` does), from the true state, again
  each time a new period opens.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from target_gym.energy.battery.env import N_PERIODS, submitted_period

#: The compass step at which a search stops, in raw action units (a baseline
#: of 2 x 0.55 MW spans 2): 1e-6 of it is 0.6 W.
STEP_TOL = 1e-6
#: Grid points per baseline of the first look (``search``).
GRID = 5
#: The share by which the search raises each energy requirement over the
#: contract's: 138 Wh. The best plan rides a requirement's edge, where the
#: float32 plant ends a few units in the last place of stored energy (0.2 Wh
#: each) off another run of itself, and one short loses the period's
#: availability. The plan found is charged at the contract's requirement.
MARGIN = 1e-4


def open_periods(params, start: int = 0) -> np.ndarray:
    """The settlement periods a plan from step ``start`` still declares: open
    at some step from ``start`` to the block's end, and inside the block."""
    T = int(params.max_steps_in_episode)
    per = int(round(float(params.period_seconds) / float(params.delta_t)))
    last = min(-(-T // per) - 1, N_PERIODS - 1)
    first = int(submitted_period(start, params))
    return np.arange(first, last + 1)


def raw_action(B, periods, t, params):
    """The raw action at step ``t`` for baselines ``B`` of ``periods``: the
    open period's, 0 for a period past the block."""
    k = submitted_period(t, params) - periods[0]
    inside = (k >= 0) & (k < periods.shape[0])
    return jnp.where(inside, B[jnp.clip(k, 0, periods.shape[0] - 1)], 0.0)[None]


def block_cost(env, params, state, keys, B, periods):
    """The env's cost from ``state`` to the block's end under baselines ``B``
    (GBP, summed), and whether any step tripped. ``keys`` are the remaining
    steps' keys."""
    periods = jnp.asarray(periods)

    def f(s, k):
        a = raw_action(B, periods, s.time, params).astype(jnp.float32)
        _, s1, r, _, info = env.step_env(k, s, a, params)
        return s1, (r, info["tripped"])

    r, trip = jax.lax.scan(f, state, keys)[1]
    return -jnp.sum(r), jnp.any(trip)


def evaluator(env, params, state, keys, periods, batch: int = 625):
    """``block_cost`` over a batch of baseline vectors, jitted at one batch
    size (a smaller batch is padded, a larger one split), so it compiles
    once."""
    one = lambda B: block_cost(env, params, state, keys, B, periods)  # noqa: E731
    run = jax.jit(jax.vmap(one))

    def f(C):
        C = np.asarray(C, np.float32)
        out_c, out_t = [], []
        for i in range(0, len(C), batch):
            part = C[i : i + batch]
            pad = np.concatenate([part, np.repeat(part[-1:], batch - len(part), 0)])
            c, t = run(jnp.asarray(pad))
            out_c.append(np.asarray(c, float)[: len(part)])
            out_t.append(np.asarray(t, bool)[: len(part)])
        return np.concatenate(out_c), np.concatenate(out_t)

    return f


def directions(n: int) -> np.ndarray:
    """The compass: each coordinate and each transfer between two periods,
    both ways."""
    eye = np.eye(n)
    pairs = [eye[i] - eye[j] for i in range(n) for j in range(n) if i != j]
    return np.concatenate([eye, -eye] + ([np.stack(pairs)] if pairs else []))


#: Random unit directions polled beside the compass at every step: with
#: them the poll is dense in the limit, so the search cannot stall at a kink
#: whose descent lies between the compass's directions (mesh adaptive direct
#: search, Audet and Dennis, SIAM J. Optim. 17, 2006).
N_RANDOM = 32


def compass(f, B0, step=0.5, tol=STEP_TOL, max_evals=4000, seed=0):
    """Pattern search from ``B0`` inside [-1, 1]: move to the best improving
    poll point at ``step``, else halve it, until it is under ``tol``. The poll
    is the compass and ``N_RANDOM`` fresh random unit directions. Returns (B,
    cost, evaluations, final step)."""
    rng = np.random.default_rng(seed)
    n = B0.shape[0]
    D0 = directions(n)
    B = np.clip(np.asarray(B0, float), -1.0, 1.0)
    fB = float(f(B[None])[0][0])
    evals = 0
    while step >= tol and evals < max_evals:
        R = rng.normal(size=(N_RANDOM, n))
        D = np.concatenate([D0, R / np.linalg.norm(R, axis=1, keepdims=True)])
        C = np.clip(B[None] + step * D, -1.0, 1.0)
        fC = f(C)[0]
        evals += 1
        i = int(np.argmin(fC))
        if fC[i] < fB:
            B, fB = C[i], float(fC[i])
        else:
            step *= 0.5
    return B, fB, evals, step


def search(env, params, state, keys, periods, starts: int = 6):
    """The grid's ``starts`` best points and the hold at 0, each refined by
    ``compass``; the best result, and every start's for the record."""
    n = len(periods)
    f = evaluator(
        env, params, state, keys, periods, batch=len(directions(n)) + N_RANDOM
    )
    f_grid = evaluator(env, params, state, keys, periods, batch=GRID**n // GRID)
    axes = np.linspace(-1.0, 1.0, GRID)
    grid = np.stack(np.meshgrid(*[axes] * n, indexing="ij"), -1).reshape(-1, n)
    costs = f_grid(grid)[0]
    first = [grid[i] for i in np.argsort(costs)[:starts]] + [np.zeros(n)]
    runs = [compass(f, B0, seed=i) for i, B0 in enumerate(first)]
    best = min(runs, key=lambda r: r[1])
    return best, runs


def ceiling_search(env, params, state, keys):
    """The ceiling's plan from the reset ``state`` with the episode's step
    ``keys``: the baselines, their cost and the search's record. The search
    runs at the requirements raised by ``MARGIN``; ``cost`` is the plan's at
    the contract's, ``cost_at_margin`` the search's own."""
    periods = open_periods(params, int(state.time))
    t0 = time.time()
    strict = params.replace(volume_hours=params.volume_hours * (1.0 + MARGIN))
    (B, planned, evals, step), runs = search(env, strict, state, keys, periods)
    cost = float(block_cost(env, params, state, keys, jnp.asarray(B), periods)[0])
    return {
        "periods": periods.tolist(),
        "baselines_raw": [float(b) for b in B],
        "cost": cost,
        "cost_at_margin": planned,
        "evaluations": evals,
        "final_step": step,
        "converged": bool(step < STEP_TOL),
        "starts": [
            {"baselines_raw": [float(b) for b in r[0]], "cost": r[1],
             "evaluations": r[2], "final_step": r[3]}
            for r in runs
        ],  # fmt: skip
        "seconds": time.time() - t0,
    }


class BaselineOracle:
    """The registry's slot: on the true state, the open periods' baselines on
    the signal's mean path, planned again each time a period opens."""

    def __init__(self, env, params):
        self.env = env
        self.params = params
        self.plan_params = params.replace(signal_std=0.0)
        self.reports: list[dict] = []
        self.reset()

    def reset(self):
        self._open, self._B, self._periods = None, None, None

    def step(self, _obs, state):
        p = self.params
        t = int(state.time)
        open_now = int(submitted_period(t, p))
        if self._open != open_now:
            self._open = open_now
            periods = open_periods(p, t)
            T = int(p.max_steps_in_episode)
            if len(periods) and t < T:
                # Any keys: at signal_std 0 the draws are multiplied by 0.
                keys = jax.random.split(jax.random.PRNGKey(0), T - t)
                (B, cost, evals, step), _ = search(
                    self.env, self.plan_params, state, keys, periods, starts=1
                )
                self._B, self._periods = B, periods
                self.reports.append(
                    {"evaluations": evals, "converged": bool(step < STEP_TOL)}
                )
        if self._B is None:
            return 0.0
        a = raw_action(jnp.asarray(self._B), jnp.asarray(self._periods), t, self.params)
        return float(a[0])

    def solver_report(self) -> dict:
        return {
            "solver_calls": len(self.reports),
            "solver_unconverged": sum(not r["converged"] for r in self.reports),
        }


def make_battery_mpc(env, params):
    """The battery's oracle slot: :class:`BaselineOracle`."""
    return BaselineOracle(env, params)
