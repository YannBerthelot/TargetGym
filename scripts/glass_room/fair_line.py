"""Glass room study: the fair line, a planner without foresight built to keep its marks.
It is the shipped oracle without a seed (``mpc_line.py``: the true state, the walks'
mean path, the events known) with three changes, none of which shows it the future:

- a new plan every knot (4 h), started from the last plan's remainder;
- no fining limit. Each planned step is charged the glass it would lose, its whole pull
  at the lost-glass price, times the chance that the fining comes out under its mark,
  the forecast's error taken as normal with the spread measured at that lead. The
  margin then comes from the price of a lost step, not from a count of sigmas;
- when no plan meets the other limits, the last plan goes on.

The spread is that of the shipped line's own forecasts (``mpc_line.py`` on the five
scenarios and seeds 100 to 104, the 154 days whose plan started inside its limits,
measured 2026-11-16): nothing before 4 h, then growing to 0.0070 of share at 16 h. A
plan made now reaches the fining after its dead time (11.8 h, the ladder's step test),
and the next plan comes 4 h later, so past 16 h a later plan can still answer and the
spread is held at its 16 h value (ours).

Run: PYTHONPATH=src python scripts/glass_room/fair_line.py SCENARIO SEED OUT_DIR [MAX_ITER]
"""

import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import lib as L  # noqa: E402
import mpc_line  # noqa: E402

from target_gym.glass_furnace import oracle as O  # noqa: E402

#: Spread of the fining's forecast error at each hour of lead, 0 to 16 h, share.
SPREAD = np.array([0.0, 0.0, 1e-5, 3e-5, 0.00012, 0.00034, 0.00069, 0.00114, 0.00164,
                   0.00216, 0.0027, 0.00327, 0.00388, 0.00456, 0.00531, 0.00614, 0.00701])  # fmt: skip
#: Under this the spread is the rounding between the planner's scan and the runner's
#: replay, which the oracle's own back-off is sized on (``oracle.FINING_BACKOFF``).
SPREAD_MIN = 2e-4


def spread(n_steps, dt):
    """The forecast's spread on each of the next ``n_steps`` steps of ``dt`` s."""
    lead = (np.arange(n_steps) + 1) * dt / 3600.0
    return np.maximum(np.interp(lead, np.arange(len(SPREAD)), SPREAD), SPREAD_MIN)


def risk(fining, mark, sd, worth):
    """The expected loss of a step, EUR: ``worth`` times the chance that a fining
    forecast at ``fining``, off by a normal error of spread ``sd``, is under ``mark``.
    """
    return worth * norm.cdf((mark - fining) / sd)


class Fair(O.GlassOracle):
    """The oracle without a seed, planning every knot on the expected bill."""

    def __init__(self, env, params):
        super().__init__(env, params, replan_hours=O.KNOT_HOURS)
        self.kept = 0

    def _plan(self, state, t):
        n = int(self.params.max_steps_in_episode) - t
        p = self.params.replace(walk_sd=(0.0,) * len(L.E.WALKS))
        keys = jax.random.split(jax.random.PRNGKey(0), n)
        again = bool(self.plans) and t - self._t0 == self.knot
        last = self.plans[-1].knots[1:] if again else None
        dt = float(p.delta_t)
        worth = float(state.pull) * dt / 1000.0 * float(p.lost_glass_price)
        sd = jnp.asarray(spread(n, dt), jnp.float32)
        scan, backoff = O._scan, O.FINING_BACKOFF

        def charged(env, params, s, k, valve):
            end, (cost, run, fining, canal, headroom) = scan(env, params, s, k, valve)
            # the tail is past the episode: held at the 16 h spread, as past 16 h
            at = sd if params is p else float(SPREAD[-1])
            run = run + risk(fining, params.fining_pass, at, worth)
            return end, (cost, run, fining, canal, headroom)

        O._scan, O.FINING_BACKOFF = charged, -1.0  # no fining limit: the charge instead
        try:
            plan = O.solve(self.env, p, state, keys, self.knot, start=last)
        finally:
            O._scan, O.FINING_BACKOFF = scan, backoff
        if plan.violation > 0.0 and again:
            self.kept += 1
            plan = plan._replace(
                knots=np.asarray(last), valve=np.repeat(last, self.knot)[:n],
                how="the last plan kept",
            )  # fmt: skip
        self.plans.append(plan)
        self._valve, self._t0 = plan.valve, t


if __name__ == "__main__":
    mpc_line.main(make=Fair, tag="_fair")
