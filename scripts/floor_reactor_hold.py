"""Achievable hold floor of the reactor under its shipped OU demand.

Run with ``uv run python scripts/floor_reactor_hold.py``. Prints ``e_floor``,
the smallest long-run mean |target - power| any controller can hold, and the
policy's rod travel, for the reward's ``e_floor`` and ``rod_travel_hold``.

The hold problem on this plant is a fast-timescale precision problem: the
demand is an Ornstein-Uhlenbeck process whose innovation, ``demand_sigma *
sqrt(delta_t)`` = 2.5e-3 of rated per second, is unpredictable, while the rods
move power at a bounded rate and are commanded once per control period of ten
physics seconds. Over one control period the demand random-walks by 7.9e-3
(sd), and the reward is the mean over the ten sub-steps, so no controller can
hold the mean sub-step error below the expectation of that walk however it
positions the rods. The mean reversion (1/theta = 6667 s) and the clip of the
demand to its range are negligible over ten seconds and are ignored, which
makes the floor very slightly optimistic -- the right direction for a floor.

Reduced model, one state: the error at the start of a control period. The
action is a power correction; the plant moves toward it at the rod-limited
rate (asymmetric: insertion at -0.0245/s, withdrawal at +0.0123/s of rated
power, the quasi-static slews of the handover's reduced reactor model with
thermal feedback, `dp_reactor_hold.py`) and stops when it gets there. Cost per
period is the mean over the ten sub-steps of E|e_k|, e_k Gaussian about the
ramp; the next period starts from the end of the walk. Solved by relative
value iteration on a 2e-4 grid -- the Gaussian kernel's sd (7.9e-3) is forty
grid cells wide, so the chain is not frozen -- and the optimal policy scored
exactly from its stationary distribution.

Sanity check that must hold: the shipped MPC's measured hold error
(``scripts/measure_hold.py``) is at or above this floor.
"""

import numpy as np
from scipy.special import erf

SIGMA = 2.5e-3  # demand innovation per second (demand_sigma * sqrt(delta_t))
N_SUB = 10  # physics sub-steps per control period (Reactor.control_period)
SLEW_UP = 0.0123  # power rise per second, rod withdrawal
SLEW_DOWN = 0.0245  # power fall per second, rod insertion

es = np.arange(-0.04, 0.04 + 1e-9, 2e-4)
ne = len(es)
ds = np.concatenate(
    [-np.geomspace(1e-4, 0.05, 30)[::-1], [0.0], np.geomspace(1e-4, 0.05, 30)]
)
nd = len(ds)


def expected_abs(mean, sd):
    """E|X| for X ~ N(mean, sd^2)."""
    sd = np.maximum(sd, 1e-12)
    z = mean / sd
    return sd * np.sqrt(2 / np.pi) * np.exp(-0.5 * z * z) + mean * erf(z / np.sqrt(2))


# Ramp path within a period for each (start error, correction): e_k before noise.
k = np.arange(1, N_SUB + 1)
path = np.empty((ne, nd, N_SUB))
for j, d in enumerate(ds):
    rate = SLEW_UP if d > 0 else SLEW_DOWN
    move = np.sign(d) * np.minimum(abs(d), rate * k)  # (N_SUB,)
    path[:, j, :] = es[:, None] + move[None, :]
COST = expected_abs(path, SIGMA * np.sqrt(k)[None, None, :]).mean(axis=2)  # (ne, nd)

# Transition: end-of-period error ~ N(path[..., -1], N_SUB * SIGMA^2), on the grid.
sd_end = SIGMA * np.sqrt(N_SUB)
P = np.exp(-0.5 * ((es[None, None, :] - path[:, :, -1][:, :, None]) / sd_end) ** 2)
P /= P.sum(axis=2, keepdims=True)  # (ne, nd, ne)

h = np.zeros(ne)
ref = ne // 2
for it in range(5000):
    Q = COST + np.einsum("ijk,k->ij", P, h)
    hn = Q.min(axis=1)
    h_new = hn - hn[ref]
    if np.abs(h_new - h).max() < 1e-10:
        break
    h = 0.5 * h + 0.5 * h_new
pol = Q.argmin(axis=1)

Pp = P[np.arange(ne), pol]
mu = np.ones(ne) / ne
for _ in range(20000):
    mu = mu @ Pp
e_floor = mu @ COST[np.arange(ne), pol]
travel = mu @ np.abs(
    np.minimum(np.abs(ds[pol]), np.where(ds[pol] > 0, SLEW_UP, SLEW_DOWN) * N_SUB)
)
start_err = mu @ np.abs(es)

print(f"reactor hold floor under the shipped OU demand, {N_SUB} s control period")
print(f"  e_floor (mean sub-step |error|, fraction of rated) = {e_floor:.5f}")
print(f"  mean |error| at period start                        = {start_err:.5f}")
print(f"  mean power travel per period (rod motion proxy)     = {travel:.5f}")
print(
    f"  reference: E|walk| over one period with no correction = "
    f"{expected_abs(np.zeros(N_SUB), SIGMA * np.sqrt(k)).mean():.5f}"
)
