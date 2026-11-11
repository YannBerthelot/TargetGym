"""The oracle (the MPC slot) for the compressor, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem, by the
smoothing protocol's inventory (docs/proposals/oracle-smoothing-protocol.md,
section 5.4, and decision S10), run by
``target_gym.oracles.controller.Oracle``:

- model: the env's step restated (``compressor_step``) with its own
  functions: the commands' clips removed and the moves bounded (T1); each
  command within the travel its actuator covers over the hold (T1r): past
  it the drive or valve runs at full rate every substep, as at it, so the
  bound loses nothing and keeps the plan off the full-rate plateau, where a
  command changes nothing and IPOPT stalled; the corner left inside the
  reach smoothed by softplus at the K5 widths (T5,
  ``data/oracle_widths/compressor_surge.json``); the valve behind its
  positioner's dead time, as the env passes it (``history``): the state
  carries the commands still in the positioner; the consumers' deviation on
  its noise-free mean path, a^t times its measured value;
- objective (version 3, charter Amendment 3): the recycle energy in kJ, and
  the header zone as a smooth hinge, the zone price per second past the band
  (T2);
- limit: the soft minimum of Phi over the step at least the surge line on
  every visited state, elastic at the trip cost per unit (T3): the oracle
  never surges, which bounds the counted-surge trip from the safe side. Not
  re-run since the rebuild;
- dropped guard: the characteristic's switch to its tangent past
  ``phi_join`` (T4, K9); every plan's substep Phi is checked against it.

The cold start holds the drive setpoint and the valve where they are.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial

from target_gym import records
from target_gym.compressor_surge.env import (
    KPA,
    _substeps,
    compute_velocity,
    live_target,
    recycle_flow,
    scheduled_setpoint,
    suction_density,
)
from target_gym.integration import integrate_dynamics
from target_gym.oracles.controller import Oracle, TaskModel, raw_of, softplus_clip
from target_gym.oracles.shooting import Limit
from target_gym.reward import trip_cost

#: The zone hinge's softplus width, Pa (ours, the owner's pick of 2026-11-11):
#: a tenth of a kPa against the 1.2 kPa band, so the plan charges about 1e-4
#: of a zone step at the setpoint, where the env charges nothing; at 1 kPa it
#: charged half a step there and pulled the header onto the setpoint.
ZONE_WIDTH = 0.1 * KPA


def commands(p, u):
    """The drive setpoint and valve opening the raw move ``u`` commands,
    unclipped (T1: the move is bounded instead)."""
    return p.N_min + 0.5 * (u[0] + 1.0) * (p.N_max - p.N_min), 0.5 * (u[1] + 1.0)


def reach(p, x, u, steps):
    """T1r, rows >= 0: each command within the travel its actuator covers
    over ``steps`` held steps from where it stands (env.py:511, :514)."""
    N_cmd, x_cmd = commands(p, u)
    hold = steps * p.delta_t
    dN, dx = N_cmd - x[3], x_cmd - x[4]
    return jnp.stack([
        hold * p.drive_rate - dN, hold * p.drive_rate + dN,
        hold * p.valve_open_rate - dx, hold * p.valve_close_rate + dx,
    ])  # fmt: skip


def reach_scale(p, steps):
    """Half the span of each raw move's reach over ``steps`` held steps."""
    hold = steps * p.delta_t
    return (
        float(2.0 * hold * p.drive_rate / (p.N_max - p.N_min)),
        float(hold * (p.valve_open_rate + p.valve_close_rate)),
    )


def history(p) -> int:
    """How many past valve commands the positioner can still pass: those
    issued within ``valve_lag`` of a step ago. The env keeps more
    (``VALVE_HISTORY``), but the newest command issued at least ``valve_lag``
    before a substep's end is among these (env.py, ``delayed``)."""
    return math.ceil(float(p.valve_lag) / float(p.delta_t)) + 1


def compressor_step(env, p, x, u, t, task, widths, deltas=None):
    """One env step from ``x`` = (m_c, dp, N, N_ramp, x, the last ``history``
    valve commands, newest first) under the raw move ``u``: ``(x_next,
    phi_min, the substep Phis, the rate limits' arguments)``. ``widths`` are
    the drive's and the valve's (0.0 is the exact clip), and ``deltas[i, j]``
    is added to site i's output at substep j (K5's probe)."""
    _, demand, clock, drift, time = task
    n, K = _substeps(env.integration_method), history(p)
    h, rho = p.delta_t / n, suction_density(p)
    N_cmd, x_cmd = commands(p, u)
    dev = drift * jnp.exp(-p.demand_theta * p.delta_t) ** t
    deltas = jnp.zeros((2, n)) if deltas is None else deltas
    # The positioner's queue, as the env's: each command with the time it was
    # issued, the step's start, so at the same float operations.
    cmds = jnp.concatenate([jnp.reshape(x_cmd, (1,)), x[5 : 5 + K]])
    issued = (time + t - jnp.arange(K + 1)) * p.delta_t
    t0 = (time + t) * p.delta_t

    def delayed(tau):
        ready = issued <= tau - p.valve_lag
        return cmds[jnp.where(jnp.any(ready), jnp.argmax(ready), K)]

    def substep(carry, j):
        m_c, dp, N, N_ramp, xv = carry
        args = jnp.stack([N_cmd - N_ramp, delayed(t0 + (j + 1) * h) - xv])
        r_N, lo_x, hi_x = (
            p.drive_rate * h,
            p.valve_close_rate * h,
            p.valve_open_rate * h,
        )
        N_ramp = N_ramp + softplus_clip(args[0], -r_N, r_N, widths[0]) + deltas[0, j]
        xv = xv + softplus_clip(args[1], -lo_x, hi_x, widths[1]) + deltas[1, j]
        velocity = partial(
            compute_velocity, action=None, N_ramp=N_ramp, x=xv, dev=dev,
            demand_levels=demand, params=p,
        )  # fmt: skip
        new, _ = integrate_dynamics(
            positions=jnp.stack([m_c, dp, N, (clock + t) * p.delta_t + j * h]),
            delta_t=h, method="rk4_1", compute_velocity=velocity,
        )  # fmt: skip
        phi = new[0] / (rho * p.A_c * p.U_r * new[2])
        return (new[0], new[1], new[2], N_ramp, xv), (phi, args)

    out, (phis, args) = jax.lax.scan(substep, tuple(x[:5]), jnp.arange(n))
    phi_min = -p.soft_min_temperature * jax.nn.logsumexp(-phis / p.soft_min_temperature)
    return jnp.concatenate([jnp.stack(out), cmds[:K]]), phi_min, phis, args


def compressor_model(env, p, zone_backoff: float = 0.0) -> TaskModel:
    """The planner problem; the zone's edge is planned ``zone_backoff`` Pa
    inside the band (K11 on the header, ``zone`` in the widths file)."""
    band = p.header_band - zone_backoff

    def step(d, x, u, t):
        w = d["widths"]  # in name order: drive_rate, valve_rate
        x1, phi_min, phis, _ = compressor_step(env, p, x, u, t, d["task"], (w[0], w[1]))
        setpoints, _, clock, *_ = d["task"]
        target = scheduled_setpoint(setpoints, 0.0, (clock + t + 1) * p.delta_t, p)
        dp = x1[1]
        power = recycle_flow(dp, x1[4], p) * dp / suction_density(p)
        cost = p.energy_price * power * p.delta_t / 1000.0
        # Each side's hinge on its own (K2's abs, p_w(x) + p_w(-x)): softplus of
        # abs keeps abs's corner at the setpoint, where the plan holds, and
        # the solver stalled on it. Each side's slope is the zone price per
        # kPa past the band.
        e, w = target - dp, ZONE_WIDTH
        past = jax.nn.softplus((e - band) / w) + jax.nn.softplus((-e - band) / w)
        cost = cost + p.zone_price * p.delta_t * (w / KPA) * past
        y = jnp.stack([phi_min])  # against the line 2W, the limit's bound
        return x1, cost, y, (dp / KPA)[None], p.phi_join - phis

    K = history(p)
    return TaskModel(
        nx=5 + K, nu=2, step=step,
        state_x=lambda s: jnp.concatenate(
            [jnp.stack([s.m_c, s.dp, s.N, s.N_ramp, s.x]), s.valve_cmds[:K]]
        ),
        data=lambda s: (
            s.setpoint_levels, s.demand_levels, s.block_clock, s.demand_dev, s.time,
        ),
        hold=lambda s: jnp.stack([raw_of(s.N_ramp, p.N_min, p.N_max), raw_of(s.x, 0.0, 1.0)]),
        tracked=lambda s: (s.dp / KPA)[None],
        resolution=(float(p.precision_floor),),
        limits=(Limit(lo=2.0 * p.W, price=float(trip_cost(p))),),
        move_rows=lambda x, u, steps: reach(p, x, u, steps),
        move_scale=lambda steps: reach_scale(p, steps),
    )  # fmt: skip


def make_compressor_surge_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``compressor_model``,
    its zone backed off by the widths file's measure."""
    zone = records.load_row("oracle_widths", "compressor_surge")["zone"]
    build = functools.partial(compressor_model, zone_backoff=zone["backoff_kpa"] * KPA)
    return Oracle(env, params, "compressor_surge", build)


#: The reward terms the ceiling's solver keeps (``pc_gym.foresight``):
#: the recycle energy; ``zone``, ``surge`` and the trip are the margins.
CEILING_TERMS = ("energy",)

#: The ceiling's margins' units: 0.1 kPa of header pressure, 0.01 of Phi.
ZONE_UNIT, PHI_UNIT = 0.1 * KPA, 0.01


def ceiling_margins(state, p):
    """The ceiling's margins (``pc_gym.foresight``): the header inside its zone,
    where ``zone`` charges nothing (``|target - dp| <= header_band``), in tenths
    of a kPa, backed off a tenth of one; and the flow forward on every substep,
    where no surge is counted (a surge is a reversal), as the step's soft
    minimum of Phi, which is below every substep's, in hundredths, backed off
    one. The price is a step outside the zone."""
    e = live_target(state, p) - state.dp
    zone = jnp.stack([p.header_band - e, p.header_band + e]) / ZONE_UNIT - 0.1
    surge = state.phi_min / PHI_UNIT - 1.0
    return jnp.concatenate([zone, surge[None]]), p.zone_price * p.delta_t
