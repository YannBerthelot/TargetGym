"""The oracle (the MPC slot) for the battery task, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem, by the
smoothing protocol's inventory (docs/proposals/oracle-smoothing-protocol.md,
section 5.4, and Amendment 3), run by ``target_gym.oracles.controller.Oracle``:

- model: the env's own ``compute_velocity`` and integrator, with the raw move's
  clip removed and the move bounded instead (T1); the fade state left out, since
  it feeds nothing but the reward; the regulation signal on its noise-free mean
  path, its AR(1) decaying onto the blocks' levels; the half-cycle under way,
  ``half_cycles`` below;
- objective: the imbalance at the meter through epigraph rows on every visited
  state, linear in |error| since e_tol = 0 and p = 1 (T2); the ripple's
  |current| likewise (T2); the calendar fade, smooth; each half-cycle's depth by
  the published stress function, its running peak an epigraph: the peak's growth
  is a second move, >= 0, and a row keeps the peak past the charge on every
  visited state, so it costs what ``half_cycle`` charges (T2). Fade stays above
  ``c_hold`` so that max(fade - c_hold, 0) is the fade less ``c_hold`` (T4, the
  guard below);
- a half-cycle's turn is a jump, in what it charges and in its anchor, so no
  plan may cross one: each turn is fixed where the planned signal reverses,
  once a reference following it has swung past ``cycle_hysteresis``, and each
  closed swing where the reference runs back past it (class X, from the solve's
  start, ours). The plan carries the two open turns nearest; a third comes from
  the reference. A plan that turns only there meets the env's rainflow count; K8
  reports any other's gap. The charge a new half-cycle jumps to at the hysteresis
  is blended by the logistic at the K5 width (T5, ``data/oracle_widths/
  battery.json``);
- limits: the state of charge, cell voltage and temperature, on every visited
  state, elastic at the trip cost per unit (T3); the inverter's draw within 0.999
  of the deliverable power at each step's end (T1s);
- dropped guards (T4): the charge, discriminant and power clips of
  ``current_for_power`` and the charge clip of ``open_circuit_voltage``, which
  those limits keep off their edges.

The reward is in dollars, so the tie-break's budget is what half a resolution
costs in one step in dollars (G10). The cold start holds the delivered power.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial

from target_gym.energy.battery.env import (
    compute_velocity,
    current_for_power,
    cycle_events,
    cycle_stress,
    degradation_rate,
    dispatch_block,
    half_cycle,
    inverter_input,
    inverter_output,
    max_deliverable_power,
    terminal_voltage,
)
from target_gym.integration import integrate_dynamics
from target_gym.oracles.controller import Oracle, TaskModel, raw_of
from target_gym.oracles.shooting import Hinge, Limit
from target_gym.reward import trip_cost


class Cycles(NamedTuple):
    """Per step of a plan, what a reference following the planned signal does to
    the half-cycle: whether it turns as that step leaves its state, the heading
    after it, whether it closes a swing there, and then the two open turns that
    close uncovers, the nearest first."""

    turned: jnp.ndarray
    heading: jnp.ndarray
    folded: jnp.ndarray
    below: jnp.ndarray


def half_cycles(p, soc, anchor, peak, turns, count, signal) -> Cycles:
    """The reference's half-cycles: it follows ``signal`` through ``half_cycle``'s
    own rainflow count, and a turn it finds goes back to the peak it left, where
    the charge reversed."""
    per_W = p.delta_t / (p.pack_kWh * 3.6e6)
    n = signal.size

    def walk(c, t):
        soc, anchor, peak, stack, count, at, heading, first, marks, folds, below = c
        soc = jnp.clip(soc - per_W * (signal[t] + p.aux_load), p.soc_min, p.soc_max)
        extends, turned, closes = cycle_events(anchor, peak, stack, count, soc, p)
        fold = closes & (count > 1)
        marks = marks.at[at].add(turned.astype(marks.dtype))
        first = jnp.where(first == 0.0, turned * jnp.sign(soc - peak), first)
        heading = jnp.where(turned, jnp.sign(soc - peak), heading)
        # The last step at the peak: the reference rests there at the window.
        at = jnp.where(extends | turned | (soc == peak), t + 1, at)
        anchor, peak, stack, count, _ = half_cycle(anchor, peak, stack, count, soc, p)
        folds = folds.at[t].set(fold.astype(folds.dtype))
        left = jnp.stack([stack[count - 1], stack[count - 2]])
        below = below.at[t].set(left)
        return (
            soc,
            anchor,
            peak,
            stack,
            count,
            at,
            heading,
            first,
            marks,
            folds,
            below,
        ), 0

    zeros = jnp.zeros(n + 1, soc.dtype)
    start = jnp.sign(peak - anchor)
    carry = (
        soc, anchor, peak, turns, count, 0, start, 0.0 * soc, zeros,
        jnp.zeros(n, soc.dtype), jnp.zeros((n, 2), soc.dtype),
    )  # fmt: skip
    (*_, first, marks, folds, below), _ = jax.lax.scan(walk, carry, jnp.arange(n))
    before = jnp.where(start == 0.0, -first, start)
    flips = jnp.cumsum(marks[:-1])
    return Cycles(marks[:-1], before * (1.0 - 2.0 * jnp.mod(flips, 2.0)), folds, below)


def battery_model(env, p) -> TaskModel:
    assert float(p.e_tol) == 0.0 and float(p.tracking_exponent) == 1.0
    per_W = float(p.imbalance_price / 1.0e6 * p.delta_t / 3600.0)
    floor = per_W * float(p.e_floor)
    dollars = float(p.fade_price * p.pack_kWh)  # per unit of fade
    n = int(p.max_steps_in_episode)
    decay = jnp.exp(-p.delta_t / p.regulation_tau)
    #: The most the charge moves in a step, as the peak's growth at raw move 1.
    grows = 2.0 * p.power_max * p.delta_t / (p.pack_kWh * 3.6e6)

    def velocity(x, action, params):
        return compute_velocity(x, action, params)[0][:3], None

    def charge(depth, width):
        """Phi(depth), but for a half-cycle still under the hysteresis, which
        ``half_cycle`` has not charged: the exact step at ``width`` = 0."""
        over = jnp.abs(depth) - p.cycle_hysteresis
        w = jnp.where(width > 0.0, width, 1.0)
        step = jnp.where(width > 0.0, jax.nn.sigmoid(over / w), over > 0.0)
        return cycle_stress(depth, p) * step

    def step(d, x, u, t):
        signal, cycles = d["task"]
        width = d["widths"][0]
        command = u[0] * p.power_max  # T1
        positions, _ = integrate_dynamics(
            positions=x[:3], delta_t=p.delta_t, method=env.integration_method,
            compute_velocity=partial(velocity, action=command, params=p),
        )  # fmt: skip
        soc, v_rc, T = positions
        current = current_for_power(inverter_input(command, p), soc, v_rc, p)
        V = terminal_voltage(current, soc, v_rc, p)
        power = inverter_output(V * current, p) - p.aux_load

        old_anchor, old_peak, last, outer = x[3], x[4], x[5], x[6]
        turned, h, folded = cycles.turned[t], cycles.heading[t], cycles.folded[t]
        grown = h * 0.5 * (u[1] + 1.0) * grows
        stays = 1.0 - turned - folded
        anchor = stays * old_anchor + turned * old_peak + folded * outer
        peak = grown + stays * old_peak + turned * x[0] + folded * last
        now = charge(peak - anchor, width)
        was = charge(old_peak - old_anchor, width)
        closing = charge(last - old_anchor, width) - was + now
        life = 0.5 * (
            stays * (now - was) + turned * now
            + folded * (closing - charge(last - outer, width))
        )  # fmt: skip
        stack = (
            stays * jnp.stack([last, outer])
            + turned * jnp.stack([old_anchor, last])
            + folded * cycles.below[t]
        )

        calendar = degradation_rate(0.0, T, p)  # a literal current: no abs
        ripple = (degradation_rate(1.0, T, p) - calendar) * current * p.delta_t
        fade = calendar * p.delta_t + p.life_fade * life - p.c_hold
        room = 0.999 * max_deliverable_power(soc, v_rc, p)
        drawn = inverter_input(command, p)
        y = jnp.stack([
            (signal[t] - power) / p.e_floor,
            dollars * ripple / floor,
            soc, V / p.n_series, T,
            (room - drawn) / p.power_max, (room + drawn) / p.power_max,
            h * (peak - soc),
        ])  # fmt: skip
        guard = (fade - p.life_fade * life + jnp.abs(ripple)) / p.c_hold + 1e-9
        x1 = jnp.concatenate([positions, jnp.stack([anchor, peak]), stack])
        return x1, dollars * fade, y, power[None], (guard,)

    @jax.jit
    def data(s):
        ts = jnp.arange(n)
        levels = s.dispatch_schedule[dispatch_block(s.time + 1 + ts, p)]
        signal = jnp.clip(
            levels + s.regulation * decay ** (ts + 1.0), -p.power_max, p.power_max
        )
        cycles = half_cycles(
            p, s.soc, s.soc_anchor, s.soc_peak, s.soc_turns, s.n_turns, signal
        )
        return signal, cycles

    price = float(trip_cost(p))
    return TaskModel(
        nx=7, nu=2, step=step,
        state_x=lambda s: jnp.stack([
            s.soc, s.v_rc, s.T_cell, s.soc_anchor, s.soc_peak,
            s.soc_turns[s.n_turns - 1], s.soc_turns[s.n_turns - 2],
        ]),
        data=data,
        hold=lambda s: jnp.stack([
            raw_of(s.power + p.aux_load, -p.power_max, p.power_max),
            -jnp.ones_like(s.soc),
        ]),
        action=lambda s, u: u[:1],
        tracked=lambda s: s.power[None],
        resolution=(float(p.precision_floor),),
        hinges=(Hinge(weight=floor, absolute=True), Hinge(weight=floor, absolute=True)),
        limits=(
            Limit(p.soc_min, p.soc_max, price),
            Limit(p.V_cell_min, p.V_cell_max, price),
            Limit(hi=p.T_max, price=price),
            Limit(lo=0.0, price=price), Limit(lo=0.0, price=price),
            Limit(lo=0.0, price=price),
        ),
        tie_budget=per_W * 0.5 * float(p.precision_floor),
    )  # fmt: skip


def make_battery_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``battery_model``."""
    return Oracle(env, params, "battery", battery_model)
