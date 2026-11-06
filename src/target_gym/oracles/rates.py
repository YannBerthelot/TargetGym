"""Planning through an actuator's rate limit, by rate.

``base.actuate`` closes a share of the gap to its command and clips that move
to the rate limit. A planner on commands sees nothing past the clip: on
``glass_furnace`` the fuel stuck, 8.84 K off the crown (PR 158).

So a plan's move on such an actuator is its rate, v in [-1, 1] of the limit, held
over the T_c (docs/proposals/oracle-smoothing-protocol.md, Amendment 3). The
position is a state of the plan, ``reached`` moves it by exactly v of the limit,
and on each task step the env is sent ``command``, the action that does that,
through ``TaskModel.action``. No clip of ``actuate`` binds on it, so no site is
left to smooth (T1r's corner goes), and ``rows`` keeps every command of the ramp
in the action box (T1). At a hold of one step every path the env allows is such
a plan. Over longer holds the plan reaches the same positions at each T_c, by a
ramp where a held command would arrive and wait.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class RateLimit(NamedTuple):
    """``base.actuate`` in a plan's moves, in raw action units: each actuator's
    ``travel`` per step at its limit, and the share ``closes`` of a gap that its
    lag closes per step (1 with no lag)."""

    travel: jnp.ndarray
    closes: jnp.ndarray
    low: jnp.ndarray
    high: jnp.ndarray

    @classmethod
    def of(cls, env, params) -> RateLimit:
        if float(params.actuator_dead_band) or float(params.actuator_stiction):
            raise ValueError("a dead band or stiction leaves no rate plan")
        space = env.action_space(params)
        rate = jnp.asarray(params.actuator_rate)
        lag = jnp.asarray(params.actuator_lag)
        if not bool(jnp.all(rate > 0)):
            raise ValueError("an actuator with no rate limit has no rate plan")
        dt = getattr(env, "control_period", 1) * params.delta_t
        closes = jnp.where(
            lag > 0, 1.0 - jnp.exp(-dt / jnp.where(lag > 0, lag, 1.0)), 1.0
        )
        low, high = jnp.asarray(space.low), jnp.asarray(space.high)
        shape = jnp.broadcast_shapes(rate.shape, low.shape)
        return cls(
            jnp.broadcast_to(rate * (high - low) * dt, shape),
            jnp.broadcast_to(closes, shape),
            jnp.broadcast_to(low, shape),
            jnp.broadcast_to(high, shape),
        )

    def reached(self, held, v):
        """The position after one step at rate ``v``."""
        return held + v * self.travel

    def command(self, held, v):
        """The raw action that moves ``held`` by ``v`` of the limit in one step."""
        return held + v * self.travel / self.closes

    def move(self, held, command):
        """The rate the env gives ``command`` from ``held``: the plan's move for
        a step someone else commanded, such as the expert's at a K5 probe."""
        return jnp.clip(self.closes * (command - held) / self.travel, -1.0, 1.0)

    def rows(self, held, v, steps):
        """Rows >= 0, affine in ``held`` and ``v``: the last command of a ramp
        held ``steps`` steps, the furthest out, stays in the box."""
        last = held + v * self.travel * (steps - 1.0 + 1.0 / self.closes)
        return jnp.concatenate([self.high - last, last - self.low])
