"""The aircraft family as a continuing plant (docs/proposals/long-strata.md, S1, S1b).

In a stream (``params.stream_mode``) an aircraft task flies a sequence of sorties.
The stream starts every event through the task's ``start_event`` hook at the
scenario's total rate (``Scenario.events_per_h``); the hook picks the kind by
``params.event_mix``, each kind's share of that rate. Thinning one Poisson clock
by fixed shares gives each kind a Poisson clock of its own, so sortie renewals,
turbulence zones and the task's own events (a clearance, a vector) each keep
their rate, independently of the others.

- A sortie starts established in cruise with a fresh mass and level (S1b). Its
  mass is held: in a stream the burn does not lower it. The weather carries on.
- A turbulence zone carries its intensity and its end in the state, so it runs
  to its end through a scenario switch. A zone that starts inside another merges
  with it: the later end and the stronger intensity.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = [
    "SORTIE",
    "TASK",
    "ZONE",
    "carry",
    "event_kind",
    "fuel_for",
    "in_stream",
    "in_window",
    "pick",
    "sortie_key",
    "sortie_mass",
    "start_zone",
]


def in_stream(params, stream_value, window_value, xp=jnp):
    """``stream_value`` in a stream (``params.stream_mode``), else
    ``window_value``. Where the params are constants (closed over, a Python
    bool) the choice is made in Python, so a window's graph is the one it had:
    a select there would turn constants XLA folds into arrays it does not,
    which moves the last bit of a window."""
    flag = params.stream_mode
    if isinstance(flag, bool):
        return stream_value if flag else window_value
    return xp.where(flag, stream_value, window_value)


def in_window(params) -> bool:
    """Whether ``params`` are a window's, known in Python: the window's graph
    must then be built exactly as it was (``in_stream``)."""
    flag = params.stream_mode
    return isinstance(flag, bool) and not flag


#: The event kinds, in the order of ``params.event_mix``.
SORTIE, ZONE, TASK = 0, 1, 2
#: Salt for a sortie's own draws, apart from every draw a reset makes.
_SORTIE = 0x534F5254


def event_kind(key, params):
    """Which kind of event the stream started: ``SORTIE``, ``ZONE`` or ``TASK``,
    with chances ``params.event_mix``."""
    sortie, zone, _ = params.event_mix
    u = jax.random.uniform(key)
    return jnp.where(u < sortie, SORTIE, jnp.where(u < sortie + zone, ZONE, TASK))


def pick(kind, sortie, zone, task):
    """The state after the event of ``kind``: one of three states of one shape."""
    return jax.tree_util.tree_map(
        lambda a, b, c: jnp.where(kind == SORTIE, a, jnp.where(kind == ZONE, b, c)),
        sortie,
        zone,
        task,
    )


def start_zone(start, end, sigma, now, key, params):
    """``(start, end, sigma)`` after a turbulence zone starts at ``now`` (s): the
    scenario's intensity ``params.turbulence_sigma``, for a length drawn over
    ``params.zone_duration_range``. Inside a zone, the two merge."""
    lo, hi = params.zone_duration_range
    length = jax.random.uniform(key, minval=lo, maxval=hi)
    inside = (now >= start) & (now < end)
    new = jnp.asarray(params.turbulence_sigma, jnp.float32)
    return (
        jnp.where(inside, start, now),
        jnp.where(inside, jnp.maximum(end, now + length), now + length),
        jnp.where(inside, jnp.maximum(sigma, new), new),
    )


def sortie_key(key):
    """The key a reset draws a sortie's mass from, apart from its other draws."""
    return jax.random.fold_in(key, _SORTIE)


def sortie_mass(key, params):
    """A sortie's all-up mass, kg: uniform over ``params.sortie_mass_range``."""
    lo, hi = params.sortie_mass_range
    return jax.random.uniform(key, minval=lo, maxval=hi)


def fuel_for(mass, params):
    """The fuel that gives the all-up mass ``mass``: the model's mass is its
    zero-fuel mass, ``initial_mass - initial_fuel_quantity``, plus the fuel, so a
    sortie heavier than ``initial_mass`` carries more than the tanks' figure.
    Read it as payload and fuel together; in a stream it is only the mass."""
    return mass - (params.initial_mass - params.initial_fuel_quantity)


def carry(fresh, state, fields):
    """``fresh`` with ``state``'s clock, instruments and ``fields``: the world a
    new sortie flies into."""
    kept = {f: getattr(state, f) for f in fields}
    return fresh.replace(time=state.time, sensor_key=state.sensor_key, **kept)
