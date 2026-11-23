"""The hold's job on ``plane3d_racetrack`` (owner's decision, 2026-11-23;
plane3d/PHYSICS.md and REALITY.md): the speed the aircraft holds at, the
maximum holding airspeed, and the fuel the hold costs.

Kept out of ``env.py``, which the patrol family's fingerprints read: a change
here re-records the four 3D airliner tasks only.
"""

import jax.numpy as jnp

from target_gym import reward as R
from target_gym.plane.dynamics import (
    calibrated_airspeed,
    economy_cost,
    scheduled_airspeed,
)
from target_gym.plane3d.env import (
    airspeed_3d,
    altitude_cost,
    check_is_terminal_3d,
    distance_to_racetrack,
    get_obs_racetrack,
    racetrack_guidance,
    wrap_angle,
)

#: Green dot, the A320's holding speed: "an estimate of the speed for best
#: lift-to-drag ratio", computed from the aircraft's weight to "minimize drag
#: and thus, the fuel consumption" in the hold [SF24]. Derived on this plant:
#: its level-flight thrust is least at 218.1 to 222.2 kt CAS at 73.5 t between
#: 3 000 and 8 000 m, and that speed goes as the square root of the mass
#: within 0.4 kt from 60 to 78 t (``scripts/plane3d_review.py polar``).
GREEN_DOT_CAS = 219.5 * 1852.0 / 3600.0
GREEN_DOT_MASS = 73_500.0


def holding_cap(target_altitude, params, xp=jnp):
    """The maximum holding airspeed at the hold's level, a CAS: 230 KIAS to
    14 000 ft and 265 above [AIM538, IFH ch. 10]."""
    low = target_altitude <= params.holding_split_altitude
    return xp.where(low, params.holding_cap_low, params.holding_cap_high)


def assigned_cas(mass, target_altitude, params, xp=jnp):
    """The CAS a hold is flown at: green dot for the aircraft's mass, or the
    maximum holding airspeed where that is less. Off a hold
    (``assigned_speed_mode`` 0), the cruise schedule's."""
    green_dot = GREEN_DOT_CAS * xp.sqrt(mass / GREEN_DOT_MASS)
    hold = xp.minimum(green_dot, holding_cap(target_altitude, params, xp))
    return xp.where(params.assigned_speed_mode == 1, hold, params.target_cas)


def assigned_tas(z, mass, target_altitude, params, xp=jnp):
    """The true airspeed of the assigned speed at height ``z``."""
    cas = assigned_cas(mass, target_altitude, params, xp)
    return scheduled_airspeed(z, cas, params.target_mach, xp)


def cap_cost(state, params, xp=jnp):
    """Past the maximum holding airspeed: the squared excess, in widths of
    ``speed_band``. Under it nothing: the sources give a maximum and no
    minimum, and the aircraft's own protection holds the low side."""
    cas = calibrated_airspeed(state.z, airspeed_3d(state, params, xp), xp)
    cap = holding_cap(state.target_altitude, params, xp)
    excess = xp.maximum(cas - cap, 0.0) / params.speed_band
    return xp.where(params.amended, excess**2, 0.0)


def fuel_cost(state, params, xp=jnp):
    """The fuel the hold burns, per second, in ``economy_scale`` kg (1 on the
    task, so kg). The time is ATC's, so nothing is charged per distance: the
    engine's flow alone, with ``cost_index`` 0 on the task."""
    tas = airspeed_3d(state, params, xp)
    return economy_cost(state.power, state.z, tas, tas, params, xp)


def reward_terms(state, params, xp=jnp, tripped=None):
    """The hold's additive cost terms, each >= 0: the bands of Amendments 2 to
    4 (level, path, the pattern's direction), the maximum holding airspeed,
    and the fuel."""
    path = R.tracking_cost(
        distance_to_racetrack(state),
        params.e_floor_path,
        params.e_tol_path,
        params.tracking_exponent,
        xp,
    )
    _, tangent, _ = racetrack_guidance(state)
    heading = R.tracking_cost(
        wrap_angle(state.psi - tangent),
        params.e_floor_heading,
        params.e_tol_heading,
        params.tracking_exponent,
        xp,
    )
    terms = {
        "tracking": altitude_cost(state, params, xp)
        + path
        + xp.where(params.amended, heading, 0.0),
        "running": cap_cost(state, params, xp),
        "economy": fuel_cost(state, params, xp),
    }
    if tripped is None:
        tripped, _ = check_is_terminal_3d(state, params, xp)
    return R.with_trip(terms, tripped, R.trip_cost(params), xp)


def observe(state, params, target_speed):
    """The hold's observation: the racetrack's, with the speed the aircraft
    holds at as its assigned speed, at index ``target_speed``."""
    obs = get_obs_racetrack(state, xp=jnp, params=params)
    speed = assigned_tas(state.z, state.m, state.target_altitude, params)
    return obs.at[target_speed].set(speed)
