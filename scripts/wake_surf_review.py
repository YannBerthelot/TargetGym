"""Measurements behind the 2026-11-23 review of ``wake_surf`` (patrol/REALITY.md).

Nothing here is registered or scored. On training seeds 100 to 109 it flies:

1. the shipped reference, with the wake and without it;
2. a candidate reference, the formation hold over the A320's own TECS tune
   (height on pitch, speed on thrust, as the 3D tasks fly it) with a slow
   spacing loop, at stations moved outboard of the shipped one and set on the
   vortex plane for their distance in trail;
3. a stand-in for the ceiling: the wake's best steady point held perfectly,
   from the wake model alone. It is a bound from a static argument, not an
   oracle, and it has no ride limit (REALITY.md, limits).

It prints each station's hold, its saving against the same seeds flown with
the wake off, and the candidate's cost over the stand-in's, each with a 95 %
interval over the ten paired seeds.

    uv run python scripts/wake_surf_review.py      # about 3 min on a server
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import target_gym as tg  # noqa: E402
from target_gym.patrol.env import (  # noqa: E402
    air_speed,
    compute_reward_terms_wake_surf,
    desired_slot_position,
    slot_delay,
    slot_error_vector,
    wake_air,
)
from target_gym.patrol.experts import (  # noqa: E402
    WAKE_SURF_HOLD,
    make_wake_surf_pid_controller,
)
from target_gym.plane.dynamics import (  # noqa: E402
    compute_air_density_from_altitude,
    compute_speed_of_sound_from_altitude,
    compute_thrust_output,
    trim_level,
)
from target_gym.plane.experts_tecs import AIRLINER  # noqa: E402

SEEDS = jnp.arange(100, 110)
STEPS, BURN_IN = 1800, 300
#: The candidate: the A320's one TECS tune, and a spacing loop slow enough to
#: leave the throttle alone (the Gulfstream's pilot held +-250 ft by hand).
CANDIDATE = WAKE_SURF_HOLD.replace(
    tecs=AIRLINER, k_ref=0.3, k_x=0.01, a_x=0.05, k_xi=0.0
)
STATIONS_OUT_M = (0.0, 7.5, 10.0, 12.5, 15.0, 20.0)


def start(env, params, seed, out_m, on_plane):
    """A reset with the station ``out_m`` further outboard and, with
    ``on_plane``, at the vortex's sink for its distance; the follower keeps
    its drawn offset from the station."""
    reset_key, run_key = jax.random.split(jax.random.PRNGKey(seed))
    _, state = env.reset_env(reset_key, params)
    drawn = jnp.stack(desired_slot_position(state))
    follower = state.follower
    offset = jnp.stack([follower.x, follower.y, follower.z]) - drawn
    speed = jnp.hypot(state.lead.x_dot, state.lead.y_dot)
    b0 = jnp.pi / 4.0 * params.wingspan
    sink = state.wake_history[0, 6] / (2.0 * jnp.pi * b0) * state.slot_back / speed
    state = state.replace(
        slot_up=jnp.where(on_plane, -sink, state.slot_up),
        slot_right=state.slot_right + jnp.sign(state.slot_right) * out_m,
    )
    x, y, z = jnp.stack(desired_slot_position(state)) + offset
    return state.replace(follower=follower.replace(x=x, y=y, z=z)), run_key


def fly(env, ctrl, gains, params, seed, out_m, on_plane):
    """One episode's per-step station errors, upwash, economy, bank and trips."""
    state, run_key = start(env, params, seed, out_m, on_plane)

    def step(carry, key):
        obs, state, memory = carry
        action, memory = ctrl.step(gains, memory, obs, jnp)
        speed = air_speed(state.lead, state.gust_history[0], params)
        follower = state.follower
        wake, _ = wake_air(
            state.wake_history,
            jnp.atleast_1d(follower.x),
            jnp.atleast_1d(follower.y),
            jnp.atleast_1d(follower.z),
            slot_delay(jnp.atleast_1d(state.slot_back), speed, params),
            params,
        )
        obs, state, _, _, info = env.step_env(key, state, action, params)
        _, right, up = slot_error_vector(state)
        economy = compute_reward_terms_wake_surf(state, params)["economy"]
        out = dict(right=right, up=up, upwash=wake[0, 2], economy=economy)
        out.update(bank=state.follower.phi, tripped=info["tripped"])
        return (obs, state, memory), out

    init = (env.get_obs(state, params), state, ctrl.init_state)
    return jax.lax.scan(step, init, jax.random.split(run_key, STEPS))[1]


def static_peak(env, params, seed):
    """Out of the wake and at its best steady point, for one seed's draw:
    ``(fuel flow in kg/s, the peak's share of it saved, the cost ratio)``."""
    state, _ = start(env, params, seed, 0.0, True)
    lead = state.lead
    speed, z = float(jnp.hypot(lead.x_dot, lead.y_dot)), float(lead.z)
    rho = float(compute_air_density_from_altitude(z))
    _, _, power = trim_level(speed, z, params.initial_mass, params)
    mach = speed / float(compute_speed_of_sound_from_altitude(z))
    sea_level = params.thrust_output_at_sea_level
    thrust = float(compute_thrust_output(power, sea_level, mach, rho))
    flow = float(params.specific_fuel_consumption) * 1e-3 * thrust
    x, y, z_slot = (float(v) for v in desired_slot_position(state))
    psi, side = float(lead.psi), float(jnp.sign(state.slot_right))
    lateral, height = np.meshgrid(np.arange(-12.0, 40.5, 0.5), np.arange(-8, 8.5, 0.5))
    across = side * lateral.ravel()
    gusts, _ = wake_air(
        state.wake_history,
        jnp.asarray(x + across * np.sin(psi)),
        jnp.asarray(y - across * np.cos(psi)),
        jnp.asarray(z_slot + height.ravel()),
        jnp.broadcast_to(slot_delay(state.slot_back, speed, params), across.shape),
        params,
    )
    # Upwash w rotates the lift forward by w / V (patrol/PHYSICS.md, the wake).
    saved = float(params.initial_mass * params.gravity) * float(gusts[:, 2].max())
    saved = saved / speed / thrust
    time_cost = float(params.cost_index) / 60.0
    return flow, saved, (flow * (1.0 - saved) + time_cost) / (flow + time_cost)


def interval(values):
    """Mean and 95 % half-width over the ten paired seeds (Student t, 9 dof)."""
    values = np.asarray(values, float)
    half = 2.262 * values.std(ddof=1) / np.sqrt(len(values))
    return f"{values.mean():.4f} +-{half:.4f}"


def main() -> int:
    env, params = tg.make("wake_surf")
    ctrl = make_wake_surf_pid_controller()

    @jax.jit
    def flown(gains, wake_strength, out_m, on_plane):
        p = params.replace(wake_strength=wake_strength)
        one = lambda seed: fly(env, ctrl, gains, p, seed, out_m, on_plane)  # noqa: E731
        return jax.vmap(one)(SEEDS)

    def row(label, out):
        held = {k: np.asarray(v)[:, BURN_IN:] for k, v in out.items()}
        rms = lambda a: np.sqrt((a**2).mean(1)).mean()  # noqa: E731
        bank = np.rad2deg(held["bank"])
        print(
            f"{label:<34} lateral {rms(held['right']):6.1f} m  height "
            f"{rms(held['up']):6.1f} m  bank rms {rms(bank):5.1f} max "
            f"{np.abs(bank).max():5.1f} deg  upwash {held['upwash'].mean():5.2f} m/s"
            f"  trips {int(np.asarray(out['tripped']).sum())}",
            flush=True,
        )
        return held["economy"].mean(1)

    row("shipped reference", flown(WAKE_SURF_HOLD, 1.0, 0.0, False))
    row("shipped reference, wake off", flown(WAKE_SURF_HOLD, 0.0, 0.0, False))
    wake_off = row("candidate, wake off", flown(CANDIDATE, 0.0, 0.0, True))

    flow, saved, ratio = map(
        np.array, zip(*(static_peak(env, params, int(s)) for s in SEEDS))
    )
    ceiling = wake_off * ratio
    print(
        f"static peak: {100 * saved.mean():.1f} % of fuel flow "
        f"({100 * saved.min():.1f} to {100 * saved.max():.1f}); fuel flow out of "
        f"the wake {3600 * flow.mean():.0f} kg/h"
    )
    fuel_share = flow / (flow + float(params.cost_index) / 60.0)
    for out_m in STATIONS_OUT_M:
        economy = row(
            f"candidate, {out_m:4.1f} m outboard", flown(CANDIDATE, 1.0, out_m, True)
        )
        cost_saving = 1.0 - economy / wake_off
        print(
            f"    cost saving {interval(cost_saving)}   fuel saving "
            f"{interval(cost_saving / fuel_share)}   cost over the static "
            f"stand-in {interval(economy / ceiling)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
