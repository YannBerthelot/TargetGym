"""Prototype of ``wake_surf``'s next version. Nothing here is registered or scored.

The owner asked for the design on 2026-11-23 (patrol/REALITY.md's property card):
a wake that drifts and is predicted, never measured; a ride limit; the trials'
distance in trail; and an oracle. Each element is a number in ``Design``, so its
effect on the reference's cost can be read alone.

- ``drift``: the wake is carried by the air the lead flew in. The plant already
  freezes the lead's gust and hands it to the follower a time in trail later
  (Taylor); here each wake sample moves with the gust the lead met where it left
  it, so the wake meanders by that gust times its age. No new number.
- ``wind_sigma``: the station is a prediction, the lead's track plus the
  follower's wind estimate times the time in trail [REALITY.md, N1 III.B]. The
  estimate is off by an error drawn once an episode, so the station is off by
  that error times the time in trail, across the track.
- The ride limit is the wake crossing [REALITY.md, N4 VI.D]: the follower's
  centreline inboard of the near vortex. It is counted here and would be a trip.
  The cabin vibration NASA measured is at 16 to 28 Hz and does not grow toward
  the core, so a plant stepped at 1 Hz neither makes it nor needs it as a limit.
- ``back_lo``, ``back_hi``: the distance in trail, m.
- ``spans``: the station, in spans from the predicted wake's centreline.

The reference. Its station follows the charter's zone rule (J1): the edge, the
closest station the hold keeps inside the protected bank with the wake known
and still, moved out by two standard deviations of the wake's own drift. Where
the station is only predicted, the reference does what NASA's crews did
[REALITY.md, N1 III.D]: it moves in or out until its roll trim, the filtered
aileron, sits at the level that station gives (``seek_station``), and it starts
the window established there.

The ceiling. No oracle has measured it, so two stand-ins bracket it. Below the
reference: the same hold told the wind and the wake's drift 4 s ahead, at the
closest station it keeps on the scored seeds with no crossing (a policy, so the
ceiling costs no more). Far below: the wake's best steady point held perfectly
(no policy does better at rest).

    uv run python scripts/wake_surf_next.py        # on a server, about 25 min
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

warnings.filterwarnings("ignore")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import target_gym as tg  # noqa: E402
from target_gym.patrol.env import (  # noqa: E402
    air_speed,
    at_delay,
    compute_reward_terms_wake_surf,
    desired_slot_position,
    slot_delay,
    slot_error_vector,
    wake_air,
    wake_sink,
)
from target_gym.patrol.experts import make_wake_surf_pid_controller  # noqa: E402
from target_gym.plane.dynamics import (  # noqa: E402
    compute_air_density_from_altitude,
    compute_speed_of_sound_from_altitude,
    compute_thrust_output,
    trim_level,
)

STEPS, BURN_IN = 1800, 300
TRAINING, HELD_OUT = jnp.arange(100, 110), jnp.arange(200, 220)
_WIND_ERROR = 7  # the key fold of the wind estimate's error
#: The wind estimate's error on each horizontal axis, m/s (ours). Family: 0.5,
#: the airspeed's share alone, to about 2, a 2 to 3 m/s vector over two axes
#: [WMO-No. 8, aircraft-based observations, 3.4.1].
WIND_SIGMA = 1.0
#: 34 to 50 spans: the trials' 4,000 ft at the near end, the 50 spans NASA calls
#: the larger end of the usable range at the far one [REALITY.md, N1 III.D].
TRIALS = dict(back_lo=1_200.0, back_hi=1_800.0)
#: The near vortex, in spans from the wake's centreline (``b0 / 2``).
CORE = np.pi / 8.0
SPAN = 35.8


#: The roll trim's filter, s, and the station's rate per radian of trim past
#: the level sought, m/s (ours): the trim loop closes in about 2 min, four
#: times the filter, as NASA stepped inward and let each step settle
#: [REALITY.md, N1 III.D].
TRIM_FILTER_S, TRIM_RATE = 30.0, 3.5
#: How far the trim loop may move the station either way, m: the box's width.
TRIM_REACH = 30.0


class Design(NamedTuple):
    """One variant of the task; the defaults are ``wake_surf-v3``."""

    spans: float = 1.25
    drift: float = 0.0
    wind_sigma: float = 0.0  # m/s, each horizontal axis
    back_lo: float = 2_800.0
    back_hi: float = 3_700.0
    wake: float = 1.0
    #: The roll trim the reference seeks, rad of aileron; 0 leaves the station
    #: where the prediction puts it (``seek_station``).
    trim: float = 0.0
    #: Seconds of foresight of the wake's drift handed to the hold; below 0,
    #: none. A stand-in for a ceiling: the wake about to arrive is already in
    #: the air, but no instrument reads it.
    preview: float = -1.0


def start(env, params, seed, design):
    """A reset moved onto ``design``: its distance in trail (the seed's own
    draw, rescaled), its station off by the wind estimate's error, the
    follower at its drawn offset from that station. A reference that seeks a
    roll trim starts established (charter B5): where its ingress would have
    left it, the station's error already taken out. Returns the state, the
    run's key and how far outboard that reference has moved its station."""
    reset_key, run_key = jax.random.split(jax.random.PRNGKey(seed))
    _, state = env.reset_env(reset_key, params)
    drawn = jnp.stack(desired_slot_position(state))
    follower, lead = state.follower, state.lead
    offset = jnp.stack([follower.x, follower.y, follower.z]) - drawn
    lo, hi = params.slot_back_range
    share = (state.slot_back - lo) / (hi - lo)
    back = design.back_lo + share * (design.back_hi - design.back_lo)
    speed = jnp.hypot(lead.x_dot, lead.y_dot)
    error = design.wind_sigma * jax.random.normal(
        jax.random.fold_in(reset_key, _WIND_ERROR), (2,)
    )
    to_right = error[0] * jnp.sin(lead.psi) - error[1] * jnp.cos(lead.psi)
    side = jnp.sign(state.slot_right)
    state = state.replace(
        slot_back=back,
        slot_right=side * design.spans * params.wingspan + to_right * back / speed,
        slot_up=-wake_sink(lead, back, params),
    )
    off = jnp.where(design.trim > 0.0, to_right * back / speed, 0.0)
    off = jnp.clip(off, -TRIM_REACH, TRIM_REACH)
    x, y, z = jnp.stack(desired_slot_position(state)) + offset
    x, y = x - off * jnp.sin(lead.psi), y + off * jnp.cos(lead.psi)
    state = state.replace(follower=follower.replace(x=x, y=y, z=z))
    return state, run_key, -side * off


def wake_frame(state, speed, params):
    """How far outboard of the wake's own centreline the follower is, as
    ``wake_air`` reads it, m."""
    delay = slot_delay(jnp.atleast_1d(state.slot_back), speed, params)
    px, py, _, rx, ry, _, _ = jnp.moveaxis(at_delay(state.wake_history, delay), -1, 0)
    follower = state.follower
    across = (follower.x - px) * rx + (follower.y - py) * ry
    return (jnp.sign(state.slot_right) * across)[0]


def seek_station(design, moved, trim, aileron, time, dt):
    """NASA's ingress as a loop: the station moves outboard while the filtered
    aileron holds more roll trim than the level sought, inboard while less. It
    waits two filter times for the filter to fill."""
    trim = trim + dt / TRIM_FILTER_S * (aileron - trim)
    rate = TRIM_RATE * (jnp.abs(trim) - design.trim)
    rate = jnp.where(time * dt < 2.0 * TRIM_FILTER_S, 0.0, rate)
    moved = jnp.clip(moved + dt * rate, -TRIM_REACH, TRIM_REACH)
    return jnp.where(design.trim > 0.0, moved, 0.0), trim


def drift_ahead(state, speed, params, ahead):
    """How far to the lead's right the wake arriving ``ahead`` seconds from
    now will have drifted, m: its age then times the gust it was left in."""
    delay = slot_delay(jnp.atleast_1d(state.slot_back), speed, params)
    gust = at_delay(state.gust_history, delay - ahead / params.delta_t)[0]
    lead = state.lead
    to_right = gust[0] * jnp.sin(lead.psi) - gust[1] * jnp.cos(lead.psi)
    return to_right * delay[0] * params.delta_t


def fly(env, ctrl, params, seed, design):
    """One episode of the reference on ``design``, per step."""
    params = params.replace(wake_strength=design.wake)
    state, run_key, moved = start(env, params, seed, design)

    def step(carry, key):
        obs, state, memory, moved, trim = carry
        speed = air_speed(state.lead, state.gust_history[0], params)
        outboard = wake_frame(state, speed, params)
        side = jnp.sign(state.slot_right)
        moved, trim = seek_station(
            design, moved, trim, state.follower.aileron, state.time, params.delta_t
        )
        foreseen = drift_ahead(state, speed, params, jnp.maximum(design.preview, 0.0))
        foreseen = jnp.where(design.preview >= 0.0, design.drift * foreseen, 0.0)
        shift = side * moved + foreseen
        action, memory = ctrl.step(ctrl.params, memory, obs.at[11].add(-shift), jnp)
        obs, state, _, _, info = env.step_env(key, state, action, params)
        # The wake moves with the air it was left in (``drift``).
        carried = design.drift * params.delta_t * state.gust_history
        state = state.replace(wake_history=state.wake_history.at[:, :3].add(carried))
        terms = compute_reward_terms_wake_surf(state, params)
        out = dict(
            outboard=outboard,
            back=slot_error_vector(state)[0],
            economy=terms["economy"],
            tracking=terms["tracking"],
            bank=state.follower.phi,
            tripped=info["tripped"],
            trim=jnp.abs(trim),
            moved=moved,
        )
        return (obs, state, memory, moved, trim), out

    init = (env.get_obs(state, params), state, ctrl.init_state, moved, 0.0 * moved)
    return jax.lax.scan(step, init, jax.random.split(run_key, STEPS))[1]


def static_ratio(env, params, seed, design):
    """The steady cost at the wake's best point over the steady cost out of
    the wake, for one seed's draw: upwash ``w`` rotates the lift forward by
    ``w / V`` (patrol/PHYSICS.md, the wake)."""
    state = start(env, params, seed, design._replace(wind_sigma=0.0, trim=0.0))[0]
    lead = state.lead
    speed, z = float(jnp.hypot(lead.x_dot, lead.y_dot)), float(lead.z)
    rho = float(compute_air_density_from_altitude(z))
    _, _, power = trim_level(speed, z, params.initial_mass, params)
    mach = speed / float(compute_speed_of_sound_from_altitude(z))
    sea_level = params.thrust_output_at_sea_level
    drag = float(compute_thrust_output(power, sea_level, mach, rho))
    flow = float(params.specific_fuel_consumption) * 1e-3 * drag
    x, y, z_slot = (float(v) for v in desired_slot_position(state))
    psi, side = float(lead.psi), float(jnp.sign(state.slot_right))
    lateral, height = np.meshgrid(np.arange(-40.0, 10.5, 0.5), np.arange(-6, 6.5, 0.5))
    across = side * lateral.ravel()
    gusts, _ = wake_air(
        state.wake_history,
        jnp.asarray(x + across * np.sin(psi)),
        jnp.asarray(y - across * np.cos(psi)),
        jnp.asarray(z_slot + height.ravel()),
        jnp.broadcast_to(slot_delay(state.slot_back, speed, params), across.shape),
        params,
    )
    weight = float(params.initial_mass * params.gravity)
    saved = weight * float(gusts[:, 2].max()) / speed / drag
    time_cost = float(params.cost_index) / 60.0
    return (flow * (1.0 - saved) + time_cost) / (flow + time_cost)


def interval(values):
    """Mean and 95 % half-width over paired seeds (Student t)."""
    from scipy import stats

    values = np.asarray(values, float)
    t = stats.t.ppf(0.975, len(values) - 1)
    return f"{values.mean():.4f} +-{t * values.std(ddof=1) / np.sqrt(len(values)):.4f}"


def summary(out):
    """Per seed: the mean cost past the burn-in, the follower's true place in
    the wake, and whether it crossed the wake or tripped."""
    held = {k: np.asarray(v)[:, BURN_IN:] for k, v in out.items()}
    outboard = np.asarray(out["outboard"]) / SPAN
    return dict(
        cost=(held["economy"] + held["tracking"]).mean(1),
        outboard=held["outboard"],
        bank_max=np.rad2deg(np.abs(held["bank"]).max(1)),
        back_max=np.abs(held["back"]).max(1),
        crossed=(outboard < CORE).any(1),
        tripped=np.asarray(out["tripped"]).any(1),
        trim=np.rad2deg(held["trim"].mean(1)),
        moved=held["moved"].mean(1),
    )


def row(label, s):
    lost = int((s["crossed"] | s["tripped"]).sum())
    print(
        f"{label:<46} cost {s['cost'].mean():.4f}  in the wake at "
        f"{s['outboard'].mean() / SPAN:.2f} b, sd {s['outboard'].std():4.1f} m  "
        f"bank max {s['bank_max'].max():4.1f} deg  along max "
        f"{s['back_max'].max():3.0f} m  roll trim {s['trim'].mean():.2f} deg  "
        f"seeds crossing or tripping {lost} of {len(s['cost'])}",
        flush=True,
    )


def main() -> int:
    env, params = tg.make("wake_surf")
    ctrl = make_wake_surf_pid_controller()
    bank_limit = float(np.rad2deg(params.protected_bank))

    @jax.jit
    def flown(seeds, design):
        return jax.vmap(lambda seed: fly(env, ctrl, params, seed, design))(seeds)

    def run(design, seeds=TRAINING, label=None):
        s = summary(flown(seeds, Design(*map(float, design))))
        if label:
            row(label, s)
        return s

    def clean(s):
        return not (s["crossed"] | s["tripped"]).any()

    def edge(base):
        """The closest station, in 0.05 span steps, the reference holds on
        every training seed inside the protected bank, the wake known and still."""
        still = base._replace(drift=0.0, wind_sigma=0.0)
        for spans in np.arange(1.0, 1.5, 0.05):
            s = run(still._replace(spans=spans))
            if clean(s) and s["bank_max"].max() < bank_limit:
                return float(spans)
        raise RuntimeError("no station held")

    def told(base, station, ref):
        """The stand-in: the hold told the wind and the wake's drift 4 s ahead,
        at the station at or inboard of the reference's, in 0.05 span steps,
        that costs least among those it keeps on these very seeds with no
        crossing. The reference itself if none does better."""
        known = base._replace(wind_sigma=0.0, preview=4.0)
        best, at = ref, None
        for spans in np.arange(station, 1.0, -0.05):
            s = run(known._replace(spans=spans), HELD_OUT)
            if not clean(s):
                break
            if s["cost"].mean() < best["cost"].mean():
                best, at = s, float(spans)
        row("stand-in: none better" if at is None else f"stand-in at {at:.2f} b", best)
        return best

    steps = (
        ("v3: wake known and still, 2.8-3.7 km", Design()),
        ("+ drift", Design(drift=1.0)),
        ("+ wind error", Design(drift=1.0, wind_sigma=WIND_SIGMA)),
        ("+ trials' distance, 1.2-1.8 km", Design(1.25, 1.0, WIND_SIGMA, **TRIALS)),
    )
    print(f"protected bank {bank_limit:.0f} deg; wind error {WIND_SIGMA} m/s an axis")
    for name, base in steps:
        print(f"\n== {name}", flush=True)
        closest = edge(base)
        known = base._replace(wind_sigma=0.0, spans=2.5)
        meander = float(run(known)["outboard"].std())
        station = closest + 2.0 * meander / SPAN
        level = run(known._replace(spans=station))["trim"].mean()
        print(
            f"edge {closest:.2f} b; the wake's own drift sd {meander:.1f} m; station "
            f"by the zone rule {station:.2f} b, where the roll trim is {level:.2f} deg",
            flush=True,
        )
        off = run(base._replace(wake=0.0), HELD_OUT, "wake off")
        placed = base._replace(spans=station)
        if base.wind_sigma:
            error = float(run(base._replace(spans=2.5))["outboard"].std())
            wide = closest + 2.0 * error / SPAN
            run(placed, HELD_OUT, f"hold at the predicted {station:.2f} b")
            label = f"hold at the predicted {wide:.2f} b (2 sd of {error:.1f} m)"
            run(base._replace(spans=wide), HELD_OUT, label)
            placed = placed._replace(trim=float(np.deg2rad(level)))
            ref = run(
                placed, HELD_OUT, f"reference: seeks {level:.2f} deg of roll trim"
            )
        else:
            ref = run(placed, HELD_OUT, f"reference: holds {station:.2f} b")
        best = told(base, station, ref)
        ratios = (static_ratio(env, params, int(seed), base) for seed in HELD_OUT)
        steady = off["cost"] * np.array(list(ratios))
        print(
            f"    reference's saving against the wake off {interval(1 - ref['cost'] / off['cost'])}\n"
            f"    room, lower bound (over the stand-in)         {interval(ref['cost'] / best['cost'])}\n"
            f"    room, upper bound (over the best steady point) {interval(ref['cost'] / steady)}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
