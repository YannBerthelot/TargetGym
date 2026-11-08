"""The reactor's control properties, measured on the plant with no controller.

Every run holds the turbine, keeps the load's wobble off and leaves the actions
fixed apart from the one move being measured:

- the axial xenon mode: rods dipped for 2.5 h, as in the Oconee II test, then back.
  Period and decay come from the turning points of delta I that follow;
- what a rod move and a boron move of the same reactivity do to the offset and to
  Tavg, and when each starts to show;
- the core's reactivity budget through a 100-50-100 day with the rods alone.

    uv run python scripts/reactor_properties.py
"""

import json

import jax
import jax.numpy as jnp
import numpy as np

from target_gym.reactor import Reactor, ReactorParams
from target_gym.reactor.env import rod_worth, shim_position, tavg_program

STEP_H = 30.0 / 3600.0


def flat(load, **changes):
    """Lined out at ``load`` with no schedule: xenon balanced there, boron critical."""
    env = Reactor()
    p = ReactorParams(
        demand_sigma=0.0, initial_xenon_power_range=(load, load), **changes
    )
    _, s = env.reset_env(jax.random.PRNGKey(0), p)
    return env, p, s


def line_out(env, p, s, load):
    """Part-load start: the turbine at ``load``, boron trimmed by hand to keep the
    rods parked, 3 h for the temperatures to settle."""
    far = jnp.asarray(1e8, s.n.dtype)
    s = s.replace(ramp_down_at=far, ramp_up_at=far)
    if load == 1.0:
        return s
    from target_gym.reactor.env import (
        rod_balance,
        steady_state_precursors,
        steady_state_temperatures,
    )

    T_fuel, T_hot, T_cold, T_sg = steady_state_temperatures(load, p)
    shim = shim_position(load, p)
    needed = rod_balance(T_fuel, 0.5 * (T_hot + T_cold), s.Xe_hat, p)
    needed = needed - (rod_worth(shim, p) - rod_worth(p.rod_park, p))
    d = s.n.dtype
    q = jnp.asarray(load, d)
    return s.replace(
        n=q, C=steady_state_precursors(q, p), T_fuel=jnp.asarray(T_fuel, d),
        T_coolant=jnp.asarray(T_hot, d), T_cold=jnp.asarray(T_cold, d),
        T_sg=jnp.asarray(T_sg, d), target_n=q, T_ref=jnp.asarray(tavg_program(q, p), d),
        boron=jnp.asarray(p.boron_ref + needed / p.boron_worth, d),
        ramp_down_at=jnp.asarray(-1e8, d), ramp_up_at=far, rod_in=jnp.asarray(shim, d),
    )  # fmt: skip


def rollout(env, p, s, actions):
    """Step ``actions`` (steps, 2); returns the state at every step."""

    def step(s, a):
        _, s, _, _, _ = env.step_env(jax.random.PRNGKey(0), s, a, p)
        return s, s

    return jax.lax.scan(step, s, jnp.asarray(actions, jnp.float32))[1]


def rods_at(steps_in):
    return 1.0 - 2.0 * steps_in / 260.0


def hold(s, n):
    """``n`` steps of actions holding the shimming group where it is, R mid-band."""
    a = np.zeros((n, 3))
    a[:, 0] = 1.0 - 2.0 * float(s.rod_in)
    return a


def _mode(load, dip=30.0, hours=120.0):
    """The axial mode's period and decay at ``load`` after a 2.5 h rod dip."""
    env, p, s = flat(load)
    s = line_out(env, p, s, load)
    n = int(hours / STEP_H)
    a = hold(s, n)
    a[: int(2.5 / STEP_H), 0] = rods_at(float(s.rod_in) * 260.0 + dip)
    tr = rollout(env, p, s, a)
    ao = np.asarray(tr.delta_I) / np.asarray(tr.n)
    trips = int(np.sum(np.asarray(tr.time) == 0))
    # Hourly means, since float32 flickers around a turning point at 30 s.
    per_hour = int(round(1.0 / STEP_H))
    hours_ao = ao[: n // per_hour * per_hour].reshape(-1, per_hour).mean(axis=1)
    t = np.arange(len(hours_ao)) + 0.5
    d = np.diff(hours_ao)
    turn = np.where((np.sign(d[1:]) != np.sign(d[:-1])) & (t[1:-1] > 3.0))[0] + 1
    points = [(float(t[i]), float(hours_ao[i])) for i in turn]
    swings = np.abs(np.diff([y for _, y in points]))
    return {
        "load": load,
        "period_h": 2.0 * float(np.mean(np.diff([x for x, _ in points]))),
        "decay_per_h": float(
            np.log(swings[2] / swings[0]) / (points[2][0] - points[0][0])
        ),
        "trips": trips,
        "turning_points": [(x, round(y, 3)) for x, y in points[:6]],
        "offset_at_end": float(ao[-1]),
    }


def _moves():
    """A rod move and a boron move of the same reactivity at full power."""
    env, p, s = flat(1.0)
    s = line_out(env, p, s, 1.0)
    steps_in = 60.0  # to be matched by boron
    rods = float(rod_worth(steps_in / 260.0, p) - rod_worth(0.0, p))
    n = int(2.0 / STEP_H)
    a = hold(s, n)
    a[:, 0] = rods_at(steps_in)
    rod_run = rollout(env, p, s, a)
    # Group R across its whole band, foot to top: its full reach.
    a = hold(s, n)
    a[:, 1] = -1.0
    r_run = rollout(env, p, s, a)
    # The same worth as boration at full flow: ppm over the rate.
    ppm = rods / p.boron_worth
    rate = p.boration_flow / p.primary_mass * (p.boric_acid_ppm - p.boron_ref)
    on = int(round(ppm / rate / 30.0))
    a = hold(s, n)
    a[:on, 2] = 1.0
    boron_run = rollout(env, p, s, a)
    t = np.arange(n) * STEP_H * 60.0

    def effect(run):
        di = np.asarray(run.delta_I)
        tav = 0.5 * np.asarray(run.T_coolant + run.T_cold)
        dT = tav - tav[0]
        when = t[np.argmax(np.abs(dT) > 0.05)]
        peak = int(np.argmax(np.abs(dT)))
        return {
            "minutes_to_5_percent_K": float(when), "peak_dTavg_K": float(dT[peak]),
            "delta_I_at_peak": float(di[peak]), "delta_I_end": float(di[-1]),
        }  # fmt: skip

    return {
        "rod_steps": steps_in, "worth_pcm": rods * 1e5, "boron_ppm": ppm,
        "boron_full_flow_min": on / 2.0, "boron_delay_min": float(s.boron_delay) / 2.0,
        "rods": effect(rod_run), "boron": effect(boron_run),
        "r_band_in": effect(r_run),
    }  # fmt: skip


def _bank():
    """How much of the bank the day's reactivity takes: rods alone, no boron."""
    p = ReactorParams()
    from target_gym.reactor.env import rod_tilt

    out = []
    for steps in (30, 60, 90, 130):
        y = steps / 260.0
        out.append({
            "steps_in": steps,
            "worth_pcm": float(rod_worth(y, p)) * 1e5,
            "rod_eps_per_cm": float(rod_tilt(y, p)),
        })  # fmt: skip
    return out


def main():
    report = {
        "axial_mode": [_mode(1.0), _mode(0.75), _mode(0.5)],
        "moves": _moves(),
        "bank": _bank(),
    }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
