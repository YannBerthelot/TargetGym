"""The boron line's delay is the plant's (owner, 2026-11-25).

Contract, from the owner's answer and OAPS section 2.1, written before the code:

- The line's delay, 5 to 15 min from the command to the core, is set by the plant's
  piping and the flow its operator runs: a number of the family member, the same on
  every reset of it, not a draw of the day. The family sweep draws it per member.
- The recipe measures it with its existing bump test (boron on Tavg, rods held):
  the reading is the line's delay plus the plant's own lag to 2 % of the response,
  which is short (the boron mixes through the primary's loops and Tavg answers its
  reactivity within a minute or two). So the reading lies between the delay and the
  delay plus 3 control steps (90 s, ours).
- The expert leads the plan's boron by the dead time its recipe measured: given a
  dead time ``theta``, the boration for a ramp down starts ``theta`` before the
  ramp (within the step the lead rounds to).
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.reactor import Reactor, ReactorParams
from target_gym.reactor.experts import make_reactor_pid_controller

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

START_OF_CYCLE = {"cycle_range": (0.0, 0.0)}


def test_the_line_delay_is_the_plants_on_every_reset():
    env, p = Reactor(), ReactorParams(boron_delay=13)
    keys = jax.random.split(jax.random.PRNGKey(0), 16)
    delays = jax.vmap(lambda k: env.reset_env(k, p)[1].boron_delay)(keys)
    np.testing.assert_array_equal(np.asarray(delays), 13)


def test_the_family_sweep_draws_each_members_delay_in_whole_steps():
    import family_sweep as sweep

    drawn = [d["boron_delay"] for d in sweep.draws(sweep.table("reactor"))]
    assert len(set(drawn)) > 1
    assert all(isinstance(x, int) and 10 <= x <= 30 for x in drawn)


@pytest.mark.medium
@pytest.mark.parametrize("delay", [10, 30])
def test_the_recipes_bump_reads_the_lines_delay(delay):
    import reactor_ladder as L

    env = Reactor()
    p = ReactorParams(
        demand_sigma=0.0, sensor_noise_scale=0.0, boron_delay=delay, **START_OF_CYCLE
    )
    q, s = L.points(env, p, 100)[0]
    _, read = L.boron_on_tavg(env, q, s)
    assert delay * 30.0 <= read <= delay * 30.0 + 90.0


@pytest.mark.parametrize("theta", [300.0, 900.0])
def test_the_expert_leads_the_plans_boron_by_its_measured_dead_time(theta):
    """A plant lined out at full power at the start of the cycle, Tavg on program;
    the plan's ramp down (8 %/h to 50 %) starts at 1 h. The boration starts
    ``theta`` ahead of it, and not a step sooner than the step it rounds to."""
    expert = make_reactor_pid_controller(theta=theta)
    p = ReactorParams(**START_OF_CYCLE)
    steps, ramp_at = 240, 3600.0
    T_avg = p.T_avg_full_load
    obs = np.array(
        [1.0, T_avg + 16.7, 1.0 - 2.0 * 3.0 / 228.0, 1.0, T_avg, T_avg, 0.0, 1000.0,
         0.0, 0.0, 1.0, 0.08 * 100.0 / 60.0, 0.5, 0.0, 1.0, 0.0],
        np.float32,
    )  # fmt: skip
    obs = np.tile(obs, (steps, 1))
    t = np.arange(steps) * 30.0
    # Hours to the ramp down and to the return, 0 once each has started.
    obs[:, 8] = np.maximum(ramp_at - t, 0.0) / 3600.0
    obs[:, 9] = np.maximum(ramp_at + 6.25 * 3600.0 + 3 * 3600.0 - t, 0.0) / 3600.0

    def body(m, o):
        a, m = expert.act(m, o)
        return m, a[1]

    flow = np.asarray(
        jax.jit(lambda o: jax.lax.scan(body, expert.init_state, o)[1])(jnp.asarray(obs))
    )
    first = t[np.argmax(flow > 1e-3)]
    assert ramp_at - theta - 30.0 <= first <= ramp_at - theta + 30.0
