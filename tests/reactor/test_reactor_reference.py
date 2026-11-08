"""The reference's xenon tool and its override's handback.

Contracts (Amendment 3's reactor row; OAPS section 1 for the operator's xenon
tool; Shinskey's bumpless transfer for the override):

- The xenon tool runs the textbook iodine and xenon equations on the observed
  power. Started in balance, it follows the plant's mean xenon through a load
  cut (the plant's means add only the tilt's small burn term), and while xenon
  builds it asks for dilution, since xenon's poison must be met by less boron.
- When the saturation override hands Tavg back to the cascade, the boron it built
  is not dropped: the boron command does not step back by the override's size.
"""

import jax
import jax.numpy as jnp
import numpy as np

from target_gym.reactor import Reactor, ReactorParams
from target_gym.reactor.experts import (
    ReferenceParams,
    ReferenceState,
    reference_step,
    xenon_feedforward,
)


def test_xenon_tool_follows_the_plants_xenon_through_a_load_cut():
    env = Reactor()
    p = ReactorParams(demand_sigma=0.0, initial_xenon_power_range=(1.0, 1.0))
    _, s = env.reset_env(jax.random.PRNGKey(0), p)
    s = s.replace(ramp_down_at=jnp.float32(0.0), ramp_up_at=jnp.float32(1e8))
    hold = jnp.array([1.0 - 2.0 * float(s.rod_in), 0.0], jnp.float32)

    def step(c, _):
        s, I, Xe, first = c
        flow, I, Xe = xenon_feedforward(I, Xe, s.n, s.boron, first, p, 30.0)
        _, s, _, _, _ = env.step_env(jax.random.PRNGKey(0), s, hold, p)
        return (s, I, Xe, False), (s.Xe_hat, Xe, flow)

    start = (s, jnp.float32(0.0), jnp.float32(0.0), True)
    _, (plant, tool, flow) = jax.lax.scan(step, start, None, length=6 * 120)
    plant, tool, flow = map(np.asarray, (plant, tool, flow))
    assert plant[-1] > 1.05  # the cut did build xenon
    # Explicit 30 s steps on hour-long constants, and no tilt term: within 2 %.
    np.testing.assert_allclose(tool[::60], plant[::60], rtol=0.02)
    assert np.all(flow[120:300] < 0.0)  # building, 1 to 2.5 h in: dilute


def test_the_overrides_handback_keeps_the_boron_it_built():
    p = ReferenceParams()
    built = 0.3
    state = ReferenceState(boron=0.1, override=built, clock=3600.0, last_e=0.0)
    state = state.replace(I_hat=1.0, Xe_hat=1.0)
    # Rods off their top stop, at full power, Tavg on program, offset 0.
    rod = p.park - 0.2
    obs = jnp.array([1.0, 600.0, rod, 1.0, 580.0, 580.0, 0.0, 1690.0, 0.0, 0.0, 1.0])
    action, after = reference_step(p, state, obs)
    assert float(after.override) == 0.0
    assert float(action[1]) > 0.1 + built - 0.05
