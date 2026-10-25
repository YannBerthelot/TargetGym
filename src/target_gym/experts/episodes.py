"""Episode boundaries for a controller carried through a jitted loop.

Kept out of ``experts/pid.py`` on purpose: that file is part of every
environment's fingerprint, and nothing here can move a return.
"""

import jax
import jax.numpy as jnp


def reset_where_done(done, state, fresh):
    """``state`` with every entry whose ``done`` is true replaced by ``fresh``.

    The jitted form of ``pid.reset()``. Inside ``lax.scan`` with an
    auto-resetting ``env.step``, nothing else clears the PID's integral when an
    episode ends, so the next episode starts with the last one's::

        pid_state = reset_where_done(terminated | truncated, pid_state, ctrl.init_state)

    ``done`` is a scalar for one environment, or ``(num_envs,)`` for a batch
    such as ``env.expert_policy``'s state; ``fresh`` may be batched or not, as
    it broadcasts. Any pytree works, so an environment's state resets the same
    way. Reset at episode ends only: a trip restart inside a continuing
    episode keeps the memory, as the scored baselines do.
    """
    done = jnp.asarray(done)

    def pick(new, old):
        old = jnp.asarray(old)
        mask = jnp.reshape(done, done.shape + (1,) * (old.ndim - done.ndim))
        return jnp.where(mask, new, old)

    return jax.tree.map(pick, fresh, state)
