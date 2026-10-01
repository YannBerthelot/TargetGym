"""Reference controller (the MPC slot) for the boiler drum task.

Each task keeps its oracle in its own package, so changing one re-records
only the tasks of that package. The shared machinery (GradientMPC,
CasadiMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every task's baseline fingerprint.
"""

import jax
import jax.numpy as jnp

from target_gym.experts.mpc import (
    GradientMPC,
    _floor_units,
    _is_v1,
)


class BoilerDrumGradientMPC(GradientMPC):
    """Gradient MPC for the drum boiler, on a quadratic objective.

    The environment's reward uses clipped tracking bands, which go flat once
    the level is more than ``level_band`` from normal -- precisely the
    situation the controller is called on to fix. Optimising it directly leaves
    no gradient exactly when one is needed, so the objective here is a
    quadratic sharing the reward's *minimiser* (level at normal, pressure on
    target, fuel low) while staying informative far from it.

    Horizon is what earns MPC its keep here. Drum level is non-minimum phase:
    the level's first move after a load change is the wrong way. A controller
    optimising over 30 steps (60 s at dt = 2 s, covering the ~35 s swell peak)
    sees the reversal coming and keeps adding feedwater through a swell, where
    a reactive loop cuts it.
    """

    def __init__(
        self,
        env,
        params,
        level_weight: float = 1.0,
        pressure_weight: float = 0.3,
        **kwargs,
    ):
        super().__init__(env, params, **kwargs)
        self.level_weight = level_weight
        self.pressure_weight = pressure_weight

    def _rollout(self, actions: jnp.ndarray, state) -> jnp.ndarray:
        key = jax.random.PRNGKey(0)
        pr = self.params

        v1 = _is_v1(pr)
        floor = _floor_units(pr)
        fail = float(getattr(pr, "failure_cost", 0.0)) / floor

        def step_fn(carry, u):
            s, done = carry
            _, new_s, r, terminated, _ = self.env.step_env(
                key, s, self._env_action(u), pr
            )
            if v1:
                level_err = new_s.level / pr.level_band
                press_err = (new_s.pressure - new_s.target_pressure) / pr.pressure_band
                fuel = new_s.Q_fuel / pr.Q_max
                cost = (
                    self.level_weight * level_err**2
                    + self.pressure_weight * press_err**2
                    + pr.fuel_weight * fuel
                )
                return (new_s, done), -cost
            # Version 2: the plant's own cost, in floor units, with a trip
            # charged for the rest of the horizon.
            obj = jnp.where(done, -fail, r / floor)
            return (new_s, jnp.logical_or(done, terminated)), obj

        (_, _), rewards = jax.lax.scan(step_fn, (state, jnp.zeros((), bool)), actions)
        return jnp.sum(rewards)


def make_boiler_drum_mpc(
    env, params, horizon: int = 30, n_iter: int = 40, lr: float = 0.05
):
    """Gradient MPC for the drum boiler.

    Gradient-based rather than CasADi: the plant is already differentiable JAX
    and its steam-property fits and voidage algebra would have to be duplicated
    symbolically for no gain. Optimises firing and feedwater jointly, which
    matters because they are coupled through pressure -- firing harder raises
    pressure, which collapses bubbles and *lowers* the level.
    """
    return BoilerDrumGradientMPC(
        env,
        params,
        action_dim=2,
        action_lb=-1.0,
        action_ub=1.0,
        horizon=horizon,
        n_iter=n_iter,
        lr=lr,
    )
