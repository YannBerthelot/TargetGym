"""Reference controller (the MPC slot) for the boiler drum task.

Each task keeps its oracle in its own package, so changing one re-records
only the tasks of that package. The shared machinery (GradientMPC,
CasadiMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every task's baseline fingerprint.
"""

import jax
import jax.numpy as jnp
from flax import struct

from target_gym.experts.mpc import (
    GradientMPC,
    _floor_units,
    _is_v1,
)
from target_gym.experts.pid import Controller, _load_gains


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
    env,
    params,
    horizon: int = 30,
    n_iter: int = 1280,
    lr: float = 0.15,
    lr_end: float = 0.005,
):
    """Gradient MPC for the drum boiler.

    Gradient-based rather than CasADi: the plant is already differentiable JAX
    and its steam-property fits and voidage algebra would have to be duplicated
    symbolically for no gain. Optimises firing and feedwater jointly, which
    matters because they are coupled through pressure -- firing harder raises
    pressure, which collapses bubbles and *lowers* the level.

    The descent budget is large because the objective is ill-conditioned and
    the normalised steps make slow progress on it. Measured in the oracle
    audit (2026-10-01) on the protocol seeds, planning on the mean steam
    demand: 40 iterations scored 19.2, 160 with a decaying step 8.2, 640 5.3
    and 1280 4.7, each doubling gaining about half what the one before did,
    at 160 s per protocol seed. A longer horizon (45) and a PID-rollout guide
    did worse or nothing. A better-conditioned optimiser would likely beat
    more iterations.
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
        lr_end=lr_end,
    )


# ---------------------------------------------------------------------------
# Three-element PID as one pure step (host NumPy and jit/vmap alike)
# ---------------------------------------------------------------------------


@struct.dataclass
class BoilerDrumPIDParams:
    """Gains and plant constants of the three-element controller.

    Every field is a Python float, so NumPy treats it as weakly typed and the
    host arithmetic stays in the observation's dtype, as it did when
    the baseline was recorded.
    """

    Kp_level: float
    Ki_level: float
    Kp_pressure: float
    Ki_pressure: float
    Kd_pressure: float
    q_feed_max: float
    Q_max: float
    h_feedwater: float
    dt: float = 2.0
    # The level trim is bounded so the feedforward always dominates.
    trim_max: float = 25.0


@struct.dataclass
class BoilerDrumPIDState:
    """Level and pressure integrators, and the last pressure error."""

    level_int: float = 0.0
    press_int: float = 0.0
    prev_press_err: float = 0.0


def boiler_drum_pid_step(params, state, obs, xp=jnp):
    """One step of the boiler drum's level and pressure PIDs, against ``xp``.

    Keep the arithmetic order: under ``numpy`` it reproduces the scored
    baseline bit for bit.

    obs: [level, pressure, q_steam, fuel_pct, feed_pct, target_level, target_pressure]
    """
    p = params
    level = obs[..., 0]
    pressure = obs[..., 1]
    q_steam = obs[..., 2]
    target_level = obs[..., 5]
    target_pressure = obs[..., 6]

    # --- Feedwater: steam-flow feedforward, trimmed by a slow level PI ---
    err_l = target_level - level
    level_int = state.level_int + err_l * p.dt
    trim = p.Kp_level * err_l + p.Ki_level * level_int
    trim_clipped = xp.clip(trim, -p.trim_max, p.trim_max)
    level_int = xp.where(trim != trim_clipped, level_int - err_l * p.dt, level_int)
    q_feed = xp.clip(q_steam + trim_clipped, 0.0, p.q_feed_max)
    feed_raw = 2.0 * (q_feed / p.q_feed_max) - 1.0

    # --- Firing: enthalpy feedforward plus a PI on pressure ---
    Q_ff = q_steam * (2.75e6 - p.h_feedwater)
    err_p = target_pressure - pressure
    press_int = state.press_int + err_p * p.dt
    deriv = (err_p - state.prev_press_err) / p.dt
    Q = Q_ff + p.Q_max * (
        p.Kp_pressure * err_p + p.Ki_pressure * press_int + p.Kd_pressure * deriv
    )
    Q_clipped = xp.clip(Q, 0.0, p.Q_max)
    press_int = xp.where(Q != Q_clipped, press_int - err_p * p.dt, press_int)
    fuel_raw = 2.0 * (Q_clipped / p.Q_max) - 1.0

    new_state = BoilerDrumPIDState(
        level_int=level_int, press_int=press_int, prev_press_err=err_p
    )
    return xp.stack([fuel_raw, feed_raw], axis=-1), new_state


def make_boiler_drum_pid_controller() -> Controller:
    """The three-element controller as a :class:`Controller`.

    Reads the same tuned gains, with the same defaults, as
    ``make_boiler_drum_stateful_pid``.
    """
    from target_gym.boiler_drum.env import BoilerDrumParams

    pr = BoilerDrumParams()
    _p = _load_gains().get("boiler_drum", {})
    params = BoilerDrumPIDParams(
        Kp_level=float(_p.get("Kp_level", 60.0)),
        Ki_level=float(_p.get("Ki_level", 0.6)),
        Kp_pressure=float(_p.get("Kp_pressure", 0.05)),
        Ki_pressure=float(_p.get("Ki_pressure", 0.002)),
        Kd_pressure=float(_p.get("Kd_pressure", 0.0)),
        q_feed_max=pr.q_feed_max,
        Q_max=pr.Q_max,
        h_feedwater=pr.h_feedwater,
        dt=pr.delta_t,
    )
    return Controller(params, BoilerDrumPIDState(), boiler_drum_pid_step)
