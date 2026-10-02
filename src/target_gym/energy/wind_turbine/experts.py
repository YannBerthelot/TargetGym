"""Reference controller (the MPC slot) for the wind turbine task.

Each task keeps its oracle in its own package, so changing one re-records
only the tasks of that package. The shared machinery (GradientMPC,
CasadiMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every task's baseline fingerprint.

Since the oracle audit (2026-10) the version-2 oracle is not a planner but a
feedback law, :class:`WindTurbineNewtonPI`: generator torque from a Newton
solve on the noise-free plant model, so that the next step's power equals the
target, and collective pitch from the shipped PI with its command slew capped
at the activity the reward leaves free. It beats the gradient planner it
replaced on every protocol seed. That planner,
:func:`make_wind_turbine_gradient_mpc`, is kept unchanged: it is still the
oracle for version-1 params, whose recorded baseline it reproduces.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from target_gym.experts.mpc import (
    GradientMPC,
    _done_value,
    _is_v1,
    _pid_rollout_plan,
    _v2_objective,
)

_WT_BARRIER_ONSET = 0.85  # fraction of the trip speed at which the barrier starts
_WT_BARRIER_WEIGHT = 10.0


def _wind_turbine_objective(state, params):
    """Smooth stand-in for the turbine's reward, with the same minimiser.

    The environment used to score power tracking as ``clip(1-err/band, 0, 1)**2``
    minus a pitch-activity penalty: a fine thing to be scored on and a useless
    thing to descend, because one step off the operating point puts the error at
    nearly four times the band, where the term is clipped flat and the only
    surviving gradient belongs to the *penalty*. The optimiser was then correctly
    guided to stop moving the pitch, and returned ~0 for the rest of the episode.

    The reward is log-scaled now and has no flat region, so that premise is gone
    -- but the surrogate is still needed, for a subtler reason. Log-scaling makes
    the reward scale-free in *value* (each halving of the error is worth the same
    increment); it does not make the *gradient* scale-free. Differentiating
    ``1 - log1p(e/f)/log1p(E/f)`` gives ``-1/((f + e) * log1p(E/f))``, which
    decays like ``1/e``: the pull toward the setpoint is weakest exactly where
    the controller is furthest from it. A quadratic in the normalised error has
    the same minimiser and a gradient that instead *grows* with the error.

    Measured over six seeds against the log-scaled reward, that difference is
    still worth almost everything: 341.9 for this surrogate against 172.1 for
    the reward itself (with the barrier below in both). So the surrogate is not a
    workaround for a broken reward any more -- it is a planner-side
    reformulation, which is a normal thing for an MPC to carry.
    """
    from target_gym.energy.wind_turbine.env import electrical_power

    power = electrical_power(state.omega, state.torque, params)
    err = (state.target_power - power) / params.power_band
    activity = jnp.abs(state.pitch_cmd - state.pitch) / params.pitch_max

    # Soft barrier on the rotor-speed trip. The environment terminates outside
    # [underspeed, overspeed] x rated, and a *hard* stop is invisible to a
    # gradient planner: ``done`` is a boolean, so masking the reward after it
    # tells the optimiser what a trip costs while giving it no derivative
    # pointing away from one. Measured, that is exactly what happened -- the
    # planned pitch command stayed at 0.000 while the predicted rotor speed ran
    # past the trip, on every horizon from 60 to 200.
    #
    # A differentiable penalty that switches on before the boundary does give a
    # gradient, and it is what makes this controller stable: over twelve seeds
    # the worst episode goes from 22 to 307 and the spread from sd 152 to 24.
    # Re-checked against the log-scaled reward, where it is still the difference
    # between controlling and tripping: 172.1 with the barrier, 52.4 without,
    # and without it 6 of 6 episodes ended early on the overspeed trip.
    # The onset matters (0.85 beats 0.80); the weight barely does (10, 30 and
    # 100 land within 0.4 of each other), which is the signature of a term that
    # is shaping the approach rather than trading against the objective.
    barrier = _wind_turbine_barrier(state, params)

    # Offset so a healthy step scores ~1 and a terminated one scores
    # ``done_value`` = 0, making an early trip cost the rest of the horizon.
    return (
        1.0
        - err**2
        - params.pitch_activity_weight * activity
        - _WT_BARRIER_WEIGHT * barrier
    )


_WT_SPEED_BOX = 0.05  # rotor speed kept within this fraction of rated
_WT_SPEED_BOX_COST = 100.0  # floor units at a 10% excursion: ten floor-widths


def _wind_turbine_barrier(state, params):
    """Squared excursion of the rotor speed past 85% of the way to either trip."""
    from target_gym.energy.wind_turbine.env import omega_rated

    ratio = state.omega / omega_rated(params)
    over = ratio / params.overspeed_factor
    under = params.underspeed_factor / jnp.maximum(ratio, 1e-6)
    return (
        jnp.maximum(over - _WT_BARRIER_ONSET, 0.0) ** 2
        + jnp.maximum(under - _WT_BARRIER_ONSET, 0.0) ** 2
    )


def _wind_turbine_speed_box(state, params):
    """Soft box keeping the rotor within 5% of rated, in floor units.

    What a turbine's own supervisory logic does above rated wind (regulate
    rotor speed with the pitch, power with the torque), and it is there for
    the planner's horizon, not for the reward. Recovering a rotor that has
    slowed costs torque now and pays off over the rotor's ~100 s of inertia,
    beyond a 15 s plan; without the box the planner sat at 0.85x rated with a
    250 kW error rather than spend the torque (seed 1 of the test episode),
    and drifted the same way over a long hold. Mild by design: a 10%
    excursion costs ten floor-widths of tracking error (25 kW), so the plan
    is still the plant's own cost -- weighted like the trip, or like a 500 kW
    error, the box dominated the tracking term and the planner braked the
    rotor with the torque instead (a 350-700 kW error for a whole horizon,
    measured on seeds 1 and 2), where the PID rides a 16% overspeed with a
    50 kW error and the plant trips only at 25%.
    """
    from target_gym.energy.wind_turbine.env import omega_rated

    ratio = state.omega / omega_rated(params)
    excess = jnp.maximum(jnp.abs(ratio - 1.0) - _WT_SPEED_BOX, 0.0) / _WT_SPEED_BOX
    return _WT_SPEED_BOX_COST * excess**2


def make_wind_turbine_gradient_mpc(
    env,
    params,
    horizon: int = 60,
    n_iter: int | None = None,
    lr: float | None = None,
    n_tail: int = 0,
    objective_fn=_wind_turbine_objective,
):
    """Gradient MPC for the NREL 5 MW turbine: the version-1 oracle, and the
    version-2 one until the oracle audit (2026-10).

    Gradient-based for the same reason as the aircraft and the column: the
    plant is already differentiable JAX, and the Cp surface is an empirical
    fit that would gain nothing from symbolic re-expression. Optimising pitch
    and torque jointly is the point -- the rate-limited pitch actuator means
    the useful move is often to start pitching *before* the rotor has
    accelerated, which a reactive loop cannot do.

    Two changes make it actually control. The objective is the smooth surrogate
    above rather than the environment's clipped reward, without which the
    controller scored a return of -0.02 against the PID's 393 at every horizon
    tried -- identical to two decimals at 20, 40 and 60, which is the signature
    of an optimiser that is not moving. The horizon is then 60 rather than 20,
    which only matters once the gradient is informative. Measured over 400-step
    episodes the return goes from -0.02 to 387, with no terminations.

    Scored over twelve seeds this reaches 385 against the PID's 392 -- 98%, and
    ahead on 7 of the 12 -- so it is on par with the PID rather than an upper
    bound over it. The remaining gap is one seed that still drops to ~307.

    Two things that look like explanations and are not, both measured: the wind
    forecast (the MPC plans with a fixed key while the episode uses its own, and
    the seed where they disagree scored *higher*), and the inner optimiser
    (Adam, and a decaying step size, are both worse here than the plain one).

    The oracle audit (2026-10) found two weaknesses in the version-2 planner,
    kept here as recorded. Its descent (500 fixed steps at lr 0.005) does not
    converge on the torque: the applied action left a mean of 1.8 kW of power
    error that its own model predicted on protocol seed 0 (0.9 kW on seeds 1
    and 2). And the move penalty is read when the solve is traced, when there
    is no previous action, so it never enters the descent; it acts only when
    the guide plan is compared. :class:`WindTurbineNewtonPI` replaced it as
    the version-2 oracle.
    """
    # Version 1 keeps the planner it was recorded with (100 iterations at
    # lr 0.02, a zero warm start), so its recorded baseline reproduces; the
    # version-2 planner needs the larger budget for the move-suppressed,
    # squared-tracking objective (measured: 500 / 0.005 is the first setting
    # that beats the PID on every seed).
    v1 = _is_v1(params)
    n_iter = (100 if v1 else 500) if n_iter is None else n_iter
    lr = (0.02 if v1 else 0.005) if lr is None else lr
    done_value = 0.0
    if not v1 and objective_fn is _wind_turbine_objective:
        from target_gym.energy.wind_turbine.env import (
            compute_reward,
            compute_reward_terms,
        )

        objective_fn = _v2_objective(
            compute_reward,
            _wind_turbine_barrier,
            terms_fn=compute_reward_terms,
            shaping_fn=_wind_turbine_speed_box,
        )
        done_value = _done_value(params, squared=True)

    from target_gym.experts.pid import make_wind_turbine_stateful_pid

    initial_plan = (
        None
        if v1
        else _pid_rollout_plan(env, make_wind_turbine_stateful_pid, horizon, 2)
    )
    guide_plan = initial_plan

    def move_penalty(u0, u_prev, p):
        # Pitch command change between solves as a fraction of pitch_max
        # (raw range 2 <-> pitch_max), in floor units of the fatigue term.
        frac = jnp.abs(u0[0] - u_prev[0]) * 0.5
        return float(p.fatigue_weight) * (frac / p.c_hold) ** 2

    return GradientMPC(
        env,
        params,
        action_dim=2,
        action_lb=-1.0,
        action_ub=1.0,
        horizon=horizon,
        n_iter=n_iter,
        lr=lr,
        n_tail=n_tail,
        done_value=done_value,
        objective_fn=objective_fn,
        initial_plan_fn=initial_plan,
        guide_plan_fn=guide_plan,
        move_penalty_fn=None if v1 else move_penalty,
    )


class WindTurbineNewtonPI:
    """Torque by a one-step Newton solve, pitch by the shipped PI with a slew cap.

    The version-2 oracle. It reads the true state, as every oracle in the MPC
    slot does (the rotor-effective wind and its mean included), and it plans
    on the mean of the turbulence: the model it inverts is the plant's own
    ``compute_next_state`` with ``turbulence_std`` zeroed, so the wind it
    predicts for the next step is the current wind's mean-reverting drift and
    nothing else. It never sees the step key's noise, so it is causal.

    **Torque.** Generator torque reaches its command within a step
    (``torque_tau`` 0.1 s against a 0.25 s step), so the power the next step
    is scored on is set almost entirely by this step's torque command.
    ``newton_iters`` Newton steps on that command, through the model, make the
    predicted electrical power after the step equal the target; JAX gives the
    derivative. The target is multiplied by the PID's own low-speed
    protection, which backs the demand off linearly between 0.60 and 0.90 of
    rated speed so that a slowing rotor can recover, and is inactive at and
    above 0.90. What is left of the tracking error is the one-step wind
    innovation, which no causal controller can see: in the oracle audit the
    power error the model predicted for the applied action was 0 W on every
    protocol seed, where the gradient planner left a mean of 0.9 to 1.8 kW
    per seed.

    **Pitch.** The shipped PI on rotor-speed error (``Kp_pitch`` and
    ``Ki_pitch`` from the PID's tuned gains), in velocity form and bumpless
    from the command the plant is already holding (``state.pitch_cmd``).
    Two changes from the PID:

    * The setpoint is ``speed_setpoint`` times rated (1.05). With torque
      holding the power, rotor speed is free to sit anywhere safe, and a
      faster rotor turns the same wind innovation into a smaller power error
      (the error scales with the speed change over the speed). At rated speed
      a slow pitch lets the rotor sag into torque saturation.
    * The command moves by at most ``pitch_slew`` degrees per step (0.13)
      while the speed ratio is inside ``[slew_lo + slew_ramp, slew_hi -
      slew_ramp]`` (1.01 to 1.09). A command ramping 0.13 deg every step
      leaves the actuator 0.052 deg behind (measured with the plant's own
      integrator), an activity ``|cmd - pitch| / pitch_max`` of 0.0013: the
      ``c_hold`` the reward leaves free. Over the outer ``slew_ramp`` (0.04)
      of the band ``[slew_lo, slew_hi]`` (0.97 to 1.13) the cap opens
      linearly to the actuator's own rate (``pitch_rate_max`` over a step,
      2 deg), and outside the band the PI is not capped at all.
      ``pitch_slew=math.inf`` removes the cap.

    The ramp is what makes the slow pitch safe. With the cap lifted only
    outside a hard 0.90-1.15 band, a gust pitched the blades up at full rate
    and the capped pitch-down afterwards was too slow: on 16 of 197 screened
    seeds the rotor then sagged to 0.82-0.89 of rated with power errors of
    10-50 kW, and the rotor reached 1.218 of rated, 2.6% short of the
    overspeed trip. Opening the cap progressively on both sides of the
    setpoint removed all of those sags but seed 142's, a lull with less power
    in the wind than its target, where every controller sags (the PID
    included), and kept the peak at 1.178, at a cost of 2.3% on the protocol
    seeds. The screen and the protocol numbers are in the task's PHYSICS.md,
    section 5.

    No optimiser and no horizon: about 0.1 ms per step after a one-off JIT
    compile of about 0.4 s per instance. After a trip the plant restarts from
    a fresh reset draw (new wind and target, rotor at rated speed, pitch at
    the reset's balance pitch; ``base.failure_kernel``), so the PI's stored
    error belongs to the pre-trip state. At rated speed the slew cap is still
    active (about 0.6 deg), so it bounds the first increment after the
    restart. No screened episode tripped.
    """

    def __init__(
        self,
        env,
        params,
        speed_setpoint: float = 1.05,
        pitch_slew: float = 0.13,
        slew_lo: float = 0.97,
        slew_hi: float = 1.13,
        slew_ramp: float = 0.04,
        newton_iters: int = 4,
    ):
        from target_gym.energy.wind_turbine.env import (
            compute_next_state,
            electrical_power,
        )
        from target_gym.experts.pid import make_wind_turbine_stateful_pid

        self.env = env
        # Plan on the mean wind whatever the caller passed: with the
        # turbulence zeroed the step key draws nothing.
        self.params = params.replace(turbulence_std=0.0)
        pid = make_wind_turbine_stateful_pid()
        self.Kp, self.Ki = float(pid.Kp_pitch), float(pid.Ki_pitch)
        self.protect_lo, self.protect_hi = float(pid.protect_lo), float(pid.protect_hi)
        self.speed_setpoint = float(speed_setpoint)
        self.pitch_slew = float(pitch_slew)
        self.slew_lo, self.slew_hi = float(slew_lo), float(slew_hi)
        self.slew_ramp = float(slew_ramp)

        p = self.params
        self._dt = float(p.delta_t)
        self._rate_step = float(p.pitch_rate_max) * self._dt  # deg per step
        self._pitch_min, self._pitch_max = float(p.pitch_min), float(p.pitch_max)
        self._rated_rpm = float(p.omega_rated_rpm)
        self._w_rated = self._rated_rpm * 2.0 * math.pi / 60.0
        eta, n_gear, t_max = float(p.eta_gen), float(p.N_gear), float(p.torque_max)
        key = jax.random.PRNGKey(0)  # draws nothing: the model's turbulence is 0
        n = int(newton_iters)

        @jax.jit
        def newton(state, pitch_raw, target):
            def p_next(u):
                s1, _ = compute_next_state(jnp.stack([pitch_raw, u]), state, p, key)
                return electrical_power(s1.omega, s1.torque, p)

            # Start from the static feedforward the PID uses.
            u0 = (
                2.0 * jnp.clip(target / (eta * n_gear * state.omega) / t_max, 0.0, 1.0)
                - 1.0
            )

            def body(_, u):
                f, d = jax.value_and_grad(p_next)(u)
                d = jnp.where(jnp.abs(d) < 1.0, 1.0, d)
                return jnp.clip(u - (f - target) / d, -1.0, 1.0)

            return jax.lax.fori_loop(0, n, body, u0)

        self._newton = newton
        self.reset()

    def reset(self):
        self._err_prev = None

    def step(self, _obs, state):
        """Return ``[pitch_raw, torque_raw]``. ``_obs`` is ignored (API symmetry)."""
        ratio = float(state.omega) / self._w_rated
        omega_rpm = float(state.omega) * 60.0 / (2.0 * math.pi)

        # Pitch: incremental PI on the speed error, slew-capped inside the band.
        err = omega_rpm - self._rated_rpm * self.speed_setpoint
        if self._err_prev is None:
            self._err_prev = err
        d = self.Kp * (err - self._err_prev) + self.Ki * err * self._dt
        self._err_prev = err
        cap = self.pitch_slew
        if ratio < self.slew_lo or ratio > self.slew_hi:
            cap = math.inf
        elif self.slew_ramp > 0.0 and math.isfinite(cap):
            # Inside the last ``slew_ramp`` of the band the cap opens linearly
            # toward the actuator's own rate, so the PI is not released at once.
            x = max(
                (ratio - (self.slew_hi - self.slew_ramp)) / self.slew_ramp,
                ((self.slew_lo + self.slew_ramp) - ratio) / self.slew_ramp,
                0.0,
            )
            cap = cap + (self._rate_step - cap) * min(x, 1.0)
        d = max(-cap, min(cap, d))
        lo, hi = self._pitch_min, self._pitch_max
        cmd = min(max(float(state.pitch_cmd) + d, lo), hi)
        pitch_raw = 2.0 * (cmd - lo) / (hi - lo) - 1.0

        # Torque: next-step power equal to the protected target.
        protection = min(
            max((ratio - self.protect_lo) / (self.protect_hi - self.protect_lo), 0.0),
            1.0,
        )
        target = float(state.target_power) * protection
        torque_raw = float(
            self._newton(state, jnp.float32(pitch_raw), jnp.float32(target))
        )
        return np.array([pitch_raw, torque_raw], dtype=np.float32)

    def solver_report(self) -> dict:
        """No solver, so nothing to report."""
        return {}


def make_wind_turbine_mpc(
    env,
    params,
    horizon: int = 60,
    n_iter: int | None = None,
    lr: float | None = None,
    n_tail: int = 0,
    objective_fn=_wind_turbine_objective,
    *,
    speed_setpoint: float = 1.05,
    pitch_slew: float = 0.13,
    slew_lo: float = 0.97,
    slew_hi: float = 1.13,
    slew_ramp: float = 0.04,
    newton_iters: int = 4,
):
    """The turbine's oracle.

    Version-2 params get :class:`WindTurbineNewtonPI`, configured by the
    keyword-only arguments. Version-1 params get
    :func:`make_wind_turbine_gradient_mpc` with ``horizon``, ``n_iter``,
    ``lr``, ``n_tail`` and ``objective_fn``; those five are accepted on every
    call and configure only the gradient planner.
    """
    if _is_v1(params):
        return make_wind_turbine_gradient_mpc(
            env,
            params,
            horizon=horizon,
            n_iter=n_iter,
            lr=lr,
            n_tail=n_tail,
            objective_fn=objective_fn,
        )
    return WindTurbineNewtonPI(
        env,
        params,
        speed_setpoint=speed_setpoint,
        pitch_slew=pitch_slew,
        slew_lo=slew_lo,
        slew_hi=slew_hi,
        slew_ramp=slew_ramp,
        newton_iters=newton_iters,
    )
