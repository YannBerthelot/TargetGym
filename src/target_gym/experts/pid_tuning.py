"""
Gradient-based PID gain tuner using JAX autodiff.

All target_gym environments are fully differentiable (pure JAX RK4 integration),
so we can backpropagate through a complete closed-loop rollout to minimise ITAE
(Integral of Time-Absolute Error) simultaneously across many setpoints.

Workflow
--------
1. Build a loss function with ``make_siso_pid_loss_fn`` or ``make_mimo_pid_loss_fn``.
2. Pass it to ``tune_pid_gains`` (or ``tune_mimo_pid_gains`` for MIMO).
3. Tuned gains are returned and can be copied into the factory defaults in pid.py.

Per-env convenience wrappers (``tune_cstr_pid``, etc.) do all of the above in one
call and print the resulting gains — run the module directly to re-tune everything:

    python -m target_gym.experts.pid_tuning
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from target_gym.experts.pid import (
    MIMOPIDParams,
    PIDParams,
    mimo_pid_reset,
    mimo_pid_step,
    pid_reset,
    pid_step,
)

# ---------------------------------------------------------------------------
# Loss function builders
# ---------------------------------------------------------------------------


def make_siso_pid_loss_fn(
    env,
    params,
    state_index: int,
    setpoint_index: int,
    reset_fn,
    target_range: tuple[float, float],
    n_targets: int = 16,
    n_steps: int | None = None,
    seed: int = 0,
):
    """
    Build a JAX-differentiable ITAE loss for a SISO PID.

    The loss is vmapped over ``n_targets`` uniformly-spaced setpoints, so the
    optimiser sees tracking performance across the full target range at once.

    Args:
        env            : JAX environment instance.
        params         : Environment parameter struct (must have ``delta_t``).
        state_index    : Index of the controlled state in ``env.get_obs()``.
        setpoint_index : Index of the setpoint in ``env.get_obs()``.
        reset_fn       : ``(key, params, target) -> EnvState`` — returns the
                         initial state with the given target set. Must be
                         JAX-traceable (no Python branching on traced values).
        target_range   : ``(lo, hi)`` — setpoint sweep range.
        n_targets      : Number of setpoints to vmap over.
        n_steps        : Rollout length (defaults to ``params.max_steps_in_episode``).
        seed           : PRNG seed for ``env.reset_env``.

    Returns:
        ``loss_fn(Kp, Ki, Kd) -> scalar`` — fully JIT/grad-able.
    """
    if n_steps is None:
        n_steps = int(params.max_steps_in_episode)
    targets = jnp.linspace(target_range[0], target_range[1], n_targets)
    key = jax.random.PRNGKey(seed)
    dt = float(params.delta_t)

    def loss_fn(Kp: float, Ki: float, Kd: float) -> float:
        pid_params = PIDParams(
            Kp=Kp,
            Ki=Ki,
            Kd=Kd,
            dt=dt,
            state_index=state_index,
            setpoint_index=setpoint_index,
            action_min=-1.0,
            action_max=1.0,
        )
        pid_state0 = pid_reset(pid_params)

        def run_one_target(target):
            init_env_state = reset_fn(key, params, target)

            def step_fn(carry, _):
                env_state, pid_state = carry
                obs = env.get_obs(env_state, params)
                action, new_pid_state = pid_step(pid_params, pid_state, obs)
                _, new_env_state, _, _, _ = env.step_env(
                    key, env_state, action[0], params
                )
                error = obs[setpoint_index] - obs[state_index]
                return (new_env_state, new_pid_state), jnp.abs(error)

            (_, _), abs_errors = jax.lax.scan(
                step_fn, (init_env_state, pid_state0), None, length=n_steps
            )
            # ITAE: weight errors by normalised time — penalises slow settling
            weights = (jnp.arange(n_steps, dtype=jnp.float32) + 1.0) / n_steps
            return jnp.mean(weights * abs_errors)

        return jnp.mean(jax.vmap(run_one_target)(targets))

    return loss_fn


def make_mimo_pid_loss_fn(
    env,
    params,
    reset_fn,
    obs_indices: tuple[tuple[int, int], tuple[int, int]],
    target_ranges: tuple[tuple[float, float], tuple[float, float]],
    n_targets: int = 8,
    n_steps: int | None = None,
    seed: int = 0,
):
    """
    Build a JAX-differentiable ITAE loss for a 2×2 MIMO PID.

    Args:
        env          : JAX environment instance.
        params       : Environment parameter struct.
        reset_fn     : ``(key, params, t1, t2) -> EnvState``.
        obs_indices  : ``((state1_idx, sp1_idx), (state2_idx, sp2_idx))``.
        target_ranges: ``((lo1, hi1), (lo2, hi2))``.
        n_targets    : Number of setpoint *pairs* to vmap over (grid).
        n_steps      : Rollout length (defaults to ``params.max_steps_in_episode``).
        seed         : PRNG seed.

    Returns:
        ``loss_fn(Kp1, Ki1, Kd1, Kp2, Ki2, Kd2) -> scalar``.
    """
    if n_steps is None:
        n_steps = int(params.max_steps_in_episode)

    t1_vals = jnp.linspace(target_ranges[0][0], target_ranges[0][1], n_targets)
    t2_vals = jnp.linspace(target_ranges[1][0], target_ranges[1][1], n_targets)
    # Cartesian product via meshgrid → (n_targets^2, 2)
    T1, T2 = jnp.meshgrid(t1_vals, t2_vals, indexing="ij")
    target_pairs = jnp.stack([T1.ravel(), T2.ravel()], axis=-1)

    (s1_idx, sp1_idx), (s2_idx, sp2_idx) = obs_indices
    key = jax.random.PRNGKey(seed)
    dt = float(params.delta_t)

    def loss_fn(Kp1, Ki1, Kd1, Kp2, Ki2, Kd2):
        mimo_params = MIMOPIDParams(
            pid1=PIDParams(
                Kp=Kp1,
                Ki=Ki1,
                Kd=Kd1,
                dt=dt,
                state_index=s1_idx,
                setpoint_index=sp1_idx,
                action_min=-1.0,
                action_max=1.0,
            ),
            pid2=PIDParams(
                Kp=Kp2,
                Ki=Ki2,
                Kd=Kd2,
                dt=dt,
                state_index=s2_idx,
                setpoint_index=sp2_idx,
                action_min=-1.0,
                action_max=1.0,
            ),
        )
        mimo_state0 = mimo_pid_reset(mimo_params)

        def run_one_pair(pair):
            t1, t2 = pair[0], pair[1]
            init_env_state = reset_fn(key, params, t1, t2)

            def step_fn(carry, _):
                env_state, pid_state = carry
                obs = env.get_obs(env_state, params)
                action, new_pid_state = mimo_pid_step(mimo_params, pid_state, obs)
                _, new_env_state, _, _, _ = env.step_env(key, env_state, action, params)
                e1 = obs[sp1_idx] - obs[s1_idx]
                e2 = obs[sp2_idx] - obs[s2_idx]
                return (new_env_state, new_pid_state), jnp.abs(e1) + jnp.abs(e2)

            (_, _), abs_errors = jax.lax.scan(
                step_fn, (init_env_state, mimo_state0), None, length=n_steps
            )
            weights = (jnp.arange(n_steps, dtype=jnp.float32) + 1.0) / n_steps
            return jnp.mean(weights * abs_errors)

        return jnp.mean(jax.vmap(run_one_pair)(target_pairs))

    return loss_fn


# ---------------------------------------------------------------------------
# Generic optimisers
# ---------------------------------------------------------------------------


def tune_pid_gains(
    loss_fn,
    init_Kp: float,
    init_Ki: float,
    init_Kd: float,
    n_grad_steps: int = 500,
    lr: float = 0.05,
    l2_reg: float = 0.0,
    verbose: bool = False,
) -> tuple[float, float, float]:
    """
    Optimise SISO PID gains by Adam gradient descent on a JAX loss function.

    Gains are parametrised in log-magnitude space with a fixed sign:
        Kp = sign(init_Kp) * exp(log_kp)
    This keeps magnitudes strictly positive and prevents sign flips during
    optimisation while keeping the gradient numerically well-conditioned.

    For systems where the controller saturates most of the time (e.g. CSTR),
    the ITAE loss alone is nearly flat for large Kp — the gains drift to
    impractically large values without improving tracking.  Set ``l2_reg > 0``
    (e.g. 1e-8) to add an L2 penalty on the gain magnitudes, which keeps them
    in a reasonable range.

    Args:
        loss_fn      : ``f(Kp, Ki, Kd) -> scalar``, must be JIT/grad-able.
        init_Kp/Ki/Kd: Initial gains.  Sign is preserved throughout.
        n_grad_steps : Adam steps.
        lr           : Adam learning rate.
        l2_reg       : L2 regularisation coefficient on gain magnitudes
                       (added as ``l2_reg * (Kp² + Ki² + Kd²)`` to the loss).
                       Prevents runaway in saturation-dominated systems.
        verbose      : Print loss and gains every 50 steps.

    Returns:
        ``(Kp, Ki, Kd)`` — tuned gains.
    """
    sign_Kp = float(jnp.sign(jnp.array(init_Kp))) or 1.0
    sign_Ki = float(jnp.sign(jnp.array(init_Ki))) or 1.0
    sign_Kd = float(jnp.sign(jnp.array(init_Kd))) or 1.0

    log_kp0 = float(jnp.log(abs(init_Kp) + 1e-8))
    log_ki0 = float(jnp.log(abs(init_Ki) + 1e-8))
    log_kd0 = float(jnp.log(abs(init_Kd) + 1e-8))

    def wrapped_loss(log_gains):
        log_kp, log_ki, log_kd = log_gains
        Kp = sign_Kp * jnp.exp(log_kp)
        Ki = sign_Ki * jnp.exp(log_ki)
        Kd = sign_Kd * jnp.exp(log_kd)
        tracking = loss_fn(Kp, Ki, Kd)
        reg = l2_reg * (Kp**2 + Ki**2 + Kd**2)
        return tracking + reg

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    log_gains = jnp.array([log_kp0, log_ki0, log_kd0])
    opt_state = optimizer.init(log_gains)
    grad_fn = jax.jit(jax.value_and_grad(wrapped_loss))

    for step in range(n_grad_steps):
        loss_val, grads = grad_fn(log_gains)
        updates, opt_state = optimizer.update(grads, opt_state)
        log_gains = optax.apply_updates(log_gains, updates)
        if verbose and step % 50 == 0:
            lg = log_gains
            print(
                f"  step {step:4d}: loss={float(loss_val):.6f}  "
                f"Kp={float(sign_Kp * jnp.exp(lg[0])):10.4f}  "
                f"Ki={float(sign_Ki * jnp.exp(lg[1])):10.4f}  "
                f"Kd={float(sign_Kd * jnp.exp(lg[2])):10.6f}"
            )

    lg = log_gains
    return (
        float(sign_Kp * jnp.exp(lg[0])),
        float(sign_Ki * jnp.exp(lg[1])),
        float(sign_Kd * jnp.exp(lg[2])),
    )


def tune_mimo_pid_gains(
    loss_fn,
    init_gains_1: tuple[float, float, float],
    init_gains_2: tuple[float, float, float],
    n_grad_steps: int = 500,
    lr: float = 0.05,
    l2_reg: float = 0.0,
    verbose: bool = False,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Optimise 6 MIMO PID gains jointly by Adam gradient descent.

    Returns:
        ``((Kp1, Ki1, Kd1), (Kp2, Ki2, Kd2))`` — tuned gains.
    """
    all_inits = list(init_gains_1) + list(init_gains_2)
    signs = [float(jnp.sign(jnp.array(g))) or 1.0 for g in all_inits]
    log_inits = [float(jnp.log(abs(g) + 1e-8)) for g in all_inits]

    def wrapped_loss(log_gains):
        gains = [signs[i] * jnp.exp(log_gains[i]) for i in range(6)]
        reg = l2_reg * sum(g**2 for g in gains)
        return loss_fn(*gains) + reg

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    log_gains = jnp.array(log_inits)
    opt_state = optimizer.init(log_gains)
    grad_fn = jax.jit(jax.value_and_grad(wrapped_loss))

    for step in range(n_grad_steps):
        loss_val, grads = grad_fn(log_gains)
        updates, opt_state = optimizer.update(grads, opt_state)
        log_gains = optax.apply_updates(log_gains, updates)
        if verbose and step % 50 == 0:
            g = [signs[i] * float(jnp.exp(log_gains[i])) for i in range(6)]
            print(
                f"  step {step:4d}: loss={float(loss_val):.6f}  "
                f"pid1=({g[0]:.3f},{g[1]:.3f},{g[2]:.4f})  "
                f"pid2=({g[3]:.3f},{g[4]:.3f},{g[5]:.4f})"
            )

    g = [signs[i] * float(jnp.exp(log_gains[i])) for i in range(6)]
    return (g[0], g[1], g[2]), (g[3], g[4], g[5])


# ---------------------------------------------------------------------------
# Per-environment convenience tuners
# ---------------------------------------------------------------------------


def tune_cstr_pid(
    n_grad_steps: int = 600,
    lr: float = 0.02,
    verbose: bool = True,
) -> tuple[float, float, float]:
    """Tune CSTR PID gains (Kp, Ki, Kd) via gradient descent."""
    from target_gym import CSTR, CSTRParams

    env = CSTR(integration_method="rk4_1")
    params = CSTRParams()

    loss_fn = make_siso_pid_loss_fn(
        env,
        params,
        state_index=0,  # Ca
        setpoint_index=2,  # target_Ca
        reset_fn=lambda key, p, target: env.reset_env(key, p)[1].replace(
            target_CA=target
        ),
        target_range=tuple(params.target_CA_range),
        n_targets=16,
        n_steps=int(params.max_steps_in_episode),
    )
    if verbose:
        print("Tuning CSTR PID...")
    Kp, Ki, Kd = tune_pid_gains(
        loss_fn,
        init_Kp=-100.0,
        init_Ki=-20.0,
        init_Kd=-0.5,
        n_grad_steps=n_grad_steps,
        lr=lr,
        l2_reg=1e-8,
        verbose=verbose,
    )
    if verbose:
        print(f"  → CSTR: Kp={Kp:.4f}, Ki={Ki:.4f}, Kd={Kd:.6f}")
    return Kp, Ki, Kd


def tune_first_order_pid(
    n_grad_steps: int = 400,
    lr: float = 0.05,
    verbose: bool = True,
) -> tuple[float, float, float]:
    """Tune FirstOrderSystem PID gains via gradient descent."""
    from target_gym import FirstOrderParams, FirstOrderSystem

    env = FirstOrderSystem(integration_method="rk4_1")
    params = FirstOrderParams()

    loss_fn = make_siso_pid_loss_fn(
        env,
        params,
        state_index=0,  # x
        setpoint_index=1,  # target_x
        reset_fn=lambda key, p, target: env.reset_env(key, p)[1].replace(
            target_x=target
        ),
        target_range=tuple(params.target_x_range),
        n_targets=16,
        n_steps=int(params.max_steps_in_episode),
    )
    if verbose:
        print("Tuning FirstOrder PID...")
    Kp, Ki, Kd = tune_pid_gains(
        loss_fn,
        init_Kp=2.0,
        init_Ki=0.5,
        init_Kd=0.01,
        n_grad_steps=n_grad_steps,
        lr=lr,
        verbose=verbose,
    )
    if verbose:
        print(f"  → FirstOrder: Kp={Kp:.4f}, Ki={Ki:.4f}, Kd={Kd:.6f}")
    return Kp, Ki, Kd


def tune_four_tank_pid(
    n_grad_steps: int = 400,
    lr: float = 0.05,
    verbose: bool = True,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Tune FourTank MIMO PID gains via gradient descent."""
    from target_gym import FourTank, FourTankParams

    env = FourTank(integration_method="rk4_1")
    params = FourTankParams()

    loss_fn = make_mimo_pid_loss_fn(
        env,
        params,
        reset_fn=lambda key, p, t1, t2: env.reset_env(key, p)[1].replace(
            target_h1=t1, target_h2=t2
        ),
        obs_indices=((0, 4), (1, 5)),  # (h1, target_h1), (h2, target_h2)
        target_ranges=(
            tuple(params.target_h1_range),
            tuple(params.target_h2_range),
        ),
        n_targets=6,  # 6×6 = 36 pairs
        n_steps=int(params.max_steps_in_episode),
    )
    if verbose:
        print("Tuning FourTank MIMO PID...")
    (Kp1, Ki1, Kd1), (Kp2, Ki2, Kd2) = tune_mimo_pid_gains(
        loss_fn,
        init_gains_1=(5.0, 1.0, 0.0),
        init_gains_2=(5.0, 1.0, 0.0),
        n_grad_steps=n_grad_steps,
        lr=lr,
        verbose=verbose,
    )
    if verbose:
        print(f"  → FourTank pid1: Kp={Kp1:.4f}, Ki={Ki1:.4f}, Kd={Kd1:.6f}")
        print(f"  → FourTank pid2: Kp={Kp2:.4f}, Ki={Ki2:.4f}, Kd={Kd2:.6f}")
    return (Kp1, Ki1, Kd1), (Kp2, Ki2, Kd2)


def tune_plane_pid(
    n_grad_steps: int = 500,
    lr: float = 0.02,
    verbose: bool = True,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Tune Airplane2D MIMO PID (power + stick) via gradient descent.

    Both loops close on altitude error (target_altitude - z).  pid1 controls
    power (coarse, slow), pid2 controls stick (fast pitch correction).
    Obs layout: [x_dot, z, z_dot, theta, theta_dot, gamma, target_altitude, power, stick]
                  idx 0  1    2      3        4        5          6             7      8
    """
    from target_gym.plane.env_jax import Airplane2D, PlaneParams

    env = Airplane2D(integration_method="rk4_1")
    params = PlaneParams()
    n_steps = int(params.max_steps_in_episode)
    dt = float(params.delta_t)

    # obs indices
    z_idx = 1  # state: altitude
    tgt_idx = 6  # setpoint: target_altitude

    targets = jnp.linspace(
        params.target_altitude_range[0],
        params.target_altitude_range[1],
        8,
    )
    key = jax.random.PRNGKey(0)

    # ITAE weights (computed once, outside the traced function)
    weights = (jnp.arange(n_steps, dtype=jnp.float32) + 1.0) / n_steps

    def loss_fn(Kp1, Ki1, Kd1, Kp2, Ki2, Kd2):
        mimo_params = MIMOPIDParams(
            pid1=PIDParams(
                Kp=Kp1,
                Ki=Ki1,
                Kd=Kd1,
                dt=dt,
                state_index=z_idx,
                setpoint_index=tgt_idx,
                action_min=-1.0,
                action_max=1.0,
            ),
            pid2=PIDParams(
                Kp=Kp2,
                Ki=Ki2,
                Kd=Kd2,
                dt=dt,
                state_index=z_idx,
                setpoint_index=tgt_idx,
                action_min=-1.0,
                action_max=1.0,
            ),
        )
        mimo_state0 = mimo_pid_reset(mimo_params)

        def run_one_target(target):
            init_state = env.reset_env(key, params)[1].replace(target_altitude=target)

            def step_fn(carry, _):
                env_state, pid_state = carry
                obs = env.get_obs(env_state)
                action, new_pid_state = mimo_pid_step(mimo_params, pid_state, obs)
                _, new_env_state, _, _, _ = env.step_env(key, env_state, action, params)
                error = obs[tgt_idx] - obs[z_idx]
                return (new_env_state, new_pid_state), jnp.abs(error)

            (_, _), abs_errors = jax.lax.scan(
                step_fn, (init_state, mimo_state0), None, length=n_steps
            )
            return jnp.mean(weights * abs_errors)

        return jnp.mean(jax.vmap(run_one_target)(targets))

    if verbose:
        print("Tuning Plane MIMO PID (power + stick)...")
    (Kp1, Ki1, Kd1), (Kp2, Ki2, Kd2) = tune_mimo_pid_gains(
        loss_fn,
        init_gains_1=(0.0005, 1e-5, 0.001),
        init_gains_2=(0.0005, 1e-5, 0.001),
        n_grad_steps=n_grad_steps,
        lr=lr,
        verbose=verbose,
    )
    if verbose:
        print(f"  → Plane pid1 (power): Kp={Kp1:.6f}, Ki={Ki1:.6f}, Kd={Kd1:.6f}")
        print(f"  → Plane pid2 (stick): Kp={Kp2:.6f}, Ki={Ki2:.6f}, Kd={Kd2:.6f}")
    return (Kp1, Ki1, Kd1), (Kp2, Ki2, Kd2)


# ---------------------------------------------------------------------------
# 3D plane tuners  (heading / circle / figure-8)
# ---------------------------------------------------------------------------
#
# Each 3D task has its own JAX-differentiable closed-loop rollout that
# mirrors the python-stateful controllers in pid.py. The controllers are
# purely functional for tracing (no Python-side accumulators).


def _freeze_if_done(prev_state, new_state, done):
    """Return new_state when not done, else freeze on prev_state.

    The plane3d ISA air-density model has ``rho = (1 - z*L/T0)**5.26 / (R*T)``
    which goes complex (→NaN) once z > 44_331 m. After termination
    (``z <= min_alt`` or ``z >= max_alt``) the dynamics keep being integrated,
    and divergent trajectories quickly cross that threshold, NaN the state,
    and poison the gradient. Freezing the state on terminate keeps the
    rollout finite for autodiff.
    """
    return jax.tree.map(
        lambda old, new: jnp.where(done, old, new), prev_state, new_state
    )


def _safe_step_env(env, key, env_state, action, params, done, safe_state):
    """Step env with a NaN-safe gradient on terminate.

    Plain ``where(done, prev, step_env(prev, ...))`` masks the *forward* value
    correctly but JAX's VJP for ``where`` evaluates both branches, so a NaN
    coming out of ``step_env`` (e.g. from divergent post-terminal dynamics)
    multiplied by 0 still yields NaN and poisons the Adam moments.

    Trick: substitute a *tame* state for ``env_state`` *before* calling
    ``step_env`` whenever ``done`` is True. ``step_env`` therefore always sees
    a finite, well-conditioned input and produces finite gradients regardless
    of how badly the closed-loop trajectory diverges after termination. The
    outer ``where`` then freezes the visible state on the pre-done value.
    """
    safe_state = jax.lax.stop_gradient(safe_state)
    effective_state = jax.tree.map(
        lambda old, safe: jnp.where(done, safe, old),
        env_state,
        safe_state,
    )
    obs, new_state, reward, new_done, info = env.step_env(
        key,
        effective_state,
        action,
        params,
    )
    new_state = jax.tree.map(
        lambda old, new: jnp.where(done, old, new),
        env_state,
        new_state,
    )
    return obs, new_state, reward, new_done, info


def _build_plane3d_alt_stick_scan(Kp_alt, Ki_alt, Kd_alt, dt):
    """
    Returns (init_state, step_fn) for the altitude→stick PID loop, pure JAX.

    step_fn: (obs, pid_carry) -> (stick, new_carry)
    obs uses plane3d layout where z is at index 2 and target_altitude at 10.
    """

    def init():
        return (jnp.zeros(()), jnp.zeros(()))  # integral, prev_err

    def step(obs, carry):
        integral, prev_err = carry
        e = obs[10] - obs[2]
        new_integral = integral + e * dt
        deriv = (e - prev_err) / dt
        u = Kp_alt * e + Ki_alt * new_integral + Kd_alt * deriv
        u_cl = jnp.clip(u, -1.0, 1.0)
        new_integral = jnp.where(u == u_cl, new_integral, integral)
        return u_cl, (new_integral, e)

    return init, step


def tune_plane3d_heading_pid(
    n_grad_steps: int = 400,
    lr: float = 0.02,
    verbose: bool = True,
    n_targets: int = 8,
):
    """
    Tune heading-task PID: altitude PID (3) + heading PID (3) + bank P (1) = 7 gains.

    Closed-loop objective: ITAE on altitude error + angular heading error.

    The controller is vmapped over (altitude, heading) setpoint pairs and
    differentiated through the full episode.
    """
    from target_gym.plane3d.env_jax import Plane3DHeading

    env = Plane3DHeading(integration_method="rk4_1")
    params = env.default_params
    # Truncate rollout to avoid NaN gradients from BPTT over 10 000 steps.
    n_steps = min(int(params.max_steps_in_episode), 2_000)
    dt = float(params.delta_t)

    alt_lo, alt_hi = params.target_altitude_range
    hdg_lo, hdg_hi = params.target_heading_range
    # Cartesian grid of setpoint pairs
    a_vals = jnp.linspace(alt_lo, alt_hi, n_targets)
    h_vals = jnp.linspace(hdg_lo, hdg_hi, n_targets)
    A, H = jnp.meshgrid(a_vals, h_vals, indexing="ij")
    pairs = jnp.stack([A.ravel(), H.ravel()], axis=-1)

    key = jax.random.PRNGKey(0)
    weights = (jnp.arange(n_steps, dtype=jnp.float32) + 1.0) / n_steps
    max_bank_rad = jnp.deg2rad(25.0)
    cruise_power = 0.6

    def loss_fn(Kp_alt, Ki_alt, Kd_alt, Kp_hdg, Ki_hdg, Kd_hdg, Kp_bank):
        alt_init, alt_step = _build_plane3d_alt_stick_scan(Kp_alt, Ki_alt, Kd_alt, dt)

        def run_one(pair):
            t_alt, t_hdg = pair[0], pair[1]
            _, st = env.reset_env(key, params)
            st = st.replace(target_altitude=t_alt, target_heading=t_hdg)
            carry0 = (
                alt_init(),
                jnp.zeros(()),
                jnp.zeros(()),
            )  # (alt_carry, hdg_int, hdg_prev)

            def step_fn(carry, _):
                env_state, (alt_carry, hdg_int, hdg_prev), done = carry
                obs = env.get_obs(env_state)
                stick, alt_carry = alt_step(obs, alt_carry)

                hdg_err = jnp.arctan2(
                    jnp.sin(obs[11] - obs[9]), jnp.cos(obs[11] - obs[9])
                )
                new_hdg_int = hdg_int + hdg_err * dt
                hdg_deriv = (hdg_err - hdg_prev) / dt
                desired_bank = (
                    Kp_hdg * hdg_err + Ki_hdg * new_hdg_int + Kd_hdg * hdg_deriv
                )
                desired_bank = jnp.clip(desired_bank, -max_bank_rad, max_bank_rad)
                bank_err = obs[6] - desired_bank
                aileron = jnp.clip(Kp_bank * bank_err, -1.0, 1.0)
                # Anti-windup on heading integrator
                new_hdg_int = jnp.where(jnp.abs(aileron) >= 1.0, hdg_int, new_hdg_int)
                action = jnp.array([cruise_power * 2.0 - 1.0, stick, aileron])
                _, new_env_state, _, new_done, _ = _safe_step_env(
                    env,
                    key,
                    env_state,
                    action,
                    params,
                    done,
                    st,
                )
                done = jnp.logical_or(done, new_done)

                alt_err = jnp.abs(obs[10] - obs[2]) / (alt_hi - alt_lo + 1e-3)
                hdg_cost = jnp.abs(hdg_err) / jnp.pi
                # Saturate per-step cost when done so divergent post-terminal
                # trajectories can't dominate the loss
                step_cost = jnp.where(done, 2.0, alt_err + hdg_cost)
                return (
                    new_env_state,
                    (alt_carry, new_hdg_int, hdg_err),
                    done,
                ), step_cost

            (_, _, _), costs = jax.lax.scan(
                step_fn, (st, carry0, jnp.array(False)), None, length=n_steps
            )
            return jnp.mean(weights * costs)

        return jnp.mean(jax.vmap(run_one)(pairs))

    all_inits = (0.0005, 1e-5, 0.001, 0.5, 0.0, 0.0, -2.0)
    signs = [float(jnp.sign(jnp.array(g))) or 1.0 for g in all_inits]
    log_inits = [float(jnp.log(abs(g) + 1e-8)) for g in all_inits]

    def wrapped(log_gains):
        gains = [signs[i] * jnp.exp(log_gains[i]) for i in range(7)]
        return loss_fn(*gains)

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    log_gains = jnp.array(log_inits)
    opt_state = optimizer.init(log_gains)
    grad_fn = jax.jit(jax.value_and_grad(wrapped))

    if verbose:
        print("Tuning Plane3D heading PID...")
    for step in range(n_grad_steps):
        loss_val, grads = grad_fn(log_gains)
        updates, opt_state = optimizer.update(grads, opt_state)
        log_gains = optax.apply_updates(log_gains, updates)
        if verbose and step % 50 == 0:
            g = [signs[i] * float(jnp.exp(log_gains[i])) for i in range(7)]
            print(
                f"  step {step:4d}: loss={float(loss_val):.4f}  "
                f"alt=({g[0]:.4g},{g[1]:.4g},{g[2]:.4g})  "
                f"hdg=({g[3]:.3f},{g[4]:.3f},{g[5]:.3f})  bank={g[6]:.3f}"
            )

    g = [signs[i] * float(jnp.exp(log_gains[i])) for i in range(7)]
    return (g[0], g[1], g[2]), (g[3], g[4], g[5]), g[6]


def tune_plane3d_circle_pid(
    n_grad_steps: int = 400,
    lr: float = 0.02,
    verbose: bool = True,
    n_targets: int = 6,
):
    """
    Tune circle-task PID: altitude PID (3) + radial PID (3) + bank P (1) = 7 gains.

    Closed-loop objective: ITAE on altitude error + radial distance to circle.
    """
    from target_gym.plane3d.env_jax import Plane3DCircle

    env = Plane3DCircle(integration_method="rk4_1")
    params = env.default_params
    # Truncate rollout: the circle task accumulates state drift over long
    # episodes, and BPTT through the full 10 000 steps produces gradients
    # that overflow to NaN within a few Adam steps.  2 000 steps (~200 s)
    # is enough to evaluate tracking quality while keeping gradients stable.
    n_steps = min(int(params.max_steps_in_episode), 2_000)
    dt = float(params.delta_t)

    alt_lo, alt_hi = params.target_altitude_range
    r_lo, r_hi = params.target_radius_range
    a_vals = jnp.linspace(alt_lo, alt_hi, n_targets)
    r_vals = jnp.linspace(r_lo, r_hi, n_targets)
    A, R = jnp.meshgrid(a_vals, r_vals, indexing="ij")
    pairs = jnp.stack([A.ravel(), R.ravel()], axis=-1)

    # We randomise the starting angle for each pair so the tuner sees
    # diverse initial conditions.
    n_seeds = 4
    seeds = jnp.arange(n_seeds)

    weights = (jnp.arange(n_steps, dtype=jnp.float32) + 1.0) / n_steps
    max_bank_rad = jnp.deg2rad(25.0)
    target_bank_rad = jnp.deg2rad(15.0)
    cruise_power = 0.6

    def loss_fn(Kp_alt, Ki_alt, Kd_alt, Kp_rad, Ki_rad, Kd_rad, Kp_bank):
        alt_init, alt_step = _build_plane3d_alt_stick_scan(Kp_alt, Ki_alt, Kd_alt, dt)

        def run_one(pair_and_seed):
            pair, seed = pair_and_seed
            t_alt, t_rad = pair[0], pair[1]
            key = jax.random.PRNGKey(seed)
            _, st = env.reset_env(key, params)
            st = st.replace(target_altitude=t_alt, target_radius=t_rad)
            carry0 = (alt_init(), jnp.zeros(()), jnp.zeros(()))

            def step_fn(carry, _):
                env_state, (alt_carry, rad_int, rad_prev), done = carry
                obs = env.get_obs(env_state)
                stick, alt_carry = alt_step(obs, alt_carry)

                rel_x, rel_y = obs[11], obs[12]
                radius = obs[13]
                dist = jnp.sqrt(rel_x * rel_x + rel_y * rel_y + 1e-8)
                rad_err = dist - radius

                new_rad_int = rad_int + rad_err * dt
                rad_deriv = (rad_err - rad_prev) / dt
                bank_corr = Kp_rad * rad_err + Ki_rad * new_rad_int + Kd_rad * rad_deriv
                desired_bank = jnp.clip(
                    target_bank_rad + bank_corr, -max_bank_rad, max_bank_rad
                )
                bank_err = obs[6] - desired_bank
                aileron = jnp.clip(Kp_bank * bank_err, -1.0, 1.0)
                new_rad_int = jnp.where(jnp.abs(aileron) >= 1.0, rad_int, new_rad_int)
                action = jnp.array([cruise_power * 2.0 - 1.0, stick, aileron])
                _, new_env_state, _, new_done, _ = _safe_step_env(
                    env,
                    key,
                    env_state,
                    action,
                    params,
                    done,
                    st,
                )
                done = jnp.logical_or(done, new_done)

                alt_err = jnp.abs(obs[10] - obs[2]) / (alt_hi - alt_lo + 1e-3)
                rad_cost = jnp.abs(rad_err) / (r_hi + 1e-3)
                step_cost = jnp.where(done, 2.0, alt_err + rad_cost)
                return (
                    new_env_state,
                    (alt_carry, new_rad_int, rad_err),
                    done,
                ), step_cost

            (_, _, _), costs = jax.lax.scan(
                step_fn, (st, carry0, jnp.array(False)), None, length=n_steps
            )
            return jnp.mean(weights * costs)

        # vmap over pairs × seeds
        all_pairs = jnp.repeat(pairs, n_seeds, axis=0)
        all_seeds = jnp.tile(seeds, pairs.shape[0])
        return jnp.mean(jax.vmap(run_one)((all_pairs, all_seeds)))

    all_inits = (0.0005, 1e-5, 0.001, 1e-5, 0.0, 0.0, -2.0)
    signs = [float(jnp.sign(jnp.array(g))) or 1.0 for g in all_inits]
    log_inits = [float(jnp.log(abs(g) + 1e-8)) for g in all_inits]

    def wrapped(log_gains):
        gains = [signs[i] * jnp.exp(log_gains[i]) for i in range(7)]
        return loss_fn(*gains)

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    log_gains = jnp.array(log_inits)
    opt_state = optimizer.init(log_gains)
    grad_fn = jax.jit(jax.value_and_grad(wrapped))

    if verbose:
        print("Tuning Plane3D circle PID...")
    for step in range(n_grad_steps):
        loss_val, grads = grad_fn(log_gains)
        updates, opt_state = optimizer.update(grads, opt_state)
        log_gains = optax.apply_updates(log_gains, updates)
        if verbose and step % 50 == 0:
            g = [signs[i] * float(jnp.exp(log_gains[i])) for i in range(7)]
            print(
                f"  step {step:4d}: loss={float(loss_val):.4f}  "
                f"alt=({g[0]:.4g},{g[1]:.4g},{g[2]:.4g})  "
                f"rad=({g[3]:.4g},{g[4]:.4g},{g[5]:.4g})  bank={g[6]:.3f}"
            )

    g = [signs[i] * float(jnp.exp(log_gains[i])) for i in range(7)]
    return (g[0], g[1], g[2]), (g[3], g[4], g[5]), g[6]


def tune_plane3d_figure8_pid(
    n_grad_steps: int = 400,
    lr: float = 0.02,
    verbose: bool = True,
    n_targets: int = 6,
):
    """
    Tune figure-8 task PID: altitude PID (3) + heading P (1) + bank P (1) = 5 gains.

    The figure-8 PID tracks the twisted 3D lemniscate.  Obs layout:
      [... psi(9), target_alt(10), target_radius(11),
       nearest_dx(12), nearest_dy(13), nearest_dz(14), tangent_heading(15), ...]
    Altitude PID drives nearest_dz → 0.  Heading blends tangent with correction.
    """
    from target_gym.plane3d.env_jax import Plane3DFigureEight

    env = Plane3DFigureEight(integration_method="rk4_1")
    params = env.default_params
    n_steps = min(int(params.max_steps_in_episode), 2_000)
    dt = float(params.delta_t)

    alt_lo, alt_hi = params.target_altitude_range
    r_lo, r_hi = params.target_radius_range
    a_vals = jnp.linspace(alt_lo, alt_hi, n_targets)
    r_vals = jnp.linspace(r_lo, r_hi, n_targets)
    A, R = jnp.meshgrid(a_vals, r_vals, indexing="ij")
    pairs = jnp.stack([A.ravel(), R.ravel()], axis=-1)

    key = jax.random.PRNGKey(0)
    weights = (jnp.arange(n_steps, dtype=jnp.float32) + 1.0) / n_steps
    max_bank_rad = jnp.deg2rad(25.0)
    cruise_power = 0.6
    cost_cap = 5.0

    def loss_fn(Kp_alt, Ki_alt, Kd_alt, Kp_hdg, Kp_bank):

        def run_one(pair):
            t_alt, t_rad = pair[0], pair[1]
            _, st = env.reset_env(key, params)
            st = st.replace(target_altitude=t_alt, target_radius=t_rad)
            # carry: (alt_int, alt_prev)
            carry0 = (jnp.zeros(()), jnp.zeros(()))

            def step_fn(carry, _):
                env_state, (alt_int, alt_prev), done = carry
                obs = env.get_obs(env_state, params)

                # Altitude PID on nearest_dz (obs[14])
                alt_err = obs[14]
                new_alt_int = alt_int + alt_err * dt
                alt_d = (alt_err - alt_prev) / dt
                stick = jnp.clip(
                    Kp_alt * alt_err + Ki_alt * new_alt_int + Kd_alt * alt_d,
                    -1.0,
                    1.0,
                )
                # Anti-windup
                new_alt_int = jnp.where(
                    jnp.abs(stick) >= 1.0,
                    alt_int,
                    new_alt_int,
                )

                # Heading: blend tangent (obs[15]) with correction (obs[12:14])
                nearest_dx, nearest_dy = obs[12], obs[13]
                tangent_hdg = obs[15]
                lat_dist = jnp.sqrt(nearest_dx**2 + nearest_dy**2 + 1e-6)
                blend = jnp.clip(lat_dist / (0.05 * t_rad + 1e-3), 0.0, 1.0)
                corr_hdg = jnp.arctan2(nearest_dy, nearest_dx)
                bx = blend * jnp.cos(corr_hdg) + (1.0 - blend) * jnp.cos(tangent_hdg)
                by = blend * jnp.sin(corr_hdg) + (1.0 - blend) * jnp.sin(tangent_hdg)
                desired_hdg = jnp.arctan2(by, bx)

                psi = obs[9]
                hdg_err = jnp.arctan2(
                    jnp.sin(desired_hdg - psi),
                    jnp.cos(desired_hdg - psi),
                )
                desired_bank = jnp.clip(Kp_hdg * hdg_err, -max_bank_rad, max_bank_rad)
                aileron = jnp.clip(Kp_bank * (obs[6] - desired_bank), -1.0, 1.0)

                action = jnp.array([cruise_power * 2.0 - 1.0, stick, aileron])
                _, new_env_state, _, new_done, _ = _safe_step_env(
                    env,
                    key,
                    env_state,
                    action,
                    params,
                    done,
                    st,
                )
                done = jnp.logical_or(done, new_done)

                # Cost: 3D distance to curve
                curve_dist = jnp.sqrt(obs[12] ** 2 + obs[13] ** 2 + obs[14] ** 2 + 1e-8)
                curve_err = curve_dist / (r_hi + 1e-3)
                raw_cost = jnp.minimum(curve_err, cost_cap)
                step_cost = jnp.where(done, 2.0, raw_cost)
                return (new_env_state, (new_alt_int, alt_err), done), step_cost

            (_, _, _), costs = jax.lax.scan(
                step_fn, (st, carry0, jnp.array(False)), None, length=n_steps
            )
            return jnp.mean(weights * costs)

        return jnp.mean(jax.vmap(run_one)(pairs))

    # Gains: (Kp_alt, Ki_alt, Kd_alt, Kp_hdg, Kp_bank)
    all_inits = (0.0005, 1e-5, 0.001, 0.5, -2.0)
    signs = [float(jnp.sign(jnp.array(g))) or 1.0 for g in all_inits]
    log_inits = [float(jnp.log(abs(g) + 1e-8)) for g in all_inits]

    def wrapped(log_gains):
        gains = [signs[i] * jnp.exp(log_gains[i]) for i in range(5)]
        return loss_fn(*gains)

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    log_gains = jnp.array(log_inits)
    opt_state = optimizer.init(log_gains)
    grad_fn = jax.jit(jax.value_and_grad(wrapped))

    if verbose:
        print("Tuning Plane3D figure-8 PID (heading-chasing)...")
    for step in range(n_grad_steps):
        loss_val, grads = grad_fn(log_gains)
        finite = jnp.isfinite(grads)
        safe_grads = jnp.where(finite, grads, 0.0)
        updates, new_opt_state = optimizer.update(safe_grads, opt_state)
        any_bad = jnp.logical_not(jnp.all(finite))
        log_gains = jnp.where(
            any_bad, log_gains, optax.apply_updates(log_gains, updates)
        )
        opt_state = jax.tree.map(
            lambda old, new: jnp.where(any_bad, old, new), opt_state, new_opt_state
        )
        if verbose and step % 50 == 0:
            g = [signs[i] * float(jnp.exp(log_gains[i])) for i in range(5)]
            print(
                f"  step {step:4d}: loss={float(loss_val):.4f}  "
                f"alt=({g[0]:.4g},{g[1]:.4g},{g[2]:.4g})  "
                f"hdg={g[3]:.3f}  bank={g[4]:.3f}"
            )

    g = [signs[i] * float(jnp.exp(log_gains[i])) for i in range(5)]
    return (g[0], g[1], g[2]), g[3], g[4]


# ---------------------------------------------------------------------------
# Tune all environments and save gains to data/pid_gains.json
# ---------------------------------------------------------------------------


def tune_all_and_save(verbose: bool = True, force: bool = False) -> None:
    """Run all PID tuning routines and persist the results to data/pid_gains.json.

    Results are saved incrementally after each environment so a crash mid-way
    does not discard completed work.  Pass ``force=True`` to re-tune envs that
    already have gains in the file.
    """
    import json
    import time

    from target_gym.experts.pid import _GAINS_FILE

    # Load any partial results that may already exist.
    _GAINS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if _GAINS_FILE.exists():
        with open(_GAINS_FILE) as f:
            gains: dict = json.load(f)
    else:
        gains = {}

    def _save():
        with open(_GAINS_FILE, "w") as f:
            json.dump(gains, f, indent=2)
        import target_gym.experts.pid as _pid_mod

        _pid_mod._gains_cache = gains

    def _run(key, label, tune_fn, encode_fn):
        if key in gains and not force:
            if verbose:
                print(
                    f"[pid_tuning] {label}: already tuned, skipping (use force=True to re-tune)"
                )
            return
        if verbose:
            print(f"\n[pid_tuning] === {label} ===")
        t0 = time.time()
        result = tune_fn()
        elapsed = time.time() - t0
        gains[key] = encode_fn(result)
        _save()
        if verbose:
            print(
                f"[pid_tuning] {label} done in {elapsed:.1f}s → saved to {_GAINS_FILE}"
            )

    _run(
        "cstr",
        "CSTR",
        lambda: tune_cstr_pid(verbose=verbose),
        lambda r: {"Kp": r[0], "Ki": r[1], "Kd": r[2]},
    )
    _run(
        "first_order",
        "FirstOrder",
        lambda: tune_first_order_pid(verbose=verbose),
        lambda r: {"Kp": r[0], "Ki": r[1], "Kd": r[2]},
    )
    _run(
        "four_tank",
        "FourTank",
        lambda: tune_four_tank_pid(verbose=verbose),
        lambda r: {
            "pid1": {"Kp": r[0][0], "Ki": r[0][1], "Kd": r[0][2]},
            "pid2": {"Kp": r[1][0], "Ki": r[1][1], "Kd": r[1][2]},
        },
    )
    _run(
        "plane",
        "Plane (Airplane2D)",
        lambda: tune_plane_pid(verbose=verbose),
        lambda r: {
            "pid1": {"Kp": r[0][0], "Ki": r[0][1], "Kd": r[0][2]},
            "pid2": {"Kp": r[1][0], "Ki": r[1][1], "Kd": r[1][2]},
        },
    )
    _run(
        "plane3d_heading",
        "Plane3D Heading",
        lambda: tune_plane3d_heading_pid(verbose=verbose),
        lambda r: {
            "alt": {"Kp": r[0][0], "Ki": r[0][1], "Kd": r[0][2]},
            "hdg": {"Kp": r[1][0], "Ki": r[1][1], "Kd": r[1][2]},
            "bank": {"Kp": r[2]},
        },
    )
    _run(
        "plane3d_circle",
        "Plane3D Circle",
        lambda: tune_plane3d_circle_pid(verbose=verbose),
        lambda r: {
            "alt": {"Kp": r[0][0], "Ki": r[0][1], "Kd": r[0][2]},
            "rad": {"Kp": r[1][0], "Ki": r[1][1], "Kd": r[1][2]},
            "bank": {"Kp": r[2]},
        },
    )
    _run(
        "plane3d_figure8",
        "Plane3D Figure-8",
        lambda: tune_plane3d_figure8_pid(verbose=verbose),
        lambda r: {
            "alt": {"Kp": r[0][0], "Ki": r[0][1], "Kd": r[0][2]},
            "hdg": {"Kp": r[1], "Ki": 0.0, "Kd": 0.0},
            "bank": {"Kp": r[2]},
        },
    )

    if verbose:
        print(f"\n[pid_tuning] All environments tuned. Gains at {_GAINS_FILE}")


# ---------------------------------------------------------------------------
# Run all tuners and print gains ready to paste into pid.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Tune PID gains for all target_gym environments."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-tune envs that already have gains in the file.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Tune a single env by key (cstr, first_order, four_tank, plane, "
        "plane3d_heading, plane3d_circle, plane3d_figure8). Omit to tune all.",
    )
    args = parser.parse_args()

    if args.env is not None:
        # Single-env mode: find and run just the requested tuner.
        _single_map = {
            "cstr": tune_cstr_pid,
            "first_order": tune_first_order_pid,
            "four_tank": tune_four_tank_pid,
            "plane": tune_plane_pid,
            "plane3d_heading": tune_plane3d_heading_pid,
            "plane3d_circle": tune_plane3d_circle_pid,
            "plane3d_figure8": tune_plane3d_figure8_pid,
        }
        if args.env not in _single_map:
            print(f"Unknown env '{args.env}'. Choices: {list(_single_map)}")
        else:
            print(f"Tuning {args.env} only...")
            _single_map[args.env](verbose=True)
    else:
        tune_all_and_save(verbose=True, force=args.force)
