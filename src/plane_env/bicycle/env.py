from typing import Sequence

import numpy as np
from flax import struct
from jax.tree_util import Partial as partial

try:
    import chex
    import jax
    import jax.numpy as jnp
except Exception:
    jax = None
    jnp = None
    chex = None

EPS = 1e-8


@struct.dataclass
class EnvState:
    omega: float  # tilt angle [rad]
    omega_dot: float  # tilt angular velocity
    theta: float  # steering angle [rad]
    theta_dot: float  # steering angular velocity
    psi: float  # heading angle [rad]
    x_f: float  # front wheel x
    y_f: float  # front wheel y
    x_b: float  # back wheel x
    y_b: float  # back wheel y
    last_d: float  # last displacement action
    t: int


@struct.dataclass
class EnvParams:
    c: float = 0.66
    dCM: float = 0.30
    h: float = 0.94
    l: float = 1.11
    Mc: float = 15.0
    Md: float = 1.7
    Mp: float = 60.0
    r: float = 0.34
    v: float = 10.0 / 3.6
    g: float = 9.81

    max_torque: float = 2.0
    max_disp: float = 0.02

    delta_t: float = 0.05

    max_tilt_deg: float = 12.0
    max_steps_in_episode: int = 1000

    use_goal: bool = False
    goal_x: float = 0.0
    goal_y: float = 100.0
    goal_radius: float = 10.0

    tiny: float = 1e-6
    discrete_actions: bool = False


# -------- Physics helpers --------


def total_mass(p: EnvParams, xp=jnp):
    return p.Mc + p.Mp + 2 * p.Md


def inertia_bicycle_and_cyclist(p: EnvParams, xp=jnp):
    return (13.0 / 3.0) * p.Mc * (p.h**2) + p.Mp * ((p.h + p.dCM) ** 2)


def tyre_inertia_Idc(p: EnvParams, xp=jnp):
    return p.Md * (p.r**2)


def tyre_inertia_Idv(p: EnvParams, xp=jnp):
    return 1.5 * p.Md * (p.r**2)


def tyre_inertia_Idl(p: EnvParams, xp=jnp):
    return 0.5 * p.Md * (p.r**2)


def phi_total(omega, d, p: EnvParams, xp=jnp):
    return omega + xp.arctan(d / p.h)


def radius_front(theta, p: EnvParams, xp=jnp):
    s = xp.abs(xp.sin(theta))
    return xp.where(s < p.tiny, 1e6, p.l / s)


def radius_back(theta, p: EnvParams, xp=jnp):
    t = xp.abs(xp.tan(theta))
    return xp.where(t < p.tiny, 1e6, p.l / t)


def radius_CM(theta, p: EnvParams, xp=jnp):
    tan_theta = xp.tan(theta)
    denom = xp.where(xp.abs(tan_theta) < p.tiny, p.tiny, tan_theta)
    return xp.sqrt((p.l - p.c) ** 2 + (p.l**2) / (denom**2))


def tyre_angular_velocity(p: EnvParams, xp=jnp):
    return p.v / p.r


def theta_ddot_from_eq3(T, omega_dot, p: EnvParams, xp=jnp):
    Idv = tyre_inertia_Idv(p, xp=xp)
    Idl = tyre_inertia_Idl(p, xp=xp)
    sigma_dot = tyre_angular_velocity(p, xp=xp)
    return (T - Idv * sigma_dot * omega_dot) / (Idl + EPS)


def omega_ddot_from_eq2(omega, theta, theta_dot, d, p: EnvParams, xp=jnp):
    I_tot = inertia_bicycle_and_cyclist(p, xp=xp)
    M = total_mass(p, xp=xp)
    phi = phi_total(omega, d, p, xp=xp)

    term_gravity = M * p.h * p.g * xp.sin(phi)

    Idc = tyre_inertia_Idc(p, xp=xp)
    sigma_dot = tyre_angular_velocity(p, xp=xp)

    r_f = radius_front(theta, p, xp=xp)
    r_b = radius_back(theta, p, xp=xp)
    r_CM = radius_CM(theta, p, xp=xp)

    term_centrifugal = (p.v**2) * (
        p.Md * p.r / (r_f + EPS) + p.Md * p.r / (r_b + EPS) + M * p.h / (r_CM + EPS)
    )

    sgn_theta = xp.sign(theta)
    term_cross = Idc * sigma_dot * theta_dot + sgn_theta * term_centrifugal

    return (term_gravity - xp.cos(phi) * term_cross) / (I_tot + EPS)


# -------- Dynamics update (RK4) --------


def derivatives(state: EnvState, action, p: EnvParams, xp=jnp):
    a_T, a_d = action
    T = a_T * p.max_torque
    d = a_d * p.max_disp

    theta_dd = theta_ddot_from_eq3(T, state.omega_dot, p, xp=xp)
    omega_dd = omega_ddot_from_eq2(
        state.omega, state.theta, state.theta_dot, d, p, xp=xp
    )

    return theta_dd, omega_dd, T, d


def rk4_step(state: EnvState, action, p: EnvParams, xp=jnp):
    dt = p.delta_t

    def f(s: EnvState):
        theta_dd, omega_dd, _, d = derivatives(s, action, p, xp)
        return dict(
            omega=s.omega_dot,
            omega_dot=omega_dd,
            theta=s.theta_dot,
            theta_dot=theta_dd,
            psi=(p.v * xp.sin(s.theta)) / p.l,  # ψ̇ per steering geometry
        )

    # RK4 integration for core dynamics
    k1 = f(state)
    k2 = f(
        state.replace(
            omega=state.omega + 0.5 * dt * k1["omega"],
            omega_dot=state.omega_dot + 0.5 * dt * k1["omega_dot"],
            theta=state.theta + 0.5 * dt * k1["theta"],
            theta_dot=state.theta_dot + 0.5 * dt * k1["theta_dot"],
            psi=state.psi + 0.5 * dt * k1["psi"],
        )
    )
    k3 = f(
        state.replace(
            omega=state.omega + 0.5 * dt * k2["omega"],
            omega_dot=state.omega_dot + 0.5 * dt * k2["omega_dot"],
            theta=state.theta + 0.5 * dt * k2["theta"],
            theta_dot=state.theta_dot + 0.5 * dt * k2["theta_dot"],
            psi=state.psi + 0.5 * dt * k2["psi"],
        )
    )
    k4 = f(
        state.replace(
            omega=state.omega + dt * k3["omega"],
            omega_dot=state.omega_dot + dt * k3["omega_dot"],
            theta=state.theta + dt * k3["theta"],
            theta_dot=state.theta_dot + dt * k3["theta_dot"],
            psi=state.psi + dt * k3["psi"],
        )
    )

    omega_new = state.omega + (dt / 6.0) * (
        k1["omega"] + 2 * k2["omega"] + 2 * k3["omega"] + k4["omega"]
    )
    omega_dot_new = state.omega_dot + (dt / 6.0) * (
        k1["omega_dot"] + 2 * k2["omega_dot"] + 2 * k3["omega_dot"] + k4["omega_dot"]
    )
    theta_new = state.theta + (dt / 6.0) * (
        k1["theta"] + 2 * k2["theta"] + 2 * k3["theta"] + k4["theta"]
    )
    theta_dot_new = state.theta_dot + (dt / 6.0) * (
        k1["theta_dot"] + 2 * k2["theta_dot"] + 2 * k3["theta_dot"] + k4["theta_dot"]
    )
    psi_new = state.psi + (dt / 6.0) * (
        k1["psi"] + 2 * k2["psi"] + 2 * k3["psi"] + k4["psi"]
    )

    # update wheel positions
    dx = p.v * dt * xp.cos(psi_new)
    dy = p.v * dt * xp.sin(psi_new)

    xf_new, yf_new = state.x_f + dx, state.y_f + dy
    xb_new, yb_new = state.x_b + dx, state.y_b + dy

    _, _, _, d = derivatives(state, action, p, xp)

    return state.replace(
        omega=omega_new,
        omega_dot=omega_dot_new,
        theta=theta_new,
        theta_dot=theta_dot_new,
        psi=psi_new,
        x_f=xf_new,
        y_f=yf_new,
        x_b=xb_new,
        y_b=yb_new,
        last_d=d,
        t=state.t + 1,
    )


# -------- Env functions --------


def check_is_terminal(state: EnvState, p: EnvParams, xp=jnp):
    max_tilt_rad = xp.deg2rad(p.max_tilt_deg)
    terminated = xp.abs(state.omega) > max_tilt_rad
    truncated = state.t >= p.max_steps_in_episode
    return terminated, truncated


def compute_reward(state: EnvState, p: EnvParams, xp=jnp):
    terminated, truncated = check_is_terminal(state, p, xp=xp)
    default = xp.where(terminated, -1.0, 0.0)
    if p.use_goal:
        dx = state.x_f - state.x_b
        dy = state.y_f - state.y_b
        heading = xp.arctan2(dy, dx)
        gx = p.goal_x - state.x_f
        gy = p.goal_y - state.y_f
        goal_dir = xp.arctan2(gy, gx)
        g = xp.abs(xp.arctan2(xp.sin(goal_dir - heading), xp.cos(goal_dir - heading)))
        within_goal = (gx**2 + gy**2) <= (p.goal_radius**2)
        step_reward = (4.0 - 2.0 * g) * 0.00004
        reward = xp.where(terminated, -1.0, xp.where(within_goal, 0.01, step_reward))
        return reward
    else:
        return default


def get_obs(state: EnvState, p: EnvParams, xp=jnp):
    omega_ddot = omega_ddot_from_eq2(
        state.omega, state.theta, state.theta_dot, state.last_d, p, xp=xp
    )
    return xp.stack(
        [state.omega, state.omega_dot, omega_ddot, state.theta, state.theta_dot], axis=0
    ).astype(xp.float32)


def compute_next_state(action: Sequence[float], state: EnvState, p: EnvParams, xp=np):
    action = xp.clip(action, -1.0, 1.0)
    return rk4_step(state, action, p, xp=xp)
