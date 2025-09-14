from typing import Any, Dict, Sequence, Tuple

from flax import struct
from jax.tree_util import Partial as partial

try:
    import jax
    import jax.numpy as jnp
except Exception:
    jax = None
    jnp = None

# Import your integrator
from plane_env.integration import (
    compute_velocity_and_pos_from_acceleration_integration,
)

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
    last_d: float  # last displacement action (normalized)
    t: int
    # for rendering
    torque: float = jnp.nan
    displacement: float = jnp.nan


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

    use_dense: bool = True
    target_tilt: float = 0.0


# -------- Physics helpers (kept same as original) --------


def total_mass(p: EnvParams):
    return p.Mc + p.Mp + 2 * p.Md


def inertia_bicycle_and_cyclist(p: EnvParams):
    return (13.0 / 3.0) * p.Mc * (p.h**2) + p.Mp * ((p.h + p.dCM) ** 2)


def tyre_inertia_Idc(p: EnvParams):
    return p.Md * (p.r**2)


def tyre_inertia_Idv(p: EnvParams):
    return 1.5 * p.Md * (p.r**2)


def tyre_inertia_Idl(p: EnvParams):
    return 0.5 * p.Md * (p.r**2)


def phi_total(omega, d, p: EnvParams):
    return omega + jnp.arctan(d / p.h)


def radius_front(theta, p: EnvParams):
    s = jnp.abs(jnp.sin(theta))
    return jnp.where(s < p.tiny, 1e6, p.l / s)


def radius_back(theta, p: EnvParams):
    t = jnp.abs(jnp.tan(theta))
    return jnp.where(t < p.tiny, 1e6, p.l / t)


def radius_CM(theta, p: EnvParams):
    tan_theta = jnp.tan(theta)
    denom = jnp.where(jnp.abs(tan_theta) < p.tiny, p.tiny, tan_theta)
    return jnp.sqrt((p.l - p.c) ** 2 + (p.l**2) / (denom**2))


def tyre_angular_velocity(p: EnvParams):
    return p.v / p.r


def theta_ddot_from_eq3(T, omega_dot, p: EnvParams):
    Idv = tyre_inertia_Idv(p)
    Idl = tyre_inertia_Idl(p)
    sigma_dot = tyre_angular_velocity(p)
    return (T - Idv * sigma_dot * omega_dot) / (Idl + EPS)


def omega_ddot_from_eq2(omega, theta, theta_dot, d, p: EnvParams):
    I_tot = inertia_bicycle_and_cyclist(p)
    M = total_mass(p)
    phi = phi_total(omega, d, p)

    term_gravity = M * p.h * p.g * jnp.sin(phi)

    Idc = tyre_inertia_Idc(p)
    sigma_dot = tyre_angular_velocity(p)

    r_f = radius_front(theta, p)
    r_b = radius_back(theta, p)
    r_CM = radius_CM(theta, p)

    term_centrifugal = (p.v**2) * (
        p.Md * p.r / (r_f + EPS) + p.Md * p.r / (r_b + EPS) + M * p.h / (r_CM + EPS)
    )

    sgn_theta = jnp.sign(theta)
    term_cross = Idc * sigma_dot * theta_dot + sgn_theta * term_centrifugal

    return (term_gravity - jnp.cos(phi) * term_cross) / (I_tot + EPS)


# -------- New compute_acceleration compatible with integrator --------
# velocities: [omega_dot, theta_dot, psi_dot]
# positions:  [omega, theta, psi]
def compute_acceleration_bike(
    velocities: jnp.ndarray,
    positions: jnp.ndarray,
    action: Sequence[float],
    params: EnvParams,
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """
    Returns accelerations and metrics.

    - velocities: [omega_dot, theta_dot, psi_dot]
    - positions:  [omega, theta, psi]
    - action: (a_T, a_d) where a_T in [-1,1] scaled to torque, a_d in [-1,1] scaled to displacement
    """
    # unpack
    omega_dot, theta_dot, psi_dot = velocities
    omega, theta, psi = positions

    a_T, a_d = action
    T = a_T * params.max_torque
    d = a_d * params.max_disp

    # compute second derivatives
    theta_dd = theta_ddot_from_eq3(T, omega_dot, params)
    omega_dd = omega_ddot_from_eq2(omega, theta, theta_dot, d, params)

    # psi_dot = (p.v * sin(theta)) / p.l
    # psi_dd is time derivative of psi_dot = (p.v * cos(theta) * theta_dot) / p.l
    psi_dd = (params.v * jnp.cos(theta) * theta_dot) / (params.l + EPS)

    accelerations = jnp.array([omega_dd, theta_dd, psi_dd], dtype=jnp.float32)
    metrics = {
        "T": T,
        "d": d,
        "omega_dd": omega_dd,
        "theta_dd": theta_dd,
        "psi_dd": psi_dd,
    }
    return accelerations, metrics


# -------- Utilities: mapping state <-> vectors --------
def state_to_vec(state: EnvState):
    velocities = jnp.array(
        [state.omega_dot, state.theta_dot, (p.v * jnp.sin(state.theta)) / p.l]
    )  # psi_dot computed from state
    positions = jnp.array([state.omega, state.theta, state.psi])
    return velocities, positions


def vecs_to_state(
    state: EnvState,
    velocities: jnp.ndarray,
    positions: jnp.ndarray,
    metrics: Dict[str, Any],
    params: EnvParams,
) -> EnvState:
    omega_dot_new, theta_dot_new, psi_dot_new = velocities
    omega_new, theta_new, psi_new = positions

    # update wheel positions based on new psi
    dx = params.v * params.delta_t * jnp.cos(psi_new)
    dy = params.v * params.delta_t * jnp.sin(psi_new)
    xf_new, yf_new = state.x_f + dx, state.y_f + dy
    xb_new, yb_new = state.x_b + dx, state.y_b + dy
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
        last_d=metrics.get("d", state.last_d).squeeze(),
        t=state.t + 1,
    )


# -------- compute_next_state that uses the generic integrator --------
@partial(jax.jit, static_argnames=("integration_method",))
def compute_next_state(
    action: Sequence[float],
    state: EnvState,
    params: EnvParams,
    integration_method: str = "rk4_1",
):
    """
    action: (a_T, a_d) normalized in [-1,1]
    """
    # clip action
    a_T = jnp.clip(action[0], -1.0, 1.0)
    a_d = jnp.clip(action[1], -1.0, 1.0)
    action_clipped = (a_T, a_d)

    # Build compute_acceleration closure matching (velocities, positions) -> (acc,metrics)
    def _accel_fn(velocities, positions):
        return compute_acceleration_bike(velocities, positions, action_clipped, params)

    # Build initial vectors from state (note: compute psi_dot from theta)
    psi_dot0 = (params.v * jnp.sin(state.theta)) / (params.l + EPS)
    velocities0 = jnp.array(
        [state.omega_dot, state.theta_dot, psi_dot0], dtype=jnp.float32
    )
    positions0 = jnp.array([state.omega, state.theta, state.psi], dtype=jnp.float32)

    v_new, p_new, metrics = compute_velocity_and_pos_from_acceleration_integration(
        velocities=velocities0,
        positions=positions0,
        delta_t=params.delta_t,
        compute_acceleration=_accel_fn,
        method=integration_method,
    )

    # map back to EnvState
    new_state = vecs_to_state(state, v_new, p_new, metrics, params).replace(
        torque=a_T, displacement=a_d
    )
    return new_state, metrics


# -------- Terminal, reward, observation (unchanged semantics) --------
def check_is_terminal(state: EnvState, params: EnvParams):
    max_tilt_rad = jnp.deg2rad(params.max_tilt_deg)
    terminated = jnp.abs(state.omega) > max_tilt_rad
    truncated = state.t >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward(state: EnvState, params: EnvParams):
    terminated, truncated = check_is_terminal(state, params)
    default = jnp.where(terminated, -1.0, 0.0)

    dx = state.x_f - state.x_b
    dy = state.y_f - state.y_b
    heading = jnp.arctan2(dy, dx)
    gx = params.goal_x - state.x_f
    gy = params.goal_y - state.y_f
    goal_dir = jnp.arctan2(gy, gx)
    g = jnp.abs(jnp.arctan2(jnp.sin(goal_dir - heading), jnp.cos(goal_dir - heading)))
    within_goal = (gx**2 + gy**2) <= (params.goal_radius**2)
    step_reward = (4.0 - 2.0 * g) * 0.00004
    reward = jnp.where(terminated, -1.0, jnp.where(within_goal, 0.01, step_reward))

    reward_dense = terminated, _ = check_is_terminal(state, params)
    max_tilt_diff = params.max_tilt_deg
    true_reward = (
        (max_tilt_diff - jnp.abs(params.target_tilt - jnp.rad2deg(state.omega)))
        / max_tilt_diff
    ) ** 4
    reward_dense = jnp.where(
        terminated,
        -1.0 * params.max_steps_in_episode,
        true_reward,
    )
    return (reward * params.use_goal + (1 - params.use_goal) * default) * (
        1 - params.use_dense
    ) + reward_dense * params.use_dense


def get_obs(state: EnvState, params: EnvParams):
    omega_ddot = omega_ddot_from_eq2(
        state.omega, state.theta, state.theta_dot, state.last_d, params
    )

    return jnp.stack(
        [state.omega, state.omega_dot, omega_ddot, state.theta, state.theta_dot], axis=0
    ).astype(jnp.float32)
