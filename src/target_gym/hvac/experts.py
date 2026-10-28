"""Reference controller (the MPC slot) for the HVAC task.

Each task keeps its oracle in its own package, so changing one re-records
only the tasks of that package. The shared machinery (GradientMPC,
CasadiMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every task's baseline fingerprint.
"""

import jax.numpy as jnp
import numpy as np
from flax import struct

from target_gym.experts.mpc import (
    CasadiMPC,
    _is_v1,
    _smooth_max,
)
from target_gym.experts.pid import Controller, pid_update

try:
    import casadi
    import do_mpc
except ImportError:  # CasadiMPC refuses to build without them
    pass


# ---------------------------------------------------------------------------
# Building HVAC
# ---------------------------------------------------------------------------


class HVACCasadiMPC(CasadiMPC):
    """
    CasADi MPC for the single-zone building (ISO 13790 5R1C).

    States : [T_mass, Q_emitter]      Input: u_raw in [-1, 1] -> [0, Q_heat_max]

    Only two differential states: the 5R1C air and surface nodes have no
    capacitance, so they are substituted in closed form exactly as the plant
    does. The model is therefore an almost-exact copy of the plant rather than
    a reduction -- unusual here, and possible because the building model is
    genuinely low-order.

    Where the MPC earns its advantage is **anticipation**. Setpoint schedule,
    outdoor temperature, solar gain and occupancy gain all enter as
    time-varying parameters over the horizon, so the controller can pre-heat
    ahead of the morning setback recovery and back off ahead of a sunny
    afternoon. A PID sees none of that until it has already happened, and with
    a 43 h thermal time constant "already happened" is far too late.
    """

    SCALING = {"_x": {"T_mass": 20.0, "Q_emitter": 800.0}}

    def __init__(self, env, params, horizon: int = 24, mpc_dt: float = None):
        super().__init__(
            env, params, horizon=horizon, mpc_dt=mpc_dt or float(params.delta_t)
        )

    def _build_mpc(self):
        p = self.params
        from target_gym.hvac.env import zone_conductances

        c = zone_conductances(p)
        H_is, H_ms, H_w, H_em, H_ve = (
            c["H_tr_is"],
            c["H_tr_ms"],
            c["H_tr_w"],
            c["H_tr_em"],
            c["H_ve"],
        )
        A_tot, A_m, C_m = c["A_tot"], c["A_m"], c["C_m"]

        model = do_mpc.model.Model("continuous")
        T_mass = model.set_variable("_x", "T_mass")
        Q_emitter = model.set_variable("_x", "Q_emitter")
        u_raw = model.set_variable("_u", "u_raw")
        T_out = model.set_variable("_tvp", "T_out")
        phi_int = model.set_variable("_tvp", "phi_int")
        phi_sol = model.set_variable("_tvp", "phi_sol")
        model.set_variable("_tvp", "target_T")
        model.set_variable("_tvp", "occupied")

        Q_command = 0.5 * (u_raw + 1.0) * p.Q_heat_max

        # Gain split (ISO 13790), mirroring hvac.env.split_gains.
        phi_ia = 0.5 * phi_int
        remainder = 0.5 * phi_int + phi_sol
        phi_m = (A_m / A_tot) * remainder
        phi_st = (1.0 - A_m / A_tot - H_w / (9.1 * A_tot)) * remainder

        # Algebraic air/surface nodes in closed form (see solve_air_and_surface).
        denom_air = H_is + H_ve
        a = H_is / denom_air
        b = (H_ve * T_out + phi_ia + Q_emitter) / denom_air
        T_surface = (H_ms * T_mass + H_w * T_out + phi_st + H_is * b) / (
            H_ms + H_w + H_is * (1.0 - a)
        )
        T_air = a * T_surface + b

        model.set_rhs(
            "T_mass",
            (H_ms * (T_surface - T_mass) + H_em * (T_out - T_mass) + phi_m) / C_m,
        )
        model.set_rhs("Q_emitter", (Q_command - Q_emitter) / p.emitter_tau)
        model.set_expression("T_air", T_air)
        model.setup()

        mpc = do_mpc.controller.MPC(model)
        mpc.set_param(
            n_horizon=self.horizon,
            t_step=self.mpc_dt,
            n_robust=0,
            store_full_solution=False,
        )

        # Objective mirrors env.compute_reward: squared comfort band minus
        # normalised energy. Minimising -reward.
        T_air_post = model.aux["T_air"]
        target_post = model.tvp["target_T"]
        err = target_post - T_air_post
        # Quadratic in the error rather than a copy of the reward. The reward
        # is clipped, and both failure modes of copying it have been observed:
        # dropping the clip makes the quadratic turn back *upward* past
        # 2*comfort_band so large errors score better (the MPC stopped heating
        # entirely), while keeping the clip makes the objective *flat* out
        # there so the solver sees no gradient at all. What matters is a shared
        # minimiser with a usable gradient everywhere, which a quadratic gives.
        comfort = -((err / (2.0 * p.comfort_band)) ** 2)
        energy = model.x["Q_emitter"] / p.Q_heat_max
        if _is_v1(p):
            mpc.set_objective(
                lterm=-comfort + float(p.energy_weight) * energy, mterm=-comfort
            )
        else:
            # Version 2: the plant's own cost in euros per step -- a
            # quadratic on the air temperature outside the occupied comfort
            # band, or below the setback target at night, plus the gas. The
            # dead zone is smoothed by ``_SMOOTH_K`` for IPOPT; the symmetric
            # quadratic above heated to the night target and ignored both the
            # tolerance and the gas, which under this cost is not a ceiling.
            hours = float(p.delta_t) / 3600.0
            occupied = model.tvp["occupied"]
            tol = float(p.comfort_tolerance)
            excess_occ = _smooth_max(casadi.sqrt(err * err + 1e-6) - tol)
            shortfall = _smooth_max(err)  # target above the air: too cold
            excess_night = (
                shortfall
                if bool(p.setback_lower_bound_only)
                else (casadi.sqrt(err * err + 1e-6))
            )
            excess = occupied * excess_occ + (1.0 - occupied) * excess_night
            comfort_cost = float(p.comfort_price) * hours * excess**2
            gas_cost = float(p.gas_price) * model.x["Q_emitter"] / 1000.0 * hours
            mpc.set_objective(lterm=comfort_cost + gas_cost, mterm=comfort_cost)
        mpc.set_rterm(u_raw=1e-3)
        mpc.bounds["lower", "_u", "u_raw"] = -1.0
        mpc.bounds["upper", "_u", "u_raw"] = 1.0

        self._current_step = 0
        self._setpoint_occupied = float(p.setpoint_occupied)
        tvp_tpl = mpc.get_tvp_template()

        def tvp_fun(_t):
            from target_gym.hvac.env import (
                internal_gain,
                is_occupied,
                outdoor_temperature,
                scheduled_setpoint,
                solar_gain,
            )

            for k in range(self.horizon + 1):
                t = self._current_step + k
                # Forecast uses the deterministic daily cycle; the stochastic
                # weather deviation is unknown, so it is left at zero and
                # rejected by receding-horizon feedback.
                tvp_tpl["_tvp", k, "T_out"] = float(outdoor_temperature(t, 0.0, p))
                tvp_tpl["_tvp", k, "phi_int"] = float(internal_gain(t, p))
                tvp_tpl["_tvp", k, "phi_sol"] = float(solar_gain(t, p))
                tvp_tpl["_tvp", k, "target_T"] = float(
                    scheduled_setpoint(t, self._setpoint_occupied, p)
                )
                tvp_tpl["_tvp", k, "occupied"] = float(is_occupied(t, p))
            return tvp_tpl

        mpc.set_tvp_fun(tvp_fun)
        self._apply_scaling(mpc)
        mpc.set_param(nlpsol_opts=self._quiet_ipopt())
        mpc.setup()
        return mpc

    def _extract_x0(self, state):
        return np.array([float(state.T_mass), float(state.Q_emitter)])

    def _update_setpoint(self, state):
        self._current_step = int(state.time)
        self._setpoint_occupied = float(state.setpoint_occupied)


def make_hvac_mpc(env, params, horizon: int = 24):
    """CasADi/IPOPT MPC for the single-zone building.

    With delta_t = 900 s, horizon = 24 gives 6 h of lookahead -- enough to see
    the morning setback recovery and the solar peak, which is where the
    anticipation advantage over PID comes from.
    """
    return HVACCasadiMPC(env, params, horizon=horizon)


# ---------------------------------------------------------------------------
# The expert: ASHRAE Guideline 36 zone heating with optimal start
# ---------------------------------------------------------------------------


@struct.dataclass
class G36Params:
    """The published constants, and the schedule a BMS is programmed with.

    The loop is G36's zone heating PI as LBL's Modelica Buildings library
    implements it (``G36/ThermalZones/ControlLoops.mo``: ``kHeaCon = 0.1``
    per K, ``TiHeaCon = 900`` s, output 0-1). Optimal start is LBL's
    ``OBC/Utilities/OptimalStart`` with its defaults: ``tOptMax`` 3 h, three
    days of slopes, ``uLow`` 0 K and ``uHigh`` 0.5 K, ``thrOptOn`` 60 s, and
    its floors of 1 K/h on the mean slope and 120 s on a learning interval.
    """

    kp: float = 0.1
    Ti: float = 900.0
    dt: float = 900.0
    occupied_start_h: float = 7.0
    occupied_end_h: float = 22.0
    setpoint_design: float = 21.0
    t_opt_max: float = 10800.0
    n_day: int = struct.field(pytree_node=False, default=3)
    u_low: float = 0.0
    u_high: float = 0.5
    thr_opt_on: float = 60.0
    slope_min: float = 1.0 / 3600.0
    t_learn_min: float = 120.0


@struct.dataclass
class G36State:
    """The PI's integral and optimal start's memory."""

    integral: float = 0.0
    #: The occupied setpoint last seen, the warm-up's aim.
    setpoint_occupied: float = 21.0
    #: The last ``n_day`` learned slopes (K/s) and how many there are.
    slopes: tuple = (0.0, 0.0, 0.0)
    n_slopes: int = 0
    #: This morning's warm-up time (s), set ``t_opt_max`` before occupancy.
    t_opt: float = 0.0
    #: A warm-up being timed: 1 while learning, its start deficit and age.
    learning: float = 0.0
    dT_start: float = 0.0
    t_learning: float = 0.0
    #: A slope learned today, pushed at midnight (``has_pending`` 1).
    pending: float = 0.0
    has_pending: float = 0.0
    #: The previous step's hour and time to occupancy, to see crossings.
    prev_hour: float = 0.0
    prev_to_occ: float = 1e9
    prev_opt_on: float = 0.0


def g36_step(params, state, obs, xp=jnp):
    """One step of the G36 zone heating loop with optimal start.

    obs: ``[T_air, T_out, heat_pct, solar_norm, sin_h, cos_h, target_T]``.
    The clock is read from ``sin_h, cos_h``, as a BMS knows the time.
    """
    p = params
    T_air, target = obs[..., 0], obs[..., 6]
    hour = xp.mod(xp.arctan2(obs[..., 4], obs[..., 5]) * 24.0 / (2.0 * np.pi), 24.0)
    occupied = (hour >= p.occupied_start_h) & (hour < p.occupied_end_h)
    to_occ = xp.mod(p.occupied_start_h - hour, 24.0) * 3600.0
    sp_occ = xp.where(occupied, target, state.setpoint_occupied)

    # Midnight: push the slope learned that morning into the moving mean.
    midnight = (hour < state.prev_hour) & (state.has_pending > 0.5)
    slopes = xp.asarray(state.slopes)
    pushed = xp.concatenate([slopes[1:], xp.asarray(state.pending)[..., None]], -1)
    slopes = xp.where(midnight, pushed, slopes)
    n_slopes = xp.where(
        midnight, xp.minimum(state.n_slopes + 1, p.n_day), state.n_slopes
    )
    has_pending = xp.where(midnight, 0.0, state.has_pending)

    # ``t_opt_max`` before occupancy: sample the zone, set the warm-up time.
    sample = (~occupied) & (to_occ <= p.t_opt_max) & (state.prev_to_occ > p.t_opt_max)
    # The moving mean over the slopes learned so far (newest last); none
    # learned yet reads 0, which the floor turns into 1 K/h.
    learned = xp.arange(p.n_day) >= p.n_day - n_slopes
    mean = xp.sum(xp.where(learned, slopes, 0.0), -1) / xp.maximum(n_slopes, 1)
    slope = xp.maximum(mean, p.slope_min)
    deficit = sp_occ - T_air
    t_new = xp.where(deficit > p.u_high, xp.minimum(deficit / slope, p.t_opt_max), 0.0)
    t_opt = xp.where(sample, t_new, state.t_opt)
    opt_on = (~occupied) & (t_opt > p.thr_opt_on) & (to_occ < t_opt)

    # Learning: time each warm-up from its start until the zone reaches the
    # occupied setpoint (or ``t_opt_max`` has passed).
    # It starts when the warm-up does, on the rising edge of ``opt_on``.
    start = opt_on & (state.prev_opt_on < 0.5) & (state.learning < 0.5)
    learning = xp.where(start, 1.0, state.learning)
    dT_start = xp.where(start, deficit, state.dT_start)
    t_learning = xp.where(start, 0.0, state.t_learning + p.dt)
    done = (learning > 0.5) & (
        (T_air >= sp_occ - p.u_low) | (t_learning >= p.t_opt_max)
    )
    pending = xp.where(
        done, dT_start / xp.maximum(t_learning, p.t_learn_min), state.pending
    )
    has_pending = xp.where(done, 1.0, has_pending)
    learning = xp.where(done, 0.0, learning)

    # The heating PI on the setpoint in force.
    setpoint = xp.where(opt_on, sp_occ, target)
    e = setpoint - T_air
    heat, integral, _ = pid_update(
        e, state.integral, 0.0, p.kp, p.kp / p.Ti, 0.0, p.dt, 0.0, 1.0, xp
    )
    new_state = G36State(
        integral=integral,
        setpoint_occupied=sp_occ,
        slopes=slopes,
        n_slopes=n_slopes,
        t_opt=t_opt,
        learning=learning,
        dT_start=dT_start,
        t_learning=t_learning,
        pending=pending,
        has_pending=has_pending,
        prev_hour=hour,
        prev_to_occ=to_occ,
        prev_opt_on=xp.where(opt_on, 1.0, 0.0),
    )
    return xp.stack([2.0 * heat - 1.0], axis=-1), new_state


def make_hvac_pid_controller() -> Controller:
    """The expert: G36 zone heating with optimal start, as a
    :class:`~target_gym.experts.pid.Controller`. Every constant is the
    published one (:class:`G36Params`); nothing is tuned on seeds."""
    from target_gym.hvac.env import HVACParams

    pr = HVACParams()
    params = G36Params(
        dt=float(pr.delta_t),
        occupied_start_h=float(pr.occupied_start_h),
        occupied_end_h=float(pr.occupied_end_h),
        setpoint_design=float(pr.setpoint_occupied),
    )
    state = G36State(
        setpoint_occupied=params.setpoint_design,
        slopes=np.zeros(params.n_day, np.float32),
    )
    return Controller(params, state, g36_step)
