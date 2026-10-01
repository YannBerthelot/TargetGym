"""Reference controller (the MPC slot) for the HVAC task.

Each task keeps its oracle in its own package, so changing one re-records
only the tasks of that package. The shared machinery (GradientMPC,
CasadiMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every task's baseline fingerprint.
"""

import numpy as np

from target_gym.experts.mpc import (
    CasadiMPC,
    _is_v1,
    _smooth_max,
)

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
