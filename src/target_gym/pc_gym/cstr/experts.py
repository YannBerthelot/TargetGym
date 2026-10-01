"""Reference controller (the MPC slot) for the CSTR task.

Each task keeps its oracle in its own package, so changing one re-records
only the tasks of that package. The shared machinery (GradientMPC,
CasadiMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every task's baseline fingerprint.
"""

import numpy as np

from target_gym.experts.mpc import CasadiMPC

try:
    import casadi
    import do_mpc
except ImportError:  # CasadiMPC refuses to build without them
    pass


# ---------------------------------------------------------------------------
# CSTR
# ---------------------------------------------------------------------------


class CSTRCasadiMPC(CasadiMPC):
    """
    CasADi MPC for CSTR.

    States : [C_a, T]
    Input  : u_raw ∈ [-1, 1]  →  T_c ∈ [T_c_min, T_c_max]
    ODE    :
        dC_a/dt = q/V*(Caf - C_a) - k0*exp(-EA/R/T)*C_a
        dT/dt   = q/V*(Ti - T) + (-ΔHr)*rA/(ρ·C) + UA*(T_c - T)/(ρ·C·V)
    """

    SCALING = {"_x": {"C_a": 1.0, "T": 300.0}}

    def _build_mpc(self):
        p = self.params
        model = do_mpc.model.Model("continuous")

        C_a = model.set_variable("_x", "C_a")
        T = model.set_variable("_x", "T")
        u_raw = model.set_variable("_u", "u_raw")
        target_CA = model.set_variable("_p", "target_CA")

        # Action scaling
        T_c = p.T_c_min + 0.5 * (u_raw + 1.0) * (p.T_c_max - p.T_c_min)
        rA = p.k0 * casadi.exp(-p.EA_over_R / T) * C_a

        model.set_rhs("C_a", p.q / p.V * (p.Caf - C_a) - rA)
        model.set_rhs(
            "T",
            p.q / p.V * (p.Ti - T)
            + (-p.deltaHr) * rA / (p.rho * p.C)
            + p.UA * (T_c - T) / (p.rho * p.C * p.V),
        )
        model.setup()

        mpc = do_mpc.controller.MPC(model)
        mpc.set_param(
            n_horizon=self.horizon,
            t_step=self.mpc_dt,
            n_robust=0,
            store_full_solution=False,
        )

        lterm = (target_CA - C_a) ** 2
        mpc.set_objective(lterm=lterm, mterm=lterm)
        mpc.set_rterm(u_raw=1e-4)

        mpc.bounds["lower", "_u", "u_raw"] = -1.0
        mpc.bounds["upper", "_u", "u_raw"] = 1.0

        self._target_CA = float(p.target_CA_range[0])
        p_tpl = mpc.get_p_template(1)

        def p_fun(_t):
            p_tpl["_p", 0, "target_CA"] = self._target_CA
            return p_tpl

        mpc.set_p_fun(p_fun)
        self._apply_scaling(mpc)
        mpc.set_param(nlpsol_opts=self._quiet_ipopt())
        mpc.setup()
        return mpc

    def _extract_x0(self, state):
        return np.array([float(state.C_a), float(state.T)])

    def _update_setpoint(self, state):
        self._target_CA = float(state.target_CA)


def make_cstr_mpc(env, params, horizon: int = 5):
    """CasADi/IPOPT MPC for CSTR — matches the PC-gym oracle (N=5).

    With delta_t=0.25 s (PC-gym standard: tsim=25s, N=100), horizon=5 gives
    1.25 s lookahead — about one residence time (V/q=1 s).
    """
    return CSTRCasadiMPC(env, params, horizon=horizon)
