"""Reference controller (the MPC slot) for the first-order task.

Each task keeps its oracle in its own package, so changing one re-records
only the tasks of that package. The shared machinery (GradientMPC,
CasadiMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every task's baseline fingerprint.
"""

import numpy as np

from target_gym.experts.mpc import CasadiMPC

try:
    import do_mpc
except ImportError:  # CasadiMPC refuses to build without them
    pass


# ---------------------------------------------------------------------------
# FirstOrderSystem
# ---------------------------------------------------------------------------


class FirstOrderCasadiMPC(CasadiMPC):
    """
    CasADi MPC for FirstOrderSystem.

    State : [x]
    Input : u_raw ∈ [-1, 1]  →  u ∈ [u_min, u_max]
    ODE   : dx/dt = (K·u - x) / tau
    """

    def _build_mpc(self):
        p = self.params
        model = do_mpc.model.Model("continuous")

        x = model.set_variable("_x", "x")
        u_raw = model.set_variable("_u", "u_raw")
        target = model.set_variable("_p", "target_x")

        u = p.u_min + 0.5 * (u_raw + 1.0) * (p.u_max - p.u_min)
        model.set_rhs("x", (p.K * u - x) / p.tau)
        model.setup()

        mpc = do_mpc.controller.MPC(model)
        mpc.set_param(
            n_horizon=self.horizon,
            t_step=self.mpc_dt,
            n_robust=0,
            store_full_solution=False,
        )

        lterm = (target - x) ** 2
        mpc.set_objective(lterm=lterm, mterm=lterm)
        mpc.set_rterm(u_raw=1e-4)

        mpc.bounds["lower", "_u", "u_raw"] = -1.0
        mpc.bounds["upper", "_u", "u_raw"] = 1.0

        self._target_x = float(p.target_x_range[0])
        p_tpl = mpc.get_p_template(1)

        def p_fun(_t):
            p_tpl["_p", 0, "target_x"] = self._target_x
            return p_tpl

        mpc.set_p_fun(p_fun)
        self._apply_scaling(mpc)
        mpc.set_param(nlpsol_opts=self._quiet_ipopt())
        mpc.setup()
        return mpc

    def _extract_x0(self, state):
        return np.array([float(state.x)])

    def _update_setpoint(self, state):
        self._target_x = float(state.target_x)


def make_first_order_mpc(env, params, horizon: int = 5):
    """CasADi/IPOPT MPC for FirstOrderSystem — matches the PC-gym oracle (N=5)."""
    return FirstOrderCasadiMPC(env, params, horizon=horizon)
