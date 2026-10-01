"""Reference controller (the MPC slot) for the four-tank task.

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
# FourTank
# ---------------------------------------------------------------------------


class FourTankCasadiMPC(CasadiMPC):
    """
    CasADi MPC for FourTank.

    States : [h1, h2, h3, h4]
    Inputs : [v1_raw, v2_raw] each ∈ [-1, 1]  →  [v1, v2] ∈ [v_min, v_max]
    ODE    : four-tank gravity-drain dynamics (see env.py)
    """

    SCALING = {"_x": {"h1": 0.25, "h2": 0.25, "h3": 0.25, "h4": 0.25}}

    def _build_mpc(self):
        p = self.params
        model = do_mpc.model.Model("continuous")

        h1 = model.set_variable("_x", "h1")
        h2 = model.set_variable("_x", "h2")
        h3 = model.set_variable("_x", "h3")
        h4 = model.set_variable("_x", "h4")
        v1_raw = model.set_variable("_u", "v1_raw")
        v2_raw = model.set_variable("_u", "v2_raw")
        target_h1 = model.set_variable("_p", "target_h1")
        target_h2 = model.set_variable("_p", "target_h2")

        # Action scaling: raw ∈ [-1,1] → physical ∈ [v_min, v_max]
        v1 = p.v_min + 0.5 * (v1_raw + 1.0) * (p.v_max - p.v_min)
        v2 = p.v_min + 0.5 * (v2_raw + 1.0) * (p.v_max - p.v_min)

        eps = 1e-6  # avoid sqrt(0)
        sq = casadi.sqrt
        g2 = casadi.sqrt(2.0 * p.g)

        dh1 = (
            -(p.a1 / p.A1) * g2 * sq(casadi.fmax(h1, eps))
            + (p.a3 / p.A1) * g2 * sq(casadi.fmax(h3, eps))
            + (p.gamma1 * p.k1 / p.A1) * v1
        )
        dh2 = (
            -(p.a2 / p.A2) * g2 * sq(casadi.fmax(h2, eps))
            + (p.a4 / p.A2) * g2 * sq(casadi.fmax(h4, eps))
            + (p.gamma2 * p.k2 / p.A2) * v2
        )
        dh3 = (
            -(p.a3 / p.A3) * g2 * sq(casadi.fmax(h3, eps))
            + ((1 - p.gamma2) * p.k2 / p.A3) * v2
        )
        dh4 = (
            -(p.a4 / p.A4) * g2 * sq(casadi.fmax(h4, eps))
            + ((1 - p.gamma1) * p.k1 / p.A4) * v1
        )

        model.set_rhs("h1", dh1)
        model.set_rhs("h2", dh2)
        model.set_rhs("h3", dh3)
        model.set_rhs("h4", dh4)
        model.setup()

        mpc = do_mpc.controller.MPC(model)
        mpc.set_param(
            n_horizon=self.horizon,
            t_step=self.mpc_dt,
            n_robust=0,
            store_full_solution=False,
        )

        lterm = (target_h1 - h1) ** 2 + (target_h2 - h2) ** 2
        mpc.set_objective(lterm=lterm, mterm=lterm)
        mpc.set_rterm(v1_raw=1e-4, v2_raw=1e-4)

        mpc.bounds["lower", "_u", "v1_raw"] = -1.0
        mpc.bounds["upper", "_u", "v1_raw"] = 1.0
        mpc.bounds["lower", "_u", "v2_raw"] = -1.0
        mpc.bounds["upper", "_u", "v2_raw"] = 1.0

        # Keep levels above minimum to avoid sqrt(0)
        # Both termination bounds, and soft.
        #
        # The plant does not clip these levels, it *ends the episode* when any
        # of them reaches h_min or h_max. Only the lower bound was here, and it
        # was hard, which is backwards on both counts. Hard was wrong because a
        # hard bound the plant can walk the initial state onto makes the NLP
        # infeasible at x0, and IPOPT answers that with a restoration phase and
        # hundreds of iterations rather than an action. Missing h_max was worse:
        # the controller was blind to half of a termination condition it is
        # scored on, so it had no reason not to overflow a tank.
        #
        # Input bounds stay hard, because the optimiser owns those and can
        # always satisfy them. State bounds get slacks, which is the usual
        # division of labour.
        for h in (h1, h2, h3, h4):
            mpc.set_nl_cons(
                f"{h.name()}_min",
                -h,
                ub=-float(p.h_min),
                soft_constraint=True,
                penalty_term_cons=1e3,
            )
            mpc.set_nl_cons(
                f"{h.name()}_max",
                h,
                ub=float(p.h_max),
                soft_constraint=True,
                penalty_term_cons=1e3,
            )

        self._target_h1 = float(p.target_h1_range[0])
        self._target_h2 = float(p.target_h2_range[0])
        p_tpl = mpc.get_p_template(1)

        def p_fun(_t):
            p_tpl["_p", 0, "target_h1"] = self._target_h1
            p_tpl["_p", 0, "target_h2"] = self._target_h2
            return p_tpl

        mpc.set_p_fun(p_fun)
        self._apply_scaling(mpc)
        mpc.set_param(nlpsol_opts=self._quiet_ipopt())
        mpc.setup()
        return mpc

    def _extract_x0(self, state):
        return np.array(
            [float(state.h1), float(state.h2), float(state.h3), float(state.h4)]
        )

    def _update_setpoint(self, state):
        self._target_h1 = float(state.target_h1)
        self._target_h2 = float(state.target_h2)


def make_four_tank_mpc(env, params, horizon: int = 10, mpc_dt: float = None):
    """CasADi/IPOPT MPC for FourTank.

    PC-gym's oracle is N=5 at the environment's own step, and that is what this
    shipped: a horizon covering 5 s of a plant whose tracking error takes ~198 s
    to close (``scripts/audit_mpc_horizons.py`` puts it at ratio 0.03, the worst
    in the suite). It settled 47x worse than the PID and drove a tank to a
    terminal state in one episode of two.

    The fix is covered *time*, not more decision variables -- measured, raising
    the horizon to 20 steps at the environment's own step is still 54x worse,
    while any configuration reaching ~200 s drives the settled error to zero
    with no terminations. So the prediction step is decoupled from the
    environment's: ten steps of 20 dt cover 200 s for a tenth of the decision
    variables that would otherwise take.
    """
    if mpc_dt is None:
        mpc_dt = 20.0 * float(params.delta_t)
    return FourTankCasadiMPC(env, params, horizon=horizon, mpc_dt=mpc_dt)
