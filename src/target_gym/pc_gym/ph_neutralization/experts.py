"""Reference controller (the MPC slot) for the pH neutralisation task.

Each task keeps its oracle in its own package, so changing one re-records
only the tasks of that package. The shared machinery (GradientMPC,
CasadiMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every task's baseline fingerprint.
"""

import numpy as np

from target_gym.experts.mpc import CasadiMPC
from target_gym.pc_gym.ph_neutralization.env import BUFFER_OU_THETA

try:
    import do_mpc
except ImportError:  # CasadiMPC refuses to build without them
    pass


# ---------------------------------------------------------------------------
# pH neutralisation
# ---------------------------------------------------------------------------


class PHCasadiMPC(CasadiMPC):
    """
    CasADi MPC for the pH neutralisation CSTR.

    States : [Wa, Wb]  reaction invariants     Input: u_raw -> base flow q3
    Algebraic: pH, defined implicitly by the charge balance.

    The invariants mix *linearly*, so the only nonlinearity is the titration
    curve -- and that is exactly where an MPC should beat a PID. Expressing pH
    as an algebraic variable constrained by the charge balance lets IPOPT see
    the curve directly and take its gradient, instead of a fixed-gain
    controller having to compromise across a ~45x gain variation.

    The buffer flow q2 is not in the observation, but this is the oracle and
    it reads the state, as it already does for the equally hidden Wa and Wb.
    The model takes q2 as a time-varying parameter, forecast from its current
    value by the env's own OU mean decay toward nominal. Planning on the
    nominal q2 instead, as this controller once did, made almost all of its
    hold error: reading q2 cut the protocol cost from 3.06 to 0.096 and the
    hold error from 0.0146 to 0.0022 pH (oracle audit, 2026-10-01, seeds 0-2).
    What remains is the OU innovation itself, which no causal controller can
    predict: a planner told the realised next q2 tracks to about 1e-5 pH, so
    the whole remaining tracking cost is that one-step innovation.
    """

    SCALING = {"_x": {"Wa": 3e-4, "Wb": 3e-4}, "_z": {"pH": 7.0}}

    def __init__(self, env, params, horizon: int = 20, mpc_dt: float = None):
        self._q2_forecast = np.full(horizon + 1, float(params.q2_nominal))
        super().__init__(
            env, params, horizon=horizon, mpc_dt=mpc_dt or float(params.delta_t)
        )

    def _build_mpc(self):
        p = self.params
        model = do_mpc.model.Model("continuous")

        Wa = model.set_variable("_x", "Wa")
        Wb = model.set_variable("_x", "Wb")
        pH = model.set_variable("_z", "pH")
        u_raw = model.set_variable("_u", "u_raw")
        model.set_variable("_tvp", "target_pH")
        q2 = model.set_variable("_tvp", "q2")  # forecast, see _update_setpoint

        q3 = p.q3_min + 0.5 * (u_raw + 1.0) * (p.q3_max - p.q3_min)

        model.set_rhs(
            "Wa",
            (p.q1 * (p.Wa1 - Wa) + q2 * (p.Wa2 - Wa) + q3 * (p.Wa3 - Wa)) / p.V,
        )
        model.set_rhs(
            "Wb",
            (p.q1 * (p.Wb1 - Wb) + q2 * (p.Wb2 - Wb) + q3 * (p.Wb3 - Wb)) / p.V,
        )

        # Charge balance as the algebraic constraint defining pH.
        carbonate = (1.0 + 2.0 * 10.0 ** (pH - p.pK2)) / (
            1.0 + 10.0 ** (p.pK1 - pH) + 10.0 ** (pH - p.pK2)
        )
        model.set_alg(
            "charge_balance",
            Wa + 10.0 ** (pH - 14.0) - 10.0 ** (-pH) + Wb * carbonate,
        )
        model.setup()

        mpc = do_mpc.controller.MPC(model)
        mpc.set_param(
            n_horizon=self.horizon,
            t_step=self.mpc_dt,
            n_robust=0,
            store_full_solution=False,
        )

        # Tracking cost is a plain quadratic in the error, NOT a copy of the
        # environment's clipped reward. The two must share a *minimiser*, but
        # the reward's clip is flat once the error exceeds the tracking band,
        # and a flat objective gives IPOPT no gradient: the solver optimised
        # the only live term (reagent cost) and railed the valve shut, driving
        # pH away from the setpoint at ~3.9 pH mean error. A quadratic has the
        # same minimiser and a usable gradient everywhere.
        pH_post = model.z["pH"]
        target_post = model.tvp["target_pH"]
        u_post = model.u["u_raw"]
        err = target_post - pH_post
        tracking = -((err / p.tracking_band) ** 2)
        q3_post = p.q3_min + 0.5 * (u_post + 1.0) * (p.q3_max - p.q3_min)
        reagent = (q3_post - p.q3_min) / (p.q3_max - p.q3_min)
        # do-mpc's terminal cost may only reference differential states, and
        # pH here is algebraic -- so the tracking term lives entirely in the
        # stage cost and mterm is a symbolic zero.
        mpc.set_objective(
            lterm=-tracking + float(p.reagent_cost_weight) * reagent,
            mterm=0.0 * model.x["Wa"],
        )
        mpc.set_rterm(u_raw=1e-3)
        mpc.bounds["lower", "_u", "u_raw"] = -1.0
        mpc.bounds["upper", "_u", "u_raw"] = 1.0
        mpc.bounds["lower", "_z", "pH"] = 0.0
        mpc.bounds["upper", "_z", "pH"] = 14.0

        self._target = float(sum(p.target_pH_range) / 2.0)
        tvp_tpl = mpc.get_tvp_template()

        def tvp_fun(_t):
            for k in range(self.horizon + 1):
                tvp_tpl["_tvp", k, "target_pH"] = self._target
                tvp_tpl["_tvp", k, "q2"] = self._q2_forecast[k]
            return tvp_tpl

        mpc.set_tvp_fun(tvp_fun)
        self._apply_scaling(mpc)
        mpc.set_param(nlpsol_opts=self._quiet_ipopt())
        mpc.setup()
        return mpc

    def _extract_x0(self, state):
        return np.array([float(state.Wa), float(state.Wb)])

    def _update_setpoint(self, state):
        self._target = float(state.target_pH)
        self._q2_forecast = q2_forecast(
            float(state.q2), self.params, self.horizon, self.mpc_dt
        )

    def step(self, _obs, state):
        """Seed the algebraic pH before solving.

        The DAE's algebraic variable needs a starting point on the titration
        curve. Left at do-mpc's default the solver begins far off it, where the
        charge balance is nearly flat in pH, and converges somewhere useless --
        which showed up as ~3.9 pH mean error, worse than a constant valve.
        The plant's measured pH is the obvious guess.
        """
        self._update_setpoint(state)
        x0 = self._extract_x0(state)
        if not self._initialized:
            self._mpc.x0 = x0
            self._mpc.z0 = np.array([float(state.pH)])
            self._mpc.set_initial_guess()
            self._initialized = True
        guess = self._save_guess()
        u = np.array(self._mpc.make_step(x0)).flatten()
        if not self._record_solve():
            u = self._fallback(guess, u)
        self._last_u = u
        return float(np.clip(u, -1.0, 1.0)[0])


def q2_forecast(q2: float, params, horizon: int, mpc_dt: float = None) -> np.ndarray:
    """The buffer flow's OU mean over the plan, from its value now.

    The env advances q2 by an Euler OU step before integrating, so control
    interval k (of ``mpc_dt`` seconds, the env step by default) starts
    ``1 + k * mpc_dt / delta_t`` updates ahead:
    ``nominal + (q2 - nominal) * (1 - theta * delta_t) ** (1 + k * mpc_dt / delta_t)``.
    That is the unclipped mean. A q2 in range keeps it in range, so the clip
    is only a guard; near ``q2_min`` the env clips each noisy sample, which
    puts the true conditional mean slightly above this forecast.
    """
    dt = float(params.delta_t)
    steps = float(mpc_dt) / dt if mpc_dt else 1.0
    decay = 1.0 - BUFFER_OU_THETA * dt
    nominal = float(params.q2_nominal)
    k = np.arange(horizon + 1)
    mean = nominal + (q2 - nominal) * decay ** (1.0 + k * steps)
    return np.clip(mean, float(params.q2_min), float(params.q2_max))


def make_ph_mpc(env, params, horizon: int = 20):
    """CasADi/IPOPT MPC for the pH neutralisation CSTR.

    With delta_t = 5 s, horizon = 20 gives 100 s of lookahead -- slightly more
    than one residence time (V/q_total ~ 88 s), so the controller can see a
    change work through the tank.
    """
    return PHCasadiMPC(env, params, horizon=horizon)
