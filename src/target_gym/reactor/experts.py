"""Reference controller (the MPC slot) for the reactor task.

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
# Reactor
# ---------------------------------------------------------------------------


class ReactorCasadiMPC(CasadiMPC):
    """
    CasADi MPC for the nuclear reactor (point kinetics + xenon + thermal).

    States : [n, C_1..6, T_fuel, T_coolant, I_hat, Xe_hat, rho_ext]  (12)
    Input  : rho_rate -- the control-rod *speed*, not its position.

    Two modelling choices matter, and the original version got both wrong:

    **Xenon is in the model.** ``I_hat``/``Xe_hat`` and the reactivity term
    ``-rho_Xe_full * (Xe_hat - 1)`` were previously omitted entirely, so the
    "oracle" was blind to the multi-hour iodine/xenon swing that the
    environment docstring calls the dominant control challenge. Equilibrium
    xenon worth (~2500 pcm) dwarfs total rod authority (500 pcm), so an MPC
    that cannot see it is not an oracle -- the reported PID-vs-MPC gap was
    measuring model mismatch rather than controller quality.

    **Rod speed is a rate limit, not a position bound.** The environment moves
    ``rho_ext`` toward the demanded position at ``rod_speed_withdraw`` /
    ``rod_speed_insert``, asymmetrically. Treating the action as a position the
    plant reaches instantly lets the MPC plan trajectories it cannot fly. Here
    ``rho_ext`` is a *state* and the input is its rate, bounded by the two rod
    speeds -- an exact, and conveniently linear, encoding of the constraint.

    ``mpc_dt`` defaults to ``delta_t * control_period``: the environment holds
    one action across ``control_period`` physics sub-steps, so planning at the
    raw physics step would model a control authority that does not exist.
    """

    SCALING = {
        "_x": {
            "n": 1.0,
            "C0": 100.0,
            "C1": 400.0,
            "C2": 100.0,
            "C3": 70.0,
            "C4": 5.0,
            "C5": 0.8,
            "T_fuel": 1000.0,
            "T_coolant": 600.0,
            "I_hat": 1.0,
            "Xe_hat": 1.0,
            "rho_ext": 0.002,
        }
    }

    def __init__(self, env, params, horizon: int = 20, mpc_dt: float = None):
        if mpc_dt is None:
            control_period = getattr(env, "control_period", 1)
            mpc_dt = float(params.delta_t) * float(control_period)
        super().__init__(env, params, horizon=horizon, mpc_dt=mpc_dt)

    def _build_mpc(self):
        p = self.params
        from target_gym.reactor.env import (
            BETA_I,
            BETA_TOT,
            LAMBDA_I,
            LAMBDA_IODINE,
            LAMBDA_XENON,
            N_GROUPS,
        )

        model = do_mpc.model.Model("continuous")

        n = model.set_variable("_x", "n")
        C = [model.set_variable("_x", f"C{i}") for i in range(N_GROUPS)]
        T_fuel = model.set_variable("_x", "T_fuel")
        T_coolant = model.set_variable("_x", "T_coolant")
        I_hat = model.set_variable("_x", "I_hat")
        Xe_hat = model.set_variable("_x", "Xe_hat")
        rho_ext = model.set_variable("_x", "rho_ext")
        rho_rate = model.set_variable("_u", "rho_rate")
        model.set_variable("_tvp", "target_n")

        # Rod position integrates the commanded rod speed.
        model.set_rhs("rho_ext", rho_rate)

        # Reactivity: rod + thermal feedback (Doppler + moderator) + xenon.
        rho_feedback = p.alpha_fuel * (T_fuel - p.T_fuel_ref) + p.alpha_coolant * (
            T_coolant - p.T_coolant_ref
        )
        rho_xenon = -p.rho_Xe_full * (Xe_hat - 1.0)
        rho = rho_ext + rho_feedback + rho_xenon

        # Point kinetics
        sum_lambda_C = sum(float(LAMBDA_I[i]) * C[i] for i in range(N_GROUPS))
        model.set_rhs("n", ((rho - BETA_TOT) / p.Lambda_gen) * n + sum_lambda_C)
        for i in range(N_GROUPS):
            model.set_rhs(
                f"C{i}",
                (float(BETA_I[i]) / p.Lambda_gen) * n - float(LAMBDA_I[i]) * C[i],
            )

        # Two-node thermal model
        P_thermal = p.P_thermal_ref * n
        Q_fuel_to_cool = p.UA * (T_fuel - T_coolant)
        Q_flow_out = p.m_dot_cp * (T_coolant - p.T_inlet)
        model.set_rhs("T_fuel", (P_thermal - Q_fuel_to_cool) / p.C_fuel)
        model.set_rhs("T_coolant", (Q_fuel_to_cool - Q_flow_out) / p.C_coolant)

        # Iodine / xenon chain (normalised; matches reactor.env.compute_velocity)
        lam_sum = LAMBDA_XENON + p.sigma_phi0
        a_coeff = p.gamma_ratio * lam_sum / (1.0 + p.gamma_ratio)
        b_coeff = lam_sum / (1.0 + p.gamma_ratio)
        model.set_rhs("I_hat", LAMBDA_IODINE * (n - I_hat))
        model.set_rhs(
            "Xe_hat",
            a_coeff * n + b_coeff * I_hat - (LAMBDA_XENON + p.sigma_phi0 * n) * Xe_hat,
        )
        model.setup()

        mpc = do_mpc.controller.MPC(model)
        # Collocation is needed: PKE is stiff (|prompt eigenvalue| ~60/s at
        # rho_ext=rho_ext_max), so explicit integrators inside the NLP would
        # require tiny substeps. do-mpc's orthogonal collocation is implicit
        # and A-stable, handling the stiffness without substepping.
        mpc.set_param(
            n_horizon=self.horizon,
            t_step=self.mpc_dt,
            n_robust=0,
            store_full_solution=False,
            state_discretization="collocation",
            collocation_type="radau",
            collocation_deg=3,
            collocation_ni=1,
        )

        n_post = model.x["n"]
        rho_ext_post = model.x["rho_ext"]
        target_post = model.tvp["target_n"]

        # Objective mirrors env.compute_reward: a Gaussian tracking term and a
        # rod-position penalty. The environment's reward is
        # ``exp(-0.5*(err/band)^2) - w*|rho_ext|/rho_scale``; exp() is flat far
        # from the target and gives IPOPT almost no gradient there, so we
        # minimise the equivalent quadratic ``(err/band)^2`` instead -- same
        # minimiser, far better conditioned.
        err = target_post - n_post
        tracking_cost = (err / p.reward_band) ** 2
        rho_scale = float(max(abs(p.rho_ext_min), abs(p.rho_ext_max)))
        rod_penalty = (
            float(p.rod_motion_weight)
            * casadi.sqrt(rho_ext_post * rho_ext_post + 1e-12)
            / rho_scale
        )
        mpc.set_objective(lterm=tracking_cost + rod_penalty, mterm=tracking_cost)
        mpc.set_rterm(rho_rate=1e2)

        # Rod *speed* limits (asymmetric: insertion is faster than withdrawal).
        mpc.bounds["lower", "_u", "rho_rate"] = -float(p.rod_speed_insert)
        mpc.bounds["upper", "_u", "rho_rate"] = float(p.rod_speed_withdraw)
        # Rod travel limits.
        mpc.bounds["lower", "_x", "rho_ext"] = float(p.rho_ext_min)
        mpc.bounds["upper", "_x", "rho_ext"] = float(p.rho_ext_max)

        # With OU demand the future is unknown; hold the current target across
        # the horizon.
        self._current_target = float(sum(p.target_n_range) / 2.0)
        tvp_tpl = mpc.get_tvp_template()

        def tvp_fun(_t):
            for k in range(self.horizon + 1):
                tvp_tpl["_tvp", k, "target_n"] = self._current_target
            return tvp_tpl

        mpc.set_tvp_fun(tvp_fun)
        self._apply_scaling(mpc)
        mpc.set_param(nlpsol_opts=self._quiet_ipopt())
        mpc.setup()
        return mpc

    def _extract_x0(self, state):
        return np.concatenate(
            [
                np.array([float(state.n)]),
                np.asarray(state.C, dtype=float),
                np.array(
                    [
                        float(state.T_fuel),
                        float(state.T_coolant),
                        float(state.I_hat),
                        float(state.Xe_hat),
                        float(state.rho_ext),
                    ]
                ),
            ]
        )

    def _update_setpoint(self, state):
        self._current_target = float(state.target_n)

    def step(self, _obs, state):
        """Return the raw action in [-1, 1] the environment expects.

        The NLP optimises a rod *rate*, but ``step_env`` takes a demanded rod
        *position*. Integrating one step of the optimal rate gives the position
        to demand; because the environment rate-limits toward that demand with
        the same bounds the NLP respects, the realised motion matches the plan.
        """
        self._update_setpoint(state)
        x0 = self._extract_x0(state)
        if not self._initialized:
            self._mpc.x0 = x0
            self._mpc.set_initial_guess()
            self._initialized = True
        guess = self._save_guess()
        u = np.array(self._mpc.make_step(x0)).flatten()
        if not self._record_solve():
            u = self._fallback(guess, u)
        self._last_u = u
        rho_rate = float(u[0])

        p = self.params
        rho_next = float(
            np.clip(
                float(state.rho_ext) + rho_rate * self.mpc_dt,
                p.rho_ext_min,
                p.rho_ext_max,
            )
        )
        span = p.rho_ext_max - p.rho_ext_min
        raw = 2.0 * (rho_next - p.rho_ext_min) / span - 1.0
        return float(np.clip(raw, -1.0, 1.0))


def make_reactor_mpc(env, params, horizon: int = 20):
    """CasADi/IPOPT MPC for the nuclear reactor (point-kinetics + thermal feedback).

    With delta_t=0.5 s, horizon=20 gives 10 s of lookahead — enough to feel
    the fastest delayed-neutron group (λ≈3.0/s → τ≈0.33 s) and several
    slow-group time constants (λ≈0.012/s → τ≈80 s) are still visible via
    the integrator-like precursor dynamics. Longer horizons make the NLP
    expensive without meaningfully improving near-term tracking.
    """
    return ReactorCasadiMPC(env, params, horizon=horizon)
