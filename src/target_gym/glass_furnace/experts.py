"""Reference controller (the MPC slot) for the glass furnace task.

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
# GlassFurnace
# ---------------------------------------------------------------------------


#: Gain of the crown heat-rate disturbance estimate: the fraction of each
#: step's one-step crown prediction error folded into the estimate. 0.02 is a
#: 50-step time constant, against the crown's 132. The error it averages is
#: small and clean (standard deviation 0.01 to 0.03 K a step on protocol seed
#: 0, alternating with the reversal half cycle) next to the mismatch it
#: estimates: over the scored window of protocol seeds 0-2 the estimate sat
#: between +0.14 and +0.19 K a step (means 0.169 / 0.150 / 0.172). The plant
#: runs hotter than this reduced model on every seed.
_FURNACE_DISTURBANCE_GAIN = 0.02
#: Clamp on that estimate, in K/s (0.6 K a step, more than three times the
#: largest value measured), so a bad solve cannot run it away.
_FURNACE_DISTURBANCE_LIMIT = 0.02


#: Checker nodes per chamber in the *controller's* regenerator model. The plant
#: carries four; the controller carries two, and ``_extract_x0`` averages the
#: plant's nodes down in pairs.
#:
#: This is the one knob that decides what the furnace baseline costs to record.
#: The regenerator is two thirds of the controller's state vector, and IPOPT's
#: cost grows superlinearly in problem size: at the plant's four nodes the NLP
#: has 3168 variables and a bad seed ran 5.5 s per step, which is hours per
#: episode against a plant that steps in 92 microseconds.
#:
#: Two nodes rather than one averaged stack. The single stack was what this
#: model carried before, and it left a 2-6 K standing offset because everything
#: downstream of the checker temperatures is nonlinear in them; two nodes keep
#: the hot/cold split that sets air preheat, which is where that error came
#: from, at half the states.
_FURNACE_MPC_REGEN_NODES = 2


class GlassFurnaceCasadiMPC(CasadiMPC):
    """
    CasADi MPC for the regenerative glass furnace.

    Prediction model (8 states): ``[T_crown, T_melt, T_work, m_batch]`` and
    both regenerator chambers at two checker nodes each, plus the flame
    temperature as an *algebraic* variable. The plant has four nodes per
    chamber; ``_extract_x0`` averages them in pairs.

    What the controller knows ahead of time, as time-varying parameters over
    the horizon:

    * the target schedule, so it pre-moves before a scheduled step;
    * the reversal: which chamber preheats the air, and the firing dip while
      the valves change over, both deterministic in time;
    * the fuel already in the pipeline, which fixes the first
      ``FUEL_DEAD_TIME_STEPS`` intervals of every plan;
    * the loads: the pull at its AR(1) conditional mean, ``m_pull + rho^(k+1)
      d_t`` with ``d_t`` read from the state, and the pulsed batch charge
      ``charge_rate_now`` at that pull. Both are what the plant applies, on
      the mean;
    * a heat-rate disturbance on the crown, estimated online (see ``step``).

    Three things changed in the oracle audit (2026-10). The numbers are the
    protocol's window cost per step on seeds 0-2 (``eval.evaluate_controller``
    after its 800-step burn-in, as ``scripts/evaluate_baselines.py`` records
    it; lower is better), against 0.2996 / 1.7834 / 0.3263 (mean 0.8031) for
    the previous version:

    * **The applied input is the first one the plan can move.** The first
      ``FUEL_DEAD_TIME_STEPS`` intervals burn the pipeline, so do-mpc's u_0
      and u_1 never enter the predicted dynamics, and only the move penalty
      set them: the plant received ``u_prev + (u_2 - u_prev)/3``, a filter
      the plan did not model. Applying ``u_2``: 0.7327 mean.
    * **The loads are fed.** Plus the pull and charge above: 0.7271 mean, and
      better than the previous version on every seed.
    * **The offset-free correction is a heat-rate disturbance, not a
      setpoint bias.** See ``step``. 0.1784 / 1.0547 / 0.1709, mean 0.4680
      at the earlier 30-minute horizon: 42% below the previous version, and
      40-48% below it on each seed, with zero trips and zero solver
      failures. Keeping the setpoint bias and adding anti-windup to it
      instead gave 0.2454 / 1.4248 / 0.3064 (mean 0.6589).
    * **The horizon is one hour.** 0.1683 / 0.8449 / 0.1675, mean 0.3936:
      51% below the previous version, better than the 30-minute horizon on
      every seed (see ``make_glass_furnace_mpc``).

    What is left is mostly authority. On seed 1 a -5.5 K trim meets a low
    pull that puts the target below the crown's equilibrium at minimum fuel:
    0.82 of its 1.05 per window step is spent with the fuel burning at its
    minimum and the crown still above target. On seeds 0 and 2 the running
    (fuel) term is 26% and 37% of the cost, and 35% and 64% of the tracking
    cost falls on steps with the fuel burning at its maximum and the crown
    below target: the reversal dips. Those shares were measured at the
    30-minute horizon; the one-hour horizon took 20% off seed 1 by
    pre-cooling earlier.

    The objective is a saturating surrogate of the reward's squared error
    (``_build_mpc``); it ignores the reward's running cost.
    """

    SCALING = {
        "_x": {
            "T_crown": 1000.0,
            "T_melt": 1000.0,
            "T_work": 1000.0,
            "m_batch": 10000.0,
            **{f"T_rA{i}": 1000.0 for i in range(_FURNACE_MPC_REGEN_NODES)},
            **{f"T_rB{i}": 1000.0 for i in range(_FURNACE_MPC_REGEN_NODES)},
        },
        "_z": {"T_gas": 2000.0},
    }

    def _build_mpc(self):
        p = self.params
        from target_gym.glass_furnace.env import (
            FUEL_DEAD_TIME_STEPS,
            M_PULL_AR_RHO,
            N_SETPOINTS,
            charge_rate_now,
            firing_fraction,
        )

        if not hasattr(self, "_pipeline"):
            nominal = 0.5 * (p.fuel_min + p.fuel_max)
            self._pipeline = np.full(FUEL_DEAD_TIME_STEPS, nominal)

        model = do_mpc.model.Model("continuous")
        SB = 5.670374419e-8
        K = 273.15

        T_crown = model.set_variable("_x", "T_crown")
        T_melt = model.set_variable("_x", "T_melt")
        T_work = model.set_variable("_x", "T_work")
        m_batch = model.set_variable("_x", "m_batch")
        # Both regenerator chambers, at two nodes each against the plant's
        # four (see ``_FURNACE_MPC_REGEN_NODES``). An earlier version collapsed
        # the two alternating chambers onto one stack "at the cycle average",
        # which is exact only if everything downstream is linear in these
        # temperatures. It is not: measured by check 13 of the model review
        # checklist, the plant ran +0.0275 K per control interval hotter than
        # that model, one-signed on 71% of settled steps, and multiplied by the
        # crown's 132-step time constant that is the 2-6 K standing offset
        # which put this MPC 16% behind its own PID.
        from target_gym.glass_furnace.env import N_REGEN_NODES

        n_regen = _FURNACE_MPC_REGEN_NODES
        T_rA = [model.set_variable("_x", f"T_rA{i}") for i in range(n_regen)]
        T_rB = [model.set_variable("_x", f"T_rB{i}") for i in range(n_regen)]
        T_gas = model.set_variable("_z", "T_gas")  # algebraic: quasi-steady flame
        u_raw = model.set_variable("_u", "u_raw")
        model.set_variable("_tvp", "target_T_crown")
        # Firing actually released, as a fraction of commanded: 1 away from a
        # reversal, near zero while the valves change over. Deterministic in
        # time and therefore known to the controller: the dip is a periodic
        # upset a predictive controller can plan through and a PID can only
        # react to.
        firing = model.set_variable("_tvp", "firing")
        # Fuel already in the pipeline. The plant applies what was commanded
        # FUEL_DEAD_TIME_STEPS ago, so the first intervals of any plan are
        # already decided and nothing the optimiser does can change them.
        # ``commit`` is 1 over those intervals and 0 after.
        u_committed = model.set_variable("_tvp", "u_committed")
        commit = model.set_variable("_tvp", "commit")
        # The reversal is a deterministic function of time, so the oracle knows
        # it exactly rather than averaging it away. 0 -> A preheats air.
        a_is_air = model.set_variable("_tvp", "a_is_air")
        # The loads, on the mean. The pull is the nominal pull plus the AR(1)
        # disturbance's conditional mean, rho^(k+1) d_t with d_t read from the
        # state; the batch charge is the pusher's step-averaged pulse train at
        # that pull (``charge_rate_now``), a pure function of time. Both used
        # to be the nominal pull and a continuous charge, and a setpoint bias
        # absorbed the difference.
        m_pull_k = model.set_variable("_tvp", "m_pull_k")
        charge_k = model.set_variable("_tvp", "charge_k")
        # Heat-rate disturbance on the crown, K/s, constant over the horizon.
        # Estimated in ``step``.
        q_crown = model.set_variable("_tvp", "q_crown")

        m_fuel_free = p.fuel_min + 0.5 * (u_raw + 1.0) * (p.fuel_max - p.fuel_min)
        m_fuel = firing * (commit * u_committed + (1.0 - commit) * m_fuel_free)
        m_air = p.AFR * (1.0 + p.excess_air) * m_fuel
        m_gas = m_fuel + m_air

        eps = p.eps_regen_node

        def _duties(nodes):
            """Exhaust and air duties for one chamber, mirroring the plant.

            Exhaust enters at the hot end and works down; air enters at the cold
            end and works up. Each node exchanges with the stream passing it at
            per-node effectiveness ``eps_regen_node``.
            """
            t_in = T_gas
            q_exh = []
            for node in nodes:  # hot end first
                t_out = t_in - eps * (t_in - node)
                q_exh.append(m_gas * p.c_p_gas * (t_in - t_out))
                t_in = t_out
            t_stack = t_in

            t_in = p.T_ambient
            q_air_rev = []
            for node in reversed(nodes):  # cold end first
                t_out = t_in + eps * (node - t_in)
                q_air_rev.append(-m_air * p.c_p_air * (t_out - t_in))
                t_in = t_out
            return q_exh, list(reversed(q_air_rev)), t_in, t_stack  # noqa: E501

        qA_exh, qA_air, TA_air_out, _ = _duties(T_rA)
        qB_exh, qB_air, TB_air_out, _ = _duties(T_rB)

        # A chamber does one duty at a time, never both at half rate.
        QA = [a_is_air * qa + (1.0 - a_is_air) * qe for qa, qe in zip(qA_air, qA_exh)]
        QB = [(1.0 - a_is_air) * qb + a_is_air * qe for qb, qe in zip(qB_air, qB_exh)]
        T_air = a_is_air * TA_air_out + (1.0 - a_is_air) * TB_air_out
        UA_node = p.U_regen * p.A_regen / n_regen
        # The checker stack's total heat capacity is fixed by the plant; the
        # controller just divides it into fewer, larger nodes.
        C_regen_node = p.C_regen_node * (N_REGEN_NODES / n_regen)

        coverage = m_batch / p.m_batch_full
        melt_open = 1.0 - p.batch_shield * coverage

        T_gas_K = T_gas + K
        T_crown_K = T_crown + K
        T_melt_K = T_melt + K
        T_work_K = T_work + K

        A_eff = p.A_crown + p.A_melt * melt_open + p.A_work
        Q_in = m_fuel * p.LHV + m_air * p.c_p_air * (T_air - p.T_ambient)
        sink_K4 = (
            p.A_crown * T_crown_K**4
            + p.A_melt * melt_open * T_melt_K**4
            + p.A_work * T_work_K**4
        )
        sink_T = p.A_crown * T_crown + p.A_melt * melt_open * T_melt + p.A_work * T_work
        # Algebraic constraint: the flame's own energy balance closes.
        model.set_alg(
            "flame_balance",
            Q_in
            - p.eps_rad * SB * (A_eff * T_gas_K**4 - sink_K4)
            - p.h_conv * (A_eff * T_gas - sink_T)
            - m_gas * p.c_p_gas * (T_gas - p.T_ambient),
        )

        Q_rad_gc = p.eps_rad * SB * p.A_crown * (T_gas_K**4 - T_crown_K**4)
        Q_rad_gm = p.eps_rad * SB * p.A_melt * (T_gas_K**4 - T_melt_K**4) * melt_open
        Q_rad_gw = p.eps_rad * SB * p.A_work * (T_gas_K**4 - T_work_K**4)
        Q_rad_cm = p.eps_rad * SB * p.A_melt * (T_crown_K**4 - T_melt_K**4) * melt_open
        Q_rad_cw = p.eps_rad * SB * p.A_work * (T_crown_K**4 - T_work_K**4)
        Q_conv_gc = p.h_conv * p.A_crown * (T_gas - T_crown)
        Q_conv_gm = p.h_conv * p.A_melt * (T_gas - T_melt) * melt_open
        Q_conv_gw = p.h_conv * p.A_work * (T_gas - T_work)

        Q_wall_c = p.U_wall * p.A_wall_crown * (T_crown - p.T_ambient)
        Q_wall_m = p.U_wall * p.A_wall_melt * (T_melt - p.T_ambient)
        Q_wall_w = p.U_wall * p.A_wall_work * (T_work - p.T_ambient)
        Q_cool_w = p.UA_work_cooling * (T_work - p.T_ambient)

        Q_to_batch = (
            p.batch_shield
            * coverage
            * p.eps_rad
            * SB
            * p.A_melt
            * ((T_gas_K**4 - T_melt_K**4) + (T_crown_K**4 - T_melt_K**4))
        )
        melt_rate = Q_to_batch / p.dH_fusion

        cp_melt = p.c_p_glass_a + p.c_p_glass_b * T_melt
        cp_work = p.c_p_glass_a + p.c_p_glass_b * T_work

        model.set_rhs(
            "T_crown",
            (Q_rad_gc + Q_conv_gc - Q_rad_cm - Q_rad_cw - Q_wall_c) / p.C_crown
            + q_crown,
        )
        model.set_rhs(
            "T_melt",
            (
                Q_rad_gm
                + Q_conv_gm
                + Q_rad_cm
                - Q_wall_m
                - melt_rate * p.dH_fusion
                + m_pull_k * cp_melt * (p.T_batch_in - T_melt)
            )
            / (p.C_melt * cp_melt / p.c_p_glass_a),
        )
        model.set_rhs(
            "T_work",
            (
                Q_rad_gw
                + Q_conv_gw
                + Q_rad_cw
                - Q_wall_w
                - Q_cool_w
                + m_pull_k * cp_work * (T_melt - T_work)
            )
            / (p.C_work * cp_work / p.c_p_glass_a),
        )
        model.set_rhs("m_batch", charge_k - melt_rate)
        for i in range(n_regen):
            model.set_rhs(
                f"T_rA{i}",
                (QA[i] - UA_node * (T_rA[i] - p.T_ambient)) / C_regen_node,
            )
            model.set_rhs(
                f"T_rB{i}",
                (QB[i] - UA_node * (T_rB[i] - p.T_ambient)) / C_regen_node,
            )
        model.setup()

        mpc = do_mpc.controller.MPC(model)
        mpc.set_param(
            n_horizon=self.horizon,
            t_step=self.mpc_dt,
            n_robust=0,
            store_full_solution=False,
        )

        u_post = model.u["u_raw"]
        T_crown_post = model.x["T_crown"]
        target_post = model.tvp["target_T_crown"]
        m_fuel_post = p.fuel_min + 0.5 * (u_post + 1.0) * (p.fuel_max - p.fuel_min)

        # Share the minimiser of env.compute_reward, not its shape.
        #
        # This previously normalised the error by the crown's whole 250 K
        # envelope, six times flatter than the band the reward discriminates
        # over. Against an unchanged fuel penalty the controller duly sold
        # tracking for fuel, and lost to the PID on 7 of 10 seeds.
        #
        # The old form was also non-monotonic: ``((scale - err)/scale)**2`` turns
        # back upward past ``err = scale``, so beyond twice it the objective
        # preferred *more* error. A plain squared normalised error is monotone,
        # smooth, and minimised in the same place as the reward.
        #
        # ``tracking_band``, the same field name the four-tank, the
        # distillation column and the pH loop carry: the error at which
        # tracking is bad for this plant. It read ``params.tracking_scale``,
        # 40 K, left over from when the reward was ``clip(1 - err/40, 0, 1)**2``
        # -- the reward has been log-scaled for a while and that field was dead.
        # 40 K against errors of about 1 K put the tracking term at 6e-4 while
        # the fuel penalty stayed O(1), so the objective was nearly flat in the
        # direction being scored, which is both why this MPC trails its own PID
        # and why IPOPT struggles on it.
        scale = float(p.tracking_band)
        fuel_span = float(p.fuel_max - p.fuel_min)
        err = target_post - T_crown_post
        err_abs = casadi.sqrt(err * err + 1e-4)  # smooth |err|
        # Bounded rather than a bare quadratic. Normalising by 40 K instead of
        # 250 K is what fixes the weighting, but it also makes an unbounded
        # ``e**2`` reach 6.25 at a 100 K error where the old term gave 0.36, and
        # IPOPT does not cope: one seed of ten went from ~48 s to over 400 s.
        # ``e**2/(1+e**2)`` has the same minimiser and the same slope near zero,
        # is monotone in the error, and saturates instead of diverging -- which
        # also matches the environment's own reward, itself bounded in [0, 1].
        e_norm = err_abs / scale
        tracking_cost = e_norm**2 / (1.0 + e_norm**2)
        fuel_pen = float(p.fuel_cost_weight) * (m_fuel_post - p.fuel_min) / fuel_span
        mpc.set_objective(lterm=tracking_cost + fuel_pen, mterm=tracking_cost)
        mpc.set_rterm(u_raw=1e-3)

        mpc.bounds["lower", "_u", "u_raw"] = -1.0
        mpc.bounds["upper", "_u", "u_raw"] = 1.0

        default_target = float(sum(p.target_T_crown_range) / 2.0)
        self._target_schedule = np.full(N_SETPOINTS, default_target)
        self._current_step = 0
        self._pull_disturbance = 0.0
        self._q_crown = 0.0
        self._pred_crown = None
        # Overridden by make_glass_furnace_mpc; a default here so a directly
        # constructed instance still behaves.
        self._q_gain = _FURNACE_DISTURBANCE_GAIN
        self._max_steps = int(p.max_steps_in_episode)
        self._n_setpoints = int(N_SETPOINTS)
        p_rev = float(p.reversal_period)
        tvp_tpl = mpc.get_tvp_template()

        def tvp_fun(_t):
            for k in range(self.horizon + 1):
                future = self._current_step + k
                slot = min(
                    (future * self._n_setpoints) // self._max_steps,
                    self._n_setpoints - 1,
                )
                tvp_tpl["_tvp", k, "target_T_crown"] = float(
                    self._target_schedule[slot]
                )
                # The reversal is deterministic in time, so the oracle supplies
                # its exact phase across the whole horizon rather than averaging
                # it away: 1 while chamber A preheats the air, 0 while B does.
                cycles = (future * self.mpc_dt) / p_rev
                tvp_tpl["_tvp", k, "a_is_air"] = float(1.0 - (np.floor(cycles) % 2.0))
                tvp_tpl["_tvp", k, "firing"] = float(firing_fraction(future, p, xp=np))
                # Interval k runs from t+k to t+k+1, and the plant draws that
                # interval's pull after one more AR(1) step, so its mean is
                # rho^(k+1) d_t. Same 0.1 kg/s floor as the plant.
                m_pull = max(
                    float(p.m_pull) + M_PULL_AR_RHO ** (k + 1) * self._pull_disturbance,
                    0.1,
                )
                tvp_tpl["_tvp", k, "m_pull_k"] = m_pull
                tvp_tpl["_tvp", k, "charge_k"] = float(
                    charge_rate_now(m_pull, future, p, xp=np)
                )
                tvp_tpl["_tvp", k, "q_crown"] = float(self._q_crown)
                # The first FUEL_DEAD_TIME_STEPS intervals burn what is already
                # in the pipeline. Beyond that the optimiser decides.
                committed = k < len(self._pipeline)
                tvp_tpl["_tvp", k, "commit"] = 1.0 if committed else 0.0
                tvp_tpl["_tvp", k, "u_committed"] = (
                    float(self._pipeline[k]) if committed else 0.0
                )
            return tvp_tpl

        mpc.set_tvp_fun(tvp_fun)
        self._apply_scaling(mpc)
        mpc.set_param(nlpsol_opts=self._quiet_ipopt())
        mpc.setup()
        return mpc

    @staticmethod
    def _coarsen(nodes) -> np.ndarray:
        """The plant's checker nodes averaged onto the controller's, hot end first.

        Both chambers are kept, because they alternate duty and averaging them
        together is what left a standing offset. What is coarsened is the depth
        resolution within a chamber, which is a smooth profile down the stack:
        contiguous groups average to the temperature a node of that size would
        hold, and the group heat capacity is scaled to match in ``_build_mpc``.
        """
        x = np.asarray(nodes, dtype=float)
        n = _FURNACE_MPC_REGEN_NODES
        if x.size == n:
            return x
        return x.reshape(n, x.size // n).mean(axis=1)

    def _extract_x0(self, state):
        """The plant's regenerator state, both chambers, coarsened in depth.

        This once averaged the two chambers together *and* resampled four nodes
        onto three; that cost 2-6 K of standing offset, because everything
        downstream of the checker temperatures is nonlinear in them and the two
        chambers are never at the same temperature. Both chambers are kept now.
        The depth resolution is halved instead, which is what makes the NLP
        affordable: see ``_FURNACE_MPC_REGEN_NODES``.
        """
        return np.concatenate(
            [
                np.array(
                    [
                        float(state.T_crown),
                        float(state.T_melt),
                        float(state.T_work),
                        float(state.m_batch),
                    ]
                ),
                self._coarsen(state.T_rA),
                self._coarsen(state.T_rB),
            ]
        )

    def reset(self):
        """Clear the disturbance estimate as well as the warm start.

        Without this the estimate earned on one episode is carried into the
        next, where it starts from another plant state.
        """
        super().reset()
        self._q_crown = 0.0
        self._pred_crown = None

    def _update_setpoint(self, state):
        self._target_schedule = np.asarray(state.target_schedule, dtype=float)
        self._current_step = int(state.time)
        # What the plant will burn over the next intervals regardless of what is
        # decided now. An upper-bound controller reads the true state, and this
        # is part of it.
        self._pipeline = np.asarray(state.fuel_pipeline, dtype=float)
        # The pull disturbance is in the state too; its future innovations are
        # not, so the plan uses its conditional mean (see ``tvp_fun``).
        self._pull_disturbance = float(state.m_pull_disturbance)

    def step(self, _obs, state):
        """One receding-horizon step: estimate, solve, apply the first free move.

        **Offset-free correction.** The model is reduced (two checker nodes a
        chamber against four, a flame solved continuously against the plant's
        frozen per step, collocation against RK4), and a finite-horizon MPC
        with a mismatched model settles with an offset. The plant runs hotter
        than the model, about +0.15 K a step on the crown. That is estimated
        here as a heat-rate disturbance on the crown balance: each step, the
        crown the last solve predicted for now is compared with the crown
        measured, and a fraction ``_q_gain`` of the difference is added to the
        estimate, which the next plan carries over its whole horizon.

        This replaces a setpoint bias that integrated the tracking error
        (target minus crown) and shifted every future target by it. That bias
        had three faults the measurement showed. It integrated against the
        current slot while the plan pre-moved toward the next one, so it
        pushed against anticipation (seed 1: from -0.55 to +0.65 K over the 30
        steps before a -5.5 K trim). It wound up while the fuel sat at its
        bound (seed 1: -22 K, holding fuel at minimum for about 90 steps after
        the crown had fallen below target). And it was dropped at every slot
        change and re-learned, though it sat at -0.6 to -1.6 K within every
        slot on every seed, so each trim started with the model's offset
        uncorrected. A prediction error does none of these: it does not see the
        target, a saturated input is in the prediction, and the mismatch it
        measures does not change at a trim, so it never resets.

        **Applied input.** The plan's first ``FUEL_DEAD_TIME_STEPS`` inputs
        never reach the predicted dynamics (those intervals burn the
        pipeline), so the input applied is the plan's input at that index,
        the first one that does. do-mpc's ``u_prev``, which its move penalty
        compares the next plan with, is set to it. Returning do-mpc's u_0, as
        the base class does, gave the plant ``u_prev + (u_2 - u_prev)/3``.
        """
        from target_gym.glass_furnace.env import FUEL_DEAD_TIME_STEPS

        if self._pred_crown is not None:
            e1 = float(state.T_crown) - self._pred_crown
            self._q_crown = float(
                np.clip(
                    self._q_crown + self._q_gain * e1 / self.mpc_dt,
                    -_FURNACE_DISTURBANCE_LIMIT,
                    _FURNACE_DISTURBANCE_LIMIT,
                )
            )
        self._update_setpoint(state)
        x0 = self._extract_x0(state)
        m = self._mpc
        if not self._initialized:
            m.x0 = x0
            m.set_initial_guess()
            self._initialized = True
        guess = self._save_guess()
        m.make_step(x0)
        ok = self._record_solve()
        u = np.array(
            m.opt_x_num["_u", FUEL_DEAD_TIME_STEPS, 0] * m._u_scaling
        ).flatten()
        if ok:
            # The crown this plan predicts for the next step, for the next
            # estimate. Node k of do-mpc's collocation grid is
            # ``_x[k, scenario, -1]``.
            self._pred_crown = float(
                m.opt_x_num["_x", 1, 0, -1, "T_crown"] * m._x_scaling["T_crown"]
            )
        else:
            u = self._fallback(guess, u)
            self._pred_crown = None
        m._u0.master = casadi.DM(u)
        self._last_u = u
        return float(np.clip(u, -1.0, 1.0)[0])


def make_glass_furnace_mpc(
    env,
    params,
    horizon: int = 120,
    disturbance_gain: float = _FURNACE_DISTURBANCE_GAIN,
):
    """CasADi/IPOPT MPC for the GlassFurnace.

    With delta_t = 30 s, horizon = 120 is one hour of lookahead, about 0.9 of
    the crown's 3960 s (132-step) open-loop time constant. That is enough to
    see each scheduled setpoint change and pre-cool or pre-heat for it, which
    a PID cannot do, and to start pre-cooling early enough for a large trim.

    Measured in the oracle audit (2026-10) on protocol seeds 0-2: 0.1683 /
    0.8449 / 0.1675, mean 0.3936, against 0.1784 / 1.0547 / 0.1709 (mean
    0.4680) at 30 minutes (horizon 60): better on every seed, mostly seed 1
    (-20%), where it starts pre-cooling earlier. It costs 1.7 to 1.9 times
    the solve time of the 30-minute horizon.

    ``disturbance_gain`` is the gain of the crown heat-rate estimate
    (``GlassFurnaceCasadiMPC.step``). The result is flat in it: 0.05 gave
    1.0514 on seed 1 against 1.0547 at the default 0.02.
    """
    mpc = GlassFurnaceCasadiMPC(env, params, horizon=horizon)
    mpc._q_gain = float(disturbance_gain)
    return mpc
