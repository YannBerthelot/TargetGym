# Cold review: distillation

Reviewer had not seen the design. Read REALITY.md first, then env.py, env_jax.py, PHYSICS.md and the shared base.
Paths below are relative to `src/target_gym/`. REALITY.md source tags (D1 to D6) are kept as written there.

## Summary

- The objective is two-sided setpoint tracking of both products. Targets are drawn per episode. The real job is to hold both impurities inside one-sided zones at the least reflux and boilup [D4 p.5]. Over-purifying is free on the plant and costly here.
- Composition limits are trips with a 4 h restart charge. On the real splitter the product specs are control zones, a quality cost, not a trip [D4 p.5 Table 1]. The real hydraulic limit, flooding [D6], is not modelled at all.
- The only disturbance is a slow drift in feed composition. The bad days the sources show are a 20% feed-rate step and a boilup upset from steam pressure [D1 p.33 Fig. 18, p.24]. Feed rate is constant in the task.
- Reflux and boilup jump anywhere in their range in one minute. The real splitter's MPC moves reflux at most about 0.1% of its range per minute [D4 p.5 Table 2].
- The drums hold 0.5 kmol. The source says 10 kmol (reboiler) and 32.1 kmol (condenser) for realistic studies [D2]. So the product compositions react far faster than on a real column.
- The tray temperature is a perfect function of composition, because pressure is constant. On a real column pressure "significantly affects" tray temperature [D3]. At alpha 1.5 the source doubts that inferential temperature control works at all [D3].

## Observations

1. **Analyser timing.** The task reports yD and xB every 5 min, 5 min old, so 5 to 10 min of total delay (pc_gym/distillation/env_jax.py:31, 58-62). Sources give 3 to 30 min [D5], and say 5 or 10 min cycles hardly differ on a PP splitter [D3]. This is inside the range. **Minor.** It sits at the fast end, which makes the task a little easier than a typical column.
2. **Analyser noise of 4e-4.** env_jax.py:59, 61. REALITY.md gives no analyser scatter. **Unknown.**
3. **Tray temperature ignores pressure.** T_15 is a linear function of x on stage 15 (env.py:360-363), and pressure is constant (PHYSICS.md:33-35). Real tray temperature moves with column pressure, and needs a pressure correction [D3]. Inferential control is not feasible below alpha 1.4 and only usually works above 2.0 [D3]. Column A sits at 1.5 [D2]. **Major.** The task's inner temperature loop is a near-perfect composition sensor. A good controller here leans on it more than a real one could. The missing physics is pressure dynamics and its effect on bubble point, which D3 names.
4. **No thermocouple lag.** The tray reading has no `lag` (env_jax.py:60). REALITY.md: response time not found. **Unknown.**
5. **One tray temperature.** env.py:69, get_obs at env.py:366-387. Skogestad recommends at least three per section [D1 p.34]. **Minor.** Fewer sensors makes the task a little harder, and it is a design choice an operator could also make.
6. **No pressure, level or DP readings.** The operator reads column pressure, condenser and reboiler levels and section DP [D1 p.25; D4 p.5; D6]. The task has none, because pressure and levels are not states. **Minor** for pressure and levels while they are held perfectly. **Major** together with the missing flood limit (limits, item 3), since DP is how an operator sees flooding coming [D6].
7. **Flows reported exactly.** L_pct and V_pct are the commands (env.py:375-376). Real reflux and heat-pump flows have their own flow controllers, FC 2 and FC 3 [D4 p.5]. **Minor.**

## Actuators

1. **No move limit.** The action maps straight to L and V each step (env.py:316-321). The base actuator rate and lag are 0 (base.py, EnvParams `actuator_rate`, `actuator_lag`). On the real splitter reflux runs 200 to 320 t/h with moves of at most 0.13 t/h per minute. Heat-pump flow runs 220 to 350 t/h with moves of at most 0.15 t/h per minute [D4 p.5 Table 2]. **Major.** A bang-bang or high-gain policy is allowed here and impossible there. That changes what a good controller does on an ill-conditioned plant, where large coordinated moves are the obvious exploit. The source's limits are MPC move bounds, not valve hardware. So the missing fact is the operating practice on move size, and D4 states it.
2. **Boilup is a direct flow.** V is set exactly (env.py:320). The source names a boilup disturbance from steam pressure [D1 p.24, Fig. 12], and the real splitter boils up with a heat pump, so boilup depends on the compressor [D4 p.2]. **Major** as a disturbance (see below). **Minor** as an actuator.
3. **Flow box.** L 2.20 to 3.40 and V 2.70 to 3.90 kmol/min (env.py:94-97) are "ours, from the target box" (PHYSICS.md:101-118). REALITY.md has no Column A limits. The real splitter's reflux range is 1.6 to 1 max over min [D4 p.5 Table 2]. **Rule 12 candidate.** The upper V is tied to an assumed 80% of flood with no source.
4. **Distillate clip.** D is clipped to 0.1 to 0.9 (env.py:100-101, 255-259, 274). Where it binds, the mass balance breaks (PHYSICS.md:248-250). No source describes this. **Rule 12 candidate.**

## Limits

1. **Composition trips.** yD at or below 0.90, or xB at or above 0.10, trips the column (env.py:104-105, 390-394), with a restart charge of 240 steps (env.py:165-169). On the real splitter the product limits are zones: propane in propylene at most 0.45 mol%, propylene in propane at most 2 mol% [D4 p.5 Table 1]. They are a cost or quality limit, not a trip. **Major.** It adds a cliff that the real column does not have. That pushes a good policy toward caution. The 240-step restart has no source (env.py:168 says provisional). **Rule 12 candidate.**
2. **No product zone.** e_tol is 0 (env.py:161), and the cost is symmetric. The real zone has one side only: impurity below the limit is in spec [D4 p.5 Table 1]. **Major,** see objective.
3. **No flooding.** Nothing limits V or L by hydraulics. Real columns flood as duty rises, and DP noise warns of it [D6]. Whether flooding is a trip, an alarm or a cost: not found. **Major** as a missing constraint, **unknown** as to its form. The task's V_max stands in for it with no source.
4. **No weeping, level or pressure limits.** REALITY.md: not found. **Unknown.**

## Time scales

1. **Step of 1 min.** env.py:138. The real splitter's MPC samples at 1 min [D4 p.5]. Skogestad's simulations use a 1 min measurement delay [D1 p.25]. **Matches.**
2. **Dominant time constant about 194 min.** PHYSICS.md:66 claims it. This matches Column A [D1 p.14-15]. Column A is a model, not a plant [D1 p.13]. The real PP splitter has about 2 h [D3] and a 111 min dead time from reflux to the bottom product [D4 p.5 G2]. The task has no such dead time: its reflux reaches the reboiler in 2.46 min. **Minor** for the task as Column A. **Major** if the task is read as an industrial splitter, since a 111 min dead time is a different control job. The LV scheme "performs poorly with large dead-times" [D1 section 6.5].
3. **Drum holdups.** M_drum is 0.5 kmol (env.py:79). D2 suggests 10 kmol in the reboiler and 32.1 kmol in the condenser "for more realistic studies". **Major.** The condenser then adds about 10 min of first-order lag on yD (32.1 kmol over about 3.2 kmol/min). The task's top product answers a reflux move almost at once, which makes the top loop easier than on a plant.
4. **Episode of 400 min, about 2 dominant time constants.** env.py:141. REALITY.md gives no episode length. The 40 min recovery in D1 Fig. 18 fits inside it. **Minor.**

## Operating mode

1. **Two-point control to a moving target.** Each episode draws new targets in yD 0.980 to 0.995 and xB 0.005 to 0.020 (env.py:129-130; env_jax.py:141-150). Real practice is often one-point control (xB by LV) or no composition control at all [D1 sections 6.2, 6.4]. Where two products are controlled, the MPC holds zones at fixed specs [D4 p.5]. No source says the spec changes from run to run. **Major.** The per-episode target spread is a **rule 12 candidate.** It makes the benchmark a regulator for many operating points, not one.
2. **Cascade structure available.** The tray temperature allows the temperature cascade that D1 p.33 recommends. **Matches**, subject to Observations item 3.

## Disturbances

1. **Feed composition only, as a slow OU drift.** Stationary sd 0.03, reversion 1/500 min, clipped to 0.40 to 0.60 (env.py:57-60, 87-89, 323-334). PHYSICS.md:98 labels the amplitude TUNED. The sources show a zF step from 0.5 to 0.6 [D1 Fig. 18]. REALITY.md gives no amplitude or spectrum of real feed drift. **Unknown** for the shape, **rule 12 candidate** for the size.
2. **No feed-rate change.** F is fixed at 1.0 (env.py:84). D1 Fig. 18 shows a 20% feed step as the large upset. On the real splitter feed moves 10 to 45 t/h and the MPC pushes it up [D4 p.5 Table 2]. **Major.** Feed rate is the disturbance with the larger effect, about 4 times the zF step in D1 Fig. 18. On a plant it is measured (FC 1), so feedforward is possible. The task offers neither the disturbance nor the reading.
3. **No boilup upset.** The steam-pressure disturbance [D1 p.24, Fig. 12] is absent. **Major,** for the same reason: it hits the bottom directly and is what the temperature loop exists for.
4. **No analyser failure.** D1 section 6.5 discusses it. **Minor.** It is a bad-day case, not a normal one.

## Shape of the episode

1. **Start near steady state, off target.** The reset uses the nominal converged profile, with L drawn in 2.65 to 2.76 and V = L + 0.5 (env_jax.py:128-176). The targets differ from 0.99 and 0.01, so every episode opens with a transition. A real column runs continuously at its spec [D4 p.5]. **Minor.** It puts a setpoint change into every episode that the plant rarely sees.
2. **Instant restart after a trip.** The plant returns to the reset profile on the next step (base.py, failure_kernel). REALITY.md has no restart procedure. **Unknown,** and a **rule 12 candidate** together with the 240-step charge.

## Objective

1. **What is scored.** It scores the squared error of each product against its target over 4e-4, plus V over 3.282, plus the trip charge (env.py:397-417, 159-169). The operator is paid to keep both impurities inside their zones at least reflux and boilup [D4 p.5; REALITY.md, the operator's job]. **Major.** On the plant the best point sits just inside each zone, at least energy. Here the best point is exactly on target, and purer product is penalised as much as off-spec product. The energy weight of 1 against an error unit of 4e-4 is ours (env.py:116-126 comment). So the trade between purity and energy is not the plant's.
2. **Reflux not charged.** Only V is charged. In LV with D = V - L the two move together, so this is **minor**.
3. **Score reads the true state.** The plant is paid on analysed product (PHYSICS.md:401-405). **Minor.**

## Parts no source in REALITY.md supports (rule 12 candidates)

- Composition trips at yD 0.90 and xB 0.10, and the 240-step restart (env.py:104-105, 165-169).
- Per-episode targets in 0.980 to 0.995 and 0.005 to 0.020 (env.py:129-130).
- Flow limits L 2.20 to 3.40 and V 2.70 to 3.90 (env.py:94-97).
- The D clip to 0.1 to 0.9 (env.py:100-101).
- The OU feed drift's size and rate (env.py:57-60, 87-89).
- The analyser and RTD noise figures (env_jax.py:59-61). REALITY.md has none.
- The boilup running weight and c_hold (env.py:163-164).

## Proposed behaviour tests

Each test runs the task with `sensor_noise_scale = 0` and the feed drift off, unless it says otherwise. Each tolerance is fixed now, before any run.

1. **Nominal steady state** [D2]. Hold L = 2.706 and V = 3.206 at zF 0.5 from the reset profile for 2000 min. Read yD, xB and D. Pass if yD is 0.99 and xB is 0.01, each within 10% of the impurity (yD in 0.989 to 0.991, xB in 0.009 to 0.011), and D within 0.5%. Reason: D2 prints the impurities to one significant figure as design specs. 10% of the impurity is tight enough to catch a wrong alpha or stage count.
2. **Reflux step, open loop** [D1 p.15 Fig. 5]. Step L by +0.1% at constant V from steady state. Read the final changes in yD and xB and the time to 63%. Pass if yD rises by 0.0015 to 0.0025 (figure: about 0.002), xB by 0.0025 to 0.0045 (figure: about 0.0035), and the 63% time is 175 to 215 min (text: 194). Reason: the end values are read off a plot, so 25% either side. The time constant is printed to three digits, so 10%.
3. **Boilup step, open loop** [D1 p.15 Fig. 5]. Same step on V at constant L. Pass if each final change has the opposite sign to test 2, and its size is 0.8 to 1.25 times test 2's. Reason: the source says "almost the same size". The tolerance is read as within 25%.
4. **Internal flows, D constant** [D1 p.15 Fig. 6]. Raise L and V together by 10%. Pass if both products get purer, and the gain per unit flow change is 30 to 300 times smaller than test 2's. Pass also if the 63% time is 7 to 30 min (text: about 15 min, first order). Reason: the text says "about 100 times" and "about 15 min". A factor of about 3 either side is all a statement in words supports.
5. **Hydraulic delay to the bottom** [D1 p.13 Fig. 4, p.15 Fig. 6]. Step L with `delta_t` set to 0.05 min. Read the time at which xB has moved 2% of its change over the first 10 min. Pass if it is 1.9 to 3.1 min. Reason: the source gives 2.46 and 2.54 min. The band covers both, plus the spread of the 39 lags.
6. **Slow linear modes** [D1 p.15 section 4.4]. Take the eigenvalues of the Jacobian of `compute_velocity` at the nominal point. Pass if the three slowest time constants are 193.9, 12.0 and 3.5 min, each within 5%. Reason: they are printed to three digits from the same model.
7. **Gain matrix and RGA** [D1 p.9 eq. 31]. Use small steady-state steps (plus or minus 0.003 kmol/min) in L and V. Pass if each gain entry is within 2% of [[0.8754, -0.8618], [1.0846, -1.0982]] and lambda11 is within 5% of 35.94. Reason: the entries are printed to four digits. The RGA magnifies entry errors about 36 times, so it gets a wider band.
8. **Feed-rate step under LV PI control** [D1 p.33 Fig. 18]. Close two PI loops with D1's own tunings and a 1 min measurement delay. Then step F by +20%. Pass if the dip in yD is 3.0e-3 to 5.6e-3 (figure: about 4.3e-3), the peak in xB is 2.5e-3 to 4.5e-3 (figure: about 3.5e-3), and both are back within 20 to 60 min (figure: about 40). Reason: values are read off a plot, so 30% on size and 50% on time. This test needs F to move during a run. Today F is a fixed parameter, so the test also checks that feed rate can be a disturbance. D1's PI tunings are not in REALITY.md. They must be taken from D1 before the test is run.
9. **Feed-composition step under the same loops** [D1 p.33 Fig. 18]. Step zF from 0.5 to 0.6. Pass if the peak deviation of each product is 0.5e-3 to 2e-3 (figure: about 1e-3), and less than half of test 8's. Reason: the figure gives one rough size. A factor of 2 is what it supports. The ratio to test 8 is the source's main point.
10. **Analyser delay** [D5]. Step the true yD (by an L step) and log the analyser channel. Pass if the time from the step to the first changed reading is always between 3 and 30 min. Reason: the band is D5's stated range. The test checks that the reading cannot be fresher or staler than a real column analyser.

No test is proposed from D3 (2 h) or D4 (111 min dead time), because those figures are for a different column. They belong to a decision on whether the task claims to be Column A or an industrial splitter.
