# Wind turbine: cold review against REALITY.md

Reviewer had not seen the design. Compared only against `REALITY.md` and its sources [S1] to [S15].
Files read: `env.py`, `env_jax.py`, `PHYSICS.md`, the `row_wind_turbine` row of
`scripts/reference_ladder.py` on `feat/reference-ladder`, and the trip kernel in `base.py` that the
task calls. Line numbers are for this worktree.

## Mismatches

Ranked by how much each changes what an agent learns or what the score measures.

1. **No tower at all, so no tower motion.** The task is one rigid mass, `J dw/dt = tau_aero - N
   tau_gen` (`env.py:9-14`, `env.py:323-340`); PHYSICS.md D2 says so. The plant has a first tower
   fore-aft mode at 0.3240 Hz and side-side at 0.3120 Hz, drivetrain torsion at 0.6205 Hz and blade
   collective flap at 0.6993 Hz [REALITY 4, S1 Table 9-1]. Bending moments carry over 99 % of tower
   bottom fatigue [REALITY 3, S5 Sect. 1], and the controller's job includes damping tower and
   drivetrain motion [REALITY 6, S1 Sec. 7, S5]. Effect: no thrust force, no tower state, no way to
   excite or damp a structural mode. An agent learns to pitch at whatever bandwidth tracks power,
   which on the real machine would pump the tower. The central trade-off of above-rated pitch
   control (speed regulation against tower loads, S5's 2.1 % tower DEL for 1.26 % blade DEL) is
   absent, and the score cannot reward a controller that respects it.
2. **No nacelle accelerometer, and no tower-damping loop.** The observation is rotor speed, pitch,
   torque, electrical power and target (`env.py:412-429`). Real turbines measure tower-top
   acceleration (CART fore-aft and side-side, M5000 fed to collective pitch; sensors "already
   there" on most turbines) [REALITY 1, S3 PDF p. 5, S5 Sect. 3], and damp fore-aft with pitch and
   side-side with torque [REALITY 2, S3, S5 Eq. 6 and 7]. Effect: follows from item 1; the second
   real feedback signal of a Region 3 controller is missing.
3. **Objective is a stepping curtailment dispatch, not the plant's normal above-rated job.** Four
   iid uniform levels in 3.5 to 5 MW, each held 25 s (`env.py:68-69`, `env.py:136-138`,
   `env.py:385-397`, `env_jax.py:143-149`). On a normal day above rated, pitch holds rated speed and
   torque holds rated power [REALITY 5, S1 Sec. 7]. Curtailment in the sources is a set point
   followed within a TSO-set time and tolerance [S10 Art. 13(6)], a single de-rate step then a
   frequency event or an AGC signal from real ACE data [S11 Figs. ES-8, ES-9], under ramp limits of
   1 to 10 MW/min, 5 MW/min averaged over 10 min [S12 Sects. 1, 2]. A 1.5 MW move every 25 s is
   about 3.6 MW/min sustained in steps, with each step taken in under a second. Effect: the score
   is dominated by step transients that no grid code issues, and the agent learns an AGC-like
   tracking skill on a schedule no source supports. In derated operation ROSCO also lowers the
   speed, torque and minimum pitch references together [S9 Ex. 29]; the task keeps speed pinned at
   rated through its speed band (`env.py:178-182`), so a derated agent cannot run the rotor slower.
4. **Trip handling: an instant reset at rated with a fresh wind.** Over 1.25 x rated or under 0.40
   x rated is a trip (`env.py:110`, `env.py:119`, `env.py:432-439`); with `continuing` true
   (`base.py:59`) the plant restarts at once as `reset_env` builds it (`base.py:481-485`,
   `env_jax.py:151-176`). In the plant, above a stored permissible speed the pitch system feathers
   on battery power, independent of the main controller [REALITY 3, S4]; shutdown is staged
   feathering [S9 Ex. 30, S2 Sect. 5.4]; start-up from 90 deg is a free-wheel and torque steps
   (ROSCO demo over 400 s) [S9 Ex. 32]. The overspeed trip level in percent of rated is "not
   found" [REALITY 3]. Effect: the 1.25 trip and its price (`restart_steps`, `env.py:184-185`) rest
   on a figure REALITY could not find. The safety system that would act first (feather) is not
   modelled, so an agent is never shown what the plant does at overspeed.
5. **Underspeed trip has no counterpart.** The 0.40 x rated trip is justified in a comment as
   aerodynamic stall (`env.py:111-119`). REALITY gives a 6.9 rpm minimum rotor speed as an operating
   range (0.57 x rated) [S1 Table 1-1] and a Region 1.5 torque ramp that unloads the generator at
   low speed [S1 Sec. 7.2]; ROSCO shuts down on a generator speed threshold [S9 Ex. 30]. No source
   calls low rotor speed a trip. Effect: the task punishes over-torquing with a trip, where the
   plant's torque law would never ask for that torque.
6. **Controller step 0.25 s against 80 to 100 Hz.** `delta_t = 0.25` (`env.py:143`). NREL 5 MW
   simulations run at 0.0125 s [S1 Sec. 8] and CART's controller at 100 Hz [S3 PDF p. 5]; commercial
   rates "not found" [REALITY 4]. 4 Hz still gives 12 samples per 0.324 Hz tower period, so a tower
   mode could be represented and damped at this step, but the second fore-aft mode (2.9003 Hz) is
   above the 2 Hz Nyquist and would alias. Effect: fine for the rigid plant, would need checking if
   item 1 is fixed. The comment `1200 steps = 5 min` (`env.py:142`) disagrees with the 400 steps on
   the next line.
7. **Turbulence model.** An Ornstein-Uhlenbeck point wind, 1/theta = 20 s, constant 1.2 m/s std,
   uniform mean 13.5 to 20 m/s (`env.py:65-67`, `env.py:104-107`, `env.py:360-371`). IEC gives
   sigma1 = Iref (0.75 Vhub + 5.6), rising with wind, length scale 42 m above 60 m hub height,
   and discrete events: extreme operating gust (10.5 s), direction change (6 s), coherent gust
   (10 s), shear (12 s) [REALITY 4 and 5, S13]. Load cases span 3 to 25 m/s [S2 Table 2]. Effect:
   turbulence is the same size at 20 m/s as at 13.5 m/s, there are no gusts, and the band from
   rated to 13.5 m/s and from 20 to 25 m/s cut-out is never visited. The near-rated band is where
   ROSCO's set-point smoothing and peak shaving act [S2 Sects. 5.2, 5.3], so the hardest real
   transition is excluded. Iref is not given in REALITY, so the class's absolute level is open.
8. **The wind is hidden entirely.** `v_wind` is not observed (`env.py:412-419`). A real turbine
   carries a nacelle anemometer, biased by the rotor and corrected by a transfer function, residual
   0.36 m/s RMSE, worse in high turbulence [REALITY 1, S6]; ROSCO estimates rotor-effective wind at
   0.48 m/s RMSE [S2 Sect. 5.1]. Effect: the task is a harder estimation problem than the plant
   poses. A biased, noisy anemometer channel would be the real one.
9. **Speed is read raw, without the plant's sensor path.** Rotor speed is observed with white noise
   (`env_jax.py:57-62`); no filter or delay in the plant. The real feedback is generator speed
   through a 0.25 Hz low-pass set at a quarter of the edgewise frequency [REALITY 1, S1 p. 18, S2
   Sect. 2.1]. That filter belongs to the controller, so leaving it to the agent is fair; but the
   reason for it (edgewise 1.08 Hz content in the speed signal) is absent, so a filter-free agent
   is not penalised. Measurement delay is "not found" [REALITY 1], and the task assumes none.
10. **Actuator ranges and lags.** Pitch 0 to 40 deg (`env.py:92`) against 0 to 90 deg feather
    [S1 p. 26]; Cp clipped at zero (`env.py:288`) so pitching does not brake. Pitch lag 0.2 s and
    torque lag 0.1 s (`env.py:94`, `env.py:97`) are marked ours in PHYSICS.md 3; S1 sets its pitch
    actuator at 30 Hz to be negligible [S1 Sec. 7.4], CART's torque loop is 500 rad/s [S3], and a
    commercial pitch bandwidth is "not found". Torque is floored at 0 (`env.py:357`, `env.py:384`)
    where CART's converter goes from minus to plus rating [S3]. No pitch acceleration limit (CART
    150) [S3]. Pitch rate 8 deg/s, torque 47,400 N m and 15,000 N m/s match S1. Effect: the
    agent cannot feather to stop or brake the rotor; the torque lag is about 50 times CART's.
11. **Fatigue cost is pitch travel, not structural load.** `fatigue` charges pitch travel above the
    oracle's hold level (`env.py:465-471`, `c_hold` and `fatigue_weight` at `env.py:176-177`).
    REALITY ties fatigue to tower bending DEL [S5 Sect. 1] and to the tower against blade DEL trade
    [S5 Sect. 3.2]. Effect: pitch activity is a weak proxy without a tower; S5's own tower-damping
    loop adds pitch motion (under 0.5 deg) to cut tower load, which this cost would penalise.
12. **Episode is 100 s from equilibrium.** 400 steps of 0.25 s (`env.py:144`), started at rated
    speed with the balancing pitch (`env_jax.py:151-176`). Real statistics are 10 min aggregates
    [S7, S6] and simulated load cases are 10 min with 6 seeds per wind speed [S2 Table 2, Fig. 6].
    Effect: shorter than one SCADA interval and one ramp-limit window [S12]; each episode sees about
    five 20 s turbulence time constants, so the wind barely samples its distribution.
13. **No grid events.** No frequency response, droop, deadband, grid loss or islanding [REALITY 5,
    S10 Tables 4 and 5, S12 Sects. 2 and 3, S14]. Effect: optional for a pitch task, but the task's
    framing is grid dispatch, and the dispatch carries none of the grid's shape.
14. **Steady Cp surface, no rotor dynamics.** S1 schedules gains on a power sensitivity that doubles
    at 6.302336 deg [REALITY 4, S1 pp. 22-25]. PHYSICS.md D1 shows the steady table runs nearly
    through the origin instead. Effect: the published gains do not place the published poles on
    this plant, so the reference controller had to be refit (PHYSICS.md 5), and the agent learns
    a pitch sensitivity that no S1 figure supports.

## Unsupported parts

No source in REALITY.md supports these; by the rule they go or need a source:

- The dispatch: 4 iid levels, 25 s blocks, 3.5 to 5 MW (`env.py:68-69`, `env.py:136-138`).
- Turbulence time constant 20 s and constant 1.2 m/s std (`env.py:67`, `env.py:105`); the mean
  range 13.5 to 20 m/s (`env.py:104`); clip bounds 4 and 28 m/s (`env.py:106-107`).
- Overspeed trip at 1.25 x rated (`env.py:110`): REALITY marks this level "not found".
- Underspeed trip at 0.40 x rated (`env.py:119`).
- Speed band 1.10 and weight 200 (`env.py:178-182`).
- Pitch lag 0.2 s, torque lag 0.1 s (`env.py:94`, `env.py:97`): pitch bandwidth "not found".
- Pitch range capped at 40 deg (`env.py:92`) and Cp clipped at 0 (`env.py:288`).
- Control step 0.25 s (`env.py:143`): commercial rates "not found", the two sources say 80 and 100 Hz.
- Instant restart after a trip (`base.py:481-485`) and its 10 min price (`env.py:184-185`).
- Imbalance price 100 $/MWh, linear tracking with zero tolerance, pitch-travel fatigue, `c_hold`
  (`env.py:167-190`). S10 says tolerance is TSO-set but gives no value.
- Sensor sigmas and the 1 kW meter (`env_jax.py:57-62`, PHYSICS.md Readings): from the Odgaard
  benchmark and IEC class 0.1S, neither in REALITY; power sensor location, rate and filtering are
  "not found".
- No sensor delay: REALITY marks measurement delay "not found", and the task relies on zero.
- Episode length 100 s (`env.py:144`).

## Proposed behaviour tests

Each states the source, the quantity, the experiment and a tolerance fixed in advance.

1. **Rated torque.** Source: S1 p. 19 and Table 7-2, 43,093.55 N m at 1173.7 rpm generator speed,
   5 MW electrical, 94.4 % efficiency. Quantity: generator torque that gives 5 MW at rated speed.
   Experiment: evaluate from parameters. Tolerance 0.01 %: the four inputs are exact published
   figures, so only rounding remains.
2. **Peak Cp.** Source: S1 p. 19, Cp 0.482 at tip speed ratio 7.55 and 0 deg. Quantity: the
   maximum of the plant's Cp over tip speed ratio at 0 deg and where it falls. Experiment: sweep
   the surface. Tolerance: 0.01 in Cp and 0.5 in tip speed ratio (one table node): the surface is
   tabulated at 0.5 steps, so a node can be half a step off, and 0.01 is the rounding of a
   two-figure Cp.
3. **Region 3 hold with the published controller.** Source: S1 Sec. 7, pitch holds rated speed
   1173.7 rpm and torque holds 5 MW. Experiment: zero turbulence, constant winds 13, 16, 20, 25 m/s,
   target 5 MW, NREL baseline PI with the published gains (KP(0) 0.01882681 s, KI(0) 0.008068634,
   theta_K 6.302336 deg) run as a fixed recipe, 600 s. Quantity: mean speed and power over the last
   60 s. Tolerance: 1 % on each: with integral action and steady wind the only residual is table
   interpolation, well under 1 %.
4. **Closed-loop speed response.** Source: S1 pp. 22-25, PI placed for 0.6 rad/s and damping 0.7.
   Experiment: as test 3 at 15 m/s, then a 1 m/s wind step. Quantity: rotor speed peak time and
   overshoot. Expected from a second-order system: peak time pi / (0.6 sqrt(1 - 0.49)) = 7.3 s,
   overshoot 4.6 %. Tolerance: peak time within 25 %, overshoot under 15 %: S1 placed poles on a
   single-mass model without the speed filter and actuator lags, which shift the response by that
   order; a plant whose pitch sensitivity departs from S1's (item 14) fails it.
5. **Pitch sensitivity schedule.** Source: S1 pp. 22-25, power sensitivity to pitch doubles at
   6.302336 deg. Experiment: at rated speed, along the Region 3 pitch curve from 12 to 25 m/s,
   step pitch by 0.1 deg with speed and wind frozen, read dP/dtheta. Quantity: theta_K from a line
   fit of dP/dtheta against theta. Tolerance: 25 %: S1's own figure is a linear fit through
   scattered points (Fig. 7-4).
6. **Actuator limits are hard saturations.** Source: S1 Table 7-2: pitch rate 8 deg/s, torque
   47,402.91 N m, torque rate 15,000 N m/s; pitch 0 to 90 deg (S1 p. 26). Experiment: full-range
   step commands on each actuator. Quantity: largest rate and value over the run. Tolerance: no
   excess at all for the rates and torque (hard saturations); pitch must reach 90 deg.
7. **Tower fore-aft mode.** Source: S1 Table 9-1, 0.3240 Hz fore-aft, 0.3120 Hz side-side.
   Experiment: zero turbulence, 15 m/s, constant pitch and torque at balance, a 1 deg pitch step.
   Quantity: peak of the tower-top acceleration spectrum over 600 s. Tolerance: 0.02 Hz: twice the
   spectral resolution of a 100 s window, covering the 0.012 Hz gap between the two modes. The task
   has no tower state, so this fails by absence until item 1 is addressed.
8. **Tower damping suppresses the tower peak.** Source: S5 Sect. 3.3 and 4, tower-frequency
   acceleration "fully suppressed" with pitch modulation usually under 0.5 deg. Experiment: test 3's
   recipe in turbulence at 15 m/s, with and without S5's band-pass plus 3p-notch loop, same seeds.
   Quantity: tower-top acceleration power within 0.30 to 0.35 Hz, and RMS pitch modulation in that
   band. Tolerance: power reduced at least tenfold (read "fully suppressed" as 10 dB or more), pitch
   modulation RMS under 0.5 deg. Requires test 7 first.
9. **Turbulence grows with wind.** Source: S13 quoting IEC 61400-1, sigma1 = Iref (0.75 Vhub + 5.6).
   Experiment: hold the plant, record the hidden wind over 20 records of 10 min at means 13.5 and
   20 m/s. Quantity: ratio of the standard deviations. Expected 20.6 / 15.725 = 1.31. Tolerance:
   10 %: a 10 min record of a 20 s process has about 30 independent samples, so one record's std
   errs about 13 %, and 20 pooled records bring that near 3 %; 10 % leaves room. Iref itself (the
   turbulence class) is for the owner to choose.
10. **Extreme operating gust shape.** Source: S13, magnitude the smaller of 1.35 (Ve1 - Vhub) and
    3.3 sigma1 / (1 + 0.1 D / 42 m), with Ve1 = 0.8 x 1.4 Vref, duration 10.5 s. Experiment: request
    one gust at 15 m/s, D = 126 m. Quantity: gust amplitude and duration in the wind trace.
    Tolerance: 1 % on amplitude, duration exact to one step: the formula is deterministic. Plant
    response to the gust has no source ("not found"), so no plant tolerance is set.
11. **Overspeed safety system feathers.** Source: S4, above a stored permissible speed the pitch
    system feathers independently of the main controller. Experiment: 20 m/s, torque command 0
    (a grid loss), pitch command held at balance. Quantity: pitch after the speed passes the
    permissible level, and rotor speed afterwards. Tolerance: pitch rises at the 8 deg/s limit
    (within one step) to 90 deg, and speed falls monotonically after feather starts. The trip
    level is "not found", so the test reads it from a parameter that must cite a source.
12. **A de-rate is a reference change, not only a power target.** Source: S9 Ex. 29, a power rating
    R sets speed, torque and minimum pitch references; S11 Fig. ES-8, a de-rate step. Experiment:
    ROSCO-style rating step from 1.0 to 0.9 at 15 m/s, zero turbulence. Quantity: steady speed,
    torque and power after the step. Tolerance: power within 1 % of 4.5 MW; speed and torque at the
    rating's references within 1 %, same reasoning as test 3. Ramp settling has no source number
    (S10 leaves it to the TSO), so none is set.
