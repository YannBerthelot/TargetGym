# Cold review: four_tank

Summary of the mismatches that matter most:

- The valve split, gamma1 = gamma2 = 0.2, is far deeper into the non-minimum phase region than any documented rig setting. Its RGA is -0.067 against -0.64 (Lund P+) and -0.28 (Delaware), and the cross gains are about 4.5 times the direct ones against about 1.6 on the rig [S1, S5]. It is a harder, different plant.
- The upper tanks sit above the lower ones (about 0.29 m over 0.15 m at mid target) and are slower than them. On every rig the upper tanks run well below the lower ones and are faster [S1, S5].
- A low-level trip at 5 cm with a priced restart has no source. The rig's floor is an empty tank (h >= 0) and its ceiling is overflow at 25 cm [S2, S4]; no trip or interlock was found.
- The disturbances are steady flows added to or drawn from the lower tanks. The rigs use a cup poured into a lower tank, an extra outlet opened in an upper tank, and pumps drawing from the upper tanks [S4, S5]. The upper-tank disturbance, which hurts most in the non-minimum phase case [S4, p. 10], is missing.
- Episodes start from random levels with no pump history. Rig runs start at steady state and then take setpoint steps of about 5 percentage points [S4, S1].

Sources are cited by their REALITY.md tags ([S1] to [S7]). File and line refer to this folder unless stated. Rig time constants below are derived here from T = (A/a) sqrt(2h/g) [S1] at the task's mid targets (h1 0.15 m, h2 0.21 m), which give h3 about 0.29 m, h4 about 0.20 m, and pumps at 7.2 and 6.3 V.

## Observations

1. **All four levels are read by an industrial radar.** Task: h1 to h4 through a guided wave radar model, 0.5 mm sigma, 1 mm step, no lag (env_jax.py:49-51; env.py:289-292). Plant: lower levels as voltages, y = kc h, kc 0.50 V/cm at Lund [S1, p. 2046]. Only the lower sensors are wired at KTH; upper levels are read by eye [S4, p. 6 and p. 12]. Lund's interface shows all four [S2, Figure 11]. Effect: reading the upper tanks makes a cascade possible, which is easier where the rig gives only two outputs. Minor, since one rig shows four.
2. **No offset jumps, no calibration error.** Plant: level signals show offset jumps, and scale and screen disagree because the sensors are not calibrated exactly [S4, footnotes 3 and 4, p. 6]. Task: zero-mean white noise only. Effect: easier. Minor.
3. **Sensor type and lag.** The radar transmitter is not the rig's instrument; the rig's sensor type and time constant were not found. Unknown, and a rule 12 candidate for the instrument choice.

## Actuators

4. **Pumps are ideal.** Task: voltage maps to flow at once and linearly, no lag, rate limit or dead zone (env.py:212-231; env_jax.py:77 with base defaults off). Plant: flow k v proportional to voltage, with k varying slightly with operating point [S1, p. 2046; S2, p. 123]; air in the tubes even with pumps on [S4, footnote 2]. Pump stroke or lag not found. Minor, mostly unknown.
5. **Voltage range.** Task: 0 to 10 V (env.py:40-41), rest near 6 to 7 V. Plant: 0 to 100 % of a 15 V maximum at KTH [S4, p. 15]; Lund maximum not found; nominal 50 to 60 % [S4, pp. 5-6; S5, Table 3]. Saturation is a soft rule on the rig: keep steps small so pumps do not saturate [S4, p. 9]. Comparable headroom. Minor.
6. **Unequal pumps.** Task: k1 = 8.5e-4, k2 = 9.5e-4 (env.py:26-27). Plant: the widely copied linear model is wrong when pumps differ [S6, abstract]. What the correction is was not read. Unknown.

## Limits

7. **Low-level trip at 5 cm, with restart cost.** Task: any level at or below 0.05 m trips, is charged restart_steps times failure_cost, and restarts (env.py:44, 114-118, 295-299). Plant: h >= 0 is the physical floor [S2, p. 121]; overflow trip or interlock not found. Effect: the task's binding constraint, on the upper tanks (PHYSICS.md D2), shapes the target box and the controller. On the rig an empty tank is only a nonlinearity. Major, rule 12 candidate.
8. **No tank height.** Task: h_max 1.5 m (env.py:46), unreachable. Plant: full at 25 cm on KTH [S4, p. 15]; tank level 1 means full in [S3, Figure 4]. At the task's levels a 25 cm upper tank would overflow (h3 up to about 0.45 m at reset, env.py:82). Different plant scale: the task's tanks have no sourced height. Major if overflow is part of the rig's behaviour; overflow handling itself not found.
9. **No pump cost.** Task: tracking only (env.py:302-328). Plant: Delaware DMC put a cost on pump speed with a minimum level constraint [S5, p. 6]; otherwise not found. Minor.

## Time scales

10. **Upper tanks slower than lower ones.** Task: derived T1 about 50 s, T2 about 69 s, T3 about 122 s, T4 about 80 s (env.py:29-37 at the levels above). Plant: in every published set the upper tanks are the faster pair, (T3, T4) = (23, 30), (39, 56), (32, 39), (53, 55) s at Lund [S1, pp. 2047-2048] and 38.5, 31.1 s at Delaware [S5, Table 3], against lower tanks of 48 to 112 s. Effect: the cross path through the upper tanks is slow here, which deepens the inverse response and the lag of the crossed loops. Major.
11. **Step and episode.** Task: 1 s step, 1200 s episode (env.py:89-90). Plant: sampling not found; Delaware PRBS switched at most every 35 s [S5, p. 3]. Non-minimum phase step responses are plotted over 2200 to 3000 s and settle in several hundred to about 1000 s [S3, Figure 4; S1, Figure 6]. The loads arrive at step 581 (env.py:98), so the scored hold is 620 s. Effect: one non-minimum phase setpoint transient of the rig would not fit. Minor for the task as built (no setpoint step); see item 15.

## Operating mode

12. **The valve split is not a rig setting.** Task: gamma1 = gamma2 = 0.2 (env.py:23-24), sum 0.4, RGA -0.067, cross gains about 4.5 times the direct ones (PHYSICS.md section 2). Plant: Lund P+ is 0.43, 0.34 (sum 0.77, RGA about -0.64), P- is 0.70, 0.60 [S1, p. 2046]; Delaware 0.333, 0.307 (sum 0.64, RGA about -0.28) [S5, Table 3]. Lund G+(0) has cross gains about 1.6 times the direct ones [S1, eq. (3)]. Effect: a harder and qualitatively different control job; the documented rig's P+ case allows a decentralized PI with one loop of gain -0.12 [S1, p. 2048], which this split does not resemble. Major.
13. **The split is fixed for the task.** Task: one setting, always non-minimum phase. Plant: set by hand before each run, never during it [S3, p. 2; S4, pp. 13-14], with both minimum and non-minimum phase cases run. Fixing gamma within a run matches the rig. Offering only one setting is a design choice, not a mismatch in behaviour. Minor.
14. **No manual mode or bumpless transfer.** Plant: manual and automatic switched in the interface; bumpless transfer is crucial [S3, p. 6; S4, p. 15]. Minor.

## Disturbances

15. **Load shape and place.** Task: from step 581, each lower tank gets an unmeasured flow drawn uniformly within plus or minus 2.5e-4 m3/s, held 300 s (env.py:95-101, 181-199, 216, 222). Plant: a cup of water poured into a lower tank (a pulse), an extra outlet opened in an upper tank (a level-dependent draw) [S4, Exercise 4.1.1, p. 9]; submersible pumps draw from tanks 3 and 4 [S5, pp. 1-2]. The extra outlet can cause problems in the non-minimum phase case [S4, p. 10]. Effect: the task tests rejection of steady lower-tank inflows and never the upper-tank draw that excites the non-minimum phase path. Size, period and uniform shape are rule 12 candidates. Major.
16. **Setpoint steps are absent.** Task: targets fixed per episode (env_jax.py:125-130). Plant: setpoint steps of about 5 percentage points under automatic control [S4, p. 9; S1, Figures 5-6; S3, Figure 4]. Effect: the documented test of the rig, a step in r1 with the y2 excursion it causes, is never run. Major.

## Shape of the episode

17. **Random start, no pump history.** Task: four levels drawn independently, pumps recorded off (env_jax.py:113-142; env.py:80-87). Plant: start pumps at 50 %, wait for steady state, then test [S4, pp. 5-6]. Effect: the first 581 steps are a start-up from an arbitrary, not physically settled, state, which the rig never runs. Minor, since that window is outside the loads.
18. **Trip restarts.** See item 7; no source for restart within a run.

## Objective

19. **Quadratic in mm on both lower tanks.** Task: (e / 1 mm)^2 per tank, no band (env.py:111-113, 312-324). Plant: hold the lower levels at setpoint [S1, p. 2046]. No spec band was found. Supported in form; the 1 mm scale is the radar's, not the rig's. Minor.

## Parts no source in REALITY.md supports (rule 12 candidates)

- All plant constants A, a, k, gamma (env.py:23-37). PHYSICS.md already says they are PC-gym's, not the rig's.
- The 5 cm low-level trip, h_max 1.5 m, failure_cost and restart_steps (env.py:44-46, 114-118).
- The load model: size, start, period, uniform draw, place (env.py:95-101).
- The guided wave radar instrument and its figures (env_jax.py:49-51).
- The 0 to 10 V pump range (env.py:40-41).

## Proposed behaviour tests

Each test names its source, what it measures, how it excites the plant, and a tolerance fixed now. Tests 1 and 2 check the model's structure with the rig's constants set as parameters; tests 3 to 6 check the shipped task against the rigs.

1. **Rig steady state, Lund P+ and P-.** Source: [S1, table p. 2046]. Measure: steady h1 to h4. Excite: set A, a, k, gamma to the Lund P+ values, hold both pumps at 3.15 V until settled, loads off; repeat for P- at 3.00 V. Tolerance: each level within 5 % of 12.6, 13.0, 4.8, 4.9 cm (P+) and 12.4, 12.7, 1.8, 1.4 cm (P-), or 0.1 cm, whichever is larger. Reason: the table rounds to 0.1 cm and k varies slightly with operating point [S2, p. 123].
2. **Rig time constants.** Source: [S1, table p. 2047]. Measure: time constants from small open-loop steps of each pump around the P+ and P- steady states of test 1. Tolerance: within 10 % of (62, 90, 23, 30) s and (63, 91, 39, 56) s. Reason: a finite step on a square-root law and a first-order fit add a few percent.
3. **Relative gain sign and value.** Source: [S3, eq. (2)], lambda = gamma1 gamma2 / (gamma1 + gamma2 - 1). Measure: lambda from the steady-state gain matrix found by 2 % voltage steps on the shipped task, loads off. Tolerance: within 5 % of the formula. Reason: step size nonlinearity on the square-root outflow stays below 2 %.
4. **Cross to direct gain ratio within the rigs' range.** Source: [S1, eq. (3)] G+(0) = 1.5, 2.5, 2.5, 1.6; [S5, Table 3] gammas 0.333, 0.307. Measure: G12/G11 and G21/G22 on the shipped task, as in test 3. Tolerance: inside 1.3 to 2.6, the span from Lund P+ (about 1.6) to Delaware (about 2.1 with equal pumps), widened by 25 % because rigs differ. Expected to fail today (about 4.5).
5. **Right half-plane zero in the rigs' range.** Source: [S1, table p. 2047], P+ zero +0.013; [S5, p. 4], +0.0324. Measure: the positive transmission zero of the task's model linearized by finite differences at the mid targets. Tolerance: 0.010 to 0.040, the published span widened by 25 %. Reason as test 4.
6. **Upper tanks below and faster than lower tanks.** Source: [S1, table p. 2046] h3/h1 about 0.38 (P+), 0.15 (P-); [S5, Table 3] 0.51 and 0.42; time constants of item 10. Measure: steady h3/h1, h4/h2, and T3/T1, T4/T2 at each target corner, loads off. Tolerance: level ratios at most 0.6 and time constant ratios below 1. Reason: 0.6 is the highest published level ratio plus about 15 %; every published upper time constant is shorter than its lower one. Expected to fail today.

Not proposed, for lack of numbers in REALITY.md: the S4 extra-outlet disturbance (qualitative only), and the S1 Figure 5 and 6 closed-loop traces, which depend on the rig's PI tunings and would test a controller, not the plant.
