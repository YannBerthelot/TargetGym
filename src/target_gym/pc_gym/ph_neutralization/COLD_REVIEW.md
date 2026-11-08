# Cold review: ph_neutralization

Summary of the mismatches that matter most:

- The objective is tight quadratic tracking to 0.01 pH with no band. The operator is paid to stay inside a discharge permit (a timed limit [P8], a hard pH 5.0 floor to sewer [P9]) at least reagent [P5]. A good controller for the task is not a good controller for the plant.
- The main field disturbance, large unknown changes in the influent's strong acid concentration [P3], is absent. Acid flow and strength are constants; only an OU drift of the buffer flow moves the plant.
- Effluent pH can fall to about 3.1, below the 5.0 sewer floor [P9], and nothing trips, diverts or counts it. The field plant diverts effluent on low tank pH [P3].
- The reagent valve is ideal: instant, continuous, never closes below 10 mL/s. Real final elements are on/off pumps, timer shots or PI valves with resolution, split range and plugging limits [P3, P4].
- One perfectly mixed tank of fixed volume. Real units are staged (two or more tanks or mixers) [P3, P5, P7], and the lab rig controlled level as well as pH in each of two tanks [P1].

Sources are cited by their REALITY.md tags ([P1] to [P10]). File and line refer to this folder unless stated.

## Observations

1. **One probe, no redundancy, no faults.** Task: a single pH reading with 0.01 sigma, 0.01 step, 10 s lag and 10 s dead time (env_jax.py:52-54), from a perfectly mixed tank (env.py:258-265). Plant: the field case had three transmitters with middle-of-three selection at each stage and on the tank [P3, pp. 4-5]. Coating, junction blockage and electrolyte dilution give sluggish response and drift, and lead to overdosing [P5, p. 2; P6]. Effect: easier. The task never asks a controller to survive a slow or drifting probe, which is the failure the sources name most. Minor for the nominal task, major if robustness is the claim.
2. **Electrode lag and sampling.** Task: 10 s first-order lag. Plant: electrode time constant and sampling period not found in REALITY.md. Unknown. The 10 s dead time is in line with the 7 s mixing delay and 10 s assumption of [P7, p. 2]; that part is supported.
3. **No discharge sensor.** Task: tank pH is the scored and observed quantity (env.py:332, 352-358). Plant: a separate sensor at the point of discharge records effluent pH for inspection [P5, p. 2]. With one perfectly mixed tank the two coincide, so this is minor until staging is added.
4. **No flow measurements, but none needed.** Task: no influent or reagent flow reading; the valve position is observed (env.py:331). Plant: flow transmitters on waste and reagent [P3, Figure 2], and feedforward is a named enhancement [P3, p. 3]. Because influent flow is constant here (env.py:82), the missing reading costs nothing. It would matter once influent load varies (see Disturbances). Minor.

## Actuators

5. **Ideal continuous reagent flow.** Task: base flow follows the action exactly each 5 s step, no rate limit, lag, dead band or stiction (env_jax.py:79 with base defaults off; env.py:278-280). Plant: on/off pumps or solenoids, deadband or latching, timer "shots", or PI to an I/P and valve [P4, pp. 1-4]; split range points cause limit cycles and high resolution valves move under 0.2 % [P3, p. 4]. Effect: easier. Valve resolution near the equivalence point is where real pH loops cycle. Major.
6. **The reagent valve cannot close.** Task: q3 lives in 10 to 22 mL/s (env.py:95-96). Plant: no source gives a floor on reagent flow; a dosing pump or valve can stop. Rule 12 candidate. Effect: removes the "dose nothing" move and bounds the reachable pH.
7. **Single reagent, single element.** Task: base only. Plant: acid or caustic added as needed [P5, p. 1; P6], two final elements with different set points [P4, p. 2], several valves behind a splitter [P3]. Minor: the task's influent is always acidic, so base alone is enough.
8. **Reagent strength.** Task: the base stream is 3 mM, about the influent's own strength, so reagent flow is about equal to influent flow (env.py:82-83, 95-98). Plant: REALITY.md gives no reagent concentration or reagent-to-influent ratio, only that the ratio is tracked per shift [P3, p. 9]. Unknown. If real reagent is concentrated, reagent flow barely changes residence time, which the task's does.

## Limits

9. **No permit band, no timed limit.** Task: no trip (env.py:335-341); pH 2 and 12 are documentation only (env.py:105-107); e_tol is 0 (env.py:157). Plant: discharge outside the permitted range at most 7 h 26 min per month and no excursion over 60 min [P8], a timed limit. Effect: a different job. Major.
10. **No hard floor at pH 5.** Task: reachable steady pH spans about 3.1 to 10.6 (comment env.py:336-337) and below 5 is just a large tracking cost. Plant: no discharge to sewer below pH 5.0 [P9]; any discharge below the low limit could be a recordable violation, and effluent is diverted when tank pH gets too low [P3, pp. 4-5]. Major.
11. **No level limit.** Task: fixed volume (env.py:79). Plant: batch units lock out on wrong level [P4, p. 1]; the lab rig controlled level [P1]. Minor for a continuous overflow tank.
12. **The trip price is never used.** failure_cost and restart_steps (env.py:161-163) are unsourced and unreachable. Rule 12 candidate, harmless.

## Time scales

13. **Residence time of 88 s.** Task: 2.9 L at about 33 mL/s (env.py:79, 82, 95). Plant: lab rig residence time not found [P1, P2 abstracts only]. Industrial continuous tanks under on/off control need over 10 min retention [P4, p. 2]; the design analysis of [P7, Table 1] gives tens of m3 at 0.01 m3/s for three or four tanks. Effect: as a bench rig, unknown; as an industrial unit, the task is 7 or more times faster than the shortest tank P4 allows. Unknown pending the Henson and Seborg paper.
14. **Episode of 25 min.** Task: 300 steps of 5 s (env.py:134-135; the comment above it still says 600 steps and 50 min). Plant: the timed limit counts 60 min excursions [P8], so a 25 min episode cannot show one. Shift metrics run over hours [P3, p. 9]. Effect: the permit's time structure cannot be expressed. Major if a permit objective is added.

## Operating mode

15. **Setpoint is imposed.** Task: target drawn in 6.5 to 8.0 per episode (env.py:128; env_jax.py:129-133). Plant: reagent cost is cut by putting the set point closer to the pH limits [P3, p. 4], and a fuzzy optimizer moved the pH set points [P3, Figure 2]. Effect: the real economic decision is removed. Different job.
16. **No manual mode, no cascade.** Plant: manual override of dosing is common [P5, p. 2]; loops ran in cascade under a set point optimizer [P3]. Minor for a single-loop benchmark.

## Disturbances

17. **Influent acid strength never changes.** Task: Wa1 and q1 are constants (env.py:82-84). Plant: large unknown changes in strong acid concentration from production units [P3, p. 4]; loops went from calm to "rocking and reeling" over one lunch break [P3, p. 8]. Effect: easier and different. The task's hardest real event is absent. Major.
18. **Buffer drift shape.** Task: buffer flow is an OU process, mean 0.55, std 0.35 mL/s, correlation time about 500 s, clipped to 0 to 2.5 (env.py:73, 87-92, 285-292). Plant: unmeasured buffering changes on the bench rig [P2] and buffer changes that shift the titration curve [P1]. Their size, shape (step or drift) and timing are not in REALITY.md; the buffer step experiments are on the owner list. The OU form, its std (marked TUNED in PHYSICS.md section 3) and theta are rule 12 candidates.
19. **No sensor or valve failures.** Coating, breakage, poisoning [P5, pp. 2-3], valve plugging [P3, pp. 4-5]. Minor for the nominal task.

## Shape of the episode

20. **Start at steady state off target.** Task: starts at the steady state of a random valve in 14.5 to 16.5 mL/s and a random buffer flow (env_jax.py:119-134), so each episode opens with a setpoint step. Plant: a continuous unit runs without resets; lab rig step tests over a wide range are recorded [P1]. Acceptable for a benchmark; minor.
21. **No restarts, no diversion.** The plant diverts effluent [P3]; the task has no event at all. See item 10.

## Objective

22. **Quadratic in 0.01 pH units.** Task: (error / 0.01)^2 every step (env.py:352-358, 156-157). Plant: inside the permit band all pH values are compliant; the cost is reagent [P5, p. 2; P3, p. 4] and valve wear near the seat [P3, p. 5]. Example spec pH 7 plus or minus 1 [P7, p. 2]. Effect: rewards precision the permit does not pay for, and makes a 0.5 pH miss cost 2,500 times a 0.01 one. Major.
23. **Reagent cost is nearly constant.** Task: reagent flow over the hold flow, weight 1 (env.py:159-160, 360; reward.py:45). With a fixed target the steady reagent use is set by the chemistry, so this term only prices transients. Rule 12 candidate for the weight. The real trade (set point against reagent) is not reachable, see item 15.

## Parts no source in REALITY.md supports (rule 12 candidates)

- Rig numbers V, q1, Wa1, Wa2, Wb2, Wa3, Wb3 (env.py:79-98). REALITY.md could not open the Henson and Seborg paper; they may be right, but are unverified here.
- The 10 to 22 mL/s valve range and its non-zero floor (env.py:95-96).
- The OU buffer model, its std, mean and theta (env.py:73, 87-92).
- The 0.01 pH noise and step, and the 10 s electrode lag (env_jax.py:53).
- failure_cost, restart_steps, running_weight, c_hold (env.py:159-163).
- pH limits 2 and 12 (env.py:105-107).

## Proposed behaviour tests

Each test names its source, what it measures, how it excites the plant, and a tolerance fixed now.

1. **Mixing and measurement delay.** Source: [P7, p. 2], mixing delay about 7 s, 10 s assumed per tank. Measure: time from a step in base flow to the first change in the observed pH larger than 0.005 pH, at sensor_noise_scale 0. Excite: hold at steady state near pH 7, then step q3 by +0.5 mL/s. Tolerance: the apparent dead time lies in 5 to 20 s. Reason: the step is 5 s, so one step of quantization each side of the 7 to 10 s figure.
2. **Buffer shifts the titration curve.** Source: [P1] abstract, buffer changes shift the titration curve; [P2], unmeasured buffering changes. Measure: steady pH against q3 over the valve range, at buffer flow 0 and at the nominal flow. Excite: steady-state sweeps, no control. Tolerance: the two curves differ by at least 0.1 pH at some q3. Reason: ten times the 0.01 pH resolution, so the shift is visible to the instrument. A numeric figure awaits the owner list.
3. **Sewer floor is enforced.** Source: [P9], no discharge below pH 5.0; [P3], effluent diverted on low pH. Measure: whether the task flags a violation (trip, diversion or permit count) on any step with true effluent pH below 5.0. Excite: valve at its minimum with buffer flow 0 until pH settles below 5. Tolerance: exact, binary. Reason: the rule is a hard limit. Expected to fail today (env.py:339).
4. **Timed excursion rule.** Source: [P8], no single excursion over 60 min, at most 7 h 26 min per month. Measure: the scorer's excursion clock. Excite: drive pH outside the permit band for 59 min, then separately for 61 min, in an episode long enough to hold it. Tolerance: 59 min is not a violation, 61 min is, within one 5 s step. Reason: the clock is discrete at the step. Needs a sourced band and a longer episode; expected to fail today.
5. **The score prefers less reagent inside the band.** Source: [P5, p. 2], overdosing is costly; [P7, p. 2], spec pH 7 plus or minus 1. Measure: episode score. Excite: two constant-valve runs at nominal buffer, both settling inside pH 6 to 8, one at lower pH and less reagent, one nearer the target with more. Tolerance: ordering only, the lower-reagent run scores at least as well. Reason: inside the band the permit pays nothing for precision. Expected to fail today (e_tol = 0).

Tests that need the owner's sources first: the Henson and Seborg titration curve and buffer step traces [P2], and the Hall and Seborg step responses [P1]. P4's 10 min retention rule applies only if on/off dosing is modelled, so it is not proposed for the continuous valve.
