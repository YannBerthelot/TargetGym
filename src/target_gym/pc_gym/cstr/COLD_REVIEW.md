# Cold review: `cstr`

Summary of the mismatches that matter most:

- The task cannot run away. The coolant range stops below the ignition fold, so the half of the operator's job that keeps heat removal ahead of heat release [S5; S4] is absent.
- The agent sets the coolant temperature directly, with no jacket, no slave loop, no dead time and no cooling ceiling. The plant moves coolant flow through a cascade with about 2 min of jacket dead time [S6, S7, S1].
- The residence time is 1 min. The sources give 34 min (PARSEX) and 15 to 120 min (DuPont) [S1, S9]. The analyser's 1 min delay is a whole residence time here.
- The task holds concentration at 10 to 13 percent conversion. The documented operator holds temperature [S5, S6], and PARSEX ran at about 74 to 78 percent conversion [S1].
- The feed is hotter than the reactor (350 K feed into a 317 to 332 K reactor). PARSEX feeds at room temperature into a 58 °C reactor [S1].
- The disturbances are small uniform feed steps every 10 min. None of the documented bad days (catalyst loss, fouling, circulation or cooling failure, feed build-up) is present [S1, S2, S3].

Sources are tagged as in `REALITY.md`. File references are to this folder.

## Observations

1. **No jacket temperatures or coolant flow.** The task observes `[C_a, T, target]` (env.py:216). Operators read jacket inlet and outlet temperatures and coolant flow [S8, Fig. 6; S6; S1, Fig. 2], and T2's screen showed loop temperatures [S2, Fig. 9]. The task has no jacket state to show. Effect: a different job. The cascade the plant runs has no slave measurement to close on. Major.
2. **Concentration is read every 15 s, 1 min late, with 5e-4 mol/L scatter** (env_jax.py:48-51). A GC on a loop reactor has 7 to 10 min dead time [S11]; Bayer sampled for the lab once a day [S3]. An online analyser on a jacketed liquid CSTR: not found. Effect: easier, since closing a loop on composition becomes possible. Unknown, leaning major. Note that in residence times the task's 1 min is harder than 7 to 10 min on a 34 min plant.
3. **No sample-and-hold on the analyser.** Every step brings a new composition value (env_jax.py:49). A GC reports discretely, once per cycle [S11]. Effect: easier. Minor.
4. **No thermowell lag on T** (env_jax.py:50). WIKA measured 63 percent times of 4 s bare to 45 s in a thick well [S10, Fig. 2]. At a 15 s step this is at most three steps. Minor.
5. **No level reading.** Holdup is constant (env.py:20). Bayer ran level control near 50 percent [S3]. Minor.

## Actuators

6. **Coolant temperature is the action, with no jacket dynamics** (env.py:159-170, 186-188; PHYSICS.md D2). The plant moves fresh cooling-water flow [S1, p.4; U2 in the unstable record] or a jacket slave loop [S6, S7], recirculated and pumped [S8]. Jacket outlet dead time about 2 min in one plant [S6]. Effect: much easier. The fastest path from move to reactor is removed. Major.
7. **No cooling capacity ceiling tied to flow.** The command saturates at 295 and 302 K (env.py:30-31), a band chosen to keep the plant below ignition (PHYSICS.md section 2). On PARSEX the ceiling is the water flow range, 40 to 110 g/s [S1, Table 1]. Effect: a different job. The real ceiling is a heat-removal limit that the reactor can outrun. Major.
8. **No split range, no heating side.** The action is cooling only. Plants run split range heating and cooling on one slave [S7, S12]. Minor for a continuous unit at steady duty; unknown for startup.
9. **No valve dynamics.** `actuator_lag` and `actuator_rate` stay at 0 (base.py defaults, not set in env.py). A proportional valve on an electronic actuator moves the water [U2]. Its dynamics: not found in REALITY.md. Unknown.

## Limits

10. **No trip and no runaway.** `check_is_terminal` never fires (env.py:219-225). Bayer had high temperature and pressure interlocks on the feed valve and a safe limit of 155 °C [S3]; relief disks at T2 and Morton [S2, S4]; HSE lists crash cooling, inhibitor, quench and dump [S5, p.4]. Effect: a different job. A good controller here never has to weigh margin to runaway. Major.
11. **The coolant band is the safety device.** The task keeps the reactor safe by limiting what the agent can command (PHYSICS.md section 2). A real plant has the opposite: full cooling is always available and may not be enough [S3, S4]. Major, same root as 6 and 7.
12. **Trip cost and restart time are unsourced.** `failure_cost` 7.2e5 and `restart_steps` 240 (env.py:74-76). REALITY.md gives no restart time. Never charged, so no effect today. Rule 12 candidate.

## Time scales

13. **Residence time 1 min** (env.py:19-20, V/q). Sources: 34 min [S1, p.5], 15 to 120 min [S9]. Effect: neutral in residence-time units, but every lag fixed in seconds or minutes (analyser, jacket, thermowell) is out of proportion to the plant. Major.
14. **Step 15 s, episode 37.5 min** (env.py:46, 50). In residence times the episode is 37 residence times; PARSEX's 2 °C step settled in about 3.2 residence times (110 min over 34 min) [S1, Fig. 2]. Effect: the episode is long relative to one setpoint move. Minor.
15. **Open-loop speed.** The task's eigenvalues are -0.75 to -1.53 /min (PHYSICS.md section 2), so the reactor settles in a few minutes. PARSEX's closed loop took 110 min [S1]. Measured jacket and reactor thermal time constants: not found. Unknown.

## Operating mode

16. **Concentration is the controlled variable.** The task scores `C_a` against a target (env.py:236-244). The operator holds reactor temperature at its setpoint [S5; S6; S4]. Effect: a different job, set by the analyser one assumes. Major.
17. **Low-conversion branch.** Targets 0.87 to 0.90 mol/L from a 1.0 mol/L feed, 10 to 13 percent conversion (env.py:43; PHYSICS.md D1). PARSEX ran at about 74 percent design conversion and CA 0.33 from 1.5 kmol/m3 [S1, p.4, Tables 3 and 4]. Effect: a different plant regime. The reactor sits where the Arrhenius loop is weak. Major.
18. **Feed hotter than the reactor.** `Ti` 350 K against a 317 to 332 K reactor (env.py:27). PARSEX feeds at 23 to 25 °C into a 58 °C reactor [S1, Tables 3, 4]. Effect: the feed adds heat, so a feed-flow or feed-temperature upset acts with the opposite sign to the plant's. Major.
19. **Each episode starts off target.** C_a 0.80 to 0.85 and T 330 K at reset (env.py:44-45; env_jax.py:126-146), then a move to a random target. A real normal day is regulation at one point [S1, Tables 3, 4]. A random setpoint at every start: not found. Minor.

## Disturbances

20. **Feed steps of ±1 K and ±0.02 mol/L, uniform, every 10 min from step 67** (env.py:55-62, 137-151). Feed temperature differed by 1.8 °C between two days on PARSEX [S1]; its shape on a trace: not found. Size, shape and period are ours (PHYSICS.md section 1). Rule 12 candidate. Unknown.
21. **No catalyst or kinetic disturbance.** PARSEX's 40 percent drop in k0 gave a 1.7 °C dip [S1, Fig. 3]. Missing. Effect: easier. Major.
22. **No fouling.** UA is constant (env.py:26). Jacket scale was a credible cause of lost cooling at T2 [S2]. A measured fall of U on a continuous unit: not found. Minor.
23. **No circulation or cooling failure.** Bayer lost circulation and climbed over 2 °C/min [S3]; T2 lost cooling and burst 10 min later [S2]. Effect: easier, and the hardest day is absent. Major.
24. **No feed concentration build-up.** Bayer's cooler held only below about 0.5 percent methomyl [S3]. The task's ±2 percent feed step cannot ignite the reactor by design (PHYSICS.md section 2). Major, same root as 10.

## Shape of the episode

25. **Fixed length, no terminations.** 150 steps, truncation only (env.py:224). Real units run continuously with interlock trips [S3]. Minor.
26. **Loads keyed to the episode, not the step** (env.py:146-151). An implementation choice for fair comparison; no plant fact. Neutral.

## Objective

27. **Quadratic cost on concentration error, scaled by the analyser scatter** (env.py:236-244, 71-73). No cost on coolant, energy or product rate. Cost limits as numbers: not found [REALITY.md, Limits]. The operator's pay is temperature held and no runaway [S5]. Effect: rewards a composition tracker, never safety margin. Major for the shape, unknown for weights.

## Parts no source supports (rule 12 candidates)

- The coolant band 295 to 302 K (env.py:30-31) as the device that prevents runaway.
- The feed load sizes, uniform shape and 10 min period (env.py:55-62).
- The target band and the random target per episode (env.py:43).
- The initial state C_a 0.80 to 0.85, T 330 K (env.py:44-45).
- `restart_steps` 240 and `failure_cost` (env.py:74-76).
- The analyser figures: 1 min dead time and 5e-4 scatter (env_jax.py:49). REALITY.md finds no online analyser on such a reactor.
- The kinetics and heat capacity (env.py:19-28): REALITY.md finds no industrial plant for this chemistry. S1's values differ: E/R 10700 K, dH -200 kJ/mol, CA,in 1.5 kmol/m3 [S1, p.4]. The record cites no figure for the task's C = 0.239 J/(g K).

## Proposed behaviour tests

Each test reads the true state through `step_env` and sets params where needed. Tolerances are fixed now. None was tuned.

**T1. Rate doubles per about 10 °C [S5, p.1].**
Quantity: the ratio of the reaction rate at T + 10 K to the rate at T, from the one-step fall in C_a with the feed and jacket terms removed (q = 0, UA = 0), at the same C_a, at three temperatures across the operating range.
Excitation: start states only, one step.
Tolerance: ratio between 1.5 and 3.0. Reason: HSE gives a rule of thumb, so a factor of 1.5 either way of 2.

**T2. A 40 percent loss of activity cools the reactor by more than 1.7 K [S1, Fig. 3 text].**
Quantity: open-loop steady-state fall in T after k0 drops by 40 percent, coolant held, from a target's steady state.
Excitation: step in k0, hold 30 residence times.
Tolerance: fall of at least 1.7 K. Reason: PARSEX's closed loop dipped 1.7 °C while acting against it, so the open-loop fall must be at least that. This is a lower bound only.

**T3. Jacket dead time [S6].**
Quantity: time from a coolant command step to the first 10 percent of the reactor temperature's response.
Excitation: a step of a quarter of the action range from steady state.
Tolerance: at least 60 s. Reason: one plant reports about 2 min at the jacket outlet; half of that allows for a faster recirculated loop. Expected to fail today: a finding, not a tuning target.

**T4. Thermowell lag [S10, Fig. 2].**
Quantity: the observed T against true T during a fast transient, fitted as a first-order lag.
Excitation: a full coolant step, noise off.
Tolerance: 63 percent time between 4 and 45 s. Reason: the bare and thick-well curves bracket it. Expected to fail today (no lag).

**T5. Analyser dead time [S11].**
Quantity: the delay between a change in true C_a and the observed C_a, noise off.
Excitation: a coolant step.
Tolerance: delay at least 2 min. Reason: the only figure is a GC at 7 to 10 min; an online analyser on a liquid CSTR is not found, so only a loose floor is safe. Flag as unknown if it fails.

**T6. Loss of cooling heats the reactor at more than 2 °C/min [S3, Fig. 11].**
Quantity: mean dT/dt over the first 3 min after UA is set to 0 at a target's steady state.
Excitation: step UA to 0.
Tolerance: at least 2 K/min. Reason: Bayer rose over 2 °C/min after circulation loss. A rate many times higher is reported, not failed, since Bayer's vessel and chemistry differ.

**T7. Loss of cooling leads to a large excursion within about 10 to 12 min [S2, PDF p.17; S3, Fig. 11].**
Quantity: time from UA = 0 to a 25 K rise.
Excitation: as T6.
Tolerance: between 3 and 48 min. Reason: Bayer took 12 min for about 25 K; a factor of 4 either way covers vessel size and chemistry. Both sources are not continuous liquid CSTRs.

**T8. Closed-loop setpoint move takes a few residence times [S1, Fig. 2].**
Quantity: open-loop 95 percent time of T after a coolant step sized for a 2 K reactor rise, in residence times.
Excitation: coolant step from steady state.
Tolerance: between 1.6 and 6.4 residence times. Reason: PARSEX's closed loop settled in 3.2; open and closed loop speeds on a well-tuned reactor are usually within a factor of 2. Weak test, kept for scale.

Morton's normal heat-up of 1 to 2 °C/min [S4] is not used: it is a batch kettle.

Eight tests proposed (T1 to T8).
