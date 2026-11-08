# Cold review: compressor_surge

Reviewer had not seen the design. Read REALITY.md first, then env.py, env_jax.py, PHYSICS.md and the shared base.
Paths below are relative to `src/target_gym/compressor_surge/`. REALITY.md source tags (C1 to C9) are kept as written there.

## Summary

- Surge is a trip on the first crossing of the line, and it costs 15 min of downtime. On real machines surge is a counted event. The machine trips after 3 surges within 10 to 20 s, or at the 3rd or 4th event [C5; C3 p.3]. One excursion is survivable there and fatal here.
- The trip fires when the flow coefficient crosses the line, before any surge happens. So the task never shows the surge signatures the sources list [C4 Table 1; C8 Fig. 5]. It also cannot score a protection response.
- The job scored is chasing a header-pressure setpoint that moves every 30 s, against consumers that move every 20 s. The usual real target is a constant suction or discharge pressure [C3 p.4]. No source supports either schedule.
- Recycle energy is about 0.2% of the score (PHYSICS.md section 5). The operator's job is "as little recycle as possible" [C3 p.1; C1]. So the trade the real controller is paid to make carries almost no weight here.
- The plant's own surge cycle runs at 0.835 s (PHYSICS.md section 2). That is below the 1 to 3 s that C1 gives for large industrial machines. The flow and pressure readings have no lag, while real dP transmitters take up to 0.2 s to reach 63% [C1].

## Observations

1. **Instruments read.** The task gives header pressure, compressor flow, delivered flow, speed, speed setpoint, valve position and the setpoint (env.py:625-652). Real anti-surge systems also read suction pressure, suction temperature, motor current and vibration [C4 Table 1; C5; C3 p.3]. **Minor** in the task as built, since surge never develops (Limits, item 2). It would become **major** once surge is simulated, because those signals are how surge is detected.
2. **No transmitter lag or dead time.** Readings arrive within the step (PHYSICS.md, Readings, "Sensor dead time and sampling: none, ours"). The flow dP transmitter should have a 63% response under 0.2 s, ideally about 100 ms [C1]. Meter location changes what is seen [C8 p.10 Fig. 7]. **Minor.** At a 0.1 s step, one step of lag is the source's figure. Leaving it out makes the task a little easier near the line.
3. **Noise figures.** env_jax.py:58-64. REALITY.md gives no noise for these instruments. **Unknown.**
4. **Delivered flow is observed.** obs[2] is the consumers' flow (env.py:646). REALITY.md: a consumer header with its own valves was not found as a plant layout. **Unknown,** and part of a **rule 12 candidate** (see the list).

## Actuators

1. **Recycle valve opening.** Full open in 2 s, applied as a steady rate from the first substep (env.py:110, 557-559). A T95 under 2.5 s for full stroke [C1] matches. Positioner and actuator lags of 150 to 300 ms [C7] are left out (PHYSICS.md D6). **Minor.** A small opening answers up to 0.3 s sooner than on a real valve.
2. **No fast trip opening.** A solenoid trip opens a real valve fully in under 1 s [C1]. The backup line steps it open in discrete increments [C3 p.2]. The task's fastest opening is 2 s. **Minor.** The backup step is a controller function, but the solenoid path is hardware the task does not offer.
3. **Recycle valve closing at 10 s.** env.py:113. REALITY.md gives no closing time. **Unknown.**
4. **Drive.** The speed follows a setpoint rate-limited at 3%/s through a 1 s lag (env.py:102-107, 554-556). REALITY.md: speed and IGV dynamics not found. **Unknown.** The drive has no torque or inertia, so speed does not fall at surge onset as C4 Table 1 says it does. **Minor** while surge is a trip.
5. **Speed as the pressure actuator.** Siemens names speed and IGV as the usual performance control [C3 p.4]. **Matches.** No IGV is modelled. REALITY.md gives no IGV dynamics. **Minor.**

## Limits

1. **Single-event trip.** The task trips when the soft minimum of Phi over a step's substeps falls below the line (env.py:586-590, 655-666). Real practice counts surges first: 3 counts within 10 to 20 s for centrifugal machines [C5], shutdown at the 3rd or 4th event [C3 p.3], and three successive surges in about 10 s [C1; C4]. The first surge brings a warning, a valve opening and a shift of the control line [C3 p.2-3]. **Major.** That makes this a different job. A real controller may graze the line and recover at a cost in machine life [C3 p.1]. Here any graze costs 1e10, so a good policy keeps a wide margin and never learns recovery. The missing facts are the surge count and the per-surge wear, both stated in C3 and C5.
2. **Trip on the line, not on surge.** The surge limit line is the measured or predicted onset of surge [C3 p.2]. The task treats crossing it as the event itself, with no time for surge to develop (PHYSICS.md D4). So flow reversal, the pressure swing and the C4 Table 1 signal directions never appear in an episode. **Major.** It also leaves out the cost that C3 p.1 names for each surge: loss of service life and efficiency.
3. **Hidden line offset of 0 to 5% to the right.** env.py:135-137; env_jax.py:190-192. REALITY.md says the line is measured or predicted, and that the control line is set at commissioning [C3 p.2]. Margins of 10 to 15% [C1], about 10% [C6] and 3 to 5% of span [C7] exist because the line is uncertain. No source gives the size or the one-sided shape of that uncertainty. **Unknown,** and a **rule 12 candidate.**
4. **No choke or pressure-ratio limit.** Limiters exist on real machines [C3 p.3], with settings not found. The task's tangent-line extension past Phi 0.78 (env.py:87-89) has no limit. **Minor.**

## Time scales

1. **Surge cycle period.** The model's deep-surge cycle, with the trip off, is 0.835 s (PHYSICS.md section 2, "Deep surge in the task's own configuration"). The Helmholtz period is 0.452 s. Large industrial machines cycle at 1 to 3 s and benchtop units at about 0.1 s [C1]. A lab rig with a 0.203 m3 plenum shows mild surge at 4.0 Hz [C9]. **Minor** today, because surge never runs in the task. It becomes **major** if surge is simulated: the task's plenum of 15 m3 (env.py:79) is industrial in size, but its period is not.
2. **Control step of 0.1 s.** env.py:150. PLC task cycles of 20 and 40 ms were tested [C7]. The loop budget is under 2 s, or under 1 s close to surge [C2]. **Minor.** A 0.1 s step is inside the budget, but it cannot resolve flow reversal, which takes under 0.02 to 0.03 s [C1; C2]. The trip reads the substeps, so this does not change trips today.
3. **Episode of 120 s.** env.py:152. REALITY.md has no figure. Ten-second surge-count windows [C1; C5] would fit inside it. **Unknown.**

## Operating mode

1. **Header pressure setpoint moves.** It has four levels in 20 to 28 kPa, 30 s each, with 10 s ramps (env.py:116-123). The usual target is constant suction or discharge pressure [C3 p.4]. No source in REALITY.md shows a pressure setpoint moving every 30 s. **Major.** The scored skill becomes slewing the drive. The real one is regulating against demand while keeping off the line. This is a **rule 12 candidate.**
2. **One agent runs both loops.** Real plants run a dedicated anti-surge PI on a control line, with rate action, a backup line and setpoint shift [C3 p.2], beside a separate performance controller [C3 p.4]. The task gives the agent both actuators and no control line. That is a fair benchmark choice. But the task should say it is asking for something real plants split. **Minor.**
3. **Normal day.** On a normal day the valve is shut or nearly so [C3 p.2]. In the task 66% of blocks force recycle with no margin (PHYSICS.md section 4). **Major.** The schedule makes recycle the normal state, where the source says it is the exception.

## Disturbances

1. **Consumer demand schedule.** Six levels of opening in 0.35 to 0.95, 20 s each, with 5 s ramps, plus an OU deviation of sd 0.02 with a 5 s correlation time (env.py:124-134, 596-598). REALITY.md has the slow and the fast approach as bad days [C3 p.2]. It gives no size, rate or frequency for demand moves, and does not describe the consumer valve layout. **Unknown** for the shape, and a **rule 12 candidate** for the values.
2. **No emergency shutdown.** On an ESD the driver is cut and the recycle valve must open at once [C6]. This is the hardest surge case the sources name. The task has no driver trip. **Minor,** since a protection system handles it.
3. **No suction-side disturbance.** Suction pressure and temperature are fixed at ISA (env.py:66-69). REALITY.md names no suction disturbance. **Unknown.**

## Shape of the episode

1. **Reset.** The plant starts at 100% speed with the recycle at 0.35, off the first setpoint, which ramps from the reset pressure (env_jax.py:152-197). REALITY.md gives no startup. **Unknown.** The opening transient carries most of the tracking cost (PHYSICS.md section 5), so it is part of what is scored.
2. **Restart.** After a trip the plant restarts on the next step from a fresh draw, and the step is charged 9000 steps (15 min) of failure cost (env.py:167-169; base.py failure_kernel). Rundown after a trip is not found in REALITY.md. The 15 min is labelled ours. **Unknown,** and a **rule 12 candidate.**

## Objective

1. **What is scored.** It scores the squared header-pressure error over 0.0275 kPa, plus recycle power over 62.27 kW, plus valve travel above 0.25% per step, plus the trip charge (env.py:669-697). The operator meets the header demand, keeps right of the control line, uses as little recycle as possible, and has no counted surge [C3 p.1-2; C5]. **Major.** Two things differ. First, recycle energy is about 0.2% of the cost of the best policies (PHYSICS.md section 5). A policy that recycles twice as much loses almost nothing, so margin and energy are not traded the way the plant trades them [C1; C3 p.1]. Second, surge is priced only as a trip, never as the per-event wear C3 p.1 names.
2. **The margin is not scored.** The real target is the control line, not the surge line [C3 p.2]. The task scores neither a margin nor time spent left of a control line. **Minor,** since this is a stated design choice (PHYSICS.md section 5). But it rests on the single-event trip in Limits, item 1.

## Parts no source in REALITY.md supports (rule 12 candidates)

- The consumer header with its own valve, the 10 kPa static head and the check-valve leak (env.py:92-99).
- The moving setpoint schedule and its 10 s ramps (env.py:116-123).
- The demand schedule and its OU deviation (env.py:124-134).
- The hidden, one-sided surge-line offset of up to 5% (env.py:135-137).
- The 15 min restart charge, and the instant restart (env.py:167-169).
- Drive rate 3%/s and lag 1 s (env.py:104-107). No dynamics were found.
- Recycle valve closing in 10 s (env.py:111-113). No closing time was found.
- All instrument noise figures (env_jax.py:58-64).

## Proposed behaviour tests

Tests 1 to 4 step `compute_next_state` directly, so the trip kernel does not restart the plant. They read the substeps through `delta_t` set to 0.005 s. Each tolerance is fixed now, before any run.

1. **Surge cycle period, industrial machine** [C1]. Start at rated speed with the recycle shut and the consumer opening at its 0.2 floor. Let deep surge develop for 30 s. Read the period of m_c over the last 10 cycles. Pass if it is 1 to 3 s. Reason: that is C1's printed range for large industrial installations. The task's 15 m3 plenum puts it in that class. PHYSICS.md section 2 reports 0.835 s, so this test is expected to fail. It would be a finding, not a reason to widen the band.
2. **Flow reversal time** [C1; C2]. In the same run, read the time from m_c at +50% of its cycle maximum to -50% of its cycle minimum. Pass if it is at most 0.06 s. Reason: the sources give under 0.02 s and 0.03 s. "Large positive to large negative" is read loosely, so a factor of 2 on the larger figure is allowed.
3. **Sustained surge at closed throttle oscillates about zero flow** [C8 p.8 Fig. 5]. Same set-up. Pass if m_c goes negative in every cycle, and its cycle mean is within plus or minus 25% of its half-amplitude from zero. Reason: the figure shows flow swinging between forward and slightly reverse values. A mean near zero relative to the swing is what it shows, and 25% is a plot reading.
4. **Surge frequency falls as discharge volume rises** [C8 p.4]. Repeat test 1 with V_p at 0.5x and 2x. Pass if the period rises strictly with V_p. Reason: the source states the direction only, so the test asks only for the direction.
5. **Signal directions at surge onset** [C4 Table 1]. From an equilibrium 2% right of the line, drop the consumer opening to 0.2 with the recycle shut. Run with the trip off. Over the first half cycle after Phi crosses the line, read the sign of change in flow, discharge pressure, suction pressure, suction temperature and speed. Pass if flow and discharge pressure fall, suction pressure and suction temperature rise, and speed falls. Reason: the table gives signs only. The task has no suction states and its speed ignores load, so three of the five checks are expected to fail by construction. That names the missing physics: a suction volume, an energy balance, and drive inertia.
6. **Recycle valve stroke** [C1; C7]. Command the valve from 0 to 1 at a steady state. Pass if x reaches 95% within 2.5 s [C1], and if x has moved less than 5% of its stroke by 0.15 s and more than 5% by 0.30 s [C7]. Reason: the first figure is C1's limit. The second is C7's 150 to 300 ms positioner and actuator lag, read as the time to visible motion. With no valve dead time (PHYSICS.md D6), the second part is expected to fail.
7. **Surge is counted before it trips** [C5; C3 p.3; C1]. Drive the plant into one short surge excursion lasting under one cycle, then bring it back right of the line. Pass if the plant does not trip. Then force three surge cycles within 10 s. Pass if the plant trips no later than the end of the third cycle. Reason: 3 counts within 10 to 20 s is C5's centrifugal trip, and C3 p.3 shuts down at the 3rd or 4th event. The test takes the strictest of the sourced windows, 10 s [C1]. The current single-event trip is expected to fail the first half.

Seven tests are proposed. No test is proposed for the control-line margin (C1, C6, C7). That figure belongs to a controller, not to the plant.
