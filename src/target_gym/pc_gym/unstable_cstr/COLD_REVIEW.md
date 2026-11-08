# Cold review: `unstable_cstr`

Summary of the mismatches that matter most:

- No production reactor run at an open-loop unstable point was found. The operating mode rests on lab and pilot units only [U2; S1]. Six setpoint moves an hour across that unstable region have no source at all.
- The jacket lag is 6 s and has no dead time. One plant reports about 2 min of jacket dead time [S6], and McMillan says the gain window closes as lags approach the positive-feedback time constant [U1]. At real lags this task would not be controllable, so the 6 s choice decides what kind of task it is.
- The jacket has no energy balance and no cooling ceiling. Full cooling always pulls the jacket to 290 K, whatever the heat load. Real full cooling did not stop Bayer's or Morton's runaway [S3; S4].
- Concentration is read instantly. A GC on a reactor has 7 to 10 min dead time [S11]. The operator's documented variable is temperature [U1; S5].
- The trip restarts the plant at once from a warm reset. A real interlock cuts the feed [S3], and the hazards around it (pressure, relief, secondary reaction, accumulation) are absent.
- Runaway after cooling loss here takes seconds. The documented ones took 10 to 12 min [S2; S3].

Sources are tagged as in `REALITY.md`; the shared plant facts are in `../cstr/REALITY.md`. File references are to this folder.

## Observations

1. **Concentration is read at once, every 3 s** (env_jax.py:55; PHYSICS.md D3). GC dead time 7 to 10 min [S11]; lab sample once a day [S3]. An online analyser on a jacketed liquid CSTR: not found. Effect: easier, and it makes the scored variable directly observable. Unknown, leaning major.
2. **Reactor T through a 10 s lag** (env_jax.py:56). Within WIKA's 4 to 45 s [S10, Fig. 2]. Consistent. Note the thermowell matters more here [U1].
3. **Jacket temperature read with no lag and no dead time** (env_jax.py:57; PHYSICS.md Readings). The jacket outlet adds about 2 min of dead time in one plant [S6]. Effect: easier, since the slave measurement is fresh. Major.
4. **No pressure, no circulation flow, no coolant flow.** The task has no pressure state (PHYSICS.md section 1 omits it). At Bayer the first alarm acted on was high pressure, and circulation flow was logged [S3, Fig. 11, PDF p.48]. Effect: a different job. The operator watches pressure and flow to catch a failure; here there is nothing to catch. Major.
5. **Sensor range is unlimited.** Morton's chart saturated at 150 °C [S4]. The trip stops the reactor at 365 K first, so no effect. Minor.

## Actuators

6. **The action is a coolant temperature, reached through a valve and a 6 s jacket lag** (env.py:72-80, 284). Plants move cooling-water flow through a valve [U2; S1, p.4], usually as a cascade on jacket temperature [S6, S7]. Read as a slave setpoint the lag is plausible only if the slave is perfect and the jacket has no transport delay. PHYSICS.md section 6 itself says a real jacket of this size would lag about 21 s at 60 L/min. Effect: much easier than the plant. Major.
7. **No cooling ceiling from heat duty.** `T_j` follows the command whatever heat it removes (env.py:284; PHYSICS.md D2). On PARSEX the ceiling is the water flow range [S1, Table 1]. Operators at Bayer and Morton went to full cooling and it did not stop the runaway [S3, PDF p.48; S4, PDF p.25]. Effect: easier. In the task full cooling from inside the point of no return always wins (PHYSICS.md section 4). Major.
8. **Valve lag 2 s and slew 8 percent per second** (env.py:79-80). REALITY.md has a proportional valve but no figures [U2]. Unknown.
9. **No second handle.** PARSEX also moves feed concentration [S1, p.4]. HSE lists inhibitor, quench and dump [S5]. Effect: a narrower job than the plant's. Minor.
10. **The controller is never in manual.** McMillan says it must stay in automatic [U1]. The task is always closed by the agent. Consistent.

## Limits

11. **High trip at 365 K, then an instant warm restart** (env.py:83, 350-360; env_jax.py:136-181). Bayer's interlocks cut the feed valve on temperature and pressure [S3, PDF p.38]. Restart after a trip, as a time: not found. The 365 K value is ours, 12 K above the hottest target (PHYSICS.md section 3). Real trip points sit near a secondary reaction (T2 199 °C, Morton 195 °C onset) [S2, S4] or a safe limit (Bayer 155 °C) [S3]. Effect: a different job. The task has no secondary reaction, so the trip is an arbitrary line. Major for the restart, unknown for the value.
12. **The trip cost is an hour of restart** (env.py:130-133). Taken from cstr's own provisional figure, not a source. Rule 12 candidate.
13. **No pressure relief or emergency cooling layer.** HSE's protection layers [S5, p.4] and T2's missing backup water [S2] show what the plant has around the trip. Minor for the benchmark, since the trip ends the excursion.
14. **Extinction is free apart from tracking cost** (env.py:351-353). At Bayer, running cold with feed on built up unreacted material that then ran away [S3, PDF p.46]. The task does re-ignite from the low state under heating (PHYSICS.md section 4), so the hazard is partly there. Minor.

## Time scales

15. **Residence time 1 min, step 3 s** (env.py:53-54, 112). Sources: 34 min [S1], 15 to 120 min [S9 in the cstr record]. Effect: every lag fixed in real seconds is relatively small here. Major.
16. **Open-loop growth 19 to 43 s** (PHYSICS.md section 2). A measured positive-feedback time constant: not found. Unknown.
17. **Runaway in seconds.** Full heating trips in 15 to 60 s from every reset corner (PHYSICS.md section 4). Cooling loss to rupture took 10 min at T2 and 12 min at Bayer [S2, S3]; Bayer rose 6 K in 3 min. Both are batch or semi-continuous, and the task's heat release comes from a 209 K adiabatic rise (PHYSICS.md D1). Effect: harder in reaction speed, easier in that there is no time for anything but feedback. Major.
18. **Episode 60 min** (env.py:113). The lab PID held an unstable point for at most about 18 min [U2]. Consistent with a pilot run length; nothing longer was found.

## Operating mode

19. **Held at an unstable middle state.** Only lab and pilot units are documented there [U2; S1, p.3]. A production reactor run there: not found. Effect: the task is a lab experiment, not a plant. Major, and the owner should state it.
20. **Six levels per hour, ramped at 1/30 mol/L per min** (env.py:102-109, 228-240). A normal day is holding one point with small swings [U2; S1]. Setpoint moves on an unstable reactor: not found. Rule 12 candidate. Major.
21. **Concentration is tracked, not temperature** (env.py:374-380). The operator holds temperature and watches the rate of rise [U1; S2, PDF p.30]. Major.
22. **Warm start at each reset** (env_jax.py:136-181). Start-up errors were the cause at Bayer and Morton [S3, PDF p.46; S4, PDF p.35]. The task never starts up. Effect: easier, and the documented hard phase is absent. Major.

## Disturbances

23. **Feed temperature drift, OU, sd 2 K, correlation 10 min** (env.py:89-94, 319-321). PARSEX's feed differed by 1.8 °C between two days [S1, Tables 3, 4]. The size is plausible for day to day; the 10 min correlation is not supported. Its effect on a trace: not found. Rule 12 candidate. Unknown.
24. **No feed concentration disturbance** (PHYSICS.md D4). Bayer's runaway came from feed material building up [S3]. Major.
25. **No circulation, cooling or valve failure.** Bayer's split range closed both valves at changeover [S3, footnote 29]; T2 lost cooling [S2]; fouling at T2 [S2]. Effect: easier, and the bad day of the record is absent. Major.
26. **No activity disturbance.** PARSEX tested a 40 percent k0 drop [S1, Fig. 3]. Minor here.

## Shape of the episode

27. **Continuing episode with instant restart and a fresh schedule** (env.py:196-198; env_jax.py:107-110). Real units restart after an investigation or purge; restart time: not found. Minor for learning, since the cost is fixed.
28. **The reset clips the drift so every start is recoverable** (env.py:94). A design guarantee with no plant counterpart. Neutral.

## Objective

29. **Quadratic concentration error, no cost on cooling, a large trip cost** (env.py:363-382). The operator is paid for no runaway and product held [S5; U1]. Cost limits as numbers: not found. Unknown for weights, major for the shape: margin to the point of no return is never scored before a trip.

## Parts no source supports (rule 12 candidates)

- `tau_j` 6 s, with its owner exception (env.py:76; PHYSICS.md section 6).
- The target schedule, ramp and band (env.py:99-109).
- The drift's correlation time and the reset clip (env.py:89-94).
- `T_trip` 365 K and the instant restart, and `restart_steps` (env.py:83, 133).
- `T_c_max` 310 K (env.py:73), ours.
- The kinetics and heat capacity, as for cstr: no plant for this chemistry was found, and the record's propylene glycol pilot is not found.

## Proposed behaviour tests

Each test reads the true state through `step_env` and sets params where needed. Tolerances are fixed now. None was tuned.

**U-T1. A gain floor exists [U1].**
Quantity: whether T stays within ±2 K of T* for 10 min under a P-only loop from T to the coolant command, at a gain sweep from 0 up.
Excitation: start 0.2 K above T* at each target, drift on.
Tolerance: the lowest gains trip or leave the band, and some higher gain holds. Reason: McMillan's statement is qualitative; only the existence of a floor is tested.

**U-T2. The gain window closes as the lags grow [U1].**
Quantity: width of the holding gain range from U-T1.
Excitation: as U-T1, repeated with the thermowell lag and `tau_j` each doubled.
Tolerance: the width with longer lags is strictly smaller. Reason: McMillan's direction only.

**U-T3. A simple PID can hold the point for about 18 min within +2 and -3 °C [U2].**
Quantity: T excursion around T* over 18 min, under a P-only loop in the window of U-T1.
Excitation: the task's own drift, at each target.
Tolerance: the swing stays inside +2 and -3 K for at least one gain. Reason: the lab result shows a PID can do it; the task should not be harder than a lab rig.

**U-T4. The reactor leaves the point when the coolant is held [U1].**
Quantity: time to leave ±2 K of T* with the command held at Tc*.
Excitation: start 0.1 K off T* each side.
Tolerance: it leaves within 10 min. Reason: "unsafe to have the controller in manual". No measured divergence time was found, so only that it leaves is tested.

**U-T5. Loss of circulation heats the reactor at more than 2 °C/min [S3, Fig. 11].**
Quantity: mean dT/dt over the first 3 min, or until the trip, after UA is set to 0 at a target's steady state.
Excitation: step UA to 0.
Tolerance: at least 2 K/min. Reason: Bayer's figure, a lower bound.

**U-T6. Loss of cooling reaches its limit in minutes, not seconds [S2, PDF p.17; S3, Fig. 11].**
Quantity: time from UA = 0 to T_trip.
Excitation: as U-T5.
Tolerance: between 2.5 and 48 min. Reason: T2 took 10 min and Bayer 12 min; a factor of 4 either way covers vessel and chemistry, and both are not continuous liquid CSTRs. Expected to fail today (PHYSICS.md section 4 gives seconds): a finding.

**U-T7. The jacket heats up when the reactor runs away at a held command [S3; S4].**
Quantity: change in jacket outlet temperature during a 5 K reactor rise with the command held.
Excitation: start 2 K above T* at 0.45 mol/L.
Tolerance: at least 0.1 K. Reason: a real jacket absorbs the heat it removes, which is why full cooling failed at Bayer and Morton. Already a strict xfail in the repo (D2).

**U-T8. Jacket outlet dead time [S6].**
Quantity: time from a command step to the first 10 percent of the jacket reading's response.
Excitation: a step of a tenth of the range from steady state, with the reactor on a holding P loop.
Tolerance: at least 60 s. Reason: one plant reports about 2 min; half allows for a recirculated loop. Expected to fail today.

**U-T9. Analyser dead time [S11].**
Quantity: delay between true and observed C_a, noise off.
Excitation: a coolant step under a holding loop.
Tolerance: at least 2 min. Reason: only a GC figure exists; a loose floor. Already an xfail in the repo (D3) with 1 min.

**U-T10. Thermowell lag in range [S10, Fig. 2].**
Quantity: fitted first-order lag of observed T on true T, noise off.
Excitation: a fast transient under full heating from rest.
Tolerance: 63 percent time between 4 and 45 s. Reason: the WIKA curves bracket it.

Ten tests proposed (U-T1 to U-T10).
