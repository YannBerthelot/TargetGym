# Drum boiler: cold review against REALITY.md

Reviewer had not seen the task's design. Compared only against `REALITY.md` and
its sources [S1] to [S19]. File references are to `env.py` unless marked
`env_jax.py`. No numbers are proposed; each item names the missing physics or
operating fact and its source.

## Mismatches

Ranked by how much each changes what an agent learns or what the score says.

1. **The task lives at one load point; low and medium load are absent.**
   Task: every episode starts at the nominal balance (`env_jax.py:123`,
   `steady_state` at `q_steam_nominal`, `env.py:586-632`) and the steam demand
   reverts to nominal with 8 kg/s spread (`env.py:152`, `433-438`), so the plant
   stays near 100 % load. Plant: swell "is larger at medium load" [S1, printed
   p. 26]; swell is larger at low load [S1, p. 16 (cited here as p. 20, corrected
   2026-11-12); S2, sec. 5.2]; "most trips
   occur due to high-level trip events" at low load [S11]; below 20 to 30 % of
   flow the plant runs single-element because flows are unmeasurable [S5; S12];
   turndown can be 4:1 [S16] (REALITY sec. 2, 4, 5). Effect: the named property,
   shrink and swell, is exercised only in the regime where the sources say it is
   smallest. The hard part of real drum level control (swell at low load, the
   single to three-element switch) is never seen, so an agent learns a
   high-load problem and the score cannot reward low-load competence.

2. **The disturbance is a stationary random walk, not load changes.**
   Task: a hidden Ornstein-Uhlenbeck turbine valve, 1/theta about 200 s, 8 kg/s
   stationary std (`env.py:66-68`, `152`, `428-438`), always moving, never
   announced. Plant: utility units change load at 1 to 2 %/min, up to 5 %/min
   [S10]; coal units 1.5 to 6 %/min [S17]; industrial ramps up to 20 to 25
   %/min [S15]; at Abbott setpoints change "several times during every 24 h"
   and unmeasured user demand is the main disturbance [S3, p. 62]; under steady
   operation level control "is usually not a problem" [S2, sec. 1]
   (REALITY sec. 3, 4, 5). In coordinated or turbine-following mode the MW
   demand that moves the turbine valve is a known signal to the unit master
   [S10]. Effect: the agent faces continual small random moves instead of
   discrete ramps or steps of known shape; swell arrives as many small
   innovations rather than the 10 to 25 % steps of S1 Figs. 12 to 16. Whether
   swell at a realistic size is present in an episode depends on the OU draw,
   not on a scenario. A demand feedforward that a real coordinated controller
   has is withheld.

3. **Trips are symmetric, instantaneous and have no timing.**
   Task: `|level| >= 0.25 m` trips on the step it happens (`env.py:167`,
   `507-515`). Plant: low-low master fuel trip at -330 mm, and low or high
   level trips "for more than ten seconds" (timed) [S9]; a 95 MW unit's low
   trip at 7.7 in [S10]; high-high protects the turbine and trips the turbine
   or unit, low protects the tubes and trips the heat source [S12]; on the
   marine table high is alarm only and high-high stops the feed pump, not the
   burner [S13]; UK shell boilers have a first low cut-out that re-fires and a
   second that locks out [S8] (REALITY sec. 3). Effect: a brief swell spike is
   punished as hard as a sustained loss of water, which pushes an agent toward
   suppressing transient swell rather than riding it, the opposite of
   documented practice (raise the level before a test [S10], lower NWL ahead
   of startup [S19]). High and low having different consequences and margins is
   lost. No source in REALITY gives a high-high value.

4. **Who holds pressure, and which pressure.**
   Task: the agent tracks a random drum pressure target 82 to 88 bar
   (`env.py:161`, `env_jax.py:125-129`) with firing, at a hidden turbine valve
   whose steam follows drum pressure (`env.py:330-336`). This is boiler-following.
   Plant: four modes exist, base, boiler-following, turbine-following and
   coordinated [S10]; most plants control throttle valve or first stage pressure
   [S2, sec. 2.3]; Abbott controls a common header fed by several boilers [S3,
   p. 58]; Lu et al. use boiler-follow with sliding pressure [S2, sec. 3.2]
   (REALITY sec. 1, 2). Effect: boiler-following is a sourced mode, so the
   pairing is defensible, but the controlled variable is drum pressure, not
   throttle or header pressure, and the target is a random fixed setpoint
   rather than a sliding or load-dependent one. Half the score is pressure
   tracking at 0.05 bar, a goal REALITY does not describe for drum units.

5. **The objective is millimetre tracking; the plant's job is staying between
   trips.** Task: quadratic level cost on a 2.67 mm scale and pressure on
   0.05 bar, plus fuel as a running cost (`env.py:89-108`, `518-543`). Plant:
   "keep drum water between the low trip ... and the high trip" (REALITY
   sec. 6); costs are carryover, dry-out and tube damage [S4, p. 1; S15] and "a
   costly plant trip" [S2, sec. 1]. Effect: at the floors a 48 mm level RMS
   costs about 320 per step against about 36 for a 0.3 bar pressure error, so
   the score ranks controllers by level precision far below anything an
   operator distinguishes, while REALITY ranks them by trip margin. The relative
   weight of level against pressure has no source.

6. **Level reading: clean, white, undelayed, already compensated.**
   Task: true level plus 0.5 mm white noise, no delay or damping
   (`env_jax.py:52-58`; PHYSICS.md "Readings"). Plant: DP transmitter with a
   reference leg and pressure compensation whose constant "is determined
   empirically" [S4, p. 3; S16]; noise from "boiling effect on water surface,
   pressure waves" and line leaks, fitted as first-order coloured noise [S3,
   p. 63]; the Abbott model's level output carries a delay tau6 = 10 [S3, p. 62
   to 63]; filtering is "generally counter-productive" [S16] (REALITY sec. 1).
   Effect: the inverse response is easier to see and to differentiate than on
   a real drum; compensation error under pressure swings (which itself looks
   like level) is absent. The task's own note says the figure excludes surface
   turbulence, which S3 names as the main noise source.

7. **No feedwater flow measurement and no flow loop; feed position equals
   flow.** Task: the action is feedwater flow through a 1 s lag and rate limit;
   `feed_pct` is that flow, exact (`env.py:424-426`, `500`, `157-158`). Plant:
   three-element control reads drum level, steam flow and feedwater flow [S5;
   S12]; the level controller sets a feedwater flow controller's setpoint
   (cascade) on a valve [S4, p. 3; S5] (REALITY sec. 1, 2). Effect: the inner
   flow loop is assumed perfect and noiseless, so feedwater supply disturbances
   (pump pressure, valve characteristic) cannot occur. Drum level is easier and
   the third element of three-element control is handed over free.

8. **No bad-day events.** Task: no feed pump trip, runback, fan trip or
   feedwater limit; actions always reach the plant (`env.py:411-469`). Plant:
   ID fan runback "just over two minutes"; feed pump runback from 75 % load drops
   level about 6.5 in against a 7.7 in trip [S10]; standby pumps take over 30 s
   [S11]; all feed pumps off is a boiler trip [S9] (REALITY sec. 4, 5).
   Effect: the scenarios in which real units come closest to a level trip are
   missing, so robustness to them is neither learned nor scored.

9. **Trip consequence and restart.** Task: a trip costs `restart_steps = 7200`
   (4 h, marked provisional) times `failure_cost`, and the plant reappears at
   the balanced nominal state on the same clock (`env.py:96-100`;
   `env_jax.py:89-94`). Plant: REALITY gives no restart time; restart goes
   through startup on single-element control with NWL lowered for swell [S5;
   S19]. Effect: the trip price, which dominates any tripping policy's score,
   rests on an unsourced number, and an instant warm restart is not what a
   real trip leads to.

10. **Pressure trips at 65 and 105 bar.** Task: pressure outside 65 to 105 bar
    ends or restarts the plant (`env.py:168`, `171`, `510-512`), and the 65 bar
    figure sets `failure_cost` (`env.py:96-98`). Plant: high steam pressure stops
    the burner on the marine boiler [S13]; ASME safety valve limits "not
    confirmed" (REALITY sec. 3). No low-pressure trip is sourced. Effect: the
    lower bound is a model-validity limit (property fits, PHYSICS.md sec. 1)
    presented as a plant trip, and it scales the reward.

11. **Episode shape.** Task: 400 steps of 2 s (13 min), start balanced at
    nominal with level uniform in +-5 cm and a random pressure target
    (`env.py:73-74`, `161-162`; `env_jax.py:117-161`). Plant: S1's tests ran
    open loop over hundreds of seconds per step; Abbott's PRBS ran 5000 s
    [S3]; a normal day holds setpoints for hours [S3]. Effect: 13 min holds
    about four correlation times of the disturbance and at most a couple of
    swell-and-mass-trend cycles (S1 Fig. 16 shows the trend over 250 to 300 s);
    slow pressure behaviour (stored energy totals 690 s, S1 Table 1) is only
    partly visible.

12. **Firing dynamics.** Task: firing passes a 20 s lag and a 0.007 of span per
    second slew (`env.py:157-158`). Plant: REALITY has no firing lag or rate
    figure; S1's fuel steps show a "rapid swell" overshoot [S1, p. 23, Fig. 10],
    which needs heat release fast enough to raise riser voids before pressure
    rises. Effect: the 20 s lag may blur or remove the fuel-side rapid swell,
    one of the three inverse responses S1 documents.

13. **Control interval 2 s.** Task: `delta_t = 2.0` (`env.py:73`). Plant:
    signals sampled at 0.1 Hz in S1; the swell jump shows within one or two
    10 s samples; circulation time constant about 1 s [S1, p. 10, p. 22].
    Effect: small. A 2 s step resolves the swell; no source gives a controller
    scan time.

## Unsupported parts

Parts no source in REALITY.md ties to the plant (rule: a part nobody can tie
to a source goes, or gets a source):

- OU steam demand: its form, 8 kg/s std and 200 s time scale (`env.py:68`, `152`).
- Random drum pressure target 82 to 88 bar (`env.py:161`), and the initial
  level spread +-5 cm (`env.py:162`).
- Symmetric +-0.25 m instantaneous level trip (`env.py:167`): S9 and S10 give
  low trips only, one of them timed; no high-high value is sourced.
- Pressure trips at 65 and 105 bar (`env.py:168`, `171`).
- `restart_steps` 4 h and the instant restart (`env.py:99-100`).
- Reward floors 2.67 mm and 0.05 bar, the level-to-pressure weighting, and the
  fuel running cost (`env.py:89-108`): REALITY names no tracking precision and
  no fuel cost.
- Firing lag 20 s (PHYSICS.md marks it "Ours"); actuator rates 0.007 and
  0.05 of span per second and feed lag 1 s are cited in PHYSICS.md to Bell and
  Astrom 1987, which is not a REALITY.md source.
- Sensor sigmas 0.5 mm, 0.022 bar, 0.047 kg/s (`env_jax.py:54-58`): from
  datasheets outside REALITY; REALITY's only noise source [S3] describes
  coloured noise dominated by surface boiling.
- Tuned internals `f_carry`, `f_cd`, `tau_sr`, `T_d` (`env.py:133-138`): REALITY
  gives behaviour (Figs. 10 to 16) they could be tested against, not values.
- 2 s control interval (`env.py:73`).

Items REALITY.md marks "not found" or unconfirmed that the task relies on:

- Transmitter damping times: not found; the task assumes none and no delay,
  although S3's level model carries tau6 = 10.
- Alarm distances from NWL and EN limiter values: not found; the task places
  its only level limit at +-0.25 m.
- ASME safety valve set and accumulation: not confirmed; the task uses a
  105 bar trip.
- Restart time after a trip: absent from REALITY; the task prices it at 4 h.

## Proposed behaviour tests

All open-loop tests set `q_steam_noise_std = 0` and `sensor_noise_scale = 0`,
start from `steady_state` at the stated load, hold fuel and feed at that
balance (constant action) and move only the named input. Tolerances on values
read from S1 figures allow for reading error (about 0.01 m and 0.2 MPa per
grid division) and for the task's plant not being P16-G16 at every detail; they
are fixed here before any run.

1. **Swell on a steam valve step at medium load.** Source: S1 Fig. 16 and
   Fig. 12, printed pp. 25 to 27: a step of steam 52 to 67 kg/s gives a level
   jump of 0.05 to 0.07 m within one or two 10 s samples, then a mass trend
   the other way of about 0.1 m over 250 to 300 s. Measured: true level.
   Experiment: balance at 52 kg/s, step the valve so steam reaches 67 kg/s at
   the starting pressure, constant fuel and feed, 300 s. Tolerance: peak rise
   in 0.035 to 0.09 m (figure band widened by one grid reading each side);
   peak within 30 s (three samples, one more than S1 shows); level below its
   starting value by 300 s (sign of the mass trend, which the figure shows
   unambiguously).

2. **Pressure ramp on the same step.** Source: S1 Fig. 12: drum pressure moves
   8.2 to 9.1 MPa in roughly linear ramps of about 0.6 MPa per 250 s. Measured:
   drum pressure. Experiment: as test 1. Tolerance: drop over 250 s in 3 to 9
   bar (factor 1.5 either side, since the figure's ramps vary between steps);
   drop over the second 125 s at least 0.7 of the first 125 s, because the
   source calls the ramps roughly linear and an exponential with time constant
   near 250 s would give 0.6.

3. **Swell larger at medium load than at high load.** Source: S1 printed
   p. 26, "the shrink and swell effect is larger at medium load", and Figs. 15
   and 16. Measured: peak level rise per kg/s of steam step. Experiment: test 1
   at 52 kg/s and again at 100 kg/s with a 12 kg/s step (Fig. 15's range).
   Tolerance: medium-load value strictly greater; ordinal, as the source gives
   no ratio.

4. **Swell grows toward low load.** Source: S1 printed p. 20; S2 sec. 5.2;
   S11 (most trips are high-level trips at low load). Measured: peak level
   rise for a steam step of 10 % of the load. Experiment: test 1 at 50, 75 and
   100 % of nominal steam. Tolerance: monotone decreasing with load; ordinal.
   Restated 2026-11-12 with the owner's approval: S1's statement is on printed
   p. 16 and compares equal changes, so the step is 10 % of nominal at every
   load.

5. **Shrink on a load decrease is smaller and shorter than swell.** Source:
   S14: level drop on a load decrease "is typically much smaller" and "of
   shorter duration". Measured: peak level excursion and its duration above
   half peak. Experiment: equal and opposite valve steps from the same
   nominal balance. Tolerance: shrink peak magnitude below swell peak and
   shrink duration below swell duration; ordinal, as no ratio is given.

6. **Feedwater step: inverse response and level band.** Source: S1 Fig. 11,
   printed p. 24: feedwater steps of about 52 to 68 kg/s at medium load move
   level within about +-0.12 m; S1 p. 20 places a right half plane zero in the
   feed-to-level path. Measured: true level. Experiment: at the 52 kg/s
   balance, a feedwater sequence switching between 52 and 68 kg/s with
   Fig. 11's dwell (read from the figure, about 250 s), fixed valve and fuel.
   Tolerance: first 20 s of each up-step moves level down or not at all (sign
   of the zero); envelope within +-0.18 m (the figure band plus 50 %, as the
   task's feed span differs from P16-G16's).

7. **Rapid swell on a fuel step.** Source: S1 printed p. 23, Fig. 10: fuel
   steps show an overshoot the authors call "rapid swell". Measured: true
   level. Experiment: at nominal balance, a fuel step of 10 % of nominal
   firing, valve and feed fixed, 300 s. Tolerance: level shows a local maximum
   above its 300 s value before the mass trend takes over; presence only,
   since Fig. 10's magnitude is not recorded in REALITY.md. A failure here
   points at the unsourced 20 s firing lag (mismatch 12).

8. **A ramp halves level deviation against a step, closed loop.** Source: S2
   sec. 5.2 (simulation, 500 MW): a 10 %/min load ramp gave "almost halved"
   level deviation versus a step. Measured: peak absolute level deviation.
   Experiment: the task's three-element expert as a fixed recipe (its stored
   gains), a 20 % load decrease applied once as a valve step and once as a
   10 %/min ramp, noise off. Tolerance: ramp peak over step peak in 0.35 to
   0.7, centred on "almost halved" and widened because S2's plant is larger
   and its controller differs.
