# Glass furnace: physics model, provenance and validation

Reference plant: a **regenerative side-port fired float-glass furnace**, about 500 t/day,
natural gas, its melter, waist and working end in one long section. Registered as
`glass_furnace-v5` (attempt 3 of the rebuild, charter Amendments 1 to 3; v5 makes fining
a time-temperature integral).

This document is the task's contract. The plant record it rests on is `REALITY.md`
(sources S1 to S44); the cold review is `COLD_REVIEW.md`; the design, the measurements and
every decision are in `docs/proposals/glass-furnace-task.md`, from "Attempt 3" on. Method:
`docs/PHYSICS_METHODOLOGY.md`. The plants of v1 to v3 are retired (charter D2); their
contract is in this file's git history.

## 1. Model

`physical.py`, from the equations: every term is a conservation law or a sourced relation.
Temperatures are in K unless a name says otherwise.

- **Tank.** A long section on 20 columns and 8 depth layers, 1.2 m deep: the fired melter
  (21.5 m, six ports, two columns each), the unfired zone past the last port (20.5 m at
  the melter's 11 m width), the waist (7.5 m by 3.5 m) and the working end (20 m by
  7.9 m), where the refiner's cooling air takes heat from the glass.
- **Melt.** Energy in depth and length, with the long-tank flow: forward at the surface,
  back at depth, driven by buoyancy, and the pull. Advection is flux limited (Sweby, van
  Leer's limiter), conduction includes the Rosseland term. The step, 120 s, keeps the
  surface current's Courant number under 1.
- **Combustion space.** Zone radiation between the flame gas, the crown's hot face and
  the load, bay by bay; the crown is 8 brick cells by conduction. Each port takes its
  share of the firing (1, 1, 1, 1, 0.7, 0.4).
- **Regenerators.** Four checker nodes per chamber, averaged over a reversal: each node
  takes half its flue duty and half its air duty, and the 40 s firing gap enters as its
  mean (`mean_firing`).
- **Batch and foam.** The blanket, its cullet ratio and the heat it has taken from below,
  per melter column; it melts at 2.2 MJ/kg to 1450 K. Foam grows with the fining gas and
  decays with its lifetime.
- **Fining.** Each cell carries the share of its sulfate fining its glass has done,
  carried with the flow. Fining is a time-temperature integral [S45], not the peak: in
  the 1420 to 1500 C band [S31] a cell's unfined share falls at the band's share over
  `fining_time`, 1 h at the band's top (`fining_after`). Glass that cools keeps what it
  fined, since the gas it released is spent; reheating fined glass, which raises reboil
  bubbles [S45], is not modelled.

| Quantity | Value | Source |
|---|---|---|
| Geometry | 21.5 + 20.5 m, 11 m wide, six ports | W2000b |
| Waist and working end | 7.5 by 3.5 m, 20 by 7.9 m | A 1300 t/d furnace's ratios to its melter [S18], scaled (ours) |
| Glass depth | 1.2 m | Materials 17 (2024) 4989 |
| Gas absorption | 0.08 /m | Hoogendoorn, side-fired |
| Melt conductivity | 1.31 + 5.90e-4 T plus Rosseland; bands 42.8 and 459.2 /m | Liu 2016 eqs 5 to 7, Table III |
| Foam thickness | Eq 19, K = 2905 | Pilon, Fedorov, Viskanta 2001 |
| Foam lifetime | 50 s half life at 1200 C, 58 kJ/mol | Kappel 1987 Tables 2, 3 |
| Foam extinction | 267 /m, 20 % through 2 cm by two-flux | Fedorov and Pilon 2002 |
| Fining band | 1420 to 1500 C | Beerkens review [S31] |
| Sulfate | 1 wt % batch, 0.27 wt % SO3 retained | Laimbock via Pilon; Rudolph 1990 |
| Batch | 2.2 MJ/kg to 1450 K, speed from the charger | Pilon II; UV1986 |
| Regenerator casing | 1.4 W/(m^2 K), from 1726 W/m^2 | ICG slides 12 to 14 |
| Refiner crown | 6.8 W/(m^2 K), four times insulated | Ford float, seen once |
| Combustion, regenerator nodes, reversal | LHV 50 MJ/kg, 10 % excess air, 25 min reversal, 40 s gap | As v1 to v3 had them |
| Emissivity, firing shares, couple depth, height | `ours` | None found |

**The design state.** Every episode starts from `design_state.npz`, the plant settled for
30 days at the fuel that holds the crown's face at 1590 C in bay 3 (W2000b's peak) and the
refiner cooling that holds the canal 1 K over its target (`scripts/glass_slow_settle.py`):
0.4772 kg/s and 0.770 MW. `env_jax.DESIGN_STATE_SHA256` pins the file, so changing it
changes the version stamp.

**Regime of validity.** Established operation near the design pull, within the walks and
the scenarios below. Not heat-up, hot hold, a colour change or a campaign's end.

## 2. The task

### Task: `glass_furnace`

Keep a float furnace's glass fined and at forming temperature where it leaves for the
bath, at the least fuel, by setting the fuel valve. Quality is read only at the cold end,
about 18 minutes after the glass leaves; the melt answers a valve move over about two
days; and gas quality, the batch blanket and the sulfate walk unseen. Backing off too far
from the defect limit costs fuel, too little loses glass.

| obs | meaning |
|---|---|
| 0 | crown couple in bay 3, K |
| 1 | floor couple under bay 3, K |
| 2 | floor couple at the melter's end, K |
| 3 | canal, K |
| 4 | fining share read at the cold end, 0.3 h late |
| 5 | fining pass mark, 0.5 |
| 6 | refiner cooling, share of its capacity |
| 7 | valve after the crown override, kg/s |
| 8 | pull charged, kg/s |
| 9 | pull planned `pull_notice_h` ahead, kg/s |
| 10 | cullet ratio charged |
| 11 | batch line, m, every 30 min |

**Scenarios.** Five, weighted by their share of operating time (`physical_scenarios.py`):
the normal day (0.7), a planned 10 % rise or fall in pull shown 8 h ahead (0.05 each),
and an unannounced 6 % rise or fall in the gas's heating value (0.1 each), each starting
at a time the seed draws between 24 and 96 h.

**Job and score (rule 15).** Keep the glass fined and at forming temperature at the canal
at the least fuel (REALITY.md, section 6). Cost per step: fuel at 60.5 EUR/MWh [S21], plus
the charged pull at 600 EUR/t (`ours`, the owner's pick) when the canal is under 1100 C
[S16] or the fining share leaving is under 0.5. The named property: lower variance lets
the fuel drop toward the defect limit, worth 2 to 4 % of fuel in the family [S1; S2; S3].

**Limits by kind (rule 14).** Fining: a zone, priced as lost glass. Canal: a zone, held by
the refiner's cooling loop. Crown at 1620 C [S13]: protected by the plant's override, the
owner's pick and a known deviation (the record finds wear, not protection). The 1400 C
low side is not modelled: it never binds, since the hottest face sits far above it.

### Cold review, item by item (rule 13)

| # | What differs from the plant | Status | Effect on the score |
|---|---|---|---|
| 1 | The agent moves the valve; operators move crown and bottom setpoints, with firing following the crown [S4] | Known deviation, for the owner | The agent must also do a crown loop's fast work against gas swings, which a setpoint action would hand to an inner loop |
| 2 | Quality is read as the fining share, 0.3 h late with 0.005 noise; real lines count defects at the cold end [S38], with no unit or resolution found | Known deviation | The loss variable is nearly observed; only the transit and noise hide the margin |
| 3 | Crown: a protected override on the true hottest face; the record finds wear, unpriced | Known deviation (owner's pick); wear measured as is | Running hot costs only fuel. Hours over 1600 and 1610 C are reported (`glass_slow_eval.py`) |
| 4 | Lost glass at 600 EUR/t, a step on the canal and fining at that moment, on the charged pull | Known deviation; the price is the owner's | The step makes full back-off optimal; the challenge found any price from 300 to 700 EUR/t gives the same back-off |
| 5 | The fining criterion is a flow-weighted mean share of 0.5, where bubbles leave only with complete fining [S12; S31] | Known deviation; fixed before any run, as asked | Looser than the source; defects from the fastest streamlines are averaged away |
| 6 | The canal is held by a cooling loop the task builds; no source shows that loop | Known deviation | The canal never binds; overheating costs fuel only, and cooling is free |
| 7 | Reset draws the walks from the seed but starts from a state settled at nominal | Known deviation | A start transient the agent did not cause, common to all policies |
| 8 | Walks are smooth first-order processes; real pull wander is stepwise | Known deviation | Slow, smooth disturbances favour a constant valve with a margin |
| 9 | No couple lag, drift or reversal ripple; air preheat and stack had no source | Partly fixed: air preheat and stack removed | Readings are cleaner than real |
| 10 | Batch islands as a wedge; no level loop, tweel, waist coolers or boost; forward flow into the waist 2.6 times the pull against 1.2 to 2.5 [S17] | Known deviation | The refiner is slightly more coupled to the melter than the range allows |
| 11 | Unsourced numbers | Listed below with their family ranges | See the list |
| 12 | A constant valve with a margin may do nearly as well as a controller | Measured on v5 (2026-11-09): the fining PI costs 232,043 EUR a 7-day episode against 242,995 for the best constant valve (the conformance test's two seeds), and the ladder's kept rung costs 1.69 % more than the oracle's best paths, weighted (-0.3 to 2.1 % per scenario, `data/oracle_ceilings/glass_furnace.json`). No solve converged, so the room is a lower bound; on pull_down 100 and 102 and pull_up 102 the oracle's kept plan costs more than the rung. The oracle holds its last knot 24 h past the episode under the limits, because the furnace runs on; without that tail the room was 3.05 %, half of it in the last 24 h. About 0.6 point is still the episode's end: the plan drops the knot before the last, and the uncharged tail repays its fining. Another 0.6 point is the first knot, which coasts on the design state's fining margin (0.603 against a 0.5 pass) | Small room, as the family's 2 to 4 % saving says |
| 13 | Crown wear neither priced nor measured | Fixed: measured as is | Hours over 1600 and 1610 C, and the maximum, per policy |
| 14 | Cooling air power, scanner rejects and cullet remelt are not charged; the agent sees the valve after the override | Known deviation; the last is intended | Small |

### Unsourced numbers and their family ranges (rule 19)

| Number | Value | Family range or reason |
|---|---|---|
| Cold-end transit | 0.3 h | 14 to 20 min: a 50 m bath and 100 m lehr at 7.5 to 10.4 m/min [S40; S41; S42], arithmetic |
| Planned pull change | +-10 % | Inside 2 % float wander [S5] to a container plant's 29 to 42 % [S3, slide 8] |
| Gas swing | +-6 % of heating value | 3 MJ/m3 within a day [S20] on a Wobbe index of about 50, arithmetic |
| Pull ramp, notice, gas ramp, event window | 2 h, 8 h, 6 h, 24 to 96 h | Not found; ours |
| Fining pass | 0.5 | Middle of Beerkens' 1420 to 1500 C band [S31] |
| Lost glass price | 600 EUR/t | Not found; the owner's pick near one market figure seen once |
| Crown override margin and gains | 5 K, 0.0101 kg/s per K, 2 h | Ours, a protection's design; SIMC on the fuel step |
| Refiner cooling capacity, setpoint offset, gains | Twice design, 1 K, by SIMC | Not found; ours |
| Walk spreads and memories | Gas 2 %, pull 1 %, cullet 5 points, blanket 10 %, sulfate 17 %; 24 or 12 h | Gas: ENTSOG's 4 to 6 MJ/m3 read as four spreads [S20]; pull: the 2 % wander as two [S5]; cullet and blanket: the owner's +-10 points and +-20 %; sulfate: ours |
| Quality reading noise | 0.005 | Not found; ours |
| Waist and working end widths | 3.5 and 7.9 m | A 1300 t/d furnace's ratios to its melter [S18], scaled |
| Surface emissivity, port shares, couple depth and bay, combustion height | 0.8; ports 1 to 4 even then 0.7, 0.4; 36 mm in bay 3; 2.4 m | Not found; ours |
| Bubble radius | 8.3 mm | Top of the laboratory range [S32] |
| Fining time | 1 h at the band's top | Not found; ours. By Stokes at this plant's viscosity, a 0.5 mm bubble (radius ours) rises half the 1.2 m depth in 1.0 h at 1500 C and 1.7 h at 1420 C. At 3 h the plant could not pass 0.5 under its crown cap |
| Scenario weights | Normal 0.7; pull up and down 0.05 each; gas rich and lean 0.1 each | Gas swings: none at stable entry points to weekly where LNG blends [S19; S20], each 0 to 0.3. Pull changes: none on a steady float line [S5] to about 20 job changes a month on a container line [S9], each 0 to 0.3. Normal: the rest, 0.4 to 1 |

### The reference ladder's row (rule 3, for `scripts/reference_ladder.py`)

- **Registered** as `glass_furnace-v5`: `env_jax.GlassFurnace` with `GlassFurnaceParams`,
  the scored configuration as `default_params`, starting from the packaged
  `design_state.npz`. Its scenarios are `EnvSpec.scenarios`, five, weighted
  (`physical_scenarios.suite`), and their overrides `pull_change` and `gas_swing` are
  fields of the params.
- **Action:** the valve, raw in [-1, 1] mapped to 0.25 to 0.75 kg/s. Step 120 s, episode
  5040 steps (7 days).
- **Controlled variable:** the fining share read at the cold end, observation 4, its pass
  mark observation 5. Zone 0.5 to 1; the cheap side is low (less fuel, lower share), so
  the setpoint sits 2 sigma above 0.5.
- **Other limits:** the canal (held by the plant's loop) and the crown (held by the plant's
  override) need no rung of the reference.
- **Measured disturbances for feedforward:** pull (observation 8) and planned pull (9),
  cullet ratio (10), batch line (11). Observation 6 is the refiner cooling's share of its
  capacity, 7 the valve after the override.
- **Readings:** couples and canal at 0.36 K; the fining share 0.3 h late at 0.005; the
  batch line sampled every 30 min. All through `target_gym.base`'s instruments, so they
  are a function of the state and the episode's key.

### What the registration retired (charter D2)

- **The old plants:** v1 to v3's lumped crown plant (`env.py`, the old `env_jax.py`, its
  renderer) and attempt 1 (`rebuilt.py`) are deleted with their tests and scripts. The
  plant's combustion and regenerator constants moved into `physical.PhysicalParams`.
- **PID slot:** a fining PI on this plant (`experts.py`), the ladder's first rung: SIMC on
  the valve's step test, setpoint 2 sigma above the pass mark, gains in `pid_gains.json`
  under `glass_furnace`, provisional until the ladder measures its own sigma.
- **MPC slot:** the oracle (`oracle.py`), which plans the valve on the env's own step and
  reward with the episode's foresight. It does not converge on this plant (row 12), so its
  plans bound the ceiling from above; `scripts/glass_slow_oracle.py` runs the suite.
- **Scoring:** `scoring.SPEC_BAND["glass_furnace"]` is None, as HVAC's: a zone the reward
  prices itself. The legacy ceiling and settle band are one step's lost glass, 416.88 EUR,
  since any shortfall loses the step's whole pull. The consumption ratio reads the fuel
  bill.
- **Records:** the recorded rows under `src/target_gym/data/` for `glass_furnace`
  (baselines, null policies, hold measurements, credibility, oracle horizons and widths)
  describe the old plant and are stale until a re-record, which this branch does not do.

## 3. Reward (version 2)

`compute_reward = -(tracking + running + failure)`, in EUR per step, see
docs/reward-shaping.md. `running` is the fuel burnt at its heating value and price;
`tracking` is the step's charged pull at `lost_glass_price` when the canal is under
`canal_target` or the fining share leaving is under `fining_pass`, both read on the true
state. Neither goes through `reward.tracking_cost`, so there is no exponent and no hold.

| parameter | value | source |
| --- | --- | --- |
| `gas_price` | 60.5 EUR/MWh | Eurostat, EU non-household, H2 2025 [S21] |
| `lost_glass_price` | 600 EUR/t | Not found; the owner's pick near one market figure seen once |
| `canal_target` | 1373.15 K | "minimum 1100 C before entering the bath" [S16] |
| `fining_pass` | 0.5 | The middle of Beerkens' 1420 to 1500 C band [S31], fixed before any run |
| `e_floor` | 0.005 | The fining reading's scatter, set in closed form; the tolerance `TaskInfo` tells a controller. The reward does not read it |
| `failure_cost` | 0 | No trip: the crown is protected and the canal and fining are priced zones |
| `restart_steps` | 0 | No trip |

`reward_version = 1` gives 1 for a step of good glass and 0 for one lost.

## 4. Actuators and timing

| figure | value | status | source |
| --- | --- | --- | --- |
| step | 120 s | Model | the surface current's Courant number reached 1 at 180 s near the batch tip |
| valve range | 0.25 to 0.75 kg/s | Ours | about half to one and a half times the design fuel |
| valve slew | none | Ours | the valve is not what limits a furnace; the glass is |
| reversal | 25 min, 40 s gap, averaged | Read | 20 to 30 min and 30 to 60 s, Hubert, Lehigh glass lecture 3 |
| crown override | PI on the hottest face, 5 K under 1620 C | Ours, the owner's pick | a known deviation: no source shows such a protection |
| refiner cooling loop | PI on the canal, 1 K over target, up to 1.54 MW | Ours | SIMC on a 5 % cooling step (`scripts/glass_slow_settle.py tune`) |

## 5. Readings

The observation is the table in section 2. Gas quality, the blanket's thickness and the
sulfate stay hidden. `sensor_noise_scale = 0` gives the noise-free task.

| reading | sigma | timing | status | source |
| --- | --- | --- | --- | --- |
| crown couple | 0.36 K | | Read | Type B thermocouple with an E+H iTEMP TMT82 transmitter: its accuracy, stated at 2 sigma, taken as the scatter, an upper bound |
| floor couples, canal | 0.36 K | | Ours | the crown couple's figure, for want of their own |
| fining share | 0.005 | 1080 s late | Ours | the cold end is 14 to 20 min downstream: a 50 m bath and a 100 m lehr at 7.5 to 10.4 m/min [S40; S41; S42] |
| batch line | 0 | every 30 min | Read for the cadence | Klemiato 2021 |

`reward`, the override and the refiner loop read the true state.

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `glass_furnace` | 5040 | 120 | 168.0 h | 1 in [-1, 1] | 12 | 44 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are the scored configuration's (`env.default_params`),
which is what the recorded baselines use.

<!-- END GENERATED FACTS -->
