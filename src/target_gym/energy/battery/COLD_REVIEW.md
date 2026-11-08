# Battery: cold review against REALITY.md

Reviewer read REALITY.md first, then `env.py`, `env_jax.py`, `PHYSICS.md` and the
`row_battery` row of `scripts/reference_ladder.py` (branch `feat/reference-ladder`).
Every comparison is against REALITY.md and its sources only. Line numbers are in
`src/target_gym/energy/battery/`.

## Mismatches

Ranked by how much each changes what an agent learns or what the score measures.

1. **The task's core decision is the one the market rules forbid.** The task asks
   "how far to follow": shedding demand near a charge limit is how both the expert
   and the oracle survive (PHYSICS.md:23-35, 222-227). Real operators manage state of
   charge "only by submission of updated Operational Baselines", and SoC "shall not be
   managed by over and/or under delivery" (REALITY.md sec. 2 and 6, NESO-ST para.
   6.11(v) and 6.13). The legitimate lever in PJM is lowering the declared capability
   TReg (REALITY.md sec. 1, PJM-M12 sec. 4.3.2). The task has neither a baseline nor a
   capability declaration: its single action is inverter power (env_jax.py:38,
   env.py:465). Effect: the agent learns under-delivery as SoC management, which a
   real plant would be penalised or barred for. The named operating skill (keep stored
   energy inside the window by re-declaring schedule) is absent.

2. **The signal is not energy neutral, and its energy is a disturbance rather than the
   operator's schedule.** Hourly levels are drawn uniformly in +-0.8 MW and carry up to
   2.8 pack capacities in 4 h (env.py:157-158, env_jax.py:131-136, PHYSICS.md:176-180).
   Real regulation signals were "energy neutral" over 24 h, and the PNNL test cycle
   swings -1 to +1 p.u. throughout and ends at the initial SoC (REALITY.md sec. 4,
   PNNL-ESS App. B and sec. 5.3.2). In GB the energy offset of a unit is its declared
   Operational Baseline, chosen by the provider (NESO-ST para. 6.11). PHYSICS.md D5
   (line 282-285) already says the hourly levels are what make the task energy
   limited. Effect: the difficulty of the task comes from a signal shape no source
   supports. The one real bad day of this kind, a long one-way frequency deviation of
   50 mHz or more (REALITY.md sec. 5, ENTSOE-LER sec. 1), is not how it is generated.

3. **State of charge is observed almost exactly, while the plant's property of
   interest is that nobody measures it.** The observation is true `soc` plus white
   noise of sd 4.7e-5 (env.py:539, env_jax.py:57); score and trips read the true
   state (PHYSICS.md:374-377). Real BMS estimates drift: coulomb counting accumulates
   error, a 1 mV OCV error on the LFP plateau gives about 5 % SoC, temperature moves
   the estimate about +0.51 % per degC, and vendor data shows errors up to +-15 % and a
   daily gap over 12 % (REALITY.md sec. 1, ARX-LFP sec. I, eq. 5, Fig. 5; MPS-ACCURE).
   NESO checks the *reported* state of energy (NESO-SOE p. 4, 6). Effect: the
   estimation problem REALITY.md names as the point of the plant is not present.
   PHYSICS.md:363-364 admits the bias is left out.

4. **SoC limits trip and reset the pack; in the market they are a timed, priced
   limit.** Leaving 0.05 or 0.95 trips (env.py:118-119, 554) with a 1 h restart cost
   (env.py:189-191). Real SoC rules are energy volumes held at the start of each 4 h
   EFA block, with 20 % recovery per 30 min period, a unit below the minimum "deemed
   unavailable", and a whole settlement period's payment lost at 0.1 % unavailability
   (REALITY.md sec. 3, NESO-SOE Table 2, NESO-ST para. 6.12 and Schedule 3). The hard
   limits sources do name are cell voltage and cell temperature (EVE-LF280K). Effect:
   the agent learns a cliff at a SoC number rather than a requirement tied to the
   contract and the clock.

5. **The objective is a per-MWh imbalance price plus a wear price; real payment is
   availability scaled by a performance score.** Tracking cost is $100/MWh, linear in
   |error| each step (env.py:182, 572-583). PJM pays on a performance score averaged
   per interval with 10 s latency allowed, needs 75 % to qualify and bars a unit under
   40 % over 200 half-hour intervals (REALITY.md sec. 3, 4, PJM-M12 sec. 4.4.1-4.4.5).
   GB pays availability, zeroed per settlement period, scaled by a K factor from the
   worst score in the block (NESO-ST Schedule 3). Effect: a small steady error costs
   the same per watt as a large one, and losing a block's payment is never possible.
   About half the episode cost is wear (PHYSICS.md:205), priced from a figure
   REALITY.md marks "not found" (degradation cost in $ per cycle, sec. 3).

6. **The chemistry is NMC; the sourced grid cell is LFP.** OCV coefficients are
   "NMC-like" and the cell window is 2.7 to 4.25 V (env.py:87-94, 121-122). The sourced
   grid cell is LFP, 2.5 to 3.65 V, with a flat plateau over about 20 to 80 % SoC
   (REALITY.md sec. 1 and 3, EVE-LF280K items 4, 8; ARX-LFP). Effect: with the task's
   1.16 V per unit slope (PHYSICS.md:363) cell voltage reveals SoC; on LFP it would
   not, which is what makes item 3 hard. The voltage trip also lands at the wrong SoC.

7. **Temperature limits are one-sided and the same for charge and discharge.** A
   single 60 degC trip (env.py:120, 557); ambient fixed at 25 degC (env.py:101). The
   cell charges only from 0 to 55 degC and discharges from -20 to 55 degC; the
   container runs at -30 to 50 degC ambient (REALITY.md sec. 1 and 3, EVE-LF280K,
   TESLA-MP2XL). Effect: no charge lockout in cold, no low-temperature limit, and the
   upper limit sits 5 K above the cell's rated maximum.

8. **Step and signal rates are coarser than the service.** One 5 s step per
   instruction (env.py:166). Real signals come every 2 s (PJM) or 4 s (AGC); test data
   for regulation must be every second, at least twice the signal rate; GB response
   starts in 0.5 s and is full within 1 s for DC and DM (REALITY.md sec. 1, 3, 4;
   PJM-M12 4.3.2, AEMO-HPR 2.1, PNNL-ESS Foreword, NESO-WEB). Effect: the response-time
   requirements cannot be expressed. Minor for an energy-management task.

9. **No signal or telemetry delay.** The meter and signal are exact and current
   (PHYSICS.md:148-152, 348). PJM allows 10 s latency in scoring (PJM-M12 sec.
   4.4.5); SoC telemetry delay is "not found" (REALITY.md sec. 1), so the task's zero
   delay rests on nothing.

10. **No frequency, no droop, no contingency duty.** The task has no grid frequency.
    Hornsdale provides contingency response by droop "at all times", deadband +-0.15 Hz,
    1.72 % droop, alongside AGC regulation, and part of its energy is reserved
    (REALITY.md sec. 2, AEMO-HPR sec. 2.2, AUR-Y1). GB reserves 10/20/40 % of the offered
    quantity for SoC recovery; CE requires 30 min of energy at all times (NESO-SOE sec.
    3, ENTSOE-LER sec. 6). Effect: no reserve the operator must hold back, and none of
    the sourced bad days (long deviation, islanding of 16 Nov 2019) can occur.

11. **Episode end and start.** 4 h matches one EFA block (REALITY.md sec. 3), but the
    real requirement is checked at the block's start and the next block follows; the
    task ends at truncation with no terminal requirement on stored energy (env.py:559),
    so draining at the end is free. The PNNL cycle returns to the initial SoC. The
    start SoC is uniform in 0.35 to 0.75 (env.py:161); measured SoC traces at a named
    site are "not found" (REALITY.md sec. 5).

12. **Efficiency and rating.** AC round trip is 84.4 % (PHYSICS.md:97). NREL assumes
    85 %; Megapack states 92.0 to 93.7 % full depth including auxiliaries; CAISO
    "around 85 %" (REALITY.md sec. 2, 3). The task sits at the low edge of that range.
    Charge and discharge ratings are symmetric (env.py:465); Hornsdale is 80 MW charge
    against 100 MW discharge (AEMO-HPR sec. 1). Duration 1.69 h is inside the 75 min to
    10 h span sourced (sec. 4). Low impact.

13. **Cycle life.** The stress fit gives 3000 cycles at 80 % depth for NMC
    (PHYSICS.md:103, 135). The sourced LFP cell is rated 6000 or more full cycles at
    0.5C to 80 % capacity, and NREL assumes about one cycle a day for 15 years
    (REALITY.md sec. 3, EVE-LF280K item 10, NREL-ATB-2025). Wear per MWh is therefore at
    least doubled against the sourced cell, and wear is half the score.

## Unsupported parts

No source in REALITY.md supports these. By the rule, each needs a source or goes.

- Hourly dispatch levels uniform in +-0.8 MW (env.py:157-158) and the AR(1) movement,
  10 kW innovation and 900 s correlation time (env.py:159-160). PHYSICS.md marks all
  of them "ours".
- The SoC trip at 0.05 and 0.95 (env.py:118-119) and the 1 h restart (env.py:191,
  "provisional").
- The $100/MWh imbalance price (env.py:182) and the failure cost built from it
  (env.py:189).
- The wear price: `fade_price`, the rainflow stress fit, `k_cycle`,
  `cycle_hysteresis` (env.py:107-115, 184). REALITY.md marks degradation cost in $ per
  cycle "not found", and the task relies on it for about half the episode cost.
- NMC OCV coefficients and the 2.7 to 4.25 V window (env.py:87-94, 121-122).
- T_max 60 degC, C_thermal, UA_thermal, E_activation, R1 and C1 (env.py:84-106, 120).
- `pcs_loss` 3 % (env.py:78) and the 15 kW constant `aux_load` (env.py:79); the latter
  cites MESA-1 in PHYSICS.md, which is not among REALITY.md's sources.
- Initial SoC range 0.35 to 0.75 (env.py:161).
- Exact next-block level and minutes to it in the observation (env.py:544-545). No
  source describes a forecast of the regulation energy; in GB that offset is the
  operator's own baseline.
- Zero delay on signal, meter and SoC telemetry; REALITY.md marks SoC telemetry delay
  "not found".
- Unused v1 parameters kept for the version stamp (env.py:125-150) have no plant
  meaning at all.

## Proposed behaviour tests

Each states its source, the quantity, the experiment and a tolerance fixed in advance.

1. **Round-trip efficiency.** Source: NREL-ATB 85 %, Megapack 92.0 to 93.7 % full depth
   with auxiliaries at 25 degC (REALITY.md sec. 3). Quantity: AC energy out at the meter
   over AC energy in. Experiment: constant full charge from the low voltage limit to the
   high one, rest 10 min, constant full discharge back, ambient 25 degC, noise off.
   Tolerance: inside [84 %, 94.7 %], the sourced span widened by 1 point for the
   differing test conditions.
2. **Plateau sensitivity.** Source: on LFP, 1 mV of OCV error gives nearly 5 % SoC
   (ARX-LFP eq. 5). Quantity: mean dOCV/dSoC per cell between SoC 0.2 and 0.8.
   Experiment: evaluate the task's OCV at rest (v_rc = 0) over that range. Tolerance:
   0.1 to 0.4 mV per % SoC (the source's 0.2, a factor 2 each way, since "nearly").
3. **Cell voltage window.** Source: EVE-LF280K, 2.5 to 3.65 V. Quantity: cell terminal
   voltage at which the pack stops charge and discharge. Experiment: constant 0.5C
   charge to the stop, then constant 0.5C discharge to the stop. Tolerance: +-0.05 V,
   the datasheet's limits being hard numbers.
4. **Charge lockout in cold.** Source: charge only from 0 to 55 degC, discharge -20 to
   55 degC (EVE-LF280K item 8). Quantity: accepted current. Experiment: cell at -5 degC,
   command full charge, then full discharge. Tolerance: exact; charge refused or
   limited to zero, discharge accepted.
5. **SoC estimate is not the true SoC.** Source: BMS errors up to +-15 %, daily gap over
   12 % (MPS-ACCURE); +0.51 % per degC (ARX-LFP Fig. 5). Quantity: observed minus true
   SoC. Experiment (a): 400 resets, read the gap at step 0. Tolerance: its sd between
   2.5 % and 22.5 % (the +-15 % read as 2 sigma, a factor 3 each way, vendor data).
   Experiment (b): hold the pack at rest on the plateau and raise the cell 10 K.
   Tolerance: the gap moves by 5.1 % +-50 %.
6. **Ah-counting drift accumulates.** Source: Ah counting "cannot correct initial
   errors" and reaches 10 % after 14 days (ARX-LFP sec. I, MPS-ACCURE). Quantity: gap
   at the end of an episode against its start, with no voltage correction possible on
   the plateau. Experiment: energy-neutral +-1 p.u. cycle (test 7), 4 h. Tolerance: the
   gap does not shrink on average over 100 seeds (one-sided, structural).
7. **Signal energy neutrality.** Source: PJM's 24 h signals energy neutral, PNNL cycle
   -1 to +1 p.u. (PNNL-ESS App. B, sec. 5.3.2). Quantity: |integral of the target| over
   the integral of |target|, per seed. Experiment: record the target with any policy,
   regulation-only configuration. Tolerance: at most 5 % over 24 h of signal, since
   neutrality is by design and 5 % allows for a finite window.
8. **Return to initial SoC.** Source: the PNNL test ends at the initial SoC (sec. 5.3.2).
   Quantity: end SoC minus start SoC for an exact follower. Experiment: follow the
   energy-neutral cycle exactly. Tolerance: the drop equals the losses,
   (1 - RTE) / 2 times throughput over capacity using test 1's RTE, +-20 %.
9. **SoC rule is timed and priced, not a trip.** Source: energy volume held at the EFA
   block start, Energy Recovery 0.2 per 30 min, deemed unavailable below it, payment
   zeroed for the period (NESO-SOE Table 2, NESO-ST para. 6.11(iii), 6.12, Schedule 3).
   Quantity: plant state and payment. Experiment: scripted operator exports 2 MWh of a
   50 MWh-scaled volume in period 1, then holds. Tolerance: exact; no reset, the next
   period requires 48 MWh-scaled, recovery capped at 20 % per period.
10. **Sampling rate resolves the service.** Source: data every second for regulation,
    at least twice the signal rate; signals every 2 s or 4 s (PNNL-ESS Foreword,
    PJM-M12 4.3.2, AEMO-HPR 2.1). Quantity: `delta_t`. Experiment: read the scored
    configuration. Tolerance: at most 1 s, structural.
11. **Contingency droop.** Source: deadband 50 +-0.15 Hz, 1.72 % droop, about 100 ms
    response (AUR-Y1, AEMO-HPR sec. 2.2). Quantity: power change against frequency.
    Experiment: frequency step from 50.00 to 49.77 Hz with regulation held. Tolerance:
    (49.85 - 49.77) / 50 / 0.0172 = 0.093 p.u., +-10 %, reached by the first sample.
12. **Cycle life.** Source: 6000 cycles or more at 0.5C/0.5C, 25 degC, to 80 % capacity
    (EVE-LF280K item 10). Quantity: capacity fade per full cycle. Experiment: repeat
    full 0.5C charge and discharge for 50 cycles at 25 degC, read fade. Tolerance: at
    most 0.2 / 6000 = 3.33e-5 per cycle, one-sided because the datasheet says "or more".
13. **Normal-day following.** Source: Hornsdale's measured MW lies almost on the AGC
    set-point for an hour (AEMO-HPR Fig. 2); qualification needs 75 % (PJM-M12 4.4.1).
    Quantity: PJM performance score with 10 s latency. Experiment: a feedforward
    follower on an energy-neutral signal from mid SoC, 40 min (one test). Tolerance:
    score at least 75, the qualification line.
