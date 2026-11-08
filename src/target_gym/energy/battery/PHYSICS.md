# Grid battery: model, sources, limits and deviations

Version 4, rebuilt on 2026-11-08 under charter Amendments 1 and 2. The plant
record is `REALITY.md`, and its cold review `COLD_REVIEW.md`. Source tags
(NESO-ST, ARX-LFP, ...) are REALITY.md's. Nothing here is compared with an
earlier version of the task (rule 11).

## 1. The job

REALITY.md sec. 6: deliver the contracted power profile to the second and keep
the estimated stored energy inside the window the market rules require for the
whole contract block, using declared baseline changes only, never
under-delivery (NESO-ST para. 6.11 to 6.13).

Named property (rule 18): **a hidden, drifting state of charge**. Nobody
measures it; the BMS estimates it, and on the flat LFP plateau it cannot
correct its own error (ARX-LFP sec. I). It is in the model as the BMS's
report: the true value plus an initial error, an Ah-counting drift and a
temperature shift of the OCV curve (section 2). The market checks the report
(NESO-SOE p. 4, 6); the plant delivers from the true charge.

## 2. Model

The unit is a Tesla Megapack 2 XL in its 2-hour configuration, built of EVE
LF280K LFP cells. Stored energy follows CAISO's market model (CAISO-SOC eq. 1):

```
E(t + dt) = E(t) - dt (P_dis + eta P_ch),   soc = E / energy_capacity
```

with `eta` the unit's round trip, so a full cycle returns `eta` of what it took
in. The BMS lets through, at the meter, the least of the rating, the energy
left (or the room left), and the power at which a cell's terminal voltage
reaches its window, for a cell carrying `P / N`:

```
OCV(s) = 3.2 + 0.02 (s - 0.5) - 0.69 exp(-s / w) + 0.44 exp(-(1 - s) / w)
V      = OCV - I R,   P / N = V I,   2.5 V <= V <= 3.65 V
```

The plateau is the cell's nominal 3.2 V with ARX-LFP's slope; the two knees
take OCV to 2.5 V empty and 3.65 V full. Charging is refused outside 0 to 55
degC and discharge outside -20 to 55 degC (EVE-LF280K item 8).

The BMS's report:

```
reported = clip(soc + e + k_T (T_cell - 25), 0, 1),   de/dt = d
e(0) ~ N(0, 0.075),   d = +- 0.10 / 14 days
```

The regulation signal `r`, in p.u. of the contracted power, moves every 2 s
(PJM-M12 sec. 4.3.2) as an Ornstein-Uhlenbeck process pulled back onto its own
integral, so that it is energy neutral over a day (PNNL-ESS App. B):

```
x' = a x - (1 - a) S / tau_N + sigma sqrt(1 - a^2) eps,   a = exp(-2 / tau)
r  = clip(x, -1, 1),   dS/dt = r
```

## 3. Numbers

| Symbol | Value | Unit | Source |
|---|---|---|---|
| `power_max` | 1.927e6 | W | TESLA-MP2XL, 2-hour unit |
| `energy_capacity` | 3854 kWh | J | TESLA-MP2XL, 2-hour unit |
| `round_trip_efficiency` | 0.920 | - | TESLA-MP2XL, full depth with auxiliaries, 25 degC |
| `cell_capacity_Ah`, `cell_nominal_V` | 280, 3.2 | A h, V | EVE-LF280K item 3 |
| `cell_resistance` | 0.25e-3 | ohm | EVE-LF280K item 5 (AC, 1 kHz, used for DC: ours) |
| `V_cell_min`, `V_cell_max` | 2.5, 3.65 | V | EVE-LF280K item 4 |
| `T_charge_min`, `T_charge_max` | 0, 55 | degC | EVE-LF280K item 8 |
| `T_discharge_min`, `T_discharge_max` | -20, 55 | degC | EVE-LF280K item 8 |
| `ocv_plateau_slope` | 0.02 | V per unit SoC | ARX-LFP eq. 5, 1 mV for nearly 5 % |
| `ocv_knee_width` | 0.03 | - | Ours: knees inside 0-20 % and 80-100 %, outside ARX-LFP's plateau |
| `T_cell_start` | 25 | degC | TESLA-MP2XL rating temperature |
| `soc_error_std` | 0.075 | - | MPS-ACCURE, +-15 % read as 2 sigma (the reading is ours) |
| `soc_drift_rate` | 8.27e-8 | 1/s | MPS-ACCURE, 10 % after 14 days; the random sign is ours |
| `soc_temp_coeff` | 0.0051 | 1/degC | ARX-LFP Fig. 5, +0.51 % per degC |
| `T_bms_curve` | 25 | degC | ARX-LFP, the 25 degC curve |
| `reserve_fraction` | 0.4 | - | NESO-SOE sec. 3, DR |
| `volume_hours` | 1.0 | h | NESO-SOE Table 2, DR |
| `recovery_fraction` | 0.2 | - | NESO-SOE Table 2, NESO-ST para. 6.11(iii) |
| `period_seconds` | 1800 | s | NESO-GUIDE p. 5, settlement period |
| `gate_closure_seconds` | 3600 | s | NESO-GUIDE p. 36 |
| `unavailable_fraction` | 0.001 | - | NESO-ST Schedule 3 |
| `shortfall_tolerance` | 1.0 | W | Ours: float32 rounding of the delivered power |
| `signal_interval` | 2.0 | s | PJM-M12 sec. 4.3.2 |
| `signal_tau` | 300 | s | Ours: no source gives the signal's spectrum |
| `signal_std` | 0.4 | p.u. | Ours: swings reach +-1 p.u. now and then, as PNNL's cycle does |
| `neutrality_tau` | 1800 | s | Ours: the pull that makes the signal neutral (PNNL-ESS App. B) |
| `delta_t` | 1.0 | s | PNNL-ESS Foreword, data every second |
| `max_steps_in_episode` | 14400 | steps | One 4 h EFA block, NESO-SOE Table 1 |
| `e_floor`, `precision_floor` | 330 | W | The meter's scatter (Readings); KPI tables only, the score reads neither |
| `restart_steps`, `failure_cost` | 0, 0 | steps, MW h | Nothing trips (section 4) |

Derived: the contracted quantity is `power_max / 1.4` = 1.376 MW, the largest
DR offer that keeps the 40 % reserve (the choice of the largest is ours); the
baseline spans +-0.4 of it, 0.551 MW; the response volume is 1.376 MWh each
way; the unit holds `3854 / (3.2 x 0.28)` = 4301 cells.

The floor: the score has no tracking term, so there is no tracking floor. A
unit that never misses a requirement and never falls short loses nothing: the
documented minimum is 0 MW h, a closed form.

## 4. Limits (rule 14)

| Limit | Kind | Model | Source |
|---|---|---|---|
| Energy requirement at each settlement period's start, both ways | Timed and priced | Reported energy below `volume - deficit` either way: the period is lost | NESO-SOE Table 2, NESO-ST para. 6.11(iii), 6.12 |
| Recovery of the requirement | Timed | The deficit falls by 20 % of the volume per period | NESO-ST para. 6.11(iii) |
| Failing to deliver | Priced | Short for 0.1 % of a period (1.8 s): the period is lost | NESO-ST Schedule 3 |
| Cell voltage 2.5 to 3.65 V | Protected | The BMS limits power; nothing trips | EVE-LF280K; the action is not found, protection is ours |
| Cell temperature, charge 0 to 55, discharge -20 to 55 degC | Protected | The BMS refuses that direction | EVE-LF280K item 8 |
| Empty or full | Protected | Power stops at the energy left | CAISO-SOC eq. 1 |

No limit trips. REALITY.md names no trip for any of them, and rule 14 models
an unsourced kind as nothing. PJM's bar from the market (rolling score under
40 % over 200 intervals) is far longer than an episode and is not modelled.

## 5. Observation, episode and score

**Lever.** The action is the Operational Baseline, in +-0.551 MW, positive
exporting. It is submitted for the first settlement period still open: one
whose start is more than 60 min away (NESO-GUIDE p. 36). So the current period
and the next two are locked, and a decision acts 60 to 90 min later. The plant
delivers the baseline in force plus the contracted response `1.376 MW x r`;
it under-delivers only where the BMS refuses.

**Observation (rule 16).** What the EMS reads: the BMS's reported state of
charge, cell voltage and temperature, the meter, the required power and the
signal, the three locked baselines, the period's export and import
requirements in MWh, the seconds to the next period, and whether this period
is already lost. Delay: PJM allows 10 s of latency in scoring (PJM-M12 sec.
4.4.5) but names no telemetry delay, and the SoC telemetry delay is "not
found", so no reading is delayed.

**Episode (rule 17).** One 4 h EFA block at 1 s steps. It starts in
established operation: the reported energy is uniform in the window that meets
the block's requirement both ways (NESO-GUIDE p. 37, item 2), the true charge
is that less the BMS's error, the cells are at 25 degC, the baselines of the
three locked periods are 0 (ours), and the signal is at its spread. Nothing
else moves.

**Score (rule 15).** NESO pays availability per settlement period, zeroed when
the unit is unavailable for 0.1 % of it (NESO-ST Schedule 3) or deemed
unavailable for missing its energy requirement (para. 6.12). The score is the
availability lost, in MW h of contracted availability: 1.376 MW x 0.5 h =
0.688 MW h per lost period, charged once on the step the period is lost. The
price per MW h cancels in the ratio rule 8 reports. Wear is not charged:
NREL-ATB-2025 pays augmentation from fixed O&M, and REALITY.md has no $ per
cycle. Energy bought and sold through the baseline is not charged: no price is
sourced.

**Measured (2026-11-08, 100 seeds, not a gate).** A zero baseline loses 0.234
MW h per episode on average (13 seeds lose a period), the shipped expert 0.0895
(7 seeds). Every loss is a failed energy check; neither policy ever fell short
of delivery, and the true charge stayed between 0.21 and 0.83.

## 6. Behaviour tests (charter step 3)

`tests/energy/battery/test_battery_reality.py`, the audit's 13 tests with its
tolerances. All 14 fail on the old plant (tests 5a and 5b count as two).

| # | Test | Tolerance | Measured | Result |
|---|---|---|---|---|
| 1 | Round-trip efficiency | 84 % to 94.7 % | 92.0 % | Pass |
| 2 | Plateau slope, SoC 0.2 to 0.8 | 0.1 to 0.4 mV per % | 0.224 | Pass |
| 3 | Pack stops at the voltage window | +-0.05 V | 3.65, 2.50 V | Pass |
| 4 | Cold charge lockout at -5 degC | exact | charge 0, discharge full | Pass |
| 5a | sd of reported minus true at reset | 2.5 % to 22.5 % | 7.7 % | Pass |
| 5b | 10 K moves the gap | 5.1 % +- 50 % | 5.1 % | Pass |
| 6 | Ah drift does not shrink, 100 seeds, 4 h | mean end >= mean start | 0.059748 against 0.059753 | **Fail** |
| 7 | Signal energy neutral over 24 h | <= 5 % per seed | at most 2.0 % | Pass |
| 8 | Exact follower returns less the losses | +-20 % | 0.01996 against 0.02001 | Pass |
| 9 | Energy rule timed and priced, not a trip | exact | 48/50, then 50/50; 0.7 then 0.9 | Pass |
| 10 | Step resolves the service | <= 1 s | 1 s | Pass |
| 11 | Contingency droop | 0.093 p.u. +-10 % | no grid frequency | **Fail** |
| 12 | Cycle life | <= 3.33e-5 per cycle | 0 | Pass, because fade is not modelled |
| 13 | PJM score on a normal day | >= 0.75 | 0.885 (the 10 s shift costs the rest) | Pass |

**Test 6** fails by 5e-6 on the mean. The drift's sign is drawn apart from the
initial error's, so over 4 h (0.12 % of drift against a 7.5 % spread) a gap
grows only in expectation, and 100 seeds cannot show it. The test is kept
failing; its fix would need a source tying the drift to the error, which
REALITY.md does not have.

**Test 11** fails because the task has no grid frequency. GB's DR responds to
frequency (NESO-GUIDE p. 4), but no source in REALITY.md gives a frequency
process to drive it, so the signal is modelled directly as the response.

**Test 12** passes only because capacity fade is not modelled (section 7).

The job tests are in `tests/energy/battery/test_battery_job.py` (14): they fail
on the old plant and pass.

## 7. Deviations still in the model

- **The property barely reaches the score.** The market checks the report, and
  the agent reads the report. The true charge matters only where the BMS
  refuses power at empty or full, which no measured episode reached (section
  5). Charter step 4: on these numbers the plant shows the hidden state only at
  the window's ends. Effect: the task is about planning the baseline through a
  60 to 90 min delay, more than about estimation.
- **No grid frequency, no droop, no contingency duty** (test 11, audit item 10).
  The long one-way deviation of ENTSOE-LER sec. 1 cannot occur.
- **The signal's shape is ours** (`signal_tau`, `signal_std`,
  `neutrality_tau`). Only its update rate, range and neutrality are sourced.
  It is PJM-style regulation under GB rules, a hybrid.
- **No BMS recalibration.** A real BMS can correct at the voltage knees; the
  model never corrects (ARX-LFP says only that Ah counting cannot).
- **The temperature shift applies over the whole range**, where ARX-LFP
  measured it on the second plateau.
- **No thermal model.** The cells sit at 25 degC; no source gives the unit's
  thermal mass or cooling.
- **Losses are lumped on charge** (CAISO's form) and the cell sees the meter's
  power; the 0.25 mOhm is an AC figure used for the DC drop.
- **No capacity fade**, so test 12 passes trivially, and no wear cost.
- **No K factor.** Schedule 3 scales payment by the worst performance score in
  the block; the formula is not in REALITY.md.
- **Baselines are one value per period**, where a PN may change within it at 1
  min resolution (NESO-GUIDE p. 36).
- **The block ends at truncation.** The next block's start requirement is not
  checked, so ending low is free (audit item 11).
- **Asymmetric ratings** (Hornsdale charges at 80 % of discharge) are not
  modelled; the Megapack sheet gives one rating.

## 8. Readings

The BMS's own bias is in the model; the readings add each instrument's scatter
on top (`docs/proposals/measurement-noise.md`). `sensor_noise_scale = 0` gives
the noise-free task.

| reading | sigma | step | instrument | status | source |
| --- | --- | --- | --- | --- | --- |
| `soc` | 4.7e-5 | | BMS display on TI BQ79616 cell data | Not checked | [BQ79616](https://www.ti.com/lit/ds/symlink/bq79616.pdf) |
| `V_cell` | 5.5e-5 V | | TI BQ79616 cell monitor, channel 1 | Not checked | [BQ79616](https://www.ti.com/lit/ds/symlink/bq79616.pdf) |
| `T_cell` | 0.03 K | 0.003 K | 10 kOhm NTC on a BQ79616 GPIO input | Not checked | [BQ79616](https://www.ti.com/lit/ds/symlink/bq79616.pdf) |
| `power` | 330 W | | SEL-735 revenue meter | Confirmed | [SEL-735](https://selinc.com/products/735/) |

These four are not in REALITY.md; they are carried from the instrument sheets
cited here. The score reads the true state.

## 9. What the reference ladder's row needs

`scripts/reference_ladder.py`'s `row_battery` was written for the old action
(inverter power). It now needs:

- the action as a baseline in +-0.551 MW for the first period past gate
  closure, not a power;
- a rung that is not PI on power: the plant follows the signal itself. The
  natural first rung is the expert's recipe (return the reported energy to the
  middle over one period after the locked baselines), with a gain on the gap as
  the tuned number if the ladder needs one;
- the cost read from `availability`, in MW h, with no tracking term
  (`scoring.JOB_SCORED`), and no trips to count;
- 4 h episodes at 1 s: 14 400 steps.

The oracle slot (`oracle.py`) is a stand-in: the same plan on the true charge.
The old shared-solver planner (`oracles/battery.py`) modelled rainflow wear and
an imbalance price that no longer exist, and is removed; a solver row would
need a new `TaskModel` over the baselines. Neither is run for room.
