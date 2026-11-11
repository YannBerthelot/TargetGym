# Grid battery: model, sources, limits and deviations

Version 5, 2026-11-08, under charter Amendments 1 to 3. The plant record is
`REALITY.md`, its cold review `COLD_REVIEW.md`. Source tags (NESO-ST, ARX-LFP,
EMAPS-GB, ...) are REALITY.md's. The task represents a family (rule 19): grid
LFP units selling a frequency service under GB-style rules. Nothing here is
compared with an earlier version of the task (rule 11).

## 1. The job

REALITY.md sec. 6: deliver the contracted power profile to the second and keep
the estimated stored energy inside the window the market rules require for the
whole contract block, using declared baseline changes only, never
under-delivery (NESO-ST para. 6.11 to 6.13), at the least cost of energy.

Named property (rule 18, the owner's pick for Amendment 3): **a 60 to 90 min
delay on the only lever**. A baseline can be changed only for a settlement
period whose start is more than 60 min away (gate closure, NESO-GUIDE p. 36),
on 30 min periods, so a decision reaches the meter 60 to 90 min after it is
made. Measured (behaviour test 14): 3601 to 5399 s. Against it, the window
leaves about 1.1 MWh of slack, two hours at the full baseline: the delay is
half to three quarters of the time to cross the window.

Realism kept: the state of charge is hidden. The BMS reports the truth plus an
initial error, an Ah-counting drift and a temperature shift of the OCV curve
(ARX-LFP, MPS-ACCURE). The market checks the report (NESO-SOE p. 4, 6); the
plant delivers from the true charge.

## 2. Model

A Megapack 2 XL in its 2-hour configuration, of EVE LF280K LFP cells. Stored
energy follows CAISO's market model (CAISO-SOC eq. 1):

```
E(t + dt) = E(t) - dt (P_dis + eta P_ch),   soc = E / energy_capacity
```

The BMS lets through, at the meter, the least of the rating, the energy left
(or the room left), and the power at which a cell's terminal voltage reaches
its window, for a cell carrying `P / N`:

```
OCV(s) = 3.2 + 0.02 (s - 0.5) - 0.69 exp(-s / w) + 0.44 exp(-(1 - s) / w)
V      = OCV - I R,   P / N = V I,   2.5 V <= V <= 3.65 V
```

Charge is refused outside 0 to 55 degC and discharge outside -20 to 55 degC
(EVE-LF280K item 8). The BMS's report:

```
reported = clip(soc + e0 + d t + k_T (T_cell - 25), 0, 1)
e0 ~ N(0, soc_error_std),   d = +- 0.10 / 14 days
```

The regulation signal `r`, in p.u. of the contracted power, moves every 2 s
(PJM-M12 sec. 4.3.2) as an Ornstein-Uhlenbeck process around a mean `m`,
pulled back onto the integral of its departures from `m`, so that it is energy
neutral over a day (PNNL-ESS App. B):

```
x' = a x + (1 - a) (m - S / tau_N) + sigma sqrt(1 - a^2) eps,   a = exp(-2 / tau)
r  = clip(x, -1, 1),   dS/dt = r - m
```

`m` is 0, except during a long one-way deviation (section 6).

## 3. Numbers

| Symbol | Value | Unit | Source |
|---|---|---|---|
| `power_max` | 1.927e6 | W | TESLA-MP2XL, 2-hour unit |
| `energy_capacity` | 3854 kWh | J | TESLA-MP2XL, 2-hour unit |
| `round_trip_efficiency` | 0.920 | - | TESLA-MP2XL; family range 85 % (NREL-ATB-2025, CAISO-SOC) to 93.7 % (TESLA-MP2XL) |
| `cell_capacity_Ah`, `cell_nominal_V` | 280, 3.2 | A h, V | EVE-LF280K item 3 |
| `cell_resistance` | 0.25e-3 | ohm | EVE-LF280K item 5 (AC, 1 kHz, used for DC: ours) |
| `V_cell_min`, `V_cell_max` | 2.5, 3.65 | V | EVE-LF280K item 4 |
| `T_charge_min`, `T_charge_max` | 0, 55 | degC | EVE-LF280K item 8 |
| `T_discharge_min`, `T_discharge_max` | -20, 55 | degC | EVE-LF280K item 8 |
| `ocv_plateau_slope` | 0.02 | V per unit SoC | ARX-LFP eq. 5, 1 mV for nearly 5 % |
| `ocv_knee_width` | 0.03 | - | Ours, family range: knees inside 0-20 % and 80-100 %, outside the 20-80 % plateau (ARX-LFP) |
| `T_cell_start` | 25 | degC | TESLA-MP2XL rating temperature |
| `soc_error_std` | 0.075 | - | MPS-ACCURE, +-15 % read as 2 sigma; ours within the family's 0 to 40 % |
| `soc_drift_rate` | 8.27e-8 | 1/s | MPS-ACCURE, 10 % after 14 days; the random sign is ours |
| `knee_soc` | -1, 2 (window); 0.20, 0.80 (stream) | - | The plateau's ends, ARX-LFP's "roughly SOC 20% to 80%"; that the BMS corrects right there is ours, family range 0.05 to 0.34 and 0.80 to 0.93 from gauge practice on LiFePO4 (REALITY.md [ST1], [ST2]) (owner, 2026-11-11: it recalibrates at a knee) |
| `soc_temp_coeff` | 0.0051 | 1/degC | ARX-LFP Fig. 5 |
| `T_bms_curve` | 25 | degC | ARX-LFP, the 25 degC curve |
| `reserve_fraction` | 0.4 | - | NESO-SOE sec. 3, DR |
| `volume_hours` | 1.0 | h | NESO-SOE Table 2, DR |
| `recovery_fraction` | 0.2 | - | NESO-SOE Table 2, NESO-ST para. 6.11(iii) |
| `period_seconds` | 1800 | s | NESO-GUIDE p. 36, settlement period |
| `gate_closure_seconds` | 3600 | s | NESO-GUIDE p. 36 |
| `unavailable_fraction` | 0.001 | - | NESO-ST Schedule 3 |
| `shortfall_tolerance` | 1.0 | W | Ours: float32 rounding of the delivered power |
| `signal_interval` | 2.0 | s | PJM-M12 sec. 4.3.2; family 2 to 4 s (AEMO-HPR, CAISO-SOC) |
| `signal_tau` | 300 | s | Ours, family range: swings over minutes within the hour, AEMO-HPR Fig. 2 |
| `signal_std` | 0.4 | p.u. | Ours, family range: swings reach +-1 p.u. at times (PNNL-ESS sec. 5.3.2) |
| `neutrality_tau` | 1800 | s | Ours: the pull that makes the day neutral (PNNL-ESS App. B) |
| `price_base` | 74.0 | GBP/MWh | EMAPS-GB, 2024 mean (2025: 81.2) |
| `price_event_range` | 74.0, 74.0 | GBP/MWh | Normal block: no event. Scenarios in section 6 |
| `price_event_periods` | 2 | periods | EMAPS-GB quotes hourly prices: one hour |
| `price_event_first` | 3, 6 | period | Ours: inside the block, past the first gate closure |
| `availability_price` | 4.36 | GBP/MW/h | MODO-JAN24, DR's January 2024 average |
| `bias_pu_range` | 0.0, 0.0 | p.u. | Normal block: no deviation. Section 6 |
| `bias_duration_range` | 1800, 5400 | s | Ours, family range: "long-lasting" (ENTSOE-LER sec. 1), "extended periods" (NESO-ST 6.11(vi)), against TminLER of 15 to 30 min |
| `delta_t` | 1.0 | s | PNNL-ESS Foreword, data every second |
| `max_steps_in_episode` | 14400 | steps | One 4 h EFA block, NESO-SOE Table 1 |
| `e_floor`, `precision_floor` | 330 | W | The meter's scatter (Readings); KPI tables only, the score reads neither |
| `restart_steps`, `failure_cost` | 0, 0 | steps, GBP | Nothing trips (section 4) |

Derived: the contracted quantity is `power_max / 1.4` = 1.376 MW, the largest DR
offer that keeps the 40 % reserve (ours); the baseline spans +-0.551 MW; the
response volume is 1.376 MWh each way; the unit holds 4301 cells.

The floor: the score has no tracking term, so no tracking floor. There is no
documented minimum either: the response's own losses are a cost no policy
avoids, and it cancels out of the gap (rule 21). No closed form is claimed.

## 4. Limits (rule 14) and zones (rule 22)

| Limit | Kind | Model | Source |
|---|---|---|---|
| Energy requirement at each period's start, both ways | Timed and priced | Reported energy below `volume - deficit` either way: the period's availability is lost | NESO-SOE Table 2, NESO-ST para. 6.11(iii), 6.12 |
| Recovery of the requirement | Timed | The deficit falls by 20 % of the volume per period | NESO-ST para. 6.11(iii) |
| Failing to deliver | Priced | Short for 0.1 % of a period (1.8 s): the period is lost | NESO-ST Schedule 3 |
| Cell voltage 2.5 to 3.65 V | Protected | The BMS limits power | EVE-LF280K; the action is not found, protection is ours |
| Cell temperature | Protected | The BMS refuses that direction | EVE-LF280K item 8 |
| Empty or full | Protected | Power stops at the energy left | CAISO-SOC eq. 1 |

No limit trips.

**The zone** is the reported state of charge at each period's start, between
`(volume - deficit_export) / capacity` and `1 - (volume - deficit_import) /
capacity`: 0.357 to 0.643 at the block's start, with no deficit.

**Its cheap side is the low edge** (the export requirement). On a flat price,
exported energy is sold at the price stored energy is valued at, so it costs
nothing, while energy bought back loses 8 % (CAISO's form puts the round trip
on charge). A reference that backs off by rule 22 sits 2 sigma above the low
edge, sigma being the spread of the reported SoC at period starts on its
bump-test seeds. A price event moves the cheap side for its hours: high before
a spike (to sell into it), low before a negative price (to buy in it).

## 5. Observation, episode and score

**Lever.** The action is the Operational Baseline, +-0.551 MW, positive
exporting, submitted for the first period still open. The current period and
the next two are locked. The plant delivers the baseline in force plus the
contracted response, and under-delivers only where the BMS refuses.

**Observation (rule 16).** The BMS's reported state of charge, cell voltage
and temperature, the meter, the required power and the signal, the three
locked baselines, the period's export and import requirements in MWh, the
seconds to the next period, whether this period is lost, and the day-ahead
prices of this period and the next three (known before the block). No reading
is delayed: PJM allows 10 s of latency in scoring but names no telemetry delay,
and the SoC telemetry delay is "not found".

**Episode (rule 17).** One 4 h EFA block at 1 s, in established operation: the
reported energy is uniform in the zone, the true charge is that less the BMS's
error, the cells are at 25 degC, the three locked periods' baselines are 0
(ours), and the signal is at its spread.

**Score (rules 15 and 23), GBP per step, every term >= 0:**

- `availability`: a lost period costs its availability payment, 1.376 MW x
  0.5 h x GBP 4.36/MW/h = GBP 3.00 (NESO-ST Schedule 3, MODO-JAN24);
- `losses`: the energy the unit loses, at `price_base` (charging 1 MW for 1 s
  costs GBP 74.0 x 0.08 MJ / 3.6 GJ);
- `trade`: the energy's cost against `price_base`, `(price_base - price) P dt`,
  plus `|price - price_base| power_max dt`, the most any step could win against
  the base price. The offset keeps every step >= 0 and cancels out of the gap
  (rule 21). On a flat price the term is 0.

Together, on a flat price, a block pays the energy it bought at the meter less
the energy it stored, both at `price_base`, plus lost availability. Ending
the block low is therefore neutral, not free: what is sold is valued as what
is stored. Wear is not charged: NREL-ATB-2025 pays augmentation from fixed
O&M, and REALITY.md has no cost per cycle.

**Rule 23.** On 30 seeds of each scenario (100 to 129), the expert and a zero
baseline cost more than 0 on every seed. Normal block: the expert GBP 6.10 on
average (losses 6.00, availability 0.10), a zero baseline GBP 5.37. Centring
the charge costs losses, which is the operator's real trade.

## 6. Scenarios (rule 20)

`scenarios.py`; sizes and timing drawn by the seed from each range.

| Scenario | What moves | Range | Source |
|---|---|---|---|
| normal | nothing | flat GBP 74.0/MWh, neutral signal | REALITY.md sec. 5, AEMO-HPR Fig. 2, PNNL-ESS App. B |
| price spike | one hourly day-ahead price | GBP 150 to 491/MWh, first period 3 to 6 | sec. 7, EMAPS-GB (2024 high 491); sec. 5, MPS-ACCURE (Dogfish $267). Lower bound ours, twice the mean |
| negative price | one hourly price | GBP -28.2 to 0/MWh | sec. 7, EMAPS-GB (2024 low, 141 negative hours) |
| long deviation | a one-way response | 0.19 to 0.46 p.u. for 30 to 90 min, either sign | sec. 5, ENTSOE-LER sec. 1 (mean of 50 mHz or more), mapped through DR's curve, 0.015 to 0.2 Hz (NESO-ST, NESO-GUIDE p. 4): (0.05 - 0.015) / 0.185 = 0.19 to (0.10 - 0.015) / 0.185 = 0.46; the 100 mHz top is ours |
| wrong estimate | the BMS's error | sd 20 % | sec. 1 and 5, MPS-ACCURE (up to 15 %, outliers above 40 %, read as 2 sigma) |
| long deviation, wrong estimate | both | as above | sec. 5, the record's bad day |

Measured on seeds 100 to 129 (expert, then zero baseline, GBP per block):
normal 6.10, 5.37; price spike 473, 462; negative price 168, 173; long
deviation 7.03, 9.95; wrong estimate 6.17, 5.37; both 7.21, 9.85. The price
rows include the trade offset, a constant per seed. Only in the last scenario
did the true charge reach empty or full.

## 7. Behaviour tests (charter step 3)

`tests/energy/battery/test_battery_reality.py`.

| # | Test | Tolerance | Measured | Result |
|---|---|---|---|---|
| 1 | Round-trip efficiency | 84 % to 94.7 % | 92.0 % | Pass |
| 2 | Plateau slope, SoC 0.2 to 0.8 | 0.1 to 0.4 mV per % | 0.224 | Pass |
| 3 | Pack stops at the voltage window | +-0.05 V | 3.65, 2.50 V | Pass |
| 4 | Cold charge lockout at -5 degC | exact | charge 0, discharge full | Pass |
| 5a | sd of reported minus true at reset | 2.5 % to 22.5 % | 7.7 % | Pass |
| 5b | 10 K moves the gap | 5.1 % +- 50 % | 5.1 % | Pass |
| 6 | Ah counting never corrects the error, per seed | each of 100 seeds: end gap >= start gap - 1.19e-3 | holds | Pass |
| 7 | Signal energy neutral over 24 h | <= 5 % per seed | at most 2.0 % | Pass |
| 8 | Exact follower returns less the losses | +-20 % | 0.01996 against 0.02001 | Pass |
| 9 | Energy rule timed and priced, not a trip | exact | 48/50, then 50/50; 0.7 then 0.9 | Pass |
| 10 | Step resolves the service | <= 1 s | 1 s | Pass |
| 11 | Hornsdale contingency droop | dropped | - | Removed (owner, 2026-11-08) |
| 12 | Cycle life | <= 3.33e-5 per cycle | 0 | Pass, because fade is not modelled |
| 13 | PJM score on a normal day | >= 0.75 | 0.885 | Pass |
| 14 | Gate-closure delay, the named property | 3600 to 5400 s | 5399, 4800 and 3601 s, decided at 1, 600 and 1799 s into a period | Pass |

**Test 6** was restated per seed for Amendment 3, its bound written before the
rerun: the family's Ah drift over the window, 10 % in 14 days (MPS-ACCURE). The
first rerun failed by 9.7e-5 on one seed: the model summed the drift in float32
step by step. The drift is now computed from the clock, which is the same
physics without the rounding.

**Test 11** checked Hornsdale's FCAS contingency droop (AUR-Y1), another
market's service: GB's DR reaches full delivery at 0.2 Hz, so it would give 1.0
p.u. here, not 0.093. The owner dropped it on 2026-11-08, since the task does not
model that service (section 8).

**Test 12** passes only because capacity fade is not modelled (section 8).

The job tests (`test_battery_job.py`) and the scenario tests
(`test_battery_scenarios.py`) fail on the version 4 code where they test
Amendment 3 and pass.

## 8. Deviations still in the model

- **No grid frequency, no droop, no contingency duty.** The long deviation is
  put into the response directly, through DR's curve. Test 11, which checked
  Hornsdale's contingency droop (AUR-Y1), was dropped by the owner on
  2026-11-08 for that reason: it tests a service the task does not model.
- **The signal's shape is ours**, inside the family range above. It is
  PJM-style regulation under GB rules.
- **No BMS recalibration in a window**: its knees are out of reach, so the
  estimate drifts from the reset as before. A stream recalibrates (section
  "The stream"). The temperature shift applies over the whole range, where
  ARX-LFP measured it on the second plateau.
- **No thermal model**: the cells sit at 25 degC.
- **Losses are lumped on charge** (CAISO's form), and the cell sees the meter's
  power; the 0.25 mOhm is an AC figure used for the DC drop.
- **No capacity fade and no wear cost.**
- **No K factor** (Schedule 3's scaling by the worst performance score).
- **Baselines are one value per period**, where a PN may change within it at 1
  min resolution (NESO-GUIDE p. 36).
- **The next block's start requirement is not checked.** Ending low is valued
  at the base price, so it is neutral, but a real unit must start the next
  block inside its window.
- **DR's availability price is one month's average** from a vendor review, and
  DR High has cleared negative (MODO-JAN24); only DR as a whole is priced.
- **The trade offset** makes price scenarios' costs mostly constant, so room
  (reference over ceiling) is diluted there; the gap is not.

## 9. Readings

The BMS's own bias is in the model; the readings add each instrument's scatter
on top. `sensor_noise_scale = 0` gives the noise-free task.

| reading | sigma | step | instrument | status | source |
| --- | --- | --- | --- | --- | --- |
| `soc_reported` | 4.7e-5 | | BMS display on TI BQ79616 cell data | Not checked | [BQ79616](https://www.ti.com/lit/ds/symlink/bq79616.pdf) |
| `V_cell` | 5.5e-5 V | | TI BQ79616 cell monitor, channel 1 | Not checked | [BQ79616](https://www.ti.com/lit/ds/symlink/bq79616.pdf) |
| `T_cell` | 0.03 K | 0.003 K | 10 kOhm NTC on a BQ79616 GPIO input | Not checked | [BQ79616](https://www.ti.com/lit/ds/symlink/bq79616.pdf) |
| `power` | 330 W | | SEL-735 revenue meter | Confirmed | [SEL-735](https://selinc.com/products/735/) |

These four are not in REALITY.md; they are carried from the sheets cited here.
The score reads the true state.

## 10. What the reference ladder's row needs

The ladder row, `row()` in `scripts/ladder_rows/battery.py`, needs:

- the action as a baseline in +-0.551 MW for the first period past gate
  closure, not a power;
- the controlled variable as the reported SoC at period starts, and its zone
  and cheap side from section 4: the setpoint 2 sigma above the low edge
  (`ladder.zone_setpoint`, `ladder.cv_sigma`);
- a first rung that is the expert's recipe (declare the gap to the setpoint
  over one period, after the locked baselines), not PI on power: the plant
  follows the signal itself. A feedforward rung can read the day-ahead prices;
- the cost from the `availability`, `losses` and `trade` terms in GBP, with no
  tracking term (`scoring.JOB_SCORED`) and no trips;
- the six scenarios, 4 h at 1 s each: 14 400 steps.

## 11. What the oracle needs

The oracle slot (`oracle.py`) is a stand-in that builds and is not run: the
expert's plan on the true charge. A ceiling would need a planner over the
period baselines (eight moves a block, each landing two periods later) with
the true charge, the signal's future and the prices: a `TaskModel` whose step
is a whole settlement period, or a dynamic program over the period starts.
The old shared-solver planner was removed in version 4.

## Weights (charter J4)

Each scenario's real operating time, in 4 h blocks a year of 2190
(`scenarios.py`): normal 1900, negative price 100, wrong estimate 100, price
spike 60, long deviation 20, long deviation with a wrong estimate 1. All are
ours, with family ranges in `scenarios.py`; REALITY.md gives the events but
not how many blocks a year hold them.

## The stream (docs/proposals/long-strata.md, S1; owner's calls of 2026-11-11)

As a continuing plant (`target_gym.stream`) the battery runs block after block:

- **The BMS recalibrates at a voltage knee.** Once the true state of charge
  leaves the LFP plateau (at or below 20 %, at or above 80 %, ARX-LFP), the BMS
  corrects on the OCV: as the cells enter a knee the estimate's error is drawn
  afresh from the family's spread (`soc_error_std`), and its drift counts from
  the last step in the knee. A policy that never visits a knee pays for the
  drift: 10 % in 14 days (MPS-ACCURE). Reaching a knee costs availability, since
  the service's requirement keeps the reported energy near the middle. The
  drift keeps its sign, the Ah counter's bias. The error a recalibration draws
  shows from the next step.
- **The wrong estimate acts through the same mechanism**: each recalibration in
  that regime, and a stream's start in it, draws the error at sd 20 %.
- **Each regime lasts one EFA block** on average (4 h, the unit the weights
  count in) and brings the window's one event a visit. So the long deviation
  keeps its real, rare rate: 20 a year (Q8, ours, family range 4 to 50: ENTSO-E
  counts the 20 most significant over 2017 to 2021, so at least 4 a year, REALITY.md
  [ST3]).
- **A price event is cleared past the visible horizon**: an hourly price at a
  level in the regime's range, for the first period the observation does not
  yet show (four on), so a known day-ahead price never changes.
- **The day-ahead prices are a ring** of 48 periods: each period's price returns
  to the base once it has passed, so the day never runs out and an event never
  comes back a day later. The baselines share the ring: each period's is
  written while it is open, before it is read.
- **Gate closure counts steps**, and the deviation's clock counts from its own
  onset, so both stay exact however long the plant runs.
- **A stream starts with neither of the window's events.**

The knees' reading (that the BMS corrects at the plateau's ends) and the
dwells are ours, set once without tuning and open for the owner. The search
for a sourced recalibration threshold found only vendor posts and a patent
that calibrates at 99 %, none a standard. In a window nothing changes: a
scored window is bit-identical to what it was.

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `battery` | 14400 | 1 | 4.0 h | 1 in [-1, 1] | 17 | 120 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are the scored configuration's (`env.default_params`),
which is what the recorded baselines use.

<!-- END GENERATED FACTS -->
