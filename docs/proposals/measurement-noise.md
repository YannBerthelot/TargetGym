# Measurement noise in the scored tasks

A proposal written on 2026-11-05. It brings the roadmap's "Measurement noise,
anywhere" forward from 1.0, so that it lands before 0.8.0.

## Status

Accepted on 2026-11-05, with the choices listed under [Decided](#decided).

## Why now

Every controller in the suite reads the exact state. The roadmap names the
problem: a benchmark without noise asks its question "in the one regime where
derivative action is free". It also names the cost: noise invalidates every
baseline and every tuned gain.

That cost is lowest now. 14 of the 19 tasks in the planned expert-RL benchmark
still have expert protocols to run. If noise lands first, those experts are
tuned once, on realistic readings. If it lands after, they are tuned twice.

## Decided

1. **Sensors are scored, actuators are opt-in.** Sensor noise is on in every
   scored task. Dead band, stiction and action noise are built but default to
   off, so they move no baseline.
2. **Every physical reading is noisy.** Setpoints, schedules, time and the
   controller's own commands stay exact.
3. **Learners train on measured reward, and scoring uses the true cost.**
   `reward` keeps its meaning. A plant can only compute reward from its own
   sensors, so the environment also returns a measured one.
4. **Weak figures ship labelled.** Each keeps its status in the task's
   `PHYSICS.md`, and the benchmark runs at `sensor_noise_scale` 0.5 and 2 to
   show how far its results depend on them.
5. **Slow analysers hold their noise only.** The value stays current. True
   sample-and-hold would change the control problem on `distillation` and
   `cement_kiln`, and no accepted expert protocol covers it.
6. **`patrol_bearing_only` uses provisional figures on its bearing channels,**
   so every task has noise. No real sensor has been sourced for a wingman at
   150 to 300 m, and these values stay marked as such until one is. The refuter
   asked for a source for the range figure before shipping; the owner chose to
   ship it labelled.
7. **Floors and ceilings stay as they are.** Scores stay on today's scale.

## The noise model

A reading is `value + scale * sigma * epsilon`, then rounded to the
instrument's digital step.

- **`sigma`** is the short-term random scatter of one reading. It is not
  accuracy, calibration error or drift. Bias and drift are left out, because
  no controller can see them.
- **`epsilon`** is standard normal, drawn per reading and per update.
- **Rounding** applies only where a source gives a step and `sigma` does not
  already include it.
- **Slow instruments** hold `epsilon` between updates. A gas chromatograph
  cycles every 5 to 10 minutes against the column's 1-minute step, and free
  lime is a lab sample. Per-step noise would give a filter more independent
  samples than exist.
- **`scale`** is `sensor_noise_scale`, defaulting to 1. Setting it to 0 gives
  the noise-free task, and other values test how much a result depends on the
  figures below.

## Design

### Readings, not only state fields

Noise goes on a measured copy of the state, and `get_obs` runs on that copy.
Derived channels such as path error then inherit the noise consistently, and
targets stay exact. Most readings work this way. Those computed from several
quantities do not, so a task declares two kinds of reading: field readings,
applied before `get_obs`, and channel readings, applied to its output.

| Reading | Why field noise fails | Treatment |
|---|---|---|
| `patrol_bearing_only` range, azimuth, elevation | Computed from positions, so they would inherit GPS-grade noise | Channel |
| `distillation` tray temperature | Computed from `x[14]`, while the ends of `x` are analysers | Per-index sigma on `x` |
| `compressor_surge` delivered flow | `m_d` is computed, and `dp` noise alone reveals the opening | Through `demand_dev` |
| `wind_turbine` electrical power | Noisy torque times speed gives about 10.6 kW against a 1 kW sensor | Channel |
| `cement_kiln` burning zone | Noise on 16 zones biases `max(T_solid)` upward | After the max; also `T_solid[0]`, `lime[-1]` |
| `hvac` solar gain | Computed from time, which must stay exact | Channel |
| `battery` cell voltage | Computed from current, charge and a hidden state | Channel |
| `glass_furnace` crown | Writing noise back would contaminate the sensor lag | After the lag, on the copy only |

### Keys

`get_obs(state, params)` takes no key, and it is called by experts, tuners and
Ajax's `get_raw_obs`. So a reading must be a pure function of the state.
`canonical_reset` stores a `sensor_key` in the state, and noise comes from that
key, the update index and the reading. The same episode key then gives the same
noise to every policy and to the expert, which keeps paired comparisons paired.

### Reward

`reward` stays the true cost. `info["measured_reward"]` holds the tracking and
consumption terms computed on the measured copy, with the trip charge taken from
the true state. The trip must stay true because `compute_reward_terms` calls
`check_is_terminal` on its input, so a noisy copy would charge false crashes and
miss real ones. `glass_furnace` also documents that its reward scores the true
crown temperature.

On a measured copy the cost has a floor of `(sigma / e_floor)^2` per step:

| Task | Floor per step |
|---|---|
| `distillation` | 32 |
| `cstr`, `unstable_cstr` | 25 |
| `ph_neutralization` | 1 |
| `four_tank` | 0.5 |
| `boiler_drum` pressure | 0.19 |
| `glass_furnace`, `cement_kiln` | 0.13 |
| `compressor_surge` | 0.11 |
| `first_order` | 0.034 |

This affects only what learners see. The score is unchanged.

### What stays true

Trips, termination, the score and the oracle all read the true state. The
oracle stays a full-state bound, so its gap to the expert will widen.

### Actuators

`actuator_dead_band`, `actuator_stiction` and `action_noise` default to 0, in a
shared stage between the action and `compute_next_state`. Dead band and stiction
need memory, so the state also carries the actuator's held output.

## The experts

The shared PID differentiates the raw error (`experts/pid.py:221`). Commercial
controllers filter this term, commonly with a filter time of one-tenth of the
derivative time, so noise needs that filter added first. Without it the test
would measure a defect real controllers do not have. Today's gains store has
derivative action on `plane`, `plane3d_heading` and `reactor`. The filter moves
expert fingerprints.

## Floors and the ceiling

`precision_floor` is not always an instrument's resolution. `cstr` has 1e-4
where the best analyser found scatters at 5e-4, and `distillation` has 1e-4
against 4e-4. `boiler_drum` at 0.05 bar and `glass_furnace` at 1 K sit near
accuracy bands.

`precision_floor`, `e_floor` and `rho_floor` stay as they are, so scores stay on
today's scale. On the tasks above, the ceiling therefore sits below anything the
instrument could verify, and a policy cannot reach a score of 1 there.

`H` still moves, because noise and the expert protocols change the expert's hold
cost, so the benchmark's task list has to be rechecked afterwards. `FIXED_SETTLE`
already depends on the rebuilt experts and is measured again with them.

## What it invalidates

- Every environment version, recorded baseline and tuned gain.
- The 5 benchmark experts that already pass their audit, which must be verified
  again on noisy readings.
- `tg.make(name)` results, a breaking change that goes in the 0.8.0 upgrade
  notes.

## Order of work

1. **Noise core:** `base.py`, the shared PID's filter, the actuator stage.
2. **Readings by family:** aircraft, process, and industrial and energy, as
   three parallel sessions, each adding its readings and a `PHYSICS.md` section.
3. **Expert protocols,** run on noisy plants, with settles measured again.
4. **Re-record** on CI.
5. **Release 0.8.0,** then the benchmark.

## The figures

There are 67 readings. A refutation pass checked 30 groups of them, and it
confirmed 14. Each status below means:

| Status | Readings | Meaning |
|---|---|---|
| Confirmed | 31 | The refuter opened the source and the number holds |
| Corrected | 9 | The refuter changed the value or the kind of number |
| Provisional | 6 | A value is usable, but its source is weak |
| Ceiling | 5 | The source gives accuracy, so true scatter is smaller |
| No source | 7 | Nothing found; rounding only or a placeholder |
| Not checked | 9 | Sourced, but outside the refuter's limit of 30 |

**On the tracked quantity:** of the 19 benchmark tasks, 11 rest on a Confirmed
or Corrected figure. The other 8 are Provisional: the 6 altitude-tracking
aircraft tasks, `unstable_cstr` and `ph_neutralization`. Airliner altitude is
the main gap, since no air data computer publishes this figure and 0.2 m comes
from a ground test set.

### Aircraft

| Reading | Tasks | σ | Unit | Rounding | Instrument | Status | Note | Source |
|---|---|---|---|---|---|---|---|---|
| `z` | all 10 aircraft | 0.2 | m | 1 ft | Air data computer baro altitude | Provisional | Ground test set; no airborne figure | [link](https://www.avionteq.com/Document/DPST-8000-specification-sheet.pdf) |
| `x_dot` | all 10 aircraft | 0.03 | m/s | 0.125 kt | IRS/GNSS ground velocity | Ceiling | Receiver accuracy, not scatter | [link](https://docs.novatel.com/OEM7/Content/Technical_Specs_Receiver/SMART7_Performance_Specs.htm) |
| `y_dot` | 3D, bearing-only, formation, patrol | 0.03 | m/s | 0.125 kt | IRS/GNSS ground velocity | Ceiling | As x_dot | [link](https://docs.novatel.com/OEM7/Content/Technical_Specs_Receiver/SMART7_Performance_Specs.htm) |
| `z_dot` | all 10 aircraft | 0.03 | m/s | 1 ft/min | IRS inertial vertical speed | No source | Placeholder | [link](https://docs.novatel.com/OEM7/Content/Technical_Specs_Receiver/SMART7_Performance_Specs.htm) |
| `theta` | all 10 aircraft | 0 | rad | 0.010986 deg | IRS pitch attitude | Corrected | Rounding only | [link](https://www.honeywellaerospace.com/us/en/products-and-services/products/navigation-and-sensors/inertial-measurement-units/hg9900-inertial-measurement-unit) |
| `theta_dot` | all 10 aircraft | 4.3e-06 | rad/s | 0.015625 deg/s | IRS body pitch rate | Confirmed |  | [link](https://fitness.manualsonline.com/manuals/mfg/honeywell/mk_vi.html?p=254) |
| `phi` | 3D, bearing-only, formation, patrol | 0 | rad | 0.010986 deg | IRS roll attitude | Corrected | Rounding only | [link](https://www.honeywellaerospace.com/us/en/products-and-services/products/navigation-and-sensors/inertial-measurement-units/hg9900-inertial-measurement-unit) |
| `phi_dot` | 3D, bearing-only, formation, patrol | 4.3e-06 | rad/s | 0.015625 deg/s | IRS body roll rate | Confirmed |  | [link](https://fitness.manualsonline.com/manuals/mfg/honeywell/mk_vi.html?p=254) |
| `gamma` | all 10 aircraft | 0.00013 | rad |  | IRS flight-path angle | Confirmed | Recomputed from measured velocities | [link](https://docs.novatel.com/OEM7/Content/Technical_Specs_Receiver/SMART7_Performance_Specs.htm) |
| `psi` | 3D, bearing-only, formation, patrol | 0.00013 | rad |  | IRS true track angle | Confirmed | Recomputed from measured velocities | [link](https://docs.novatel.com/OEM7/Content/Technical_Specs_Receiver/SMART7_Performance_Specs.htm) |
| `x` | plane3d_circle, plane3d_racetrack, plane3d_figure8, patrol, bearing-only, formation | 0.27 | m |  | GNSS position | Corrected |  | [link](https://ntrs.nasa.gov/api/citations/20030005804/downloads/20030005804.pdf) |
| `y` | plane3d_circle, plane3d_racetrack, plane3d_figure8, patrol, bearing-only, formation | 0.27 | m |  | GNSS position | Corrected |  | [link](https://ntrs.nasa.gov/api/citations/20030005804/downloads/20030005804.pdf) |
| `power` | all 10 aircraft | 0 | fraction | 0.003 | Engine thrust indication via FADEC | No source | Rounding only | [link](https://www.law.cornell.edu/cfr/text/14/appendix-M_to_part_121) |
| `stick` | all 10 aircraft | 0 | rad | 1.6e-3 rad | Elevator surface position sensor | No source | Rounding only | [link](https://www.law.cornell.edu/cfr/text/14/appendix-M_to_part_121) |
| `aileron` | 3D, bearing-only, formation, patrol | 0 | rad | 2.6e-3 rad | Aileron surface position sensor | No source | Rounding only | [link](https://www.law.cornell.edu/cfr/text/14/appendix-M_to_part_121) |
| `lead.x` | patrol, bearing-only, formation | 0.27 | m |  | Lead GNSS fix sent over air-to-air datalink | Corrected |  | [link](https://ntrs.nasa.gov/api/citations/20030005820/downloads/20030005820.pdf) |
| `lead.y` | patrol, bearing-only, formation | 0.27 | m |  | Lead GNSS fix sent over air-to-air datalink | Corrected |  | [link](https://ntrs.nasa.gov/api/citations/20030005820/downloads/20030005820.pdf) |
| `lead.z` | patrol, bearing-only, formation | 0.2 | m | 1 ft | Lead baro altitude sent over datalink | Provisional | As z | [link](https://www.avionteq.com/Document/DPST-8000-specification-sheet.pdf) |
| `lead.x_dot` | patrol, formation | 0.03 | m/s | 0.125 kt | Lead IRS/GNSS velocity sent over datalink | Ceiling | As x_dot | [link](https://docs.novatel.com/OEM7/Content/Technical_Specs_Receiver/SMART7_Performance_Specs.htm) |
| `lead.y_dot` | patrol, formation | 0.03 | m/s | 0.125 kt | Lead IRS/GNSS velocity sent over datalink | Ceiling | As x_dot | [link](https://docs.novatel.com/OEM7/Content/Technical_Specs_Receiver/SMART7_Performance_Specs.htm) |
| `lead.z_dot` | patrol, formation | 0.03 | m/s | 1 ft/min | Lead IRS vertical speed sent over datalink | No source | Placeholder | [link](https://docs.novatel.com/OEM7/Content/Technical_Specs_Receiver/SMART7_Performance_Specs.htm) |
| `lead.psi` | patrol, bearing-only, formation | 0.00013 | rad |  | Lead true track sent over datalink | Confirmed | As psi | [link](https://docs.novatel.com/OEM7/Content/Technical_Specs_Receiver/SMART7_Performance_Specs.htm) |
| `range` | bearing-only | 15 | m |  | TCAS II range from Mode S reply timing | No source | Unsourced; provisional, marked pending a real sensor |  |
| `azimuth` | bearing-only | 0.026 | rad |  | TCAS II four-monopole direction-finding antenna | Corrected | Random part; provisional, marked pending a real sensor | [link](https://archive.ll.mit.edu/mission/aviation/publications/publication-files/ms-papers/Harman_1984_CACA_MS-6526_WW-18698A.pdf) |
| `elevation` | bearing-only | 0.0088 | rad | 25 ft altitude report | TCAS relative altitude from Mode S reports | No source | At 250 m, scales as 1/range; provisional, marked pending a real sensor |  |

### Process

| Reading | Tasks | σ | Unit | Rounding | Instrument | Status | Note | Source |
|---|---|---|---|---|---|---|---|---|
| `C_a` | cstr, unstable_cstr | 0.0005 | mol/L |  | Online UV-Vis process photometer with flow cell | Provisional | Mapping to 1 mol/L is ours | [link](https://www.ima.co.uk/wp-content/uploads/AA_DS001A_OMA300.pdf) |
| `T` | cstr, unstable_cstr | 0.04 | K |  | Pt100 RTD with HART head transmitter | Confirmed |  | [link](https://bdih-download.endress.com/file/2c4c2c5496b6c81881df53d319b2e2f5/TI01010TDE_2725-00.pdf) |
| `T_j` | unstable_cstr | 0.04 | K |  | Pt100 RTD with HART head transmitter | Confirmed |  | [link](https://bdih-download.endress.com/file/2c4c2c5496b6c81881df53d319b2e2f5/TI01010TDE_2725-00.pdf) |
| `x` | first_order | 0.0011 | - |  | None: abstract lag. Smart transmitter class by analogy | Confirmed | Abstract lag; analogy | [link](https://web-material3.yokogawa.com/GS01C31F01-01EN.pdf) |
| `h1` | four_tank | 0.0005 | m | 1 mm | Guided wave radar level transmitter | Confirmed |  | [link](https://www.emerson.com/documents/automation/73580.pdf) |
| `h2` | four_tank | 0.0005 | m | 1 mm | Guided wave radar level transmitter | Confirmed |  | [link](https://www.emerson.com/documents/automation/73580.pdf) |
| `h3` | four_tank | 0.0005 | m | 1 mm | Guided wave radar level transmitter | Confirmed |  | [link](https://www.emerson.com/documents/automation/73580.pdf) |
| `h4` | four_tank | 0.0005 | m | 1 mm | Guided wave radar level transmitter | Confirmed |  | [link](https://www.emerson.com/documents/automation/73580.pdf) |
| `x[-1]` | distillation | 0.0004 | mol frac |  | Process gas chromatograph | Confirmed |  | [link](https://www.yokogawa.com/be-nl/solutions/products-and-services/measurement/analyzers/gas-analyzers/process-gas-chromatographs/process-gas-chromatograph-gc8000/) |
| `x[0]` | distillation | 0.0004 | mol frac |  | Process gas chromatograph | Confirmed |  | [link](https://www.yokogawa.com/be-nl/solutions/products-and-services/measurement/analyzers/gas-analyzers/process-gas-chromatographs/process-gas-chromatograph-gc8000/) |
| `x[14]` | distillation | 0.003 | mol frac |  | Pt100 RTD tray sensor with HART head transmitter | Confirmed |  | [link](https://bdih-download.endress.com/file/2c4c2c5496b6c81881df53d319b2e2f5/TI01010TDE_2725-00.pdf) |
| `pH` | ph_neutralization | 0.01 | pH | 0.01 pH | Glass electrode with industrial pH transmitter | Provisional | No noise source found | [link](https://www.controlglobal.com/articles/2013/mcmillan-ph-response-accuracy-reliability) |
| `dp` | compressor_surge | 9.17 | Pa |  | Smart gauge pressure transmitter | Confirmed |  | [link](https://web-material3.yokogawa.com/GS01C31F01-01EN.pdf) |
| `m_c` | compressor_surge | 0.021 | kg/s |  | Annubar DP flow meter with fast DP transmitter | Provisional | Excludes turbulence scatter | [link](https://core.instrumart.com/assets/product-data-sheet-rosemount-dp-flow-meters-primary-elements-en-87598.pdf) |
| `demand_dev` | compressor_surge | 0.00084 | opening |  | Delivered-flow DP meter | Provisional | Excludes turbulence scatter | [link](https://core.instrumart.com/assets/product-data-sheet-rosemount-dp-flow-meters-primary-elements-en-87598.pdf) |
| `N` | compressor_surge | 5.8e-05 | fraction of rated | 1 rpm | Speed probe with machinery tachometer module | Confirmed |  | [link](https://core.instrumart.com/assets/3500-50M-Datasheet.pdf) |
| `x` | compressor_surge | 0.0015 | fraction |  | Recycle valve travel feedback in digital positioner | Ceiling | Positioner figure, not scatter | [link](https://www.emerson.com/is/content/emerson/en/final-control/flow-controls/documents/d103415x012.pdf) |

### Industrial and energy

| Reading | Tasks | σ | Unit | Rounding | Instrument | Status | Note | Source |
|---|---|---|---|---|---|---|---|---|
| `T_crown_meas` | glass_furnace | 0.36 | K |  | Type B thermocouple with Endress+Hauser iTEMP TMT82 transmitter | Confirmed | Applied after the modelled lag | [link](https://bdih-download.endress.com/file/2c4c2c5496b6c81881df53d319b2e2f5/TI01010TDE_2725-00.pdf) |
| `T_air_preheat` | glass_furnace | 0.32 | K |  | Type S thermocouple with TMT82 transmitter | Confirmed |  | [link](https://bdih-download.endress.com/file/2c4c2c5496b6c81881df53d319b2e2f5/TI01010TDE_2725-00.pdf) |
| `fuel_flow` | glass_furnace | 0.0003 | kg/s |  | Rosemount 8800D vortex gas meter | Not checked |  | [link](https://core.instrumart.com/assets/Rosemount-8800D-Series-Vortex-Flow-Meter-Datasheet.pdf) |
| `n` | reactor | 0.00019 | fraction of rated |  | Ex-core power-range ionization chambers | Corrected | At most; scales with n | [link](https://www.osti.gov/servlets/purl/7260785) |
| `T_coolant` | reactor | 0.045 | K |  | Pt100 RTD with TMT82-class transmitter | Corrected | Lower bound | [link](https://bdih-download.endress.com/file/2c4c2c5496b6c81881df53d319b2e2f5/TI01010TDE_2725-00.pdf) |
| `T_air` | hvac | 0.023 | K | 0.1 K at BMS | Digital room sensor | Confirmed |  | [link](https://cdn.sparkfun.com/assets/9/7/7/a/d/SHT45-AD1B-R2_datasheet.pdf) |
| `T_out` | hvac | 0.023 | K | 0.1 K at BMS | Shielded outdoor sensor | Not checked |  | [link](https://cdn.sparkfun.com/assets/9/7/7/a/d/SHT45-AD1B-R2_datasheet.pdf) |
| `Q_emitter` | hvac | 27 | W |  | Hydronic heat meter: Pt100 pair plus flow sensor | Not checked |  | [link](https://bdih-download.endress.com/file/2c4c2c5496b6c81881df53d319b2e2f5/TI01010TDE_2725-00.pdf) |
| `solar_gain(time)` | hvac | 1 | W/m2 |  | Kipp & Zonen SMP6 pyranometer | Not checked | No source | [link](https://spaces.awi.de/download/attachments/183009319/KippZonen_Manual_Smart_Pyranometers_DE_HMe.pdf?api=v2) |
| `level` | boiler_drum | 0.0005 | m |  | Rosemount 5300 guided-wave radar | Confirmed |  | [link](https://www.emerson.com/documents/automation/73580.pdf) |
| `pressure` | boiler_drum | 0.022 | bar |  | Yokogawa EJA530E pressure transmitter | Confirmed |  | [link](https://web-material3.yokogawa.com/GS01C31F01-01EN.pdf) |
| `q_steam` | boiler_drum | 0.047 | kg/s |  | Rosemount 8800D vortex class | Not checked |  | [link](https://core.instrumart.com/assets/Rosemount-8800D-Series-Vortex-Flow-Meter-Datasheet.pdf) |
| `lime[-1] via discharge_lime(state)` | cement_kiln | 0.00018 | mass frac | 1e-4 fraction | XRD channel of Thermo ARL X900 lab spectrometer | Confirmed | Held between samples | [link](https://documents.thermofisher.com/TFS-Assets/CAD/Application-Notes/free-lime-determination-in-clinker-an41422-en.pdf) |
| `burning_zone_temperature(state) = max(T_solid)` | cement_kiln | 1.6 | K | 0.1 K at interface | IMPAC ISR 6 two-colour ratio pyrometer | Confirmed | Noise after the max | [link](https://www.advancedenergy.com/globalassets/resources-root/data-sheets/en-op-isr6-data-sheet.pdf) |
| `T_exhaust` | cement_kiln | 0.33 | K |  | Type S thermocouple with TMT82 transmitter | Confirmed |  | [link](https://bdih-download.endress.com/file/2c4c2c5496b6c81881df53d319b2e2f5/TI01010TDE_2725-00.pdf) |
| `T_solid[0]` | cement_kiln | 1.2 | K | 0.1 K at interface | IMPAC ISR 6 ratio pyrometer | Confirmed |  | [link](https://www.advancedenergy.com/globalassets/resources-root/data-sheets/en-op-isr6-data-sheet.pdf) |
| `raw_meal` | cement_kiln | 0.13 | kg/s |  | Pfister FRW rotor weighfeeder | Not checked |  | [link](https://www.fuller-technologies.com/files/764b473acc9a2dfc00e9247773be104557c5d5b8.pdf/Fuller-PFISTER-FRW-5-Series-Rotor-Weighfeeder.pdf) |
| `omega` | wind_turbine | 0.00052 | rad/s |  | Generator-shaft speed sensor | Confirmed |  | [link](https://vbn.aau.dk/ws/files/56213377/safeprocess_2009_1.pdf) |
| `pitch` | wind_turbine | 0.2 | deg |  | Blade pitch position sensor | Confirmed |  | [link](https://vbn.aau.dk/ws/files/56213377/safeprocess_2009_1.pdf) |
| `torque` | wind_turbine | 90 | N m |  | Generator torque measurement from the converter | Confirmed |  | [link](https://vbn.aau.dk/ws/files/56213377/safeprocess_2009_1.pdf) |
| `electrical_power(omega, torque)` | wind_turbine | 1e+03 | W |  | Generator power measurement | Confirmed | Own reading | [link](https://vbn.aau.dk/ws/files/56213377/safeprocess_2009_1.pdf) |
| `soc` | battery | 4.7e-05 | fraction |  | BMS estimator on TI BQ79616 cell data | Not checked |  | [link](https://www.ti.com/lit/ds/symlink/bq79616.pdf) |
| `V_cell` | battery | 5.5e-05 | V | 190.73 uV | TI BQ79616 cell monitor | Not checked | Own reading | [link](https://www.ti.com/lit/ds/symlink/bq79616.pdf) |
| `T_cell` | battery | 0.03 | K | 152.59 uV at GPIO | 10 kOhm NTC on BQ79616 GPIO input | Not checked |  | [link](https://www.ti.com/lit/ds/symlink/bq79616.pdf) |
| `power` | battery | 330 | W |  | SEL-735 revenue meter | Confirmed |  | [link](https://selinc.com/products/735/) |

### Actuators (opt-in)

No source gave a typical stiction range: 2% is a paper's worked example.

| Actuator | Defect | Value | Confidence | Source |
|---|---|---|---|---|
| Elevator hydraulic servo valve | dead band | Threshold at most 0.5% of rated signal; at most 0.075 deg (1.3e-3 rad) if rated signal spans the model's 15 deg authority. | low | [link](https://cdn.shopify.com/s/files/1/1469/1506/files/Hydraulic_Valve_Moog_G761_761.pdf) |
| Elevator hydraulic servo valve | stiction | Valve hysteresis at most 3% of rated signal; at most 0.45 deg open loop, far less inside a closed position loop. | low | [link](https://cdn.shopify.com/s/files/1/1469/1506/files/Hydraulic_Valve_Moog_G761_761.pdf) |
| Aileron hydraulic servo valve | dead band | Threshold at most 0.5% of rated signal; at most 0.125 deg (2.2e-3 rad) on the model's 25 deg authority. | low | [link](https://cdn.shopify.com/s/files/1/1469/1506/files/Hydraulic_Valve_Moog_G761_761.pdf) |
| Aileron hydraulic servo valve | stiction | Valve hysteresis at most 3% of rated signal; at most 0.75 deg open loop, far less inside a closed position loop. | low | [link](https://cdn.shopify.com/s/files/1/1469/1506/files/Hydraulic_Valve_Moog_G761_761.pdf) |
| Engine thrust command | dead band | Thrust command and lever position recorded at 2% of full range; treat 0.02 as a ceiling, since this is a recorder minimum. | low | [link](https://www.law.cornell.edu/cfr/text/14/appendix-M_to_part_121) |
| Control valve, well-engineered assembly (coolant, base, reflux, boilup, recycle) | dead band | ≤1 % of input span; 0.25 % for fast loops | high | [link](https://spartancontrols.emersonimpactpartners.com/getattachment/2a3e9ff2-48ee-44af-b9ca-187cdc4b63ea/emerson-control-valve-handbook.pdf) |
| Control valve as found in plants | dead band | Often above 2 %: one audit found 65 % of loops above 2 %, 30 % of valves above 4 %; 5 % not uncommon | medium | [link](https://spartancontrols.emersonimpactpartners.com/getattachment/2a3e9ff2-48ee-44af-b9ca-187cdc4b63ea/emerson-control-valve-handbook.pdf) |
| Digital positioner alone (Fisher DVC6200) | dead band | Hysteresis plus deadband ±0.25 % of span, typical value | high | [link](https://www.emerson.com/is/content/emerson/en/final-control/flow-controls/documents/d103415x012.pdf) |
| Digital positioner alone (Fisher DVC6200) | action noise | Repeatability ±0.3 % of span; σ about 0.15 % taking it as 2σ | medium | [link](https://www.emerson.com/is/content/emerson/en/final-control/flow-controls/documents/d103415x012.pdf) |
| Sticking control valve | stiction | 2 % of travel, the paper's worked example only; figure is deadband plus stickband, slip jump separate | low | [link](https://skoge.folk.ntnu.no/prost/proceedings/adchem03/contents/paperpdf/oral/290.pdf) |
| Pneumatic control valve with digital positioner (gas, feedwater) | dead band | Hysteresis and deadband 0.25 % of output span, positioner only; the valve assembly adds more. | medium | [link](https://www.emerson.com/is/content/emerson/en/final-control/flow-controls/documents/d103415x012.pdf) |
| Pneumatic control valve with digital positioner (gas, feedwater) | action noise | Repeatability 0.3 % of output span; about 0.15 % of span sigma if taken as 2 sigma. | low | [link](https://www.emerson.com/is/content/emerson/en/final-control/flow-controls/documents/d103415x012.pdf) |
| PWR control rods | dead band | One step is 1/228 of stroke, 0.44 %. Expert's rod control holds still inside 1.5 F. NUREG and HRTD sources not opened. | medium | |
| Kiln fuel dosing by rotor weighfeeder | action noise | About 0.25 % of rate sigma, borrowed from the FRW raw-meal figure of 0.5 %; a coal feeder figure was not opened. | low | [link](https://www.fuller-technologies.com/files/764b473acc9a2dfc00e9247773be104557c5d5b8.pdf/Fuller-PFISTER-FRW-5-Series-Rotor-Weighfeeder.pdf) |

## Limits

- **Tasks cannot be ranked by noise burden.** Some figures are accuracy upper
  bounds and others are instrument-only lower bounds. Drum surface turbulence,
  kiln dust and hot-leg streaming are not covered.
- **White noise is a simplification.** GPS error is correlated in time.
- **Sensor lag stays unmodelled,** except on `glass_furnace`.
- **Readings are not stale.** Holding `epsilon` keeps noise from being averaged
  away, but the value stays current.
- **Sources were read through automated fetches.** Check a figure against its
  document before citing it.

## Size budget

About 560 lines of source: 80 in `base.py`, 30 for the PID filter, 100 for
actuators and about 15 per task for readings. Tests come on top. Measure it
after.
