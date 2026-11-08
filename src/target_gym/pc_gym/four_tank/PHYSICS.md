# Four-tank process: physics model, sources, limits and deviations

The plant is Johansson's quadruple-tank laboratory process (Lund, 1996) in its
non-minimum phase setting P+, the rig of Johansson and Nunes 1998 [S1].
`REALITY.md` here is its record, with sources [S1] to [S7]; `COLD_REVIEW.md`
lists how the earlier task differed. This page follows charter Amendments 1
and 2 (`docs/proposals/task-charter.md`).

**Named property: inverse response.** With gamma1 + gamma2 < 1 the plant has a
right half-plane transmission zero, +0.013 rad/s on the rig [S1, p. 2047]; each
pump fills the far upper tank more than its own lower one. It is in the model
(rule 18) and tested against the rig's range.

---

## 1. Model

Mass balances and Torricelli outflow [S1, eq. (1)]:

```
dh1/dt = -a1/A1 sqrt(2 g h1) + a3/A1 sqrt(2 g h3) + gamma1 k1/A1 v1
dh2/dt = -a2/A2 sqrt(2 g h2) + a4/A2 sqrt(2 g h4) + gamma2 k2/A2 v2
dh3/dt = -(a3 + a3,extra)/A3 sqrt(2 g h3) + (1 - gamma2) k2/A3 v2
dh4/dt = -(a4 + a4,extra)/A4 sqrt(2 g h4) + (1 - gamma1) k1/A4 v1
```

The extra outlet `a_i,extra` is the disturbance of section 5. A cup poured into
a lower tank adds its volume in one step. Levels stay in 0 to 25 cm: an empty
tank is empty [S2, p. 121], and a full one overflows back to the reservoir
[S4, p. 15].

---

## 2. Numbers: cited or ours

| Symbol | Value | Unit | Cited or ours |
|---|---|---|---|
| `A1`, `A3` | 28e-4 | m^2 | cited: 28 cm^2 [S1, p. 2046] |
| `A2`, `A4` | 32e-4 | m^2 | cited: 32 cm^2 [S1, p. 2046] |
| `a1`, `a3` | 0.071e-4 | m^2 | cited: 0.071 cm^2 [S1, p. 2046] |
| `a2`, `a4` | 0.057e-4 | m^2 | cited: 0.057 cm^2 [S1, p. 2046] |
| `g` | 9.81 | m/s^2 | cited: 981 cm/s^2 [S1, p. 2046] |
| `k1`, `k2` | 3.14e-6, 3.29e-6 | m^3/(V s) | cited: P+, 3.14 and 3.29 cm^3/(V s) [S1, p. 2046] |
| `gamma1`, `gamma2` | 0.43, 0.34 | - | cited: P+ [S1, p. 2046] |
| `v_nominal` | 3.15 | V | cited: P+ operating point [S1, p. 2046] |
| `v_min`, `v_max` | 0, 10 | V | ours: Lund maximum not found; KTH's is 15 V [S4, p. 15] |
| `h_max` | 0.25 | m | cited for the KTH copy: full at 25 cm [S4, p. 15]; Lund height not found |
| `setpoint_step` | 0.0125 | m | cited: about 5 percentage points [S4, p. 9], of the 25 cm tank |
| `setpoint_times` | 300, 1300, 2300 | steps | ours: 1000 s apart, the settling scale of the P+ response [S1, Fig. 6] |
| `leak_fraction` | 0.3 | - | ours: the extra outlet is sourced [S4, p. 9], its size is not |
| `leak_duration` | 300 | steps | ours |
| `cup_volume` | 1e-4 | m^3 | ours: 100 mL; the cup is sourced [S4, p. 9], its size is not |
| `delta_t` | 1 | s | ours: sampling not found; well under the 23 s fastest time constant |
| `max_steps_in_episode` | 3300 | steps | ours: three setpoint moves and their settling |
| `precision_floor`, `e_floor` | 1e-3 | m | ours: the stand-in reading's 1 mm step |
| `e_tol` | 0 | m | no band found [REALITY.md, Limits] |
| `tracking_exponent` | 2 | - | ours: no cost of error found |
| `failure_cost` | 125000 | reward | never charged: the rig has no trip |
| `restart_steps` | 1 | steps | never charged |
| `rho_floor`, `rho_floor_tracking` | 0 | reward | not measured on the rebuilt rig; `e_floor` is the reading's step, the documented minimum the instrument resolves, used as a scale, so `floor_is_documented_minimum` is true and the oracle's hold is a scale, not a floor |

Count: 11 cited, 11 ours.

**Finding on the source.** With the printed constants, eq. (1) settles the P-
setting at h3 = 1.63 cm, where [S1, p. 2046] prints 1.8 cm; h1, h2 and h4 agree
within 2 %. The task uses P+, whose four printed levels the model reproduces
within 2 % (test 1). P- is not used and not tested.

---

## 3. Limits (rule 14)

| Limit | Kind | Model | Source |
|---|---|---|---|
| Empty tank, h >= 0 | Physical floor | No trip: the outflow stops | [S2, p. 121] |
| Full tank, 25 cm | Physical ceiling | No trip: the tank overflows to the reservoir | [S4, p. 15]; no interlock found |
| Pump voltage | Hardware | 0 to 10 V, clipped | ours |

The earlier 5 cm low-level trip had no source and is gone. No limit is a trip,
so the task never restarts.

---

## 4. Observation (rule 16)

The four levels and the two setpoints: `[h1, h2, h3, h4, target_h1,
target_h2]`. The Lund interface shows all four levels [S2, Fig. 11]; the KTH
rig wires only the lower two and reads the upper ones by eye [S4, pp. 6, 12].

---

## 5. Episode (rule 17)

The rig starts at rest at the P+ operating point, both pumps at 3.15 V, with
the setpoints on the levels, as after "start pumps, wait for steady state"
[S4, pp. 5-6]. Then:

- **Setpoint steps** of 1.25 cm [S4, p. 9]: at 300 s on a random lower tank and
  sign, back at 1300 s, and at 2300 s on the other tank.
- **An extra outlet** opened in one upper tank for 300 s at a random time
  [S4, Exercise 4.1.1; S5, pp. 1-2]. It drives the non-minimum phase path,
  which the source says can cause problems [S4, p. 10].
- **A cup** of 100 mL poured into one lower tank at a random time [S4, p. 9].

Sizes and timing are ours (section 2). Nothing else moves.

---

## 6. Score (rule 15)

The rig's job is tracking: hold the lower levels at their setpoints with the
pumps [S1, p. 2046]. So tracking stays:

```
reward = -( (e1 / 1 mm)^2 + (e2 / 1 mm)^2 )
```

No band and no pump cost was found for the Lund rig; Delaware's DMC put a cost
on pump speed [S5, p. 6], a controller's choice, not the plant's.

---

## 7. Behaviour tests

`tests/pc_gym/four_tank/test_four_tank_reality.py`, COLD_REVIEW.md tests 1 to 6:

| Test | Source | Before | After |
|---|---|---|---|
| Lund P+ levels within 5 % or 0.1 cm | [S1, p. 2046] | fail (the 5 cm trip fired on the rig's 4.7 cm upper level) | pass |
| Lund P+ time constants within 10 % | [S1, p. 2047] | fail (same) | pass |
| Relative gain matches the formula within 5 % | [S3, eq. (2)] | pass | pass |
| Cross gains 1.3 to 2.6 times direct | [S1, eq. (3); S5] | fail (4.5) | pass |
| Right half-plane zero in 0.010 to 0.040 rad/s | [S1, p. 2047; S5, p. 4] | fail (0.041) | pass |
| Upper tanks lower and faster | [S1; S5] | fail | pass |

---

## 8. Known deviations (rule 13)

**D1. No offset jumps or calibration error** on the level signals [S4, p. 6].
Effect: easier.

**D2. Ideal pumps.** Flow is k v at once; k varies slightly with the operating
point on the rig [S1; S2, p. 123], and unequal pumps break the linear model
[S6]. Effect: easier.

**D3. The level readings are a stand-in.** A radar's figures, ours; the rig's
sensor type, noise and lag are not found.

**D4. Air in the tubes** [S4, footnote 2] is not modelled.

---

## 9. What the reference ladder row needs

- Bump tests at the P+ operating point with `setpoint_step`, `leak_fraction`
  and `cup_volume` at 0.
- PI by SIMC on the crossed pairing (lambda = -0.64), then feedforward: none
  from a measured disturbance, since the outlet and the cup are hidden; then
  cascade on the upper levels (read here); then the static decoupler.
- The stored expert's decoupler and gains were tuned on the old plant and no
  longer decouple it (`test_four_tank_decoupler.py` fails): they are redone by
  the ladder, not here.
- Score: the tracking cost of section 6.

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `four_tank` | 3300 | 1 | 55 min | 2 in [-1, 1] | 6 | 14 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are the scored configuration's (`env.default_params`),
which is what the recorded baselines use.

<!-- END GENERATED FACTS -->

## Readings

The observation reports the four levels through their instruments and the two
setpoints exactly (`docs/proposals/measurement-noise.md`).
`sensor_noise_scale = 0` gives the noise-free readings, rounding included. The
rig's own level sensors read y = kc h, kc = 0.50 V/cm [S1, p. 2046]; their type,
noise and lag are not found (REALITY.md). The figures below are a guided wave
radar's, a stand-in that is ours (deviation D3).

| reading | sigma | step | instrument | status | source |
| --- | --- | --- | --- | --- | --- |
| `h1`, `h2`, `h3`, `h4` | 0.5 mm | 1 mm | Guided wave radar level transmitter | Ours (stand-in) | [Rosemount 5300](https://www.emerson.com/documents/automation/73580.pdf) |

The expert reads these levels, and so does every tuner. The oracle plans and
acts on the true state, so it stays a full-state bound. `reward`, trips, the
score and the time-in-band and settle KPIs read the true levels.
`info["measured_reward"]` is the same cost on the readings, with any trip taken
from the true state, and a learner can train on it.

A plant resting exactly on target pays about 0.67 per step in measured reward.
The proposal's floor of 0.5 is the scatter alone, `(0.5 mm / 1 mm)^2` per tank.
Rounding to 1 mm adds `step^2 / 12` to each reading's variance, 1/12 per tank.

## Written exception to check 2

The credibility check on a sourced real figure has nothing to read here. The owner
accepted a class exception on 2026-11-06: no published figure could be found or opened
after two searches with a re-fetch check. The plan in docs/proposals/industrial-fixes.md
says what was tried.
