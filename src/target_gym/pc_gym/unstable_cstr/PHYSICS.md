# LDPE autoclave: physics model, sources, limits and deviations

A member of the LDPE autoclave family (charter Amendment 3, rule 19): ethylene
polymerised by free radicals in an adiabatic stirred autoclave, cooled only by
its cold feed, its temperature held by the initiator fed into it, at an
operating point that is open-loop unstable. `REALITY.md` here is the record;
its "Amendment 3" section lists the family's sources [L1] to [L4].

**Status.** Back from Extra by the owner's decision (Amendment 3, 2026-11-08).
The earlier Extra said no production reactor runs at an unstable point; the
LDPE family does: "the industrial reactor normally operates at an unstable
steady state" [L1]. The task is rebuilt as that family's member, with
initiator flow as the action.

**Named property: an open-loop unstable operating point.** Measured on the
model: at the 513 K setpoint the balances' Jacobian has a mode growing at
0.0057 /s (175 s), stronger at lower temperatures and weaker at higher ones,
as [L4] describes (tested).

---

## 1. Model

One perfectly mixed zone, the perfectly mixed autoclave of [L4], with the
radicals at quasi-steady state:

```
dI/dt = (I_f - I) / tau - k_d(T) I
dM/dt = (M_f - M) / tau - k_p(T) M lambda_0
dT/dt = (T_f - T) / tau + (-dH_p) k_p(T) M lambda_0 / (rho c_p)
lambda_0 = sqrt(2 f k_d(T) I / k_t(T)),   k = k0 exp(-(E + V_A P) / (R T))
```

The initiator balance is integrated exactly over each substep (it is linear
at fixed T), since decomposition grows stiff near the trip.

---

## 2. Numbers: cited, derived or ours

| Symbol | Value | Unit | Cited or ours |
|---|---|---|---|
| `V` | 0.498 | m3 | cited [L3, Table 3] |
| `feed_mass` | 8 | kg/s | cited [L3, Table 4] |
| `rho` | 499 | kg/m3 | cited [L3, Table 4] |
| residence | 31.1 | s | derived; inside the family's 20 to 90 s [L2] |
| `mw` | 28.05 | kg/kmol | ethylene, textbook |
| `T_feed` | 353 | K | cited [L1, p. 3] |
| `feed_T_wander`, `feed_T_wander_period` | 1, 600 | K, s | ours: a slow wander of the cold feed around 353 K on every day, a fifth of the 5 K upset over 10 min; no figure found. It moves the unstable point [L1, p. 2] on the normal day |
| `P` | 1600 atm | Pa | cited [L1, p. 3]; family 1250 to 1950 bar [L2] |
| `kp0`, `Ep`, `Vp` | 1.0e6, 27.1 kJ/mol, -23.1 cm3/mol | - | cited: Ham et al.'s autoclave set in [L2, Table 4] |
| `kt0`, `Et` | 6.0e8 (3e8 each way), 16.5 kJ/mol | - | cited, as above |
| `kd0`, `Ed` | 4.2e15 1/s, 153.5 kJ/mol | - | cited: peroxide X [L2, Table 3]; half-life 0.7 s at 513 K, inside 0.02 to 1 s [L2] |
| `f_eff` | 0.5 | - | ours, family: the share of radicals that start chains |
| `dH` | -93.6 | MJ/kmol | ethylene's heat of polymerisation, textbook |
| `cp` | 2500 | J/(kg K) | ours, family: the supercritical mixture's |
| `I_f_min`, `I_f_max` | 0, 1e-3 | kmol/m3 | ours: up to twice the setpoint's need (4.4e-4, a mass fraction near [L3]'s 1.2e-4) |
| `T_setpoint` | 513 | K | cited: a Politeno zone [L1, p. 3] |
| `T_trip` | 573 | K | ours: the family's 300 C top [L2], past which decomposition runs away [L3] |
| `delta_t`, `max_steps_in_episode` | 1, 1800 | s, steps | ours: 30 min, about 60 residence times |
| `event_window` | 120, 900 | s | ours |
| `setpoint_step` | 24 | K (scenario) | cited: zones at 513 and 537 K [L1] |
| `feed_T_step`, `efficiency_drop`, `feed_flow_step` | 5 K, 0.2, 0.1 (scenarios) | - | ours: the events are the family's, their sizes are not found |
| reading | 0.5 K, lag 2 s | - | ours, family: a zone thermocouple |
| derived at 513 K | conversion 12 %, initiator 1.4e-5 kmol/m3 in the reactor | - | from the balances |

---

## 3. Limits (rule 14)

| Limit | Kind | Model | Source |
|---|---|---|---|
| High temperature | Trip | 573 K: decomposition's protection vents the reactor; the restart priced as 2 h of lost production (ours) | [L2; L3] |
| Low temperature | Cost | Polymerisation stops and the reactor cools toward its feed [L1]; no trip found, so it costs only the tracking | [L1] |
| Initiator pump | Hardware | 0 to 1e-3 kmol/m3 in the feed | ours |

---

## 4. Observation (rule 16)

`[T, I_f_pct, T_setpoint]`: the zone's temperature, which the controllers read
[L1, p. 2], the initiator pump's own rate, and the setpoint. Conversion and the
initiator in the reactor are hidden: no on-line measurement of them is in the
sources.

---

## 5. Episode and scenarios (rules 17, 20 and J4)

The reactor starts on its steady state at 513 K with the initiator the PID
would hold. `scenarios.py`, each event once at a second drawn in 120 to 900:

| Scenario | Event | Source | Weight |
|---|---|---|---|
| `normal` | the feed's 1 K wander only (size ours) | [L1, p. 3] | 0.7 |
| `grade_change` | setpoint +24 K | [L1]; [L2, section 3.2] | 0.1 |
| `feed_temperature` | the feed 5 K warmer | [L1, p. 3] | 0.1 |
| `initiator_efficiency` | 20 % fewer radicals | [L2, pp. 21-22] | 0.05 |
| `throughput_step` | the feed 10 % more | [L2, section 3.2]; [L3] | 0.05 |

The weights are ours: no source says how often each occurs. Family range for
the normal day's share: 0.5 to 0.9, the rest split between the upsets.

---

## 6. Score (rule 15)

The job [L1, p. 3]: hold each zone's temperature at its setpoint, within a
short time, since the grade rides on it. Tracking is the job:

```
reward = -((T_sp - T) / 1 K)^2,   and 7200 x 96800 on a trip
```

| parameter | value | source |
| --- | --- | --- |
| `e_floor` | 1.0 K | ours: a zone thermocouple's useful resolution, the documented minimum the score resolves |
| `e_tol` | 0 | no band found |
| `tracking_exponent` | 2 | ours |
| `failure_cost` | 96800 | ours: twice the tracking cost of a reactor 220 K off |
| `restart_steps` | 7200 | ours: 2 h of lost production; no restart time found |
| `rho_floor`, `rho_floor_tracking` | 0 | not measured on this plant |
| `precision_floor` | 1.0 K | version 1 only |

---

## 7. Behaviour tests

`tests/pc_gym/unstable_cstr/test_unstable_cstr_reality.py`:

| Test | Source | Result |
|---|---|---|
| The operating point is open-loop unstable | [L1] | pass (0.0057 /s) |
| The instability weakens as the temperature rises | [L4] | pass |
| Residence 20 to 90 s, initiator half-life 0.02 to 1 s | [L2] | pass (31 s, 0.7 s) |
| Held, it leaves +-2 K within 10 min | audit U-T4 | pass |
| A gain floor exists, and a P loop holds +-2 K for 10 min | audit U-T1 | pass |
| No initiator: polymerisation stops and it cools | [L1] | pass |
| Full initiator: it trips at its high temperature | [L2; L3] | pass |
| Starts at rest on its setpoint | rule 17 | pass |

The old task's tests (the textbook jacketed reactor) are removed: their
subject is gone (owner's approval, Amendment 3).

---

## 8. Known deviations (rule 13)

**D1. One zone.** The industrial autoclave has five to eight zones with
initiator injected into several [L1; L2] and imperfect mixing, which makes it
more stable [L4]. One perfectly mixed zone is [L4]'s perfectly mixed member.

**D2. No polymer quality.** Molecular weight and branching, which the grade
specifies, are not modelled; the setpoint stands for them.

**D3. No decomposition chemistry.** The trip stands at the family's top
temperature; the decomposition reaction itself [L3] is not modelled.

---

## 9. Expert

`experts.py`: a PID from the zone temperature to the initiator feed [L1],
biased at the setpoint's initiator. Gains by SIMC for an integrating process
(the unstable mode's 175 s is slow against the test), tau_c = theta, on one
bump test (+0.02 raw) from the steady state: k' = 4.03 K/s per raw unit,
theta 4.4 s, so Kp = -0.028 per K, Ti = 35 s, no derivative
(`src/target_gym/data/pid_gains.json`). The charter asks a named published rule
for an unstable plant's loop; this is SIMC's.

## 10. What the reference ladder row needs

- Tracking of the zone temperature, no zone band; scenarios and weights as
  section 5.
- Bump tests on the initiator from the steady state with the event off; the
  plant is unstable, so a test runs closed or briefly (the expert's test is
  40 s).
- PID by SIMC for an integrating or unstable process; feedforward: none
  measured; no cascade measurement.

## 11. Oracle needs

`oracle.py` plans on the three balances with the trip as a limit and the
event held from the state. Not run.

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `unstable_cstr` | 1800 | 1 | 30 min | 1 in [-1, 1] | 3 | 10 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are the scored configuration's (`env.default_params`),
which is what the recorded baselines use.

<!-- END GENERATED FACTS -->

## Readings

The zone temperature through a thermocouple, 0.5 K scatter and a 2 s lag
(ours, family); the initiator rate and the setpoint exactly.

