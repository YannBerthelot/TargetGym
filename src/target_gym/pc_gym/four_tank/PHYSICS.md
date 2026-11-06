# Quadruple tank — physics model, provenance and validation

Reference process: **Johansson's quadruple-tank process**, the standard
laboratory MIMO benchmark with an adjustable multivariable zero. Two pumps feed
four tanks; each pump splits its flow between the lower tank on its own side
and the *diagonal* upper tank, which then drains into the other lower tank.

Contract for `target_gym.pc_gym.four_tank`. Method:
`docs/PHYSICS_METHODOLOGY.md`.

Status: ✅ validated · ⚠️ defensible but approximate · ❌ known deviation

---

## 1. Model and provenance

Mass balances with Torricelli outflow:

```
A1 dh1/dt = −a1 √(2g h1) + a3 √(2g h3) + γ1 k1 v1
A2 dh2/dt = −a2 √(2g h2) + a4 √(2g h4) + γ2 k2 v2
A3 dh3/dt = −a3 √(2g h3) + (1 − γ2) k2 v2
A4 dh4/dt = −a4 √(2g h4) + (1 − γ1) k1 v1
```

**Provenance: adapted from [PC-gym](https://github.com/MaximilianB2/pc-gym).**
Every parameter and all four ODEs were compared against PC-gym's
`model_classes.py` term for term and match exactly. The process itself is
Johansson (2000), *The Quadruple-Tank Process: A Multivariable Laboratory
Process with an Adjustable Zero*, IEEE TCST 8(3).

**The structure is the rig's, the numbers are not.** PC-gym gives no source for
its values, and they describe a far larger plant than the laboratory rig:

| Quantity | This task (PC-gym) | Johansson's rig |
|---|---|---|
| Tank area `A` | 1 m² | 28 and 32 cm² |
| Outlet area `a` | 20 to 35 cm² | 0.057 and 0.071 cm² |
| Pump gain `k` | 850 and 950 cm³/(V s) | 3.33 and 3.35 cm³/(V s), 3.14 and 3.29 non-minimum phase |
| Valve split `gamma` | 0.2, 0.2 | 0.70, 0.60, and 0.43, 0.34 non-minimum phase |

The rig's figures are Johansson's as reproduced by
[Wafi and Widjiantoro 2023](https://arxiv.org/abs/2304.04763) (Tables 1 and 2) and
[Saeed, Uddin and Katebi 2010](https://strathprints.strath.ac.uk/42140/1/quad_tank_rk.pdf).
The paper itself could not be opened here. This task's values are unchanged, and
no figure in this file is a measurement of the rig.

Deliberately **not** modelled:

| Omitted | Rationale |
|---|---|
| Pump dynamics and deadband | Voltage maps to flow instantaneously and linearly, with no rate limit or lag. Ours, for the owner to accept. |
| Level-sensor lag | None. The transmitter updates at least once a second (see Readings), the 1 s step here. |
| Valve-position drift | γ1 and γ2 are fixed, so the zero cannot be moved at run time — which is the one thing the real rig is famous for being able to do. |

**Unmeasured loads.** After the settle, each lower tank takes a flow nothing
measures (`step_load`). Each is a uniform draw held for 5 min, within
±2.5e-4 m³/s, about 4 % of the tank's flow, and the first arrives at step 581,
the task's fixed settle, so the whole path falls in the scored hold. The draw
comes from the episode's key and the clock, so every policy on one episode key
meets the same loads. All these figures are ours, for the owner to accept. The
limit is what keeps the upper tanks 30 mm above the trip at every target and
load. `load_scale = 0` switches them off.

**Regime of validity.** Levels between roughly 0.05 and 0.45 m, pump voltages
0–10 V. Torricelli outflow assumes free discharge and no orifice submergence.

---

## 2. Validation targets

| Quantity | Target | Model | |
|---|---|---|---|
| Parameters and ODEs vs PC-gym | exact match | **verbatim** | ✅ |
| Configuration | γ1 + γ2 < 1 → non-minimum phase | **0.40** | ✅ |
| RGA element λ11 = γ1γ2/(γ1+γ2−1) | negative in this regime | **−0.067** | ✅ |
| Off-diagonal gains dominate | required for the cross pairing | ~4× the diagonal | ✅ |
| Maximum reachable levels | — | h1 0.360 m, h2 0.429 m | ✅ |
| Every target individually reachable | required | yes | ✅ |
| Every target pair **jointly** reachable | required | 49/49 | ✅ |
| Voltage headroom at the top target | < 90 % | yes | ✅ |
| Upper-tank margin above the trip | ≥ 30 mm | **≥ 50 mm**, ≥ 30 mm at the worst load | ✅ |
| Every target holds under every load | pumps inside 0 to 10 V | yes | ✅ |
| Episode covers settling | ≥ 2 expert settles | 1200 steps, twice the expert's 570 s to the 1 mm resolution (v3; 500 in v2) | ✅ |

**The negative RGA is the load-bearing result**, because it dictates how the
plant must be controlled. Johansson's formula gives λ11 = −0.067 and a
numerically differentiated gain matrix at the operating point agrees:

```
G = [[0.0092, 0.0412],      dh1/dv1 is four times SMALLER than dh1/dv2
     [0.0469, 0.0131]]
```

A negative RGA element means closing one diagonal loop reverses the sign of the
other, so integral action on the obvious pairing is unstable. The shipped
controller therefore drives **v1 from h2 and v2 from h1**. This is not a tuning
preference; the diagonal pairing tripped the low-level limit within 40 steps.

---

## 3. Parameter table

| Symbol | Value | Unit | Source | |
|---|---|---|---|---|
| `gamma1`, `gamma2` | 0.2, 0.2 | – | PC-gym, not the rig; sum < 1 → non-minimum phase | ⚠️ |
| `k1`, `k2` | 8.5e-4, 9.5e-4 | m³/(V s) | PC-gym, not the rig | ⚠️ |
| `a1`–`a4` | 3.5, 3.0, 2.0, 2.5 (×10⁻³) | m² | PC-gym, not the rig | ⚠️ |
| `A1`–`A4` | 1.0 | m² | PC-gym, not the rig | ⚠️ |
| `v_min`, `v_max` | 0, 10 | V | THIS REPO — PC-gym has no bounds | ✅ |
| `h_min`, `h_max` | 0.05, 1.5 | m | THIS REPO | ⚠️ D2 |
| `tracking_band` | 0.05 | m | THIS REPO — see D1 | ✅ |
| `load_flow` | 2.5e-4 | m³/s | THIS REPO: the upper tanks' margin, for the owner to accept | ⚠️ |
| `load_start`, `load_period` | 581, 300 | steps | THIS REPO: the fixed settle, and a 5 min block, for the owner to accept | ⚠️ |
| `target_h1_range` | (0.11, 0.19) | m | THIS REPO — sized against the envelope | ✅ |
| `target_h2_range` | (0.14, 0.28) | m | THIS REPO | ✅ |
| `delta_t` | 1.0 | s | ~58 s tank time constant | ✅ |

As with the other PC-gym adaptations, the plant is upstream and everything
below `v_min` is this repository's. That distinction matters: every defect
found in this environment was in the lower half of the table, never the model.

---

## 4. Task design

### Task: `four_tank`

Hold the levels of the two lower tanks of Johansson's quadruple-tank process
with two pumps. Each pump sends most of its flow to the upper tank on the other
side, which drains into the opposite lower tank, so pairing each pump with the
tank below it is the wrong move and the response is non-minimum phase. All four
levels are measured, but after the settle the lower tanks take flows nothing
measures, so the difficulty is the coupling and the loads.

| obs | meaning |
|---|---|
| 0 | `h1`, level of lower tank 1 (m) |
| 1 | `h2`, level of lower tank 2 (m) |
| 2 | `h3`, level of upper tank 3 (m) |
| 3 | `h4`, level of upper tank 4 (m) |
| 4 | target `h1` (m) |
| 5 | target `h2` (m) |

**Why it is hard.** Two coupled loops whose obvious pairing is unstable, with
each pump sending 80 % of its flow to the tank on the *other* side. The useful
control move is a coordinated one, and a controller that treats the loops as
independent and diagonal will destabilise itself.

**Observation.** All four levels are measured. The loads on the lower tanks are
not, so a table of the target leaves an offset that only feedback removes.

**Reward** — the floor-normalised cost described in *Reward (version 2)*
below.

**The oracle.** The MPC slot is a problem definition, not a tuned
controller (docs/proposals/oracle-audit.md): the shared solver's oracle
(`target_gym.oracles`). Once every T_c, Cooper's sample time from the
horizon protocol (4 steps, `data/oracle_horizons/four_tank.json`), it plans
P = 252 moves of 4 steps each from the true state, fewer near the episode's
end, and holds the first for those 4 steps. The plan maximises the episode
reward over the horizon on the environment's own dynamics, with the true load,
read at each solve and held over the plan, so it never sees an arrival coming.
The pumps' clips
are replaced by bounds on the moves, and both tracking terms are written as
(e / e_floor)^2, the reward itself since e_tol is 0 and p is 2. Every level
on every planned step is held inside its trip band, h_min < h < h_max, as a
hard limit, elastic at the trip cost per metre. The outflow's guarded square
root, sqrt(max(h, 0)), is left as it is: the band keeps it off its edge, and
each plan's levels are checked against 0. Nothing is smoothed. IPOPT's
algorithm solves it with exact Hessians to a scaled KKT error of 1e-8, or
stops at 150 iterations. A cold start puts every planned level at the
measured one and holds the pumps the reset records (off; simulating them
held off would drain the tanks to the square root's edge), a later solve starts from the last plan shifted one sample, and a warm solve that fails or hits the cap is solved again cold,
and one that still fails applies the last plan's next move. Every solve's certificate
(KKT error, iterations, violation, and the plan replayed on `step_env`) is
summed into the record. The oracle no longer weights the protocol's scored
window: the objective is the whole episode's reward. `e_floor` is the
transmitter's resolution and `rho_floor` is 0, both documented minima, so
the oracle sets neither.

---

## 5. Known deviations

**⚠️ D1 — the reward band was three times the operating range.** It was
originally normalised by the full tank span (`h_max − h_min` = 1.45 m) *and
squared*, so a half-metre miss still scored 0.43 and a saturated controller
looked much like a working one.

Fixed by log scaling rather than by narrowing the span: the same half-metre miss
now scores 0.15, and every halving of the error is worth the same increment down
to the millimetre the transmitter resolves. The span is still the full 1.45 m,
which under a log is only the denominator and no longer flattens anything.

`tracking_band = 0.05` is left over from the narrowing that was tried first.
The clipped reward it was added for was its only reader, and nothing has read
it since the reward became log-scaled. No MPC objective ever normalised by it:
the do-mpc oracle tracked in raw metres, and the planners that replaced it
from the oracle audit (2026-10) on track in floor units (`e_floor`).
It is named here as a warning about a band the reward ignores. The glass
furnace's since-removed `tracking_scale` was one: the reward had stopped using
it while that controller went on steering by it, which cost that environment
16% against its own PID.

Recorded because it is *why* a much worse defect went unnoticed: the target range once sat entirely above the
reachable envelope — no sampled setpoint was attainable and every episode was
unwinnable — and neither the reward nor the shared effectiveness contract
registered it.

**⚠️ D2 — the level limits are not physical.** `h_max = 1.5 m` cannot be
reached by any input (the plant tops out at 0.43 m), so the high-level trip is
dead code. `h_min = 0.05 m` is live and binds on the *upper* tanks, which is
the subtle constraint: holding a high h1 with a low h2 requires a low v1, and
v1 is what feeds tank 4. The target box is sized against that.

**⚠️ D3 — the zero is fixed.** The real apparatus is celebrated for letting you
move the multivariable zero across the imaginary axis by turning two valves.
Here γ1 and γ2 are constants, so only the non-minimum-phase configuration is
available. Making them episode parameters would recover the benchmark's most
distinctive property.

**✅ D4, fixed: the plant is disturbed.** The loads of §1 replaced a plant that
was deterministic after reset, whose hold was only the PI's tail. They come from
the episode's key, not the step's, which is why the registry declares no
`disturbance_fields` for the conformance suite's PRNG-hygiene checks.

**⚠️ D5: the rig's name, not its numbers.** See §1.

---

## 6. Expert

The quadruple-tank rig is controlled with multi-loop PI, on the crossed
pairing in the non-minimum-phase setting: pump 1 on tank 2 and pump 2 on
tank 1 (Johansson's design, as applied by Saeed, Uddin and Katebi,
["Multivariable predictive PID control for quadruple tank"](https://strathprints.strath.ac.uk/42140/1/quad_tank_rk.pdf),
2010, eqs. 31-32). The static decoupler on top is ours: that source has
none, and it takes the steady-state interaction out of the loops SIMC tunes
one at a time. The expert is that pair (`make_four_tank_pid_controller`,
`decoupled_pi_step`), SIMC-tuned (Skogestad and Grimholt 2012) from step
tests by `scripts/tune_process_experts.py --envs four_tank`.

- **Steady-state gains.** With the loads off, each pump stepped 0.3 V from rest at
  each training seed's targets and run 2000 s. At the median, G(0) = [[0.00836,
  0.03728], [0.04443, 0.01249]] m/V, so the relative gain of the straight pairing
  is -0.067 and of the crossed pairing 1.067, as the closed form says.
- **Decoupler.** `D = G(0)^-1 diag(G12, G21)`, about [[-0.300, 1.067], [1.067,
  -0.239]] V per V.
- **Bias.** The PIs' outputs are added to the voltages that hold the
  targets, from the lower tanks' steady balances and this task's constants,
  PC-gym's and not the rig's (`steady_voltages`).
- **Loops.** Each decoupled loop stepped in its own `w`, one run, N3 sizing it. The
  median fits are k = 0.0368 m/V, tau1 = 120 s, theta = 52 s on h1 and k = 0.0443
  m/V, tau1 = 102 s, theta = 48 s on h2; the delay carries the inverse response.
  Pumps have no sourced dead band, so SIMC stands: Kp_1 = 31.4 V/m, Ti_1 = 120 s,
  Kp_2 = 24.0 V/m, Ti_2 = 102 s.
- **Held-out check** (seeds 200 to 209). It holds 35.8 per step (worst seed 90.1)
  against a ceiling of 2 after the fixed settle of 581 steps, and never trips. The
  best null is the table, ratio 0.47, 0 of 10 wins. The miss is a finding, and PR 8
  looks at this ceiling: `tests/pc_gym/test_expert_specs.py`.

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `four_tank` | 1200 | 1 | 20 min | 2 in [-1, 1] | 6 | 13 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are the scored configuration's (`env.default_params`),
which is what the recorded baselines use.

<!-- END GENERATED FACTS -->

## Reward (version 2)

`compute_reward = -(tracking + failure)`, one tracking term per tank, see docs/reward-shaping.md.

| parameter | value | source |
| --- | --- | --- |
| `e_floor` | 1e-3 m | documented minimum: the level transmitter's 1 mm resolution, both tanks. Measured with no load, before PR 8 re-measures it: the MPC before the oracle audit (2026-10) held 1e-5 m on both tanks after settling (`scripts/measure_hold.py`, 900 hold steps after a 270-step burn-in; PID 0.6 and 2.9 mm), and the NLP that replaced it about 2e-7 m (protocol hold cost 3.1e-8). |
| `e_tol` | 0 | no level specification band |
| `tracking_exponent` | 2 | quadratic |
| `failure_cost` | 2.92e5 | twice the reachable two-tank excursion's cost, 2 x ((0.25 / 1e-3)^2 + (0.289 / 1e-3)^2): the steady-state tops 0.360 / 0.429 m against the lowest targets |
| `restart_steps` | 600 (10 min) | restart time priced into a trip, `restart_steps x failure_cost` (refill after an overflow or dry-out; provisional); where a plant engineer would get it: the plant's restart procedure |

`rho_floor_tracking` is the oracle's best hold for tracking -- the lowest per-seed
hold cost the reference controller demonstrated, in the reward's units, which
is 1 per term where the floor is that hold and less where the floor is clamped
at the instrument resolution -- and `rho_floor` the same with consumption
charged in full; `floor_is_documented_minimum` records whether `e_floor` is a
measured/certified floor or a resolution used as a scale, and where it is a
resolution on a deterministic plant both references are 0, since exact hold is
achievable there and the resolution only sets the unit;
`failure_cost` is the per-step cost of a tripped plant, above the largest tracking
cost the envelope can produce, and `restart_steps` the time a restart would take,
so a trip costs `restart_steps x failure_cost` (`reward.trip_cost`). A trip never
ends the window (`base.failure_kernel`): the step that leaves the envelope is
charged the trip cost, with tracking and running cost zeroed, and the plant
restarts at once as `reset_env` would, on the same clock. `terminated` stays
false unless `continuing = False`, which ends the episode on that step instead; `info["tripped"]` marks the event for the evaluator. `reward_version = 1` reconstructs the
capped log-scaled reward of the previous version (`precision_floor` and the
old weights are read only by it).

## Readings

The observation reports the four levels through their instruments and the two
setpoints exactly (`docs/proposals/measurement-noise.md`).
`sensor_noise_scale = 0` gives the noise-free readings, rounding included. The
Rosemount 5300's sheet gives at least one update a second (p. 33 of the PDF), so a
reading every 1 s step with no dead time matches it.

| reading | sigma | step | instrument | status | source |
| --- | --- | --- | --- | --- | --- |
| `h1`, `h2`, `h3`, `h4` | 0.5 mm | 1 mm | Guided wave radar level transmitter | Confirmed | [Rosemount 5300](https://www.emerson.com/documents/automation/73580.pdf) |

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
