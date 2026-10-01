# Measurement protocol for learned policies

The library exists to ask whether reinforcement learning can beat classical
control on tasks that look like industrial ones. That question is easy to answer
badly: almost every choice below can be made in a way that decides the result
before any training starts. This page fixes those choices in advance and says
why each one is what it is.

Nothing here presumes an answer. The protocol has to be one that would convince a
reader who wanted the opposite conclusion.

## 1. Two questions, not one

They need different experiments and they have different audiences.

**Q1 — tabula rasa.** *Can a learned policy, starting from nothing, beat a tuned
PID?* The academic framing, and the one that tests the environments as an RL
benchmark.

**Q2 — expert-based.** *Given a tuned PID, can a learned policy improve on it?*
The industrial framing. Nobody replaces a working loop with a random network;
they ask whether learning buys anything on top of what they have. A negative
answer to Q1 and a positive answer to Q2 is a coherent and interesting result,
and it is the outcome plant engineers would care about most.

Both are run. They are reported separately and never averaged together.

## 2. What every comparison holds fixed

A number is only comparable to the PID and MPC beside it if it was produced
under the same conditions:

- the same reward, unmodified — no auxiliary shaping, no reward scaling;
- the same episode length, the environment's own `test_params`;
- the same evaluation episodes, so differences can be paired per seed;
- a deterministic policy at evaluation (the distribution's mean action), because
  the PID and MPC are deterministic and a stochastic policy would be scored on a
  different object than it is compared against.

Any deviation makes the row incomparable, and `tests/rl/test_rl_results.py`
rejects the two that can be checked mechanically.

## 3. The information asymmetry, stated up front

These three controllers do not see the same thing:

| controller | sees | has a model |
| --- | --- | --- |
| PID | observations | no |
| **learned policy** | **observations** | no |
| MPC | **true environment state** | **yes, exact** |

The MPC calls `_extract_x0(state)` and reads state fields directly, and it plans
against the real dynamics. Several environments are deliberately partially
observed — the glass furnace hides 6 of 9 dynamic states, the reactor 7 of 11,
the kiln 64 behind 8 measurements — so on those the MPC is solving a materially
easier problem than the one the agent faces.

Therefore: **the PID is the learned policy's peer, and the MPC is an upper bound
with more information.** Beating the PID is the result. Approaching the MPC is
interesting. Beating the MPC would be a claim requiring explanation, not a
victory lap, and the first hypothesis should be a simulator exploit rather than
superior control.

## 4. Metrics

**Within an environment**, the statistic is the *paired per-seed difference*
against the PID on identical evaluation episodes. Pairing removes the variance
from initial conditions and targets, which on these tasks is large: the circle
task's radius alone moved the PID's return by a factor of two.

**Across environments**, raw returns are not comparable. Test episodes
(`EnvSpec.test_params`) run from 100 steps (cstr, first_order) to 1600
(glass_furnace), and the per-step cost differs by orders of magnitude between
plants, since each plant's tracking is in its own floor-widths or its owner's
currency ([docs/reward-shaping.md](reward-shaping.md)). Compare with the
**normalised expert advantage** `NEA = (PID - x) / (PID - rho*)`, computed on
each controller's gain, its mean cost per step after the plant's burn-in as
`target_gym.eval` measures it. NEA is 1 at the floor and 0 at PID parity.
Report the tracking and running costs beside it.

**Report the interquartile mean with a 95% stratified bootstrap confidence
interval**, plus the win rate against the PID. Not the mean alone. This is
standard practice after Agarwal et al. (2021), and this repository has its own
evidence for it: the battery MPC scores `+14.0` on the mean and `−4.1` on the
median, winning 1 seed in 10. A mean alone would have published the opposite of
what happened.

## 5. Algorithms

**Headline: SAC and PPO.** One off-policy and one on-policy, so that a poor
result cannot be pinned on the quirks of a single family. Both are standard,
widely reimplemented, and what a sceptical reader will ask for. SAC is the
default choice for continuous control with dense rewards; PPO is the robust
on-policy reference and benefits most from the massively parallel environments
JAX makes cheap.

**Secondary study: an average-reward agent (ASAC or APO).** This is not
exoticism for its own sake. These tasks are *continuing* — reach a setpoint and
hold it indefinitely — and are ended by a time limit, not by achieving anything.
Discounted RL imposes an effective horizon that corresponds to nothing in the
task, and average-reward formulations are the theoretically correct fit. If the
discounted agents underperform, this distinguishes "RL cannot do this" from "the
discounted formulation was the wrong tool", which is a distinction the headline
result needs.

Not used for headline numbers: REDQ, AVG, UDRL, TD3. They are fine algorithms
and each invites "why that one?", which is a question a benchmark should not have
to answer.

## 6. Discount factor, fixed by rule and checked against physics

A single `gamma` across this suite would be indefensible: `delta_t` ranges from
0.05 s to 900 s, so `gamma = 0.99` means a five-second horizon in one
environment and a twenty-five-hour one in another.

The first rule drafted here was `gamma = exp(-delta_t / tau)`, setting the
horizon to a fixed number of the plant's own time constants. Measuring the time
constants killed it. The reactor's tracked flux answers the control rod in **one
step** -- prompt neutron response really is that fast -- so five time constants
is a five-step horizon, while the agent must hold that flux for 1200 steps
against xenon poisoning it cannot see resolve. The wind turbine and battery are
the same shape. A horizon set from the actuator response would have made three
environments myopic by construction.

**The rule is therefore the episode:**

    gamma = 1 - 1/N          (N = the environment's own episode length)

Two reasons. Evaluation scores the *undiscounted* return over exactly N steps,
so setting the effective horizon to N aligns what the agent optimises with what
it is measured on; any shorter and the agent is deliberately blind to part of
its own score. And it removes `gamma` as a free parameter — it cannot be tuned
to flatter one side, which matters for a comparison whose result is the point.

`gamma` is **not** in the hyperparameter search for that reason.

### What the measurement is still for

The physics check has not gone away, it has changed job: it verifies that the
horizon covers the plant's open-loop response, measured as the actuator-to-output
step response (time to 63.2% of the total change, the quantity relay tuning
assumes). Where it does not, that is a property of the environment worth knowing
before reading any result from it.

| environment | N | tau (steps) | 5·tau | gamma |
| --- | --- | --- | --- | --- |
| cstr | 100 | 5 | 25 | 0.99000 |
| first_order | 100 | 10 | 50 | 0.99000 |
| hvac | 720 | 62 | 310 | 0.99861 |
| plane | 280 | 23 | 115 | 0.99643 |
| plane_energy | 1200 | 23 | 115 | 0.99917 |
| plane_sine | 480 | 23 | 115 | 0.99792 |
| plane3d_heading | 200 | 14 | 70 | 0.99500 |
| plane3d_circle | 300 | 14 | 70 | 0.99667 |
| plane3d_figure8 | 400 | 38 | 190 | 0.99750 |
| plane3d_racetrack | 650 | 14 | 70 | 0.99846 |
| patrol | 200 | 13 | 65 | 0.99500 |
| patrol_bearing_only | 200 | 12 | 60 | 0.99500 |
| distillation | 200 | 8 | 40 | 0.99500 |
| glass_furnace | 1600 | 132 | 660 | 0.99938 |
| cement_kiln | 700 | 58 | 290 | 0.99857 |
| battery | 360 | 1 | 5 | 0.99722 |
| boiler_drum | 400 | 3 | 15 | 0.99750 |
| wind_turbine | 400 | 1 | 5 | 0.99750 |
| four_tank | 500 | 38 | 190 | 0.99800 |
| ph_neutralization | 300 | 14 | 70 | 0.99667 |
| reactor | 1200 | 1 | 5 | 0.99917 |
| unstable_cstr | 1200 | 6.3 to 14.4 | 32 to 72 | 0.99917 |
| compressor_surge | 1200 | 49.3 | 247 | 0.99917 |

The `unstable_cstr` row is filled by hand, because
`scripts/measure_time_constants.py` has no step response to fit on that plant
(see the next section). Its tau is the unstable time constant 1/lambda+, the
time an uncontrolled error takes to grow e-fold. Over the five targets it runs
from 19.0 s at C_a 0.45 to 43.1 s at 0.65, which is 6.3 to 14.4 steps of 3 s
(derived, `scripts/unstable_cstr_numbers.py --section targets`).

The `compressor_surge` row is filled by hand as well, from the plant's numbers
script, which times the header pressure's response to five actuator steps from
a steady operating point: speed 87.5 % of rated and recycle opening 0.5, the
commands a zero action maps to, with the consumer valve at 0.9. Its tau is the
slowest of them, a speed step from 87.5 to 105 % of rated whose pressure
reaches 63.2 % of its change in 4.93 s, 49.3 steps of 0.1 s (derived,
`scripts/compressor_surge_numbers.py --section steady`). The recycle valve
moves the pressure faster, in 0.41 to 3.57 s over the same section's three
valve steps.

### How long an episode has to be

Two time scales bind, and an episode has to clear both.

**Settling.** A first-order system reaches 98% of a step in 4 tau and 99.3% in
5 tau. This library is about reaching a target *and holding it*, so for holding
to be what the score measures rather than the approach, the episode needs room
for both: **a floor of 10 tau** -- roughly five to arrive and five to hold, so
maintenance is at least half of what is scored -- and a target nearer **15 tau**.

That is not an invented figure. Excluding four plants whose output answers the
actuator within a step or two, where the ratio is meaningless, the suite's
median was already **13.7 tau** with a cluster from 8.7 to 25. The floor names
the norm the environments mostly already followed.

**Period.** For a path-following or otherwise periodic task the binding scale is
the task's own period, not the actuator's response, and the requirement is
**one lap**. Less than a lap proves nothing about holding a path, and the model
review checklist records why: these tasks start the aircraft exactly on the
path, so a controller that flies straight ahead looks correct for a whole
episode that is shorter than one lap.

    N >= max(10 * tau_actuator, 1 * T_period)

**The period clause is one lap, not three.** It was three when the criterion was
first written, and a later pass relaxed it on the grounds that a controller
flying one lap on the path will fly the next. `plane_sine` keeps two, because it
is a frequency probe rather than a path: the first cycle sheds the initial
transient and the second is what an amplitude ratio and a phase lag are read
off.

**The actuator clause only applies where a settling time exists, which is a
minority of the suite.** `tau_actuator` is the time to 63.2% of an open-loop
step response, and that number exists only for a plant whose response is
monotone and settles. Measured across the registry:

* **Nine integrate.** A tank level, a drum level, an aircraft altitude under a
  held stick: there is no steady state to settle to, so there is no `tau_63`.
* **The aircraft oscillate.** A fixed elevator deflection excites the phugoid,
  so altitude rises and falls rather than approaching a value. Fitting a first
  order response to it returns a number that grows with the window it is
  measured in, and nothing else.
* **The glass furnace does not settle** inside eight thousand steps. Its crown
  response reads 822 steps at a 1600-step window, 1520 at 3200, 2383 at 5760 and
  3415 at 11520. Since the furnace physics gained a regenerator, a reversal
  cycle, a thermocouple lag and a fuel dead time, it has no time constant on the
  episode timescale.
* **The unstable CSTR runs away.** It is held on an open-loop unstable steady
  state, so under a constant coolant command it leaves that state and either
  trips or falls to the extinguished branch (derived,
  `scripts/unstable_cstr_numbers.py --section reach`). There is no step
  response, and its row above uses the unstable time constant instead.

So the clause binds on roughly a third of the environments, and for the rest the
episode is set by laps (the path-following aircraft), by the disturbance
timescale (the reactor, the turbine, the battery, the boiler drum) or by the
setpoint schedule (the furnace, the building, the kiln, the unstable CSTR).
**A criterion stated in terms of a quantity that does not exist for two thirds
of the suite should not be read as a universal check**, and an earlier attempt to enforce it as a test was
withdrawn for that reason: measuring `tau` inside the episode under judgement
makes the rule circular, since a longer episode sees more of the response,
reports a larger `tau`, and demands a longer episode.

**The unstable CSTR's episode length is set by its setpoint schedule.** Read
with its unstable time constant as tau, the actuator clause asks for 10 tau,
63 to 144 steps, and N = 1200 holds 83 to 189 of them. An episode is six
blocks of 200 steps (10 min), and each block has to hold a switch's settle
with room left to hold after it. The block length is three times the settle
of the cascade's default gains after the worst switch, 62 steps, rounded up
to whole minutes (derived,
`scripts/unstable_cstr_numbers.py --section settle`). The shipped cascade's
slowest settle to 2 % of a switch is 60 steps (3.0 min; derived, same
section), 0.3 of a block. `scripts/measure_hold.py` scores the hold from
step 120 of each block until the controller starts moving toward the next
level, which `target_gym.eval.anticipations` detects, so at most 80 steps of a
block are scored. Over three seeds, 18 blocks, it scored 1440 steps for the
PID and 1141 for the MPC (measured, `hold_steps_scored` in
`src/target_gym/data/hold_measurements.json`). There is no period clause,
since the task is not periodic.

**The compressor's episode clears the actuator clause.** With tau at 49.3
steps, 10 tau is 493 steps, and N = 1200 holds 24.3 tau (derived from the row
above). The episode is four setpoint blocks of 300 steps (30 s, ours), each
6.1 tau long. After a 150-step burn-in, `scripts/measure_hold.py` scores each
block from 100 steps after its change until the controller starts moving
toward the next level. Over three seeds it scored 2250 steps for the PID and
2165 for the MPC (measured, `hold_steps_scored` in
`src/target_gym/data/hold_measurements.json`). There is no period clause,
since the task is not periodic.

**Six benchmark episodes were below the original criterion and were lengthened.** The
environments themselves were fine -- their own defaults are long -- but
`EnvSpec.test_params` overrode them with much shorter ones, a compromise from
when this measurement ran inside CI. It no longer does, so the compromise is no
longer needed.

| environment | was | now | binding criterion |
| --- | --- | --- | --- |
| glass_furnace | 240 (1.8 tau) | 1600 | 12.1 tau |
| hvac | 192 (3.1 tau) | 720 | 11.6 tau |
| cement_kiln | 240 (4.1 tau) | 700 | 12.1 tau |
| plane | 200 (8.7 tau) | 280 | 12.2 tau |
| plane3d_circle | 200 (**0.76 laps**) | 800 | 3.0 laps |
| plane3d_figure8 | 200 (**0.91 laps**) | 800 | 3.6 laps |

The circle is the instructive one: at 14.3 tau it passed the settling test
comfortably and was still being scored over three-quarters of a single lap. A
single criterion would have missed it.

**And then five were shortened again.** The table above records what happened
when the criterion was first applied; it is not the current state. Applying the
same criterion a second time, after the setpoint schedules changed, found the
opposite problem: several episodes had drifted far above what it asks, and
recording them was the dominant cost in the suite. The three-period requirement
was also relaxed to one lap, on the grounds that a controller that flies one lap
on the path will fly the next.

| environment | was | now | binding criterion | recording cost |
| --- | --- | --- | --- | --- |
| plane_energy | 2400 (104.3 tau) | 1200 | 52.2 tau | 14 595 s -> ~7 300 |
| plane_sine | 800 (34.8 tau) | 480 | 2.0 periods | 4 980 s -> ~3 000 |
| plane3d_circle | 800 (3.0 laps) | 300 | 1.14 laps | 2 256 s -> ~850 |
| plane3d_racetrack | 900 (1.5 laps) | 650 | 1.08 laps | 2 010 s -> ~1 450 |
| plane3d_figure8 | 800 (3.6 laps) | 400 | 10.5 tau | 2 205 s -> ~1 100 |

`plane_sine` keeps two periods rather than one because it is a frequency probe:
the first cycle sheds the initial transient and the second is what an amplitude
ratio and a phase lag can be read off. `plane3d_figure8` is bound by settling
rather than by laps, since its 38-step time constant is the longest in the
aircraft family.

The process plants were left alone. They cost 3 to 600 s each to record, so
nothing is bought by trimming them, and the four that look like outliers on the
tau ratio -- the reactor, the boiler drum, the turbine and the battery -- are the
ones whose output answers the actuator within a step or two, where section 6
already says the ratio means nothing and the episode is set by the disturbance
timescale instead.

### A second, separate shortfall: slow dynamics that never move

Settling is about whether a controller can demonstrate holding. A different
question is whether the *slow* dynamics a task advertises actually happen inside
its episode, and for one environment they do not.

Over a full reactor episode the xenon state moves **2.5%** and iodine 1.6%. Xenon
poisoning is the reactor's headline difficulty, and at this episode length it is
effectively a constant bias rather than something to anticipate. Its actuator
response is one step, so no settling criterion catches this.

Fixing it means an episode on the order of a xenon time constant -- 17 hours
against the current 20 minutes, some fifty times longer -- which is a different
and much larger decision than lengthening the six above, with consequences for
what the task is. It is recorded here and left open rather than folded into this
change. Until then, results on the reactor should be read as measuring flux
tracking, not xenon management.

**The distillation column has a milder version of the same gap.** Its benchmark
episode is 200 steps against a 194 min dominant mode, so the slowest dynamics
get through about 63% of one response and never settle. By the actuator-to-output
measure this section uses it is 25 tau, well clear of the floor, and the
registry records the length as a deliberate cost compromise: 41 states with 16
integration substeps make it the slowest environment per step in the suite. But
the difficulty the column advertises is the hidden interior profile acting as
memory, and one time constant exercises that only partly. Read its results as
product-composition tracking under an ill-conditioned plant, which they measure
well, rather than as profile management.

## 7. Hyperparameters, and why they must be tuned

**The PID and the MPC are tuned per environment. If the learned policy is not,
the comparison is rigged in classical control's favour and the result is
worthless.**

This project has hard evidence for how much tuning matters. Re-tuning the
aircraft PIDs was worth between +26% and +139% on held-out seeds. The glass
furnace's gains sat pinned at the edge of their search grid and widening it was
worth 3.5%. A comparison against an untuned opponent measures the tuning, not
the method.

So: a **per-environment, per-algorithm hyperparameter search**, from a single
published search space, with a declared and equal budget (TPE, 64 trials, 3
seeds per trial). The space is fixed across environments so that "tuned" means
the same mechanical procedure everywhere and not the experimenter's taste.

**Tune on seeds disjoint from the ones reported.** The winner of a search is the
maximum of many noisy draws and is biased upward; reporting it on the same seeds
publishes that bias. Search on agent seeds 0–2, report on 10–39. This is the
discipline the aircraft PID tuner already follows — searched on seeds 0–2, quoted
on held-out 3–9 — and the learned side is held to the same rule.

### The search space

Published here so that "tuned" means one mechanical procedure rather than the
experimenter's taste, and so a reader can see what was and was not allowed to
vary. It is identical for every environment.

**Shared**

| parameter | values |
| --- | --- |
| hidden sizes | `(64, 64)`, `(256, 256)`, `(400, 300)` |
| activation | `tanh`, `relu` |
| learning rate | log-uniform `[1e-4, 3e-3]` |
| observation normalisation | **fixed on** — not searched |
| `gamma` | **fixed by the rule in section 6** — not searched |

**SAC**

| parameter | values |
| --- | --- |
| batch size | `128`, `256`, `512` |
| `tau` (target smoothing) | `0.005`, `0.02` |
| target entropy scale | `0.5`, `1.0` (times `-dim(A)`) |
| gradient steps per env step | `0.25`, `0.5`, `1.0` |
| `n_envs` | `16`, `64` |

**PPO**

| parameter | values |
| --- | --- |
| rollout length | `128`, `512` |
| clip range | `0.1`, `0.2`, `0.3` |
| entropy coefficient | log-uniform `[1e-5, 1e-2]` |
| GAE `lambda` | `0.9`, `0.95`, `0.99` |
| epochs per batch | `4`, `10` |
| `n_envs` | `64`, `256` |

Ranges are the conventional ones for continuous control rather than anything
bespoke; the point is that they were chosen before any result existed, not that
they are optimal. Two entries deserve their exclusions explained. `gamma` is
fixed because it sets what the agent is asked to optimise, and tuning the
objective is not tuning the agent. Observation normalisation is fixed because
the alternative is not a worse agent but a different experiment -- see below.

Budget: **64 TPE trials per environment per algorithm**, 3 agent seeds each,
selected on the mean of the trial's seeds. Roughly proportionate to what the
classical side received: the aircraft PID search is a coordinate descent over
seven gains, and the glass furnace's is a 90-point grid.

**Observation normalisation is on, always, and is not a tuned choice.**
Observations span 9e-3 to 8.4e3 across the suite and about four orders of
magnitude *within* single environments — an aircraft reports altitude in
thousands of metres beside angles in radians. Without normalisation the
experiment would partly measure a network's tolerance of unscaled inputs. Every
run uses a running normaliser, and the fact is stated rather than swept into a
hyperparameter table.

## 8. Sample budget, reported as a curve

"Can RL beat a PID" has no answer without "at what cost". A policy that wins
after 50 million environment steps has not answered the industrial question,
because nobody runs 50 million steps on a real furnace.

Train to a fixed cap and report the **learning curve**, with results tabulated at
three budgets: **1e5, 1e6 and 1e7** environment steps. The cheapest column is the
one an engineer reads; the most expensive is the one that says whether the
method can do it at all. A single final number would hide the most
decision-relevant fact in the experiment.

Wall-clock and total environment steps are recorded alongside every result.

## 9. Seeds

Two distinct axes, routinely conflated:

- the **agent seed** — network initialisation, exploration, batch order;
- the **episode seed** — initial condition and target, which is what the shipped
  baselines vary.

Variation is reported over **agent seeds**, at least 30 of them. Each trained
agent is then evaluated on the **same fixed set of episodes** the PID and MPC
were measured on, which is what makes the comparison paired.

Thirty is affordable and ten would not have been: Ajax's own seed-scaling
measurements on `Plane3DCircle` show 15.8 s for one seed against 18.4 s for a
hundred, because fixed compilation overhead amortises. This project has been
misled by two-seed measurements three separate times — the wind turbine at "98%
of the PID", the aircraft at "no crashes", and an original verdict of MPC ahead
on 14 of 16 — each overturned by widening the seed count. Cheap seeds remove the
excuse.

## 10. The expert-based arm

Three structures, answering progressively weaker versions of Q2. The first is
the headline.

**Residual policy (primary).**

    a = clip(PID(obs) + alpha * pi_theta(obs), -1, 1)

The agent learns a bounded correction to the shipped controller. This is the
right primary for four reasons: it answers Q2 literally; it is the form industry
would actually deploy, since deviation from a known-good controller is bounded by
construction; it degrades gracefully, because `pi_theta -> 0` recovers the PID
exactly; and its failure mode is visible rather than silent.

`alpha` is swept over `{0.1, 0.25, 1.0}` and always reported. The sweep is not a
detail — at `alpha = 1` the residual can overwrite the expert entirely and the
method degenerates toward tabula rasa with an unusual prior. Showing the sweep is
what separates "learning improved the PID" from "learning ignored the PID", and
without it the claim is unfalsifiable.

**Expert-guided exploration (secondary).** Expert actions mixed into exploration
on an annealed schedule; Ajax implements this directly. Answers a weaker
question — whether the expert helps the agent *find* a good policy — while
leaving the final policy unconstrained.

**Behavioural cloning then fine-tuning (tertiary).** Pretrain on expert
trajectories, then train normally. The weakest form, because the expert's
influence decays with no guarantee and the result after enough steps is
indistinguishable from tabula rasa.

## 11. Controls

Every environment's table carries two reference rows, so a result is bracketed
rather than floating:

| row | what it establishes |
| --- | --- |
| **PID** | **the peer** |
| MPC | the model-based, full-state upper bound |

These are what `src/target_gym/data/baseline_returns.json` records, ten episode seeds each, and
they are what a learned policy is compared against.

A random-policy floor and a best-constant-action bar were specified here as well
and are not recorded. The constant-action bar is not lost: the conformance suite
already asserts that every PID beats the best constant action, which is the
claim that bar existed to support, and `runners.figure_comparison` draws it when
a figure needs it. A random floor on a setpoint-tracking task is a number
everyone can predict and nobody reads.

## 12. How to read the result

**If RL loses**, the result is on trial rather than the environments, and it has
to survive "your agent was under-trained". Cross-check three or four
environments against stable-baselines3, already a dev dependency with a working
PPO smoke test. If the two agree within noise, the JAX numbers inherit that
credibility; if they do not, that is a finding worth having before publishing.

**If RL wins**, check what it beat. This library has shipped a PID pinned to the
edge of its search grid, and an expert that could not fly a third of its own
task's radius range until it was allowed to trade speed for turn radius — worth
3.5% and 31% respectively once fixed. A win against a defective baseline
measures the defect.

**If RL wins on a partially observed environment**, suspect a simulator exploit
before celebrating. The figure-8's reward once paid for tracking precision its
own error metric could not resolve — an aircraft flying the commanded curve
exactly was scored as 66 m off it. A learned policy is far better than a PID at
finding that kind of seam, and check 11 of the model review checklist exists
because of it.

## 13. What is recorded

Every run is written through `target_gym.rl_results.record_result`, which stamps
the environment fingerprint so a result cannot outlive the environment it
describes. See [docs/rl-baselines.md](rl-baselines.md).

Tags distinguish the arms: `{algorithm}/{tabula|residual-a0.25|guided}/{budget}`.
Learning curves are stored alongside the final returns, since section 8 makes
them part of the result rather than a diagnostic.
