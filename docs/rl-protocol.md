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
- the same episode length, the environment's scored configuration
  (`env.default_params`);
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

Three questions, three numbers, all from `target_gym.scoring`, whose
`score_policy` runs a policy and the expert on the same seeds. Costs are per
step and lower is better: `x` is the policy's, `b` the expert's (today the
shipped PID) and `c` the ceiling's.

**Against the expert: `b / x`** per seed, on identical evaluation episodes,
with the share of seeds the policy wins. Pairing removes the variance from
initial conditions and targets, which on these tasks is large: the circle
task's radius alone moved the PID's return by a factor of two. It covers the
whole scored episode, consumption and target changes included, so it is the
verdict.

**Across environments: the score `S = log(b / x) / log(b / c)`**, on the
tracking cost while holding. It is 0 at the expert, 1 at the ceiling and never
above, and negative below the expert. The ceiling is every tracked quantity
within its instrument's resolution, at the reward's price: as good as a real
plant can verify. Being a ratio of logarithms, `S` has no units and does not
change with the tracking exponent, so a plant priced in euros compares with
one scored in floor-widths. A linear share of the cost gap fails both tests:
costs span orders of magnitude between controllers, so it put almost any
working policy near 1.

**Room to grow: `H = log10(b / c)`**, the decades between the expert's hold and
the ceiling. 0 means the expert already tracks within the resolution, and the
task has nothing left to score.

The hold starts after a settle fixed per plant, so every controller is scored
over the same window. Beside the score, `transient_ratio` compares the cost
after each target change with the expert's, and `consumption_ratio` compares
the fuel, energy or wear, which no resolution bounds.

**Report the interquartile mean with a 95% stratified bootstrap confidence
interval**, plus the win rate against the expert. Not the mean alone. This is
standard practice after Agarwal et al. (2021), and this repository has its own
evidence for it: an earlier battery MPC scored `+14.0` on the mean and `−4.1`
on the median, winning 1 seed in 10. A mean alone would have published the
opposite of what happened.

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

A rule based on the plant's own time constant fails here: the reactor's flux
answers the control rod in one step, so five time constants would be a
five-step horizon on a task held for 864 steps. The wind turbine and battery
are the same shape.

**The rule is the episode:**

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
| reactor | 864 | 1 | 5 | 0.99884 |
| unstable_cstr | 1200 | 6.3 to 14.4 | 32 to 72 | 0.99917 |
| compressor_surge | 1200 | 49.3 | 247 | 0.99917 |
| formation | 200 | 36 | 180 | 0.99500 |

Two rows are filled by hand. `unstable_cstr` has no step response to fit (it
runs away open loop), so its tau is the unstable time constant, the time an
uncontrolled error takes to grow e-fold over its five targets
(`scripts/unstable_cstr_numbers.py --section targets`). `compressor_surge`'s
is the slowest header-pressure response to its actuator steps
(`scripts/compressor_surge_numbers.py --section steady`).

### How long an episode has to be

Two time scales bind, and an episode has to clear both.

**Settling.** This library is about reaching a target *and holding it*, so
for holding to be what the score measures, the episode needs room for both:
**a floor of 10 tau**, roughly five to arrive and five to hold.

**Period.** For a path-following or periodic task the binding scale is the
task's own period, and the requirement is **one lap**. These tasks start the
aircraft on the path, so an episode shorter than a lap cannot tell a controller
that holds the path from one that flies straight ahead. `plane_sine` keeps two
periods, because it is a frequency probe: the first sheds the initial
transient.

    N >= max(10 * tau_actuator, 1 * T_period)

**The actuator clause applies only where a settling time exists**, which is
about a third of the suite. Nine plants integrate (a tank level, a drum level,
an aircraft altitude under a held stick), the aircraft oscillate in the
phugoid, the glass furnace does not settle on the episode timescale, and the
unstable CSTR runs away. For those the episode is set by laps, by the
disturbance timescale (reactor, turbine, battery, boiler drum) or by the
setpoint schedule (furnace, building, kiln, unstable CSTR). So the
criterion is a design rule, not a test: measuring tau inside the episode under
judgement would make it circular.

### A second, separate shortfall: slow dynamics that never move

Settling asks whether a controller can demonstrate holding. A separate question
is whether the slow dynamics a task advertises happen inside its episode.

- **Reactor.** The scored episode is 2.4 h, about 0.18 xenon time constants:
  the poison moves materially while the controller works, but the full
  xenon pit (about ten hours) is out of reach. Read reactor results as flux
  tracking with partial xenon management.
- **Distillation.** The 200-step episode covers about 63% of the column's
  194-minute dominant mode, a deliberate cost compromise for the slowest
  environment per step. Read its results as product-composition tracking on an
  ill-conditioned plant, not as profile management.

## 7. Hyperparameters, and why they must be tuned

**The PID and the MPC are tuned per environment. If the learned policy is not,
the comparison is rigged in classical control's favour and the result is
worthless.**

Re-tuning the aircraft PIDs was worth 26% to 139% on held-out seeds, so a
comparison against an untuned opponent measures the tuning, not the method.

So: a **per-environment, per-algorithm hyperparameter search**, from a single
published search space, with a declared and equal budget (TPE, 64 trials, 3
seeds per trial). The space is fixed across environments so that "tuned" means
the same mechanical procedure everywhere and not the experimenter's taste.

**Tune on seeds disjoint from the ones reported.** The winner of a search is the
maximum of many noisy draws and is biased upward; reporting it on the same seeds
publishes that bias. Search on agent seeds 0–2, report on 10–39.

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

Thirty is affordable: in JAX, compilation dominates, so a hundred seeds cost
little more than one (15.8 s against 18.4 s on `Plane3DCircle` in Ajax). Small
seed counts have misled this project three times, each overturned by widening
the count.

## 10. The expert-based arm

Three structures, answering progressively weaker versions of Q2. The first is
the headline.

**Residual policy (primary).**

    a = clip(PID(obs) + alpha * (high - low) / 2 * pi_theta(obs), low, high)

The agent learns a bounded correction to the shipped controller: `pi_theta`
outputs values in [-1, 1], and `alpha` is the correction's share of the action
range. This is the
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

No random-policy floor is recorded: on a setpoint task it is a number everyone
can predict. The conformance suite already asserts that every PID beats the
best constant action.

## 12. How to read the result

**If RL loses**, the result is on trial rather than the environments, and it has
to survive "your agent was under-trained". Cross-check three or four
environments against stable-baselines3, the opt-in `agent` group
(`uv sync --group agent`) with a working PPO smoke test. If the two agree within noise, the JAX numbers inherit that
credibility; if they do not, that is a finding worth having before publishing.

**If RL wins**, check what it beat. This library has shipped a PID pinned to the
edge of its search grid, and one that could not fly a third of its task's radius
range. A win against a defective baseline measures the defect.

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
