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

A number is only comparable to the reference beside it if it was produced
under the same conditions:

- the same reward. A learner trains on `info["measured_reward"]`, the cost a
  plant could compute from its own sensors, and is scored on the true cost,
  the reward `step` returns, which the score reads (the owner's decision of
  2026-11-05, "learners train on measured reward, and scoring uses the true
  cost"). Training on the true reward would feed the agent readings no plant
  has. No auxiliary shaping; how rewards are scaled is section 6's;
- the scored configuration: every scenario of the task's suite at its weight
  (charter J4), each episode read over the task's scored window with its edge
  charge (`evaluation.episode_cost`);
- the same evaluation episodes, so differences are paired per seed: the scoring
  seeds 200 to 209 in every scenario, on which the reference's anchor line was
  recorded (section 4);
- a deterministic policy at evaluation (the distribution's mean action), because
  the reference is deterministic and a stochastic policy would be scored on a
  different object than it is compared against.

Any deviation makes the row incomparable, and `tests/rl/test_rl_results.py`
rejects the ones that can be checked mechanically.

## 3. The information asymmetry, stated up front

These three controllers do not see the same thing:

| controller | sees | has a model |
| --- | --- | --- |
| reference (charter J1) | observations | no |
| **learned policy** | **observations** | no |
| MPC | **true environment state** | **yes, exact** |

The MPC calls `_extract_x0(state)` and reads state fields directly, and it plans
against the real dynamics. Several environments are deliberately partially
observed — the glass furnace hides 6 of 9 dynamic states, the reactor 7 of 11,
the kiln 64 behind 8 measurements — so on those the MPC is solving a materially
easier problem than the one the agent faces.

Therefore: **the reference is the learned policy's peer, and the MPC is the best
known plan, with more information.** The reference is the ladder's kept rung, a
textbook recipe of PI loops built from bump tests (on the reactor, the published
rod procedure the shipped expert follows), not the shipped expert of every task.
Beating it is the result. Approaching the MPC is
interesting. Beating the MPC would be a claim requiring explanation, not a
victory lap, and the first hypothesis should be a simulator exploit rather than
superior control.

## 4. Metrics

The verdict is the charter's headline (J4), `scoring.headline`, and one
statistic decides it: the weighted-mean saving over the reference. This
protocol and the charter are the same protocol (owner, 2026-11-27, "Fixed seeds,
run-level bootstrap"). Costs are per step, in the task's money, and lower is
better.

**The saving.** Each scenario is a stratum with its weight `w_s`, its share of
operating time. `J = sum over scenarios of w_s times the mean cost in s`, over
the scoring seeds and, for an agent, over its training runs:

    saving = 1 - J_agent / J_reference

0 at the reference, above 0 cheaper, defined while `J_reference` is above 0. A
seed's cost may be negative (an edge credit) and still counts. Means are used,
not trimmed or interquartile means: a bad seed costs real money, and on the
outside review's short PPO runs the interquartile mean of per-seed ratios and
the ratio of means gave opposite signs on pH's bad day. The reference is the
ladder's kept rung on the scoring seeds, the charter record's anchor line
(`charter.load(task)["anchor"]`), not the shipped expert, which costs 2 to
47 % more on four of the five scored tasks.

**An agent is its training runs.** Train 30 runs, agent seeds 10 to 39 (section
9). Choose each run's checkpoint on the validation seeds, then score every run
once on the scoring seeds 200 to 209 of every scenario, with a deterministic
policy. The interval is a two-level bootstrap: resample the runs, then the seeds
within each scenario. Every run and the reference ran the same seeds, so one
draw of seed indices serves them all, and every scenario that shares the seeds;
resampling seeds apart for each run would leave out the spread the runs share,
and declared 29 % false wins in a simulation where the seeds carry the spread.
The percentiles are expanded so that one level reproduces a t interval
(Hesterberg's expanded percentile), since a plain percentile bootstrap
under-covers at ten seeds.

**The verdict.** An agent beats the reference when the 95 % interval of its
saving lies above 0 and it trips, per run on average, no more than the
reference in any scenario. Publish each run's own saving (`run_savings`) beside
the headline, as the 30 points or their quartiles: it shows whether a result is
the method or one lucky run. One policy on its own (a checkpoint, a classical
controller) gets a 95 % t interval on its paired per-seed costs instead,
Fieller's, which excludes a saving of 0 exactly when a paired t test would.

**Seeds are fixed.** Ten scoring seeds per scenario, looked at once. No rule
adds seeds after a look: the stop rule of 2026-11-24 looked up to five times and
declared a false win 10 to 11.5 % of the time against a nominal 2.5 %, and a
percentile bootstrap at ten seeds 5.5 %. With no real saving and a 5 % spread
per seed, one look with the t interval declares a false win 2.5 % of the time
within Monte Carlo error where paired differences are normal, and 2.1 % on the
review's design, whose differences grow with the seed's cost. The two-level
bootstrap stays at 2.1 to 2.5 % whether the runs or the seeds carry the spread
(`tests/rl/test_verdict_statistics.py`).

**Selection never sees the scoring seeds.** Checkpoints, hyperparameters and
early stopping are chosen on the validation block, seeds 300 to 349 in every
scenario (`charter.VALIDATION_SEEDS`). The scoring seeds are run once, at the
end. A result chosen on them carries the bias of its choice.

**Beside the headline**, none of them part of the verdict:

- each scenario's saving and the worst scenario;
- the trip rate per scenario;
- the probability of improvement: the weighted share of seeds where the agent
  costs less than the reference by more than the task's noise floor (below),
  with the share within it reported as `ties`;
- `travel_ratio` and `reversal_ratio`: the agent's full strokes and changes of
  direction per episode over the reference's, from each scenario's per-seed
  `travel` and `reversals` (the anchor line records the reference's). A win
  that wears a valve out twenty times faster is a different claim: the
  review's PPO runs moved pH's valves 10 to 20 times further than the expert.

**The noise floor.** One controller run through two code paths, the host `pid`
slot and the jitted `make_expert`, does not cost exactly the same on every seed.
A seed within that gap is a tie, never a win (`charter.tie_tolerance`), in the
probability of improvement and in `score_policy`'s win rates. Each scored task's
floor is the largest relative gap per seed over the scoring seeds of every
scenario (`scripts/record_charter.py --tasks <task> --noise-floor`, stored in
`data/charter/noise_floor/`):

| task | noise floor | median gap per seed |
| --- | --- | --- |
| reactor | REACTOR_FLOOR | REACTOR_MEDIAN |
| hvac | 1.1e-6 | 6e-8 |
| cstr | 2.0e-3 | 3e-8 |
| compressor_surge | 0.19 | 0.02 |
| ph_neutralization | 2.2e-7 | 4e-8 |

compressor_surge amplifies rounding: one controller through its two code paths
ends up 0.7 to 8 % apart per seed (the median in each scenario), 19 % at most,
and after a small excursion its state never reconverges with an undisturbed run
(section 6). A per-seed win on it means little below that, so read its
probability of improvement with care; its saving, a mean over every seed, is
not biased by it.

**Recording a result.** `rl_results.record_result(..., costs=..., seeds=...)`
keeps each run's cost on each scoring seed of each scenario, stamped with the
charter's fingerprints of the reference it is read against, so a re-record of
the reference marks it stale. The headline reads it against the anchor line:

```python
import numpy as np

from target_gym import charter, scoring

task = "ph_neutralization"
record = charter.load(task)
anchor = record["anchor"]["line"]["scenarios"]  # the reference on seeds 200 to 209
# One row per training run, one cost per scoring seed, in every scenario. Here a
# stand-in: 30 runs, each the reference's costs times a factor of its own.
rng = np.random.default_rng(0)
costs = {
    s: {
        "policy": np.outer(rng.uniform(0.97, 1.01, 30), v["costs"]),
        "reference": v["costs"],
    }
    for s, v in anchor.items()
}
h = scoring.headline(costs, record["weights"], tie=charter.tie_tolerance(task))
low, high = h["saving_ci"]
print(f"saving {h['saving']:.2%}, 95 % interval {low:.2%} to {high:.2%}")
print("each run:", np.round(h["run_savings"], 3))
```

### The tracking numbers, outside the scored suite

`scoring.score_policy` scores a policy against the shipped expert on one
configuration, for the tasks scored on tracking. Its `ratio` (`b / x` per seed,
an interquartile mean) and `win_rate`, the holding score
`S = log(b / x) / log(b / c)` against the ceiling `c` of every tracked quantity
within its instrument's resolution, the room `H = log10(b / c)`, the transient
and settling ratios, the travel, reversals and off-spec share, and
`seed_scores`' clipped log ratio per training seed are described in the module's
docstring. They are reported, never the verdict on a scored task: there the
headline above is the one statistic, and its `run_savings` give one number per
training run to any further test across agents, such as AdaStop. The capability
profile, a radar of both kinds of score per kind of difficulty, is not published
in this release: it was drawn from the tracking score, which the scored tasks
priced in money do not have. It returns when it is rebuilt on the saving over the
reference.

## 5. Algorithms

**Headline: SAC and PPO.** One off-policy and one on-policy, so that a poor
result cannot be pinned on the quirks of a single family. Both are standard,
widely reimplemented, and what a sceptical reader will ask for. SAC is the
default choice for continuous control with dense rewards; PPO is the robust
on-policy reference and benefits most from the massively parallel environments
JAX makes cheap.

**In the same table: an average-reward agent, APO on Ajax's PPO.** A task is a
plant that runs without end, scored on its average cost per step (charter J4),
so the average-reward formulation is the natural fit, and a discount imposes a
horizon the task does not have. The agent runs on the same budget, seeds and
scoring. If the discounted agents fall short, it tells "RL cannot do this" from
"the discounted formulation was the wrong tool". Which agents form the headline
was fixed before any result (decided 2026-11-10,
`docs/proposals/long-strata.md`, S4).

Not used for headline numbers: REDQ, AVG, UDRL, TD3. They are fine algorithms
and each invites "why that one?", which is a question a benchmark should not have
to answer.

## 6. Discount factor and the window T

`gamma` belongs to the learner, not to the task: the task's objective is J4's
average cost per step, whatever the algorithm. SAC and PPO set it by one rule,
fixed before any result (decided 2026-11-10, long strata S4), in control steps:

    gamma = 1 - 1 / (c * T),   c = 10,   T = max(t_s, the longest preview in the observation)

- `t_s` is the task's slow mode: how long the plant takes to forget where it
  started under the reference, the decay time of the extra cost a run from
  another state the plant visits still pays (`scripts/slow_modes.py`, rule R1).
  It replaces the open-loop step response `tau`, which is one step on the
  reactor's rods and misses hidden states such as cstr's composition.
- A preview is what the observation announces ahead: hvac's next-24 h outdoor
  mean, the reactor's planned ramps, up to 10.4 h from the dispatch's notice to
  the end of a day's hold. The horizon covers what the agent is told about.
- `gamma` is never searched. `c = 5` and `c = 20` are run as a check and
  reported beside the headline.
- With `gamma` this close to 1, values are roughly `J / (1 - gamma)`, so
  rewards are scaled, never shaped, the same way for every agent: by a running
  estimate of the return's spread, as the learning check (charter J7) runs.

A driver takes `T` from this table and computes `gamma` from it:

| task | step | t_s (steps) | longest preview (steps) | T (steps) | gamma |
| --- | --- | --- | --- | --- | --- |
REACTOR_ROW
| hvac | 900 s | 69 | 96 | 96 | 0.99896 |
| cstr | 15 s | not finite | none | open | open |
| compressor_surge | 0.1 s | 1599 | none | 1599 | 0.99994 |
| ph_neutralization | 5 s | 197 | none | 197 | 0.99949 |

REACTOR_NOTE cstr's slow mode is not finite: after a 5 % excursion for a
fifth of a window, its pair never reconverges under the reference, in cost or
in state (composition, temperature, flows), over ten windows. Its `T` and
`gamma` stay open until the owner extends the rule to such a plant; a result
on it states the `gamma` it used. compressor_surge's cost forgets its start
after 1599 steps while its state never reconverges, as its noise floor
suggests.

`t_s` here was measured on 2026-11-27, on the plants of `int/amendment3` at
`dceb2537`, with `scripts/slow_modes.py --bias --offset 0.05 --pre 0.2`; a task
version that changes its plant measures it again. The tasks outside the scored
suite have no `T` yet: their slow modes predate their rebuilds.

The window an episode scores is the task's own (charter J4: its evaluation's
burn-in and scored window, or the reactor's 7-day scoring runs), never set
from `gamma`.

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
- the **episode seed** — a scenario's draw: its start, its disturbances and its
  events.

Variation is reported over **agent seeds**: 30 training runs, agent seeds 10 to
39 (0 to 2 tune, section 7). Each is evaluated once on the **same fixed set of
episodes** the reference was recorded on, which is what makes the comparison
paired. Episode seeds come in blocks, the same in every scenario:

| block | episode seeds | read by |
| --- | --- | --- |
| ladder | 100 to 109 | the reference's design and the charter's four lines (J1, J3) |
| scoring | 200 to 209 | the one final scoring run, and the reference's anchor line (J4) |
| validation | 300 to 349 | every choice of checkpoint, hyperparameter or stopping point |

J2's acceptance seeds come from a hash of the commit, above 1000. Training draws
its own episodes, from the stream or from seeds outside these blocks.

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

Every scored task's page carries the charter's four lines (J3), so a result is
bracketed rather than floating:

| line | what it establishes |
| --- | --- |
| best null policy | that the task has a control problem |
| **reference** | **the peer, and the headline's anchor** |
| MPC | the model-based, full-state plan, as context |
| ceiling | the best known plan with foresight, as context |

The reference's anchor line on the scoring seeds, in the task's charter record
(`data/charter/<task>.json`), is what a learned policy is compared against. The
shipped PID and MPC rows in `src/target_gym/data/baseline_returns/` are the
tracking protocol's, ten episode seeds each.

No random-policy floor is recorded: on a setpoint task it is a number everyone
can predict. The conformance suite already asserts that every PID beats the
best constant action.

Null policies are the row below those: what does no real control, such as a
constant action or a static table of the target. If one passes for the expert, the
task has no control problem to learn. `target_gym.null_policies.null_policies(name)`
returns them as policies for `benchmark.run_policy`, and each task's results on the
held-out seeds are in `records.load("credibility")`.

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
describes. A run whose agent used the expert, in training or at deployment,
passes `expert_based=True` to stamp the expert fingerprint too, so retuning the
expert marks it stale. A run on the charter's protocol passes `costs` and `seeds`
too, each run's cost on each scoring seed of each scenario, which stamps the
charter's fingerprints of its reference (section 4). See
[docs/rl-baselines.md](rl-baselines.md).

Tags distinguish the arms: `{algorithm}/{tabula|residual-a0.25|guided}/{budget}`.
Learning curves are stored alongside the final returns, since section 8 makes
them part of the result rather than a diagnostic.
