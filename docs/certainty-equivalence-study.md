# Scoping: deterministic, scenario and oracle MPC

A feasibility report for a study separating *the value of accounting for
uncertainty* from *what is left for learning*, by inserting stronger
model-based baselines between deterministic MPC and RL.

Investigation only. Nothing in this report has been implemented, and no
environment, baseline or test was modified to produce it.

## 1. Verdict

**The study is feasible, and the largest obstacle is not the one anticipated.**

The premise that had to hold is that the plants are stochastic. They are. All
four priority environments carry a genuine per-step disturbance process, with
parameters exposed through env params, and the repository has a convention that
makes disturbance realisations reproducible and injectable almost for free. The
oracle arm, which looked like the most awkward piece, is close to a one-line
change on the right planner.

What is *not* true is the assumption that the MPC can be handed the true model.
On three of the four priority environments the shipped MPC is a hand-written
CasADi re-implementation of the dynamics, not the simulator. Any result of the
form "RL beats MPC, and it cannot be model error" does not currently hold on
pH, glass furnace or reactor. Closing that is the main build cost.

Two further things are worth knowing before designing the arms. The existing
JAX planners are **not** certainty-equivalent: they plan against one fixed
pseudo-random disturbance trajectory, which is neither the mean nor the truth.
And the MPC baselines read the **full simulator state** while the PID and any
learned policy see only the observation, which is undocumented and is the most
serious confound found.

## 2. Per-environment table (priority set)

| | pH neutralisation | Cement kiln | Glass furnace | Nuclear reactor |
|---|---|---|---|---|
| Stochastic in `step_env` | Yes | Yes | Yes | Yes |
| Disturbance process | OU on buffer flow `q2` | OU on raw-meal feed | AR(1) on pull rate | OU on **demand setpoint** |
| Declared in registry | `('q2',)` | `('raw_meal',)` | `('m_pull_disturbance',)` | **`()` — none** |
| Disturbance observed? | No, hidden | **Yes**, `raw_meal` in obs | No, hidden | Target is observed |
| MPC implementation | `PHCasadiMPC` | `SamplingMPC` (CEM) | `GlassFurnaceCasadiMPC` | `ReactorCasadiMPC` |
| Uses the true model? | **No**, do-mpc model | **Yes**, `env.step_env` | **No**, do-mpc model | **No**, do-mpc model |
| Planner's disturbance | `q2 = q2_nominal` | fixed `PRNGKey(0)` scenario | nominal + bias term | nominal |
| MPC cost = env reward? | **No**, quadratic proxy | Yes, custom objective | No | No |
| Reward symmetric in error | Yes | Yes | Yes | Yes |
| Reward quadratic | **No**, log-scaled | **No**, log-scaled | **No** | **No** |
| Constraint handling | Termination only | Termination only | Termination only | Termination only |
| Obs includes action history | **No** | **No** | **No** | **No** |
| Obs dims / hidden dims | 3 / buffer + invariants | 8 / `n_zones x 4` profile | 5 / 6 of 9 states | 4 / 9 |

## 3. Findings in detail

### 3.1 Stochasticity is present, and follows a house convention

Every priority plant draws per-step noise the same way:

```text
noise = jax.random.normal(jax.random.fold_in(key, state.time))
```

This convention exists for a reason that matters to this study, and it is
enforced by `test_disturbance_magnitude_is_independent_of_key_splitting` in the
conformance suite. Every rollout helper in the repository drives `step_env`
with the *same* key at every step. An environment drawing noise directly from
that key would redraw one identical innovation forever, collapsing a zero-mean
process into a deterministic ramp. Folding in `state.time` fixes that.

The side effect is the thing to exploit: **the disturbance realisation is a
pure function of `(key, t)`, independent of the actions taken.** That is
exactly the property the oracle arm needs, and it is already there.

Structure per environment:

- **pH**: Ornstein-Uhlenbeck on `q2`, the buffer flow, reverting to
  `q2_nominal`, driven by `q2_noise_std` and `BUFFER_OU_THETA`, clipped to
  `[q2_min, q2_max]`. It is carried in the state, so it evolves *within* an
  episode as well as being sampled at reset. It is genuinely unmeasured: the
  observation is `[pH, q3_pct, target_pH]`.
- **Cement kiln**: OU on `raw_meal`, reverting to nominal, clipped to ±50%.
  Note it **is** observed (the weighfeeder reading is in the observation).
- **Glass furnace**: AR(1) on `m_pull_disturbance` with `M_PULL_AR_RHO`,
  additive on the pull rate, hidden.
- **Reactor**: OU on `target_n`. This is a drifting **reference**, not a plant
  disturbance, and it is keyed off `state.demand_key` rather than the passed
  key.

Three corrections to the brief follow from this:

1. The plane's OU machinery is **not** needed and is welded to the aircraft
   state anyway. Each plant already carries its own disturbance. There is no
   porting job.
2. The reactor's disturbance is on the setpoint, not the plant. The four
   certainty-equivalence mechanisms still apply to the tracking-error dynamics,
   but "revealing the disturbance" means revealing the *future demand
   trajectory*, which is a different and arguably easier oracle.
3. The reactor declares `disturbance_fields=()`, so the conformance suite's two
   disturbance tests skip it despite it having an OU process. This is
   **deliberate and documented**: the `EnvSpec` docstring defines the field as
   holding a *zero-mean stochastic disturbance* and says it excludes
   "deliberately-drifting processes such as the reactor's OU demand". The
   reactor's `target_n` is a drifting setpoint, not a plant disturbance, and it
   is centred on the middle of its range rather than on zero, so it does not
   meet the field's definition. Declaring it would make the tests pass (checked:
   the constant-versus-split-key RMS ratio is exactly 1.00, because the process
   keys off `state.demand_key` rather than the passed key) at the cost of
   overloading what the field means. The residual point stands and belongs on
   the roadmap rather than in the registry: **the reactor's stochastic process
   is currently asserted by nothing.**

### 3.2 The three MPC implementations, and which model each plans with

| Implementation | Count | Internal model | Vmappable |
|---|---|---|---|
| `CasadiMPC` subclasses | 7 | Hand-written do-mpc symbolic model | No (IPOPT, sequential) |
| `GradientMPC` | 12 | **The true simulator**, `env.step_env` | Yes |
| `SamplingMPC` (CEM) | 1 | **The true simulator**, `env.step_env` | Yes, already vmapped |

(`docs/baselines.md` says `GradientMPC` covers 8 environments. That is stale;
it is 12 since the four moving-setpoint aircraft tasks were added.)

The model-fidelity split is the central design constraint. `PHCasadiMPC`
rebuilds the reaction-invariant ODEs symbolically and states plainly:

```text
q2 = p.q2_nominal  # unmeasured; nominal in the model
```

That is certainty equivalence, implemented, in one line. As a *deterministic
MPC arm* this is ideal and needs no work. As a *true-model planner* it is
disqualified, and worse, it carries model error in an unquantified direction:
different integrator, algebraic pH via a charge-balance constraint rather than
the simulator's bisection solve, and nominal buffer flow.

The JAX planners are subtler and, I think, mischaracterised in the brief. Both
`GradientMPC._rollout` and `SamplingMPC._score` set `key = jax.random.PRNGKey(0)`
and step the true environment. Because noise is `fold_in(key, state.time)` and
`state.time` advances through the plan, the planner does **not** see the mean
disturbance. It sees one specific, fixed pseudo-random trajectory, consistent
across the episode and unrelated to the realisation the environment will
produce. So the shipped JAX "MPC" is neither certainty-equivalent nor robust:
it is single-scenario-with-the-wrong-scenario. Any published deterministic-MPC
arm should either fix this or state it.

Cost functions: the CasADi controllers use a quadratic proxy, not the
environment reward, and the code explains why. The reward is log-scaled and
clipped, so it is flat once the error leaves the tracking band, and IPOPT had
no gradient to descend; on pH it optimised the only live term, the reagent
cost, railed the valve shut and sat at ~3.9 pH mean error. The proxy shares a
minimiser and has a usable gradient. This is a defensible engineering choice
and simultaneously a confound for any published comparison, since it is exactly
mechanism 2 operating inside the baseline.

Constraints: there are none in the optimisation. Every priority environment
handles its irrecoverable states purely through termination, and there is no
explicit crash penalty anywhere. `docs/reward-shaping.md` records the reasoning:
rewards are non-negative, so termination already costs the agent every step it
would have earned, and a flat penalty bought nothing that forgone reward did
not. The consequence for this study is that mechanism 1 exists only in the weak
form "a trajectory that touches the boundary ends the episode", and there is no
constraint the planner is even aware of. Neither the CasADi nor the sampling
planner has a state constraint declared.

Tuning and caching: MPC hyperparameters are set per environment in the factory
functions with the reasoning in the docstring, not searched. The caching the
README mentions is two separate things, neither of which is MPC solution
caching: an XLA persistent compilation cache in CI, and
`src/target_gym/data/baseline_returns.json`, which stores measured returns so that the
MPC-versus-PID contract is asserted from a recorded number rather than
reproduced on every test run, guarded by a source fingerprint.

### 3.3 Scenario and oracle feasibility

**The oracle arm is nearly free on the JAX planners.** The evaluation harness
calls `jax.jit(env.step_env)` directly with a constant `jax.random.PRNGKey(seed)`,
bypassing gymnax's `Environment.step` (which would split the key). So during
evaluation the disturbance is exactly `normal(fold_in(PRNGKey(seed), t))`.
Passing that same key into `SamplingMPC._score` instead of `PRNGKey(0)` makes
the planner simulate the exact realisation the environment will deliver. That
is the oracle, and it is a constructor argument plus one substitution.

One trap: because the reactor keys its demand off `state.demand_key`, which is
carried in the state and therefore copied into any planner rollout, **a
sampling planner on the reactor is automatically a demand oracle** whether you
intended it or not. Any reactor arm needs this handled explicitly.

**The scenario arm is cheap on `SamplingMPC` and awkward on CasADi.**
`SamplingMPC` already does `jax.vmap(self._score, in_axes=(0, None))` over
action samples. Adding a scenario axis is a second vmap over K keys and a mean,
maybe ten lines. Cost multiplies the rollout count by K but is parallel, so
wall clock grows far slower than K until memory saturates.

For CasADi, do-mpc has native multi-stage robust MPC via `n_robust`, and every
controller here sets `n_robust=0`. Declaring the disturbance as an uncertain
parameter and calling `set_uncertainty_values` builds a scenario tree
automatically. Two caveats: the tree is over parameters held *constant per
branch*, not over sampled disturbance trajectories, so it is robust-to-
parametric-uncertainty rather than the scenario MPC described in the brief; and
the NLP grows as `k^n_robust`.

**Cost estimates.** From the recorded per-environment timings in
`src/target_gym/data/baseline_returns.json` (ten seeds, full episodes):

| env | steps/episode | 10 seeds, MPC + PID | implied per MPC solve |
|---|---|---|---|
| pH | 300 | 23 s | ~7 ms |
| cement kiln | 700 | 196 s | ~28 ms |
| glass furnace | 1600 | 1356 s | ~85 ms |
| reactor | 1200 | 670 s | ~56 ms |

For the sampling planner these are the numbers that scale with K. Cement kiln
at K=10 is roughly 30 minutes for ten seeds, at K=30 roughly 90 minutes, before
any vmap parallelism is counted, so realistically less. This is affordable.
The furnace and reactor would need true-model planners built first, at which
point their cost is set by the new planner, not the CasADi figure above.

### 3.4 Cost asymmetry: the framing needs adjusting

Every priority environment uses `log_scaled_reward(|error|, ...)`, multiplied
by an actuator-cost factor of the form `(1 - w * usage)`. The house convention,
documented in `docs/reward-shaping.md`, is that costs *multiply* rather than
subtract.

Consequences:

- **All four rewards are symmetric in the error.** The cement kiln's free lime,
  which the brief nominates as the obvious asymmetric case, is
  `log_scaled_reward(|discharge_lime - target|)`. There is no asymmetry
  anywhere in the priority set, and I did not find one elsewhere either.
- **But mechanism 2 is still live everywhere**, because the condition for
  `E[l(y)] != l(E[y])` is that `l` is *non-quadratic*, not that it is
  asymmetric. Log-scaled tracking multiplied by a cost factor is emphatically
  non-quadratic. The brief conflates the two slightly; asymmetry would sharpen
  the test and make the direction of the bias interpretable, but it is not
  required to observe the effect.

So mechanism 2 can be studied today. If you want a clean, interpretable
asymmetry, it has to be added, and under this repository's conventions that
means justifying it in the environment's `PHYSICS.md` against the standard in
`docs/PHYSICS_METHODOLOGY.md`. Free lime is the defensible candidate: high free
lime is unsound cement and is a quality rejection, low free lime is merely
overburnt and wastes fuel. That is a real process asymmetry, not a modelling
convenience, so it would pass review. It is still a physics change and would
invalidate every recorded baseline.

### 3.5 Observations and the action queue

**Confirmed: no environment's observation contains any history of past
actions.** Each includes the *current* actuator positions (pH: valve percent;
kiln: fuel and speed percent; furnace: fuel percent; reactor: normalised
external reactivity), which is the actuator's present state, not the in-flight
queue.

The hypothesis in the brief is correct and the docstrings say so explicitly.
The cement kiln's observation is eight numbers against an `n_zones x 4` axial
state, and the docstring notes that kiln operators really do run the process on
a few readings, "which is a large part of why the job is famously hard". The
glass furnace hides six of nine dynamic states. pH's docstring states outright
that the same pH can arise from different `(Wa, Wb)` pairs whose local process
gain differs by an order of magnitude, making it a genuine POMDP.

There is **no** frame-stacking or observation-augmentation wrapper. The only
observation toggle in the repository is the aircraft's `observe_wind`, which is
constructor-level and aircraft-specific. Building an augmentation wrapper is
new work, though small.

**The confound: the MPC baselines do not use the observation at all.** Every
planner's `step` signature is `step(self, _obs, state)` with `_obs` explicitly
ignored, and `runners.mpc_policy` passes the full simulator state.
`CasadiMPC._extract_x0(state)` reads hidden state directly. So on pH the MPC
reads `Wa` and `Wb`; on the furnace it reads the glass and checker
temperatures; on the reactor it reads the xenon and iodine estimates and the
fuel temperature. The PID sees only the observation, and any learned policy
would too.

I found no acknowledgement of this anywhere in `docs/baselines.md` or
`docs/rl-protocol.md`. It is the most serious confound in the current
comparison: the published MPC-versus-PID table compares a full-state controller
against an output-feedback one and reports the difference as a controller-class
result.

### 3.6 Evaluation harness

`scripts/record_baselines.py` is the whole story. Ten seeds, `range(10)`, the
same seeds for PID and MPC, so **paired comparison is already possible**. Per
environment it stores both ten-element return vectors, an
`mpc_terminated_early` count, the episode length and a source fingerprint.

What exists that is useful here: the per-seed returns are kept, not just the
mean, so a distribution over ten samples is available; and
`mpc_terminated_early` is a constraint-violation count in the only sense the
environments define constraints.

What does not exist: IAE, overshoot, settling time, per-step constraint
violation rate, or any quantile or tail statistic. `runners.rollout` returns
tracked values, targets and rewards, so IAE and overshoot are derivable without
touching the environments, but nothing computes them. Ten seeds is also thin
for tail metrics; the recorded counts are integers out of ten.

## 4. What would need building, by effort

1. **Distributional metrics and a paired evaluation harness.** Smallest and
   needed by every arm. `rollout` already returns what IAE, overshoot and
   settling time need. Add violation rate and quantiles, and raise the seed
   count for the study runs, which is a parameter not a code change. Half a day.
2. **Oracle arm on the sampling planner.** A key argument threaded into
   `SamplingMPC._score` plus a runner that passes the evaluation seed. Hours,
   given the `fold_in(key, t)` convention. Must handle the reactor's
   state-carried demand key explicitly.
3. **Scenario arm on the sampling planner.** A vmap over K keys and a mean, plus
   a decision about whether to average the objective or use a risk measure.
   Around a day, plus compute.
4. **True-model planners for pH, glass furnace and reactor.** The real cost. A
   `SamplingMPC` instance needs only an objective, so this is mostly tuning
   horizon, sample count and elite fraction per plant, then demonstrating the
   result is a credible expert rather than a strawman. Gradient MPC is the
   cheaper option where it works, but pH's bisection solve and the furnace's
   implicit gas-temperature solve are both gradient risks, and the kiln's
   adjoint is already documented as overflowing to NaN after eight steps.
   Several days per plant, dominated by convincing yourself the planner is good.
5. **do-mpc scenario arm.** Optional, and only if you want the comparison to
   include the shipped CasADi controllers rather than replacing them.
   `n_robust` plus uncertain parameters. A day per environment, with the caveat
   that it is a different formulation from the sampled-trajectory one.
6. **An asymmetric reward variant.** Only if you want mechanism 2 isolated
   cleanly rather than merely present. Physics justification, `PHYSICS.md`
   update, model review checklist, new environment version, full baseline
   re-record. Days, and it perturbs the release.

## 5. Confounds in the existing comparison

Ordered by how much they would undermine a published result.

1. **The MPC reads the full simulator state; the PID and any policy read the
   observation.** Undocumented. On environments explicitly designed as POMDPs,
   with six or nine hidden dimensions, this is not a small effect. Any
   "MPC beats PID" or "RL beats MPC" claim inherits it.
2. **On three of four priority environments the MPC plans with a different
   model than the simulator.** Model error is unquantified and its direction is
   unknown, so it can be used to explain away any result in either direction.
3. **The JAX planners plan against a fixed, wrong disturbance scenario**, not
   the mean and not the truth. They are mislabelled as deterministic MPC.
4. **The CasADi cost is a quadratic proxy for a log-scaled reward.** Documented
   and well-motivated, and still mechanism 2 operating inside the baseline. The
   MPC is not optimising the quantity it is scored on.
5. **The reactor's stochastic demand is asserted by nothing.** Its exclusion
   from `disturbance_fields` is deliberate and correct on that field's own
   definition, so the fix is a separate check rather than a registry edit.
6. **Ten seeds.** Adequate for a mean, thin for the tails this study is about.
7. **Fingerprint blast radius.** `provenance.baseline_fingerprint` hashes
   `experts/pid.py` and `experts/mpc.py` whole, so adding a scenario planner
   invalidates all twenty recorded baselines and forces a full re-record, which
   is a nine to eleven hour job. (Since then each task's oracle moved to its
   own package, so a planner added for one task re-records only that
   package.) The roadmap already carries an item to scope
   this per environment. Doing that item *before* this study is a direct saving.

## 6. Which environment to prototype on

**Build the machinery on the cement kiln. Test the hypothesis on pH.**

The brief nominates pH, and for the hypothesis that is right: it is the
environment where the prediction is a genuine RL win, its POMDP structure is
real and documented rather than asserted, its buffer disturbance is unmeasured,
and it is the cheapest to run at roughly 7 ms per solve.

But pH is a poor place to *build*, because it has no true-model planner today
and its bisection pH solve makes the gradient route risky. You would be
debugging a new planner and a new study harness simultaneously.

The cement kiln is the only environment in the repository that already has a
true-model, gradient-free, fully vmapped planner. Scenario and oracle arms can
be built and validated there against an existing, tuned, working controller,
with no new planner to justify. Once the machinery is trusted, porting it to pH
is a `SamplingMPC` instance plus tuning.

The kiln also has an honest weakness worth knowing in advance: its disturbance
is *observed*, so the dual-effect mechanism is largely absent there. That makes
it a good machinery testbed and a poor hypothesis testbed, which is exactly the
split proposed.

## 7. Where the framing needs correcting

- **The plants are not deterministic.** All four priority environments have
  per-step stochastic disturbances with configurable parameters. The contingency
  plan in the brief is not needed.
- **"Give the MPC the true model" is not the status quo**, it is a build item,
  on three of the four priority environments.
- **The existing MPCs are not deterministic MPC** in the certainty-equivalence
  sense, except the CasADi ones. The JAX planners use one fixed sampled
  scenario.
- **Mechanism 2 needs non-quadratic, not asymmetric.** It is testable now. No
  environment has an asymmetric reward, including the cement kiln's free lime.
- **Mechanism 1 is the weakest of the four here.** No planner has a state
  constraint; irrecoverable states are handled purely by episode termination
  with no explicit penalty. To study constraint violation properly you would be
  adding constraints, not just measuring them.
- **The oracle is cheaper than expected**, because disturbances are a pure
  function of key and time by convention, and the evaluation harness uses one
  fixed key per episode.
- **The reactor is a different problem from the other three.** Its stochasticity
  is a drifting setpoint, not a plant disturbance, and any planner that rolls out
  `step_env` gets its demand oracle for free. The registry already draws this
  distinction deliberately, which is worth respecting rather than flattening.

## 8. Method note

Nothing was modified. I deliberately did not run the test suite: a full baseline
re-record was in progress on this machine while this was written, and the fast
suite would have competed with it for cores. Every claim above is from reading
source, from `src/target_gym/data/baseline_returns.json`, or from short read-only introspection
of the registry and the do-mpc version.
