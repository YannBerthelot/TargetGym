# Task charter

These are the rules every TargetGym task follows, adopted by the owner. They replace
every earlier task rule, and the docs and tests that enforce them change to match. Where
each task stands is in [task-status.md](task-status.md). How the rules came about, and
where each old rule number went, is in [charter-history.md](charter-history.md).

## The claim

TargetGym is a collection of industrial-like control tasks for benchmarking RL agents
against good classical controllers. Its tasks are ones where such a controller still
leaves room, so that we can see what reaches it. Each task stands for a family of
plants, not one plant. The README states the benchmark's one claim:

> Beating the reference on a task means handling the kind of difficulty that plants of
> its family present (its named property, its limits and its operating job) under a
> comparable cost structure. It does not mean an agent would work on any particular plant.

Every design choice is judged against that sentence. A task is realistic enough to carry
it, and no more: slim and principled beats detailed. Fidelity here is similitude. A
wind-tunnel model is trusted because it matches the dimensionless numbers that govern
its effect, not because it looks like the aircraft, and a task is held to the same
standard through its signature (B3).

## Words

- **Family:** the plants a task stands for, such as drum boilers or float-glass furnaces.
  Sources are pooled across the family. **Domain:** a group of families sharing one
  reference recipe: process, thermal, energy, aircraft.
- **Plant record:** the task's `REALITY.md`, built from primary sources (B2).
- **Named property:** what real controllers struggle with on the family: a hidden
  variable, a long delay, drift, instability, coupling, an inverse response,
  nonlinearity, a limit or an economic trade. Complex physics alone doesn't count.
- **Signature:** the quantities that make the family hard to control, each with the
  family's range and the task's value (B3).
- **`ours`:** a number with no source. It is allowed if it lies inside the family's
  documented range and the behaviour matches.
- **Scenario:** one window of operation the sources show (a normal day, a documented
  upset, a load or grade change, a known schedule), with its weight: its share of real
  operating time.
- **Reference:** the controller the domain's recipe builds (J1). It is a textbook
  recipe, not a claim about what plants run.
- **Ceiling:** the best cost an oracle finds with the true model, the true state and the
  future. It cheats on purpose. It is a limit, never a claim that an agent could reach it.
- **Cost:** the score of B5 per step, in the source's units, unavoidable cost included,
  as industry reports its savings. `J` is a task's expected cost per step over its
  weighted scenarios (J4).
- **Saving:** `1 - J_agent / J_reference`: 0 at the reference, above 0 cheaper. It is
  defined only when `J_reference` is above its measurement noise; otherwise the task is
  solved (J5).
- **Room:** the reference's cost divided by the ceiling's. It is reported as unbounded
  when the ceiling's cost is within its measurement noise of 0.
- **Null policy:** the simple policies in `null_policies.py` at the tag: constants, target
  echo, static table, table plus integrator.
- **Trip:** an end of operation, by the plant's protection or by a timed limit's overrun
  where the source ends operation. A seed with a trip counts once; the trip rate is the
  share of seeds with one.
- **Capture:** starting away from the operating point and steering into it.
  **Give-away:** product made better than its spec, which costs money. **Bump test:** a
  small step on one manipulated variable, read open loop for gain, delay and lag.
- **The tag:** the commit the owner cuts to freeze the benchmark. **Noise scale 1:** the
  sensor noise the tasks ship with.
- **Core pool:** the tasks that pass acceptance (J2). **Flagship:** the scored list,
  kept apart from the registry and chosen from the core pool (J5). **Extra:** tasks kept
  in the repository with their reason, never scored.
- **Scope:** one plant, one agent, the plant's operating job.

## Building a task

The steps are listed in the order they are first done; a later step can send a task back
to an earlier one. Each change cites a source, never an earlier version of the task, its
controllers or its results. Earlier numbers are a reference for nothing.

**B1. Name the family and its property.** Say which plants the task stands for and what
their controllers struggle with. Name the structure that makes it, such as a jacket, a
slab or a tower. If that structure can't be built from sources, the task goes to Extra.

**B2. Gather the plant record.** `REALITY.md` is built from primary sources only:
technical documents, patents, papers with plant data, operator training material,
standards, and photos and videos of plants and control rooms. With a source for each, it
states:
- what operators read: the instrument, where it sits, and its delay;
- what they move, and in which operating mode;
- each limit, and its kind (B5);
- the time scales;
- what a normal day and a bad day look like, and how often each happens;
- the operator's job, in one sentence;
- the member's operating state the task represents (for example a point in the fuel
  cycle, a season, a product grade), and for each sourced number the conditions it was
  measured at;
- for every state an agent can drive (a temperature, a pressure, a level), what stops
  it on the real plant, from any subsystem: a relief valve, a trip, a protection, a
  procedure limit.

Where no source is found, the record says "not found". It contains nothing labelled
`ours`.

**B3. Write the property card.** At the top of `REALITY.md`, the card gives the family,
the named property, the structure and the signature. Each signature entry has the
family's pooled range and the task's value. For a new task the ranges are written before
any model; an existing task's card is written from its record. A value is measured on
the plant with no controller where possible, for example by delay over lag, gain
spread, poles and zeros, or what a state kick shows in the readings. Where it can't be,
the factor is switched off and the change in the reference's cost is reported. The
entries are:
- dead time over time constant, per loop;
- the spread of the process gain across the operating range;
- disturbance bandwidth over the reference's closed-loop bandwidth, and the notice
  operators get;
- the distance from the operating point to each limit, in standard deviations of the
  reference's controlled variable, with the limit's kind;
- what is measured, with its delay, noise and sampling, against the loop's time constant;
- the actuators' rate, dead band and range, against what the job asks of them;
- the cost asymmetry: the price of the expensive side over the cheap one.

Entries that depend on the reference are filled once J1 has built it. An entry that does
not apply says why. The behaviour tests (B6) are the card's tests.

**B4. Build a model that lands the signature.** It uses sourced equations and numbers.
An `ours` number lies in the family's range at the member's operating state (B2), not
merely somewhere in the family, and is labelled with that range in `PHYSICS.md`. A
sourced number measured at other conditions than the task's is `ours`, with the source
as one end of its range. Numbers that move together across the family (a reactor's
moderator coefficient with its boron, a boiler's gain with its load) are given as one
joint choice, not as separate ranges. Choosing them that way picks a family member; it
is not tuning. Each stop the record names is modelled, or listed in `PHYSICS.md` as a
known deviation with what keeps the task from exploiting its absence. A limit carried
over from an earlier version is re-checked against its source or a physical bound (a
saturation temperature, a design pressure) whenever the plant is rebuilt. A limit with
a real source uses the source's value. Parts may be added or removed to bring the task
closer to its family. A part stays because a source shows the family behaves that way,
not because a design we copy has it, and a part nobody can tie to a source goes. Design
stops adding detail, and stops simplifying, once every signature entry is in range and
the sweep (B7) holds. Detail that moves neither is left out, even when a reviewer asks
for it.

**B5. Set the job from the record.**
- *Limits, by kind.* A *trip* ends operation, or restarts the plant as the real one
  does, at the sourced cost of the lost production and the restart. A *protected* limit
  is held by the plant's own system, as an aircraft's envelope protection holds it, with
  no trip, and costs only what the source charges for reaching it. A *timed* limit
  allows a set time to restore: time outside is priced, and an overrun has the
  consequence the source names, such as a trip or a forced derate. A *zone* (quality,
  comfort, permit) prices leaving it as the source does. A limit with no source is not
  modelled.
- *Score.* The score is the operator's job: keep inside the bands the plant must keep, at
  the least economic cost the source names (energy, reagent, wear, lost production), in
  the source's units. Tracking a setpoint is scored only where tracking is the real job.
  Precision inside a band earns nothing more unless a source says it pays, as reduced
  give-away does on a product spec. A band the plant keeps by procedure is priced on
  both sides unless a source says one side is free. "Lost production" and the other
  costs set the price of leaving a band, not which side counts. A one-sided price names
  its source. Nothing is added or removed to make a task look harder or easier.
- *Observation.* The agent reads what operators read: the same instruments, with their
  delays, and the same advance notice. A schedule operators know ahead, such as a grid
  load plan, is in the observation.
- *Scenarios.* Each task has a suite of the scenarios its family documents, typically 4
  to 10, each with its weight from the record or `ours` in the family's range. Each
  starts in established operation, not in a capture, unless a source makes capture the
  job. Inside a scenario, targets and disturbances move as the sources show: their
  sizes, rates and frequency, and the quiet stretches between. The seed draws them from
  the family's ranges. Nothing moves that the sources don't show moving. A scenario's
  sources describe plants that fit the member's operating state and equipment; a figure
  from plants run another way (another control mode, actuator or point in the cycle) is
  not a scenario for this member.

**B6. Match it.** Behaviour tests check the family's documented properties against the
record. Each names its source figure or statement, the quantity, and a tolerance written
down before the test runs. Tolerances are family ranges, never one plant's figure, with
one exception: at least one test reproduces a number printed in a source, within that
source's stated accuracy. A failing test is a finding. Its fix is a missing piece of
physics or operation found in a source, never a number moved until the test passes.

**B7. Sweep the family.** The `ours` numbers, scenario weights included, are drawn
jointly across their ranges at the member's operating state, with coupled numbers moved
together: 8 Latin-hypercube draws, on fixed seeds, over the
scenario suite. On every draw, the property's behaviour tests pass, the reference beats
every null policy on the median over seeds, and the ceiling beats the reference, except
on a solved task (J5). Room is reported per draw, as its minimum, median and maximum,
with no gate. A failure inside the range means the model is too thin or the property is
fragile. It is fixed from sources, or by narrowing a range a source supports, never by
tuning. The `ours` numbers and their ranges sit in one machine-readable table per
package, read by `scripts/family_sweep.py`. Where an oracle on all 8 draws costs too
much (J6), it runs on the 2 draws at the ends of the number the room is most sensitive
to, and the page says so. The sweep runs whenever an `ours` number changes, before any
result from the change is reported.

**B8. Cold review.** A reviewer who has not seen the design reads `REALITY.md`, then the
task, and lists every way the task behaves unlike the family. Each item is fixed from a
source, or listed in `PHYSICS.md` as a known deviation with its effect on the score. It
happens before the tag, and for an existing task, before any more plant work.

## Judging a task

**J1. The reference.** Each domain has one written recipe, run as one script at the tag
(`scripts/reference_ladder.py` for the ladder), never tuned on results and never
weakened. Room must not come from a weak controller.
- *Process, thermal, energy:* a fixed ladder of standard industrial loops. First PI
  tuned by SIMC (Skogestad's rule, with the closed-loop time constant equal to the dead
  time). Then steady-state feedforward from the plant's balances. Then a cascade, an
  outer loop setting an inner loop's setpoint on an available measurement. Then a static
  decoupler, a fixed matrix cancelling the loops' cross effects, or an override, a
  second loop that takes the actuator as a limit nears. On nominal seeds 100 to 109,
  each loop is bump-tested at its base point and at the ends of every target and load
  range it visits. It keeps the design asking the smallest controller gain, the one for
  the point where the plant is most sensitive (adopted by the owner, 2026-11-08). The
  script keeps the best rung that trips none of those seeds. Unstable plants use a named
  published rule for the inner loop.
- *Aircraft:* one tune for one airframe, with its values printed: the ArduPlane 4.0.9
  energy (TECS) and path (L1) controllers, with course hold on heading. It flies at the
  same protected limits as the agent, with no extra margin.
- *Published:* a controller the family publishes, such as a reactor's rod procedure, may
  also be a reference. Among the candidates, the one with the lowest cost on the nominal
  seeds is kept, as the ladder keeps its best rung, so the choice can only shrink room.
  A published procedure is a candidate only on the scenarios it is published for.
  Where the family runs a different procedure for some scenarios (slow ramps against
  fast ramps, say), the reference switches between them by a rule written before any
  run.
- *Zones:* on a zone, the setpoint is the edge the economics push toward, moved k
  standard deviations of the controlled variable inside, and never past the zone's
  centre. The deviation is measured on the bump-test seeds in the normal-day scenario.
  k = 2, the usual give-away practice, fixed before any room is seen.

When a reference copied from practice acts on something the score leaves free, or
ignores something it charges, the disagreement is a finding about the score (B5) before
it is one about the reference. The record decides which is wrong; matching one to the
other without it is not allowed.

**J2. Acceptance.** One script runs once from the tag for each task version, at noise
scale 1, on seeds taken from a hash of the commit: 30 seeds, or 5 per scenario if that is
more. Every scenario gets at least 5, and the rest go by weight. A task is accepted if
the reference trips on at most 1 seed in every 30, no null policy beats the reference on
more than half the seeds, and the sweep (B7) holds. Every run goes in an append-only
log, with its trip counts. This is a sanity gate, never a statement about plant
reliability. Other noise levels are reported only.

**J3. Room and context.** Every task's page shows four lines: the best null policy, the
reference, an MPC with the true model and the schedules operators know but no future
disturbances, and the ceiling. The reference is the headline's anchor (J4); the other
three are context, never anchors. Room is published per scenario and on `J`, from an
oracle run on the tagged plant and shown converged. A task whose oracle can't be run says
"not measured". Room is reported, never gated, and nothing here estimates what an agent
could reach.

**J4. The headline.** A task's scenarios are strata, each run on fixed scoring seeds,
their number set at the tag and kept apart from J2's. Its expected cost per step is
`J = sum over scenarios of w_s times the mean cost over that scenario's seeds`, with
`w_s` the scenario's weight. Means are used, not trimmed means, since a bad seed costs
real money. The headline is the saving over the reference, `1 - J_agent / J_reference`,
with a 95 % interval from a bootstrap that resamples seeds within each scenario, keeping
agent and reference paired. Beside it the page publishes the saving in money per hour,
each scenario's saving, the worst scenario, the trip rate per scenario, and the
probability of improvement: the weighted share of seeds where the agent costs less. An
agent beats the reference when the interval lies above 0 and it trips no more than the
reference in any scenario. Across the flagship, the benchmark reports the probability of
improvement and the median of the tasks' savings, with the trip rate beside them. Each
domain weighs equally: a task counts one over the number of flagship tasks in its
domain. Intervals come from the agent's training runs.

**J5. Solved tasks and the flagship.** When the reference costs 0 on most seeds, the task
first searches its family's sources for a cost operators really pay, such as fuel,
energy or wear. If none is found, it stays in the core pool as solved, with no headline,
outside the flagship. The flagship is chosen from the core pool by reading the room
table, and each choice has a written reason.

**J6. Cost.** It is measured by `benchmark_speed`, on CPU at batch 256. No task's step
time may exceed 3 times the median over tasks, and no reset or step may run a solver
loop over 100 iterations. A task over the line looks for a simple way to become much
cheaper. If there is none, it goes to Extra.

**J7. Learning check.** One PPO run per accepted task, with fixed hyperparameters. It is
reported, never gating, to show the task can be learned and to catch simulator exploits.

## Showing a task

**S1. The page.** Each task's docs page carries:
- its property card;
- at least one overlay of the task's behaviour on a published figure from its record;
- the community benchmark it derives from or matches, where one exists (Tennessee
  Eastman, Astrom-Bell, the NREL 5 MW turbine with ROSCO, BOPTEST), and the KPIs they
  share;
- what is not modelled.

**S2. Practitioner review.** A practitioner of the family reviews each task. The owner
arranges the reviews, and the reviewer is acknowledged on the task's page. Until then, the
page says "not yet reviewed by a practitioner".

**S3. The physics page.** Each task's `PHYSICS.md` gives the model, its sources, every
number with its citation or as `ours` with its family range and a one-line reason, the
limits, and the known deviations. A script prints each task's count of `ours` numbers.

## Discipline

**D1. Nothing is tuned on results.** The tasks were designed over many iterations, with
controllers and results in view, and this charter doesn't claim otherwise. Its discipline
starts at the tag: from that commit on, nothing is tuned, and every result comes from it.

**D2. Before the tag,** a design changes to match its family's sources, adding or
removing parts as needed. A redesign from scratch needs the owner's decision. Failing a
test against one plant's figure is never a reason for Extra. Extra holds the tasks
outside the claim, among them a property that is missing or can't be represented (B1),
a task with no industrial family or job, a scope beyond one plant and one agent, and a
task over the cost line (J6).

**D3. At the tag,** the score, the scenario suites with their weights, and every scoring
constant set from oracles, such as a best-known floor a score reads, freeze as plain
task numbers.

**D4. After the tag,** a task that fails acceptance goes to Extra, with no second try. A
kept task changes only for a bug. A change bumps its version, is logged with its reason,
and reruns J2 on new hash seeds.

**D5. No exceptions.** No task gets an exception, an xfail or a rule of its own. A
failure never creates a rule.

**D6. New tasks** supply a `PHYSICS.md`, a `REALITY.md` with its card, a named property,
and a domain recipe or a new recipe script, and pass J2 from their own tag. The same
rules apply to everyone.
