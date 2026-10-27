# Target MDPs

!!! note "Status"

    This page states the definition the suite is built on. It generalises the
    Setpoint MDP of the author's thesis (Berthelot, 2026, Chapter 8,
    Definitions 8.1 to 8.4) from a single state coordinate to a set, and it
    adds the two conditions a benchmark task needs that a theory chapter does
    not: an admissibility criterion on the target and a feasibility condition on
    the plant. The thesis remains the reference for the theorems.

## The setting

Most RL benchmarks pose episodic goal-reaching: get somewhere, collect a payoff,
stop. Industrial control almost never looks like that. A furnace does not
finish. A drum level does not get solved. The task is to bring a plant to a
condition and keep it there while the world pushes back, and the episode ends
because the clock ran out rather than because anything was achieved.

A **target MDP** formalises that. Optimality is not a trajectory to a goal
state. It is a *set* of states on which the reward is maximal, and the reward
decreases as the state moves away from that set.

## Definition

Start from an MDP with state $s \in S$, action $a \in A$ and transition
$s' = f(s, a, w)$, where $w$ is a disturbance (a gust, a load change, noise on a
demand signal). An observation $o = g(s)$ may hide part of the state.

A target MDP adds four things.

- An **error map** $e : O \to \mathbb{R}^k$. The error is a function of the
  observation, not of the state. This is the whole content of the admissibility
  criterion below.
- The **target set** $G = \{ s \in S : e(g(s)) = 0 \}$.
- A **tracking cost** $\ell : \mathbb{R}^k \to [0, \infty)$ with $\ell(0) = 0$,
  non-decreasing in every $|e_i|$ and strictly increasing outside a tolerance
  band (none, on most plants). Formally, if $|e_i| \le |e'_i|$ for every $i$,
  then $\ell(e) \le \ell(e')$, strictly when some $|e'_i|$ is larger and lies
  outside the band.
- A **running cost** $c : S \to [0, \infty)$, what the plant consumes (fuel,
  reagent, wear) beyond what holding the target needs.

The per-step reward is

$$
r(s) = -\Big(\ell\big(e(g(s))\big) + c(s)\Big),
$$

minus a one-off **trip cost** on a step that leaves the operating envelope (a
crash, a tank overflow, a battery at its charge limit). The plant then restarts,
so episodes end on a time limit only, and never because the target was reached.
The [reward shaping](reward-shaping.md) page gives the suite's choice of each
term. The natural optimality criterion is the long-run average reward, the gain,
with the bias breaking ties among gain-optimal policies. The thesis works in
exactly that setting.

### How this relates to the thesis

Definition 8.1 of the thesis asks for one state coordinate $i$, a
pseudo-distance $d$ and a setpoint $x$ such that a larger $d(s_i, x)$ always
means a smaller reward. That is the case $k = 1$, $e(o) = o_i - x$,
$c \equiv 0$ of the definition above. Definition 8.2 defines the target states
as the states of maximal reward, $S_{\text{target}} = \{ s : r(s) = r^* \}$.
With no running cost and no tolerance band that set is exactly $G$.

The suite needs the set form because its targets are not single coordinates. A
holding pattern is a curve in the horizontal plane, a formation slot is a point
in another aircraft's body frame, the boiler drum holds a level and a pressure
at once. It needs the cost term because several plants pay for fuel, reagent or
degradation, and those payments are part of the reward.

### Monotonicity and what the cost does to it

The tracking cost alone is monotone in the error, and every sublevel set of the
error is a sublevel set of the tracking cost. The thesis's
$\varepsilon$-target, the states within $\varepsilon$ of the maximum, is the
set $\{ s : \ell(e) \le \varepsilon \}$. That is the object the
[reward shaping](reward-shaping.md) page is about when it distinguishes a
floor from a tolerance band: scaling the error by the plant's achievable floor
keeps the sublevel sets nested and strictly ordered all the way down, while a
band collapses them below its width.

The running cost is added to the tracking cost, so a state with a smaller error
can still score worse if it consumes more. That trade is what a running cost is
for. In particular the states of maximal reward need not lie exactly on $G$, and
the maximum $r^* = 0$ is attained only if some state has zero error and consumes
nothing avoidable. No real plant holds its target with zero error, so the suite
treats the **ceiling** $\bar r = \sup_\pi \rho^\pi$, the best achievable gain,
as a property of each environment rather than assuming it equals 0.

## The admissibility criterion

Not every subset of $S$ makes a usable target. Two conditions are required, and
the second has a strong and a weak form.

**A1. The target set is fixed.** $G$ is a fixed subset of $S$, defined by a
function the agent is given. It does not depend on time, on the policy, or on
anything outside the state. Because the state includes whatever the reference
needs (a commanded altitude, a dispatch request, the lead aircraft), $G$ stays
fixed even when the setpoint moves. The setpoint is a coordinate of $s$, and
$G$ is the diagonal where the plant's output equals it.

**A2. The error is measurable.** In the strong form, $e$ is a function of the
current observation: the agent can compute its own reward at every step from
what it sees. In the weak form, $e$ is a function of the observation history: a
short memory recovers it. Every environment in this suite satisfies the strong
form except `patrol_bearing_only`, which is the weak case by design and is
labelled as such on its page. What is excluded in both forms is a target the
agent can never recover from its observations, because that turns the task into
inference of the objective, which is a different research question.

Two things follow from A1 and A2 together.

**Disturbances move the state relative to $G$, never $G$ itself.** The
disturbance $w$ enters the transition $f$. It does not enter $e$ or $G$. A gust
displaces the aircraft, a load change displaces the drum level, a fresh dispatch
request displaces the setpoint coordinate. In every case the set the agent is
scored against is the same set it was scored against a step earlier. This is
what "disturbance invariant" means on this page.

**Partial observability of the plant is allowed.** The rule is one-sided: the
error and the target are observable, how to achieve them need not be. The glass
furnace hides six of nine states, the kiln sixty-four behind eight measurements,
the battery hides its diffusion voltage and its accumulated fade. The thesis
lists partial observability and actuation delay among the six structural
properties of the industrial problem, and the suite keeps them.

### The strict reading, and why the page does not take it

A stricter reading would admit a target only if its future is a deterministic
function of the present. That would exclude the battery, whose dispatch carries
white jitter, and the reactor, whose demand follows an Ornstein-Uhlenbeck
process. The page rejects it for three reasons:

- **It classifies noise placement, not tasks.** In error coordinates a random
  setpoint and a random output disturbance give the same error dynamics, so the
  servo and regulator problems are the same control problem.
- **It excludes the industrial cases the suite exists for**: a cascade's inner
  loop, ratio control, frequency regulation, an aircraft cleared to a new
  altitude. None of these references is predictable.
- **What it protects is a different property**, anticipatability, which has its
  own section below. Folding it into admissibility would remove environments
  instead of scoring them correctly.

The loose reading is also the minimal generalisation of the thesis: putting the
setpoint into the state, so that it may evolve, leaves Definitions 8.1 and 8.2
intact.

## Where the target is fixed: augment the state

The criterion looks restrictive and mostly is not, because a target that
appears to move in one space is fixed in another, and the space is always the
same one: the plant state augmented with the reference generator's state. Every
environment in this suite makes that move, whether or not it looks like it.

- The 3D path tasks carry the pattern's centre, radius and orientation as state
  fields. `distance_to_circle` and `distance_to_racetrack` compute the error
  from the aircraft's position relative to those fields, all of which are in
  the observation. There is no clock and no phase.
- The 2D altitude patterns carry the current commanded altitude as a state
  field, advanced each step by the pattern generator, and expose it in the
  observation. The staircase, the sinusoid and the chirp all fit this way.
- The reactor carries the current demand as a state field, advanced by an
  Ornstein-Uhlenbeck step, and exposes it. The battery carries its whole
  dispatch schedule as a state field and exposes only the current request:
  the live block's level plus regulation jitter.
- The patrol tasks carry the lead aircraft's full state and define the slot in
  its body frame.

The reference generator is therefore part of every environment's specification,
and the table below records it, because it determines the ceiling.

| Environment | Reference | Generator |
| --- | --- | --- |
| `plane`, `plane3d_heading` | altitude, heading sampled at reset | static |
| `plane3d_circle`, `plane3d_racetrack`, `plane3d_figure8` | geometric set, parameters sampled at reset | static |
| `cstr`, `first_order`, `four_tank`, `ph_neutralization`, `distillation`, `glass_furnace`, `hvac`, `cement_kiln`, `boiler_drum`, `wind_turbine` | setpoint sampled at reset | static |
| `plane_energy` | staircase, eight treads at fixed intervals | scheduled, switching times not observed |
| `plane_sine` | sinusoid, 300 m amplitude, 240 s period | deterministic exosystem of order two, phase not observed |
| `reactor` | flux demand | Ornstein-Uhlenbeck, stochastic |
| `battery` | dispatch request | scheduled 300 s blocks, next level and switching times not observed, plus white jitter: stochastic |
| `patrol`, `patrol_bearing_only` | slot on a manoeuvring lead | exogenous aircraft on a route: scheduled turn rates, in the state and observed one leg at a time |
| `formation` | lead: the patrol's route; wingmen: slots on the lead | lead: scheduled turn rates, in the state and observed one leg at a time; wingmen: a co-player, the learning lead |

A sinusoid could instead be posed as "stay on this ellipse in $(z, \dot z)$",
with the reference removed from the state. That is orbital stabilisation, which
drops phase, so it is a different task. The suite carries the setpoint in the
state and keeps the phase.

## Anticipatability

A target can satisfy the criterion and still be unpredictable. The two are
different properties, and the second determines what the ceiling is.

**Static.** The set is known from the first step. Holding exactly is achievable
in principle, and the ceiling is 0, less whatever cost cannot be driven to
zero.

**Deterministic exosystem.** The reference is generated by a finite-dimensional
autonomous system whose initial state is hidden. The current setpoint is
observed, so the generator's state is identifiable from a short history: a
staircase from one tread, a sinusoid from a few samples, a linear chirp from a
few more. The internal model principle (Francis and Wonham, 1976) says what a
policy needs in order to hold zero error against such a reference: a copy of
the generator, somewhere in the loop. In RL terms, memory, unless the plant
itself carries the generator's state, which it can: an aircraft tracking a
sinusoid has a vertical speed that follows the setpoint's, so its state
encodes the phase a single observation hides. Whether a memoryless policy
falls short of the ceiling is therefore a property of the closed loop, to be
computed rather than assumed.

**Stochastic exosystem.** The reference has an irreducible random part.
`reactor` and `battery` are in this class, and both are faithful models of grid
operation. The reactor's flux demand is load-following that nobody forecasts.
The battery's dispatch is a schedule of held blocks, as a market dispatches,
with regulation jitter on top that no policy can anticipate. The set $G$ is
fixed and the current setpoint is observed, so the criterion holds, but the
maximum reward is unattainable at every transition for any policy, because the
reference moves without warning after the action is chosen.

That ceiling is a property of the environment. On such a task an absolute score
means nothing, and acquires meaning only against a controller facing the same
reference, which is the argument for shipping tuned baselines and for the
expert-normalised metric the thesis uses.

## Feasibility

A target set is only meaningful if it can be held, and neither half of that is
automatic. Feasibility is asserted per environment rather than assumed by the
definition, and the reason for keeping it separate is that infeasible target
MDPs have to exist as objects in order to be diagnosed as such. This suite has
one on record: the four-tank's sampled targets once sat entirely above the
reachable envelope, so every episode was unwinnable and both baselines sat far
from a setpoint neither could attain (deviation D1 in its physics contract).

A target MDP is **feasible** at tolerance $\varepsilon$ if two conditions hold.

**Reachability.** There is a policy and a horizon $T$ such that, from every
initial state the reset distribution can produce, the state is within the
$\varepsilon$-target by step $T$.

**Invariance.** There is a policy under which the $\varepsilon$-target is
forward invariant: for every disturbance the environment can produce, a state
inside it stays inside it. With unbounded noise this is asked with probability
at least $1 - \delta$ over the episode rather than surely. For a moving
reference it is a bandwidth condition: if the reference varies faster than the
plant can follow, no policy stays in the set and the achievable reward is
bounded below its maximum for reasons that have nothing to do with the agent.

With deterministic dynamics and $\varepsilon = 0$ this is the thesis's
Stable-SP-MDP (Definition 8.3). The thesis proves two things in that setting:
every policy that reaches and stays in the target set is gain-optimal (Theorem
8.1), and every bias-optimal policy is such a policy (Theorem 8.2). Both proofs
use deterministic transitions and exact reaching. The first direction carries
over to the $\varepsilon$ form directly. The second holds in stochastic form
whenever some policy has finite expected integrated loss, with an explicit
bound on the expected settling time; the proof is in the formalisation
document, not on this page.

## Multi-agent target MDPs

The two registered `patrol` environments define the target relative to another
aircraft. A follower holds a slot on a lead that flies a route of straight legs
and turns, so $G$ is fixed in the follower's relative coordinates and A1 holds.
The lead does not react to the follower and has no payoff of its own: it is an
autopilot whose turn rate is scheduled per leg, so it is an exogenous reference
like the staircase, and the definition does not change. Two things are new
relative to the single aircraft: a collision trips the plant, and in the
bearing-only variant the slot error is recoverable only from the history of
range and bearing, so A2 holds in its weak form.

The line between a reference and a player is this. A reference is
**exogenous** if its dynamics do not depend on the follower's state or actions
and it carries no reward of its own. Exogenous references are admitted, and
the suite treats them as part of the environment, which has one practical
consequence: whatever drives the reference, here the lead's autopilot, is
environment code and must be versioned as such.

The definition does change when the other aircraft crosses that line and
becomes a player: when it reacts to the follower, or carries its own reward. The `formation`
task, where the lead is also a learner, is that case. A
**multi-agent target MDP** is a Markov game in which each agent $i$ has an
admissible target set $G_i$ in the joint state and a reward
$r_i = -\big(\ell_i(e_i) + c_i\big)$. Three things move.

- Feasibility becomes joint. The slots must be mutually compatible, and the
  invariant set is the intersection of the agents' $\varepsilon$-targets under
  a joint policy.
- The ceiling stops being environmental. With a shared team reward it is a
  joint supremum and remains well defined. With competing rewards it is an
  equilibrium quantity and depends on what the others do.
- Anticipatability becomes strategic. The reference's next move depends on a
  policy that is itself learning.

`formation` shares one team reward, so its ceiling is a joint supremum. It is
registered in its joint form, one action for every aircraft, so the suite's
single-agent tools score it as a centralised controller would fly it; the
multi-agent view (`FormationMARL`) is the same game with one policy per
aircraft.

## Prior art

Much of this page restates things control theory has known for fifty years,
and it is worth being precise about which parts.

**Output regulation and the internal model principle.** Francis and Wonham
(1976) pose tracking of a reference generated by an exosystem, with rejection
of disturbances from the same class, and prove that any controller achieving it
robustly contains a copy of the exosystem. Isidori and Byrnes (1990) extend
this to nonlinear plants. Carrying the reference generator in the state, and
the memory requirement for the sinusoid, are that theory restated. What differs
here is the objective and the plant. The objective is a floor-normalised cost
scored against a baseline rather than asymptotic zero error, the plant is
nonlinear and unknown to the agent, and the stochastic exosystem case is
outside the classical result altogether, belonging to LQG tracking.

**Limit-cycle and orbital stabilisation.** The ellipse remark above is the
standard observation that stabilising a periodic orbit is a different problem
from tracking a periodic reference, because orbital stability does not fix
phase (Hauser and Chung, 1994; Shiriaev, Perram and Canudas-de-Wit, 2005).

**Goal-conditioned RL.** Kaelbling (1993), universal value functions (Schaul et
al., 2015) and hindsight experience replay (Andrychowicz et al., 2017) put the
goal in the observation, which is the same augmentation as A1. Target MDPs
differ in that nothing terminates on arrival, the reward is monotone in
the error rather than sparse, and the goal may be driven by an exogenous
process.

**Set invariance and safe RL.** Control-invariant sets and viability kernels
(Blanchini, 1999) are the machinery behind the feasibility condition, and safe
RL uses the same objects, through control barrier functions (Ames et al., 2017)
or Lyapunov constraints (Berkenkamp et al., 2017; Chow et al., 2018), to keep
a policy inside a safe set while it optimises something else. Here invariance
of the target set is the objective itself, and the safety constraint is
expressed as a trip cost on leaving the envelope.

**Reach-avoid, stochastic reachability and reach-and-stay.** Stochastic
reachability (Abate et al., 2008) and the temporal-logic "eventually reach and
always remain" specifications formalise the reach-then-maintain objective as a
probability to be maximised. The thesis's related-work section positions the
SP-MDP against these.

**Average-reward RL.** The gain and bias criteria are classical (Puterman,
1994; Mahadevan, 1996) and the thesis's theorems live there.

What is new, then, is narrower than the page's length suggests, and it is
three things. The admissibility criterion as a rule for benchmark design,
separated from the anticipatability of the reference. The ceiling as a
property of the environment, which is what makes baseline-relative scoring
necessary rather than merely convenient. And the thesis's result that in a
feasible target MDP bias-optimality forces the policy to reach and stay, which
is what licenses using an existing stable controller as an exploration guide.

## References

- Abate, A., Prandini, M., Lygeros, J., Sastry, S. (2008). Probabilistic
  reachability and safety for controlled discrete time stochastic hybrid
  systems. *Automatica* 44(11).
- Ames, A. D., Xu, X., Grizzle, J. W., Tabuada, P. (2017). Control barrier
  function based quadratic programs for safety critical systems. *IEEE TAC*
  62(8).
- Andrychowicz, M. et al. (2017). Hindsight experience replay. *NeurIPS*.
- Berkenkamp, F., Turchetta, M., Schoellig, A., Krause, A. (2017). Safe
  model-based reinforcement learning with stability guarantees. *NeurIPS*.
- Berthelot, Y. (2026). *Efficient Deep Reinforcement Learning for Industrial
  Process Control.* PhD thesis, Université de Lille. Chapter 8 and Appendix C.
- Blanchini, F. (1999). Set invariance in control. *Automatica* 35(11).
- Chow, Y., Nachum, O., Duenez-Guzman, E., Ghavamzadeh, M. (2018). A
  Lyapunov-based approach to safe reinforcement learning. *NeurIPS*.
- Francis, B. A., Wonham, W. M. (1976). The internal model principle of
  control theory. *Automatica* 12(5).
- Hauser, J., Chung, C. C. (1994). Converse Lyapunov functions for
  exponentially stable periodic orbits. *Systems & Control Letters* 23(1).
- Isidori, A., Byrnes, C. I. (1990). Output regulation of nonlinear systems.
  *IEEE TAC* 35(2).
- Kaelbling, L. P. (1993). Learning to achieve goals. *IJCAI*.
- Mahadevan, S. (1996). Average reward reinforcement learning: foundations,
  algorithms, and empirical results. *Machine Learning* 22.
- Puterman, M. L. (1994). *Markov Decision Processes.* Wiley.
- Schaul, T., Horgan, D., Gregor, K., Silver, D. (2015). Universal value
  function approximators. *ICML*.
- Shiriaev, A., Perram, J. W., Canudas-de-Wit, C. (2005). Constructive tool for
  orbital stabilization of underactuated nonlinear systems: virtual constraints
  approach. *IEEE TAC* 50(8).
