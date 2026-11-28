# Task protocol

*Adopted by the owner on 2026-11-27. It replaces the task charter's gates for
accepting a task: ceiling convergence, the family sweeps and J2. The charter,
`docs/proposals/task-charter.md` in the repository, stays as history.*

This page says what a task needs before it becomes v1, the version that never
changes (see the [vision](vision.md)). A task becomes v1 when it has all nine
items below. Each task becomes v1 on its own, as it passes.

1. **Job.** One sentence naming a job that a real operator is paid to do, and
   the decision the agent makes.
2. **Score.** A score in the job's own units, over a fixed episode, with
   nothing at the episode's end that a policy can game.
3. **Expert.** One expert: the controller that a written practitioner recipe
   gives, frozen as data (`tg.make_reference`).
4. **Baselines.** Published baselines: doing nothing, the expert, an MPC on the
   true model, and PPO and SAC at one cheap budget (30 seeds, shipped
   settings). The learned agents' numbers are reported whatever they score.
   They are not a pass mark.
5. **Room.** The MPC or a learned agent beats the expert, with an interval that
   excludes zero.
6. **Five checks.** The physics matches one published figure. Tripping never
   pays. The expert beats doing nothing. Runs are deterministic. `jit` and
   `vmap` work.
7. **Speed.** An episode runs in under a second on a CPU.
8. **Card and ID.** One task card and a Gymnasium ID.
9. **Freeze.** The task is frozen at v1 in a fresh namespace,
   `TargetGym/<task>-v1`, with a CI guard. Any later change becomes v2 beside
   it. Today's version stamps become development stamps in the CHANGELOG.

The foresight ceiling may still be reported beside a task's numbers, as
context.

## The owner's answers

Given by picker on 2026-11-27:

| Topic | Answer |
|---|---|
| The vision | "Adopt the vision" |
| This protocol | "Adopt the protocol" |
| PPO and SAC numbers | "Reported, not required" |
| When a task becomes v1 | "Per task, as each passes" |

## The owner's readings

Given by picker on 2026-11-28, on the v1 task pages. They say how four items
are read.

| Item | Answer | Reading |
|---|---|---|
| 5, room | "Observation-only MPC" | The room comes from an MPC on the true model that estimates the hidden state from the observations. An MPC that reads the hidden state does not count; its numbers are context, like the ceiling. |
| 7, speed | "Per episode in a batch" | The time is one episode's share of a vmapped batch, not one episode run alone. |
| 2 and 3, score and expert | "Source every price first" | A task waits until every price that trades one goal against another has a published source. The unit a task is scored in is not such a price. Its expert is frozen only after that, since the cost picks the expert's rung. |

## How a task becomes v1

Three commands, which the task's session runs in this order (added
2026-11-28).

1. `scripts/record_v1.py <task>`, on bob or flanders. It records item 4's
   baselines into `src/target_gym/data/v1_records/<task>.json`, each scored by
   `tg.score` on the scoring seeds: doing nothing, the expert, the
   observation-only MPC, and PPO and SAC at 30 runs of 204,800 steps. The MPC is
   the task's `observation_only_mpc(env, params)` in its `oracle.py`: an MPC on
   the true model, planning from `target_gym.estimation.Estimator`'s belief.
2. `scripts/generate_cards.py`. It writes the task's card, `docs/cards/<task>.md`,
   from its inputs in `docs/cards/<task>.json` and from the record. The inputs
   hold the job, the decision, the score's unit, the prices with their sources,
   and what the task does not model. The record command runs this step too.
3. `scripts/freeze_v1.py <task>`, on a clean commit. It writes the task's
   manifest, `src/target_gym/data/v1/<task>.json`, once. That registers
   `TargetGym/<task>-v1` with Gymnasium and arms the guard,
   `tests/test_v1_freeze.py`. It refuses a task whose reference is stale, whose
   record lacks a baseline or was taken on another plant, scoring or reference,
   or whose record shows no room.
