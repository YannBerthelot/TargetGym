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
