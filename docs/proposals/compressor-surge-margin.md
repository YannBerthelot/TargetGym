# compressor_surge: a margin the plant earns

Status: the owner chose option 1, the lag and the hidden line shift (2026-11-23). It is
built in the plant; its values wait on four choices ("Still open" below).

## The finding

After the reset fix (#223), what is left of the task's room is the reference's 10 %
control margin. The plant as it is does not ask for it:

- placed on the surge line itself, the reference never surges on normal and slow
  days, and costs 8.59 kJ per step on slow days against a balance floor of 8.47;
- only fast days need a margin: under 7 % the reference surges there (2 of 10 seeds
  at 6 %). Fast days weigh 0.05.

So most of the room is a fixed offset an agent can find, on a line it reads exactly.

## What the sources say the margin is for

`compressor_surge/REALITY.md`, "Why the margin is kept" (third search, 2026-11-23):

- the line's position is not known exactly and moves: field and shop curves over
  10 % apart [C1], a drift with the machine's condition that cannot be measured
  [C10], 2.6 to 7 % left by invariant coordinates at low suction pressure [C14];
- the transmitters are not instant: flow dP 100 to 150 ms, at most 200 ms; pressure
  300 to 400 ms [C10];
- the margin covers both, with the disturbance's size and speed [C10, p. 11].

The plant has neither: its readings have noise and no lag, and its surge line is the
map's. Version 2 had a hidden line offset, removed in version 4 for lack of a source.

## Prototype

`scripts/compressor_surge_margin_causes.py`. The ladder's override rung, commissioned
on the nominal plant, on seeds 100 to 109 of each scenario. `J` in kJ per step, its
floor on the variant's own surge line, and the fast-day seeds with a surge, of 10. No
run tripped.

| variant | margin | J | J floor | fast-day seeds with a surge |
|---|---|---|---|---|
| plant as it is | 0.10 / 0.08 / 0.07 / 0.06 / 0.05 | 3.04 / 2.76 / 2.64 / 2.54 / 2.47 | 1.74 | 0 / 0 / 0 / 2 / 4 |
| readings lag 0.2 s | 0.10 / 0.09 / 0.08 / 0.07 / 0.05 | 3.07 / 2.93 / 2.81 / 2.70 / 2.54 | 1.74 | 0 / 0 / 0 / 1 / 4 |
| true line 5 % right of the map | 0.10 / 0.12 / 0.15 | 2.98 / 3.28 / 3.84 | 2.10 | 1 / 0 / 0 |
| line 5 % right, lag 0.2 s | 0.10 / 0.12 / 0.15 | 3.01 / 3.31 / 3.86 | 2.10 | 1 / 0 / 0 |
| true line 5 % left of the map | 0.10 | 3.27 | 1.48 | 0 |
| line 5 % left, lag 0.2 s | 0.15 | 4.95 | 1.48 | 0 |
| inlet 268 K, map at 288 K | 0.10 | 2.77 | 1.85 | 1 |
| inlet 308 K | 0.10 | 4.16 | 1.67 | 0 |

- The line shift scales the cubic's `W` by 1.05 or 0.95 while the controller keeps the
  nominal map.
- Inlet temperature moves the true margin by about 3.7 points for 20 K at a constant
  header pressure. Practice cancels it with reduced coordinates [C10, C11], and the
  task has no suction temperature reading, so on its own it is a weak cause.

## The set-point rule on this task

Owner's rule of 2026-11-23: on a fixed grid, the lowest set point at which the
reference leaves its limit on none of its seeds. Here the priced limit is the surge
line, so "no exit" reads as no counted surge. On a 1 % grid it keeps 0.07 on the plant
as it is, 0.08 with the lag, and 0.12 with the true line 5 % right of the map (0.11
was not run). With the sourced causes in the plant the rule lands in practice's 10 to
15 % [C1], and the margin is then the plant's own.

Two cautions. Only the fast day decides the placement, one seed apart between grid
points. And on the header pressure itself the reference leaves its zone on about half
the fast-day seeds even at the zone's centre, so there the rule has no grid point
without an exit.

The table was measured on the plant before the change, on the branch of the ceiling
fixes. The script now starts from that plant (`BARE`); on a later head of the
integration branch its rows come out within 2 % of these, with the same surges.

## The decision

1. **Lag and a hidden line shift (recommended).** A first-order lag on the flow and
   pressure readings, and a per-episode shift of the true surge line against the map,
   drawn either way. The lag's value is sourced [C10]. The shift's size is ours inside
   [C14]'s 2.6 to 7 %. The property becomes a limit whose position is hidden: an agent
   shaves the margin only by reading how the plant behaves near its line.
2. **Lag only.** Every number sourced, a small effect (0.07 to 0.08).
3. **Neither.** The rule places the margin at 0.07 and the room on J is about 1.5.

Not decided here: the shift's law and range, whether a restart redraws it, and
whether the observation gains a suction temperature channel [C4, C5].

## As built, and what it gives (2026-11-23)

The plant has a first-order lag of 0.15 s on its flow reading and 0.3 s on its
pressure reading [C10], and each episode's machine has its true surge line a hidden
fraction off the map, uniform within 5 % either side. The model computes the
machine's characteristic as the map's stretched in Phi by one plus that fraction. A
trip's restart keeps the machine. The oracle and the ceiling know the true line.

The reference's margin by the owner's rule, on seeds 100 to 109: no surge on any seed
from 0.15 down to 0.09, one fast-day seed at 0.08. So 0.09.

Room on held-out seeds 200 to 209, the reference at 0.09 against the ceiling on the
true line, paired by seed, kJ per step, with a 95 % interval from a bootstrap over
seeds within each scenario:

| scenario | reference | ceiling | balance floor | room |
|---|---|---|---|---|
| normal | 0.722 | 0.129 | 0.103 | 5.57 [3.13, 14.70] |
| slow_approach | 11.10 | 8.83 | 8.64 | 1.26 [1.20, 1.32] |
| fast_approach | 12.59 | 9.45 | 9.34 | 1.33 [1.27, 1.40] |
| load_swings | 5.24 | 3.48 | 3.40 | 1.51 [1.37, 1.75] |
| J | 3.03 | 1.97 | 1.91 | 1.54 [1.45, 1.65] |

- The ceiling costs less than the reference on 40 seeds of 40, trips on none, and
  sits 3 % above the balance floor on J.
- The reference surges once, on the one held-out machine near the top of the range
  (line 4.9 % right of its map). A margin placed on ten machines is not safe on an
  eleventh: that is the margin being the plant's.

## Still open

1. The shift's size and law (built: uniform within 5 %).
2. Whether a restart keeps the machine (built: it does).
3. Whether the observation gains a suction temperature channel (built: no).
4. The lags' values (built: 0.15 s and 0.3 s).
5. **The shift can be read off the readings.** Because the whole characteristic is
   stretched, the speed that holds 24 kPa at a given flow differs by about 0.7 % for a
   5 % shift, seventy steps of the speed reading. A controller that knew the map's
   shape could invert it and find the line without going near it, which the sources
   say a plant cannot do [C10]. Keeping the stretch local to the line would hide it;
   no source gives that shape.

