# Rendering

Every environment draws a **control-room dashboard**, not a plot. A frame shows
both what the controller measures and what it is actually up against, which is
what makes the clips worth watching rather than decorative.

## Render one episode

`save_video` drives a policy through an episode and writes the frames:

```python
import target_gym as tg

env, params = tg.make("glass_furnace")

env.save_video(
    tg.make_expert("glass_furnace").host(),   # any callable taking the observation
    seed=0,
    params=params,
    folder="videos/glass_furnace",
    format="gif",             # or "mp4"
    FPS=30,
)
```

`FPS` is worth setting deliberately. `save_video` writes at the rate you pass,
and the default of 60 against a clip built at 30 makes moviepy drop every other
frame, so a 100-step episode comes out 49 frames and plays at twice the speed
the length arithmetic suggests.

**Headless machines** — CI, a container, a server — need two environment
variables set *before* importing anything, or the renderers abort with no
display:

```bash
export SDL_VIDEODRIVER=dummy   # pygame, used by the aircraft
export MPLBACKEND=Agg          # matplotlib, used by the plants
```

## What a frame shows

The industrial plants share one visual language, taken from the reactor
renderer:

| Region | What it carries |
| --- | --- |
| **Schematic** (left) | The plant, animated by live state |
| **Instrument stack** (right) | Horizontal gauge bars with limit ticks and setpoint markers |
| **Strip chart** (bottom) | The controlled variable against its setpoint |
| **Header** | Run clock and a status pill |

The schematic is drawn from state the *controller* often cannot see — riser
voidage, thermal mass, the kiln's axial profile. A gauge marked `hidden=True`
carries a dot and the frame gains a legend, so you can tell at a glance which
quantities the agent is flying blind on. On the glass furnace that is 6 of 9
states; on the reactor, 7 of 11.

## The two toolkits

There are two, and which one an environment uses is a performance decision.

**`target_gym.render_kit`** is matplotlib, and every non-aircraft environment
draws through it. Build a frame from a schematic callback, a list of `Gauge`s
and a list of `Strip`s:

```python
# doc: skip (shape of the call; the state and helpers are the caller's)
from target_gym import render_kit as rk

gauges = [rk.Gauge("POWER", f"{p:.2f}", frac=p, color=rk.CYAN)]
fig = rk.frame(
    title="MY PLANT", step=step, params=params,
    schematic=lambda ax: draw_my_plant(ax, state, params),
    gauges=gauges,
    strips=[rk.Strip(t, [rk.Series(values, "x", rk.CYAN)], ylabel="x")],
)
frame_array = rk.finish(fig)
```

**`target_gym.render_aircraft`** is pygame, because the aircraft project a
solid A320 mesh every frame and matplotlib is far too slow for that. It carries
the palette, the console chrome, the mesh and its three projections (side,
top-down, rear), and all four aircraft modules import from it rather than from
each other. They used to reach into each other — `patrol` imported eleven
private names out of `plane3d` — so a palette change in one environment
silently restyled another.

Neither module is fingerprinted: `provenance._env_sources` skips `rendering*`
modules inside an environment package, and these live outside one. Moving code
between renderers never marks a recorded baseline stale.

## Regenerating the shipped media

```bash
make videos                                  # or videos-<env>
uv run python scripts/make_gallery_clips.py  # re-quantise the console clips
make short-gifs                              # trim frames only
uv run python scripts/make_gallery_mosaic.py --set all
```

`make_gallery_clips.py` downscales and palette-quantises, which is what makes a
clip small; `make short-gifs` only trims frames and will happily keep all of
them. The mosaics prefer `*_short.gif`, so rebuilding them needs the shorts
present locally.

**What is committed:** only the five gallery mosaics, `videos/mosaic_*.webp`,
because they are the only media a published page embeds. The per-environment
clips are rendered by the `docs-deploy` workflow before it builds the site, so
a clean checkout stays light and the published pages still have pictures.

## Adding a renderer

Implement `render(self, screen, state, params, frames, clock)` on the
environment and return `(frames, screen, clock)`. Draw through `render_kit` for
a plant or `render_aircraft` for anything flying, rather than starting a third
visual language — the suite reading as one instrument suite is the point.

Mark every gauge the observation does not expose with `hidden=True`. It costs
one keyword and it is the difference between a viewer understanding the task
and assuming the agent can see what the picture shows.
