"""Attempt 3's scenario suite (charter Amendment 3, rule 20), each from REALITY.md.

A float furnace runs continuously, so an episode is a window of it: seven days of
established operation. Inside each scenario the walks and the event's start time are
drawn by the seed, inside the window the parameters give. Every event is one the record
shows; sizes the record doesn't give are ours inside the family's range, said where they
are set (``env_jax.GlassFurnaceParams``).
"""

from target_gym.scenarios import Scenario

# Shares of operating time (charter J4), ours inside the family's ranges, which the
# family sweep draws from. No source says how often a float line changes pull or meets a
# gas swing of this size:
# - gas swings: none at entry points that stay within 4 MJ/m3 over two years [S20], up to
#   weekly where LNG and biomethane blend in [S19; S20]: each direction 0 to 0.3;
# - planned pull changes: none on a float line holding its pull within 2 % [S5], up to
#   about 20 job changes a month on a container line [S9]: each direction 0 to 0.3;
# - the normal day: the rest, 0.4 to 1.
WEIGHTS = {"normal": 0.7, "pull_up": 0.05, "pull_down": 0.05}
WEIGHTS.update(gas_rich=0.1, gas_lean=0.1)

PULL_STEP = (
    0.10  # ours, inside the family's 2 % float wander to a container's 29 to 42 %
)
GAS_STEP = 0.06  # 3 MJ/m3 within a day on a Wobbe index of about 50 [S20], arithmetic

#: The stream's numbers (docs/proposals/long-strata.md, draft 4, S1): how long each
#: regime lasts (``dwell_h``), how often its event comes (``events_per_h``) and how
#: long a change holds before it goes back (``event_hold_h``). REALITY.md section 9
#: gives the sources and the searches; a number marked ours was set once, without
#: tuning.
NORMAL_DWELL_H = 7 * 24.0  # ours
PULL_DWELL_H = 7 * 24.0  # ours: a planned change and its return in one visit
PULL_HOLD_H = 3 * 24.0  # ours
GAS_DWELL_H = 7 * 24.0  # ours: a week of blended gas
#: [ST1]: at one French site, LNG came three times in about two days, each episode
#: about 8 h above half its 2 MJ/m3 rise. With the task's 6 h ramp, an 8 h hold from
#: the start keeps the swing above half its size for those 8 h.
GAS_EVENTS_PER_H = 1.5 / 24.0
GAS_HOLD_H = 8.0
#: A stream starts no event of its own window: only the stream starts events.
_NO_WINDOW_EVENT = {"pull_change_window_h": (1.0e9, 1.0e9)}


def _stream(dwell_h, events_per_h=0.0, hold_h=None):
    """A regime's stream fields: its dwell, its event rate and its event's hold."""
    held = {} if hold_h is None else {"event_hold_h": hold_h}
    return dict(
        dwell_h=dwell_h,
        events_per_h=events_per_h,
        stream_overrides={**_NO_WINDOW_EVENT, **held},
    )


def suite() -> tuple:
    """The scenarios, normal day first."""
    return (
        Scenario(
            "normal",
            {},
            "[S5] pull wanders about 2 %; [S19] gas composition weekly or daily",
            "established operation: the walks only",
            weight=WEIGHTS["normal"],
            **_stream(NORMAL_DWELL_H),
        ),
        Scenario(
            "pull_up",
            {"pull_change": PULL_STEP},
            "[S1, slide 10] pull change in float control; [S3, slide 8]; [S4]",
            "a planned 10 % rise in pull, shown ahead by its notice",
            weight=WEIGHTS["pull_up"],
            **_stream(PULL_DWELL_H, 1.0 / PULL_DWELL_H, PULL_HOLD_H),
        ),
        Scenario(
            "pull_down",
            {"pull_change": -PULL_STEP},
            "[S1, slide 10]; [S3, slide 8]; [S4]",
            "a planned 10 % fall in pull, shown ahead by its notice",
            weight=WEIGHTS["pull_down"],
            **_stream(PULL_DWELL_H, 1.0 / PULL_DWELL_H, PULL_HOLD_H),
        ),
        Scenario(
            "gas_rich",
            {"gas_swing": GAS_STEP},
            "[S20] LNG and biomethane move the Wobbe index over 3 MJ/m3 within a day",
            "an unannounced 6 % rise in the gas's heating value",
            weight=WEIGHTS["gas_rich"],
            **_stream(GAS_DWELL_H, GAS_EVENTS_PER_H, GAS_HOLD_H),
        ),
        Scenario(
            "gas_lean",
            {"gas_swing": -GAS_STEP},
            "[S20]",
            "an unannounced 6 % fall in the gas's heating value",
            weight=WEIGHTS["gas_lean"],
            **_stream(GAS_DWELL_H, GAS_EVENTS_PER_H, GAS_HOLD_H),
        ),
    )
