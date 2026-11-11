"""The cement kiln's scenario suite: its one window, the normal kiln, as the
benchmark has always scored it (one scenario, weight 1).

It is declared only to give the stream its numbers (docs/proposals/long-strata.md,
draft 4, S1): a window draws the free-lime target once, a continuing plant changes
it. No source in this folder gives how often a kiln's free-lime target is revised,
so the rate is ours, set once without tuning and open for the owner: once a shift
(8 h) on average, each change a fresh draw over ``target_lime_range``.

A second search (2026-11-11) found how often a kiln's clinker is sampled, "every
2 h" (Ali, Tabares, McGinley, J. Advanced Manufacturing and Processing 4(4), 2022,
abstract, https://www.osti.gov/pages/biblio/1893173), but no source on how often
operators or quality control revise the free-lime target or change the clinker
type. Searched: free-lime target revisions, clinker type changes, kiln sampling
practice. The rate stays ours; it is slower than the lab's 2 h result.
"""

_SHIFT_H = 8.0


def cement_kiln_scenarios() -> tuple[dict, ...]:
    return (
        dict(
            name="normal",
            overrides={},
            source="PHYSICS.md, section 4",
            description="the kiln held on its free-lime target as the raw meal "
            "swings",
            weight=1.0,
            events_per_h=1.0 / _SHIFT_H,
        ),
    )
