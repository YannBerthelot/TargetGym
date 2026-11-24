"""A table read smoothly: one copy for a plant and for the planner that models it.

A plant that reads a chart between its nodes with straight lines has a kink at
every node. A planner that differentiates the plant then either meets the kinks
or plans on a smoothed copy of the table, and its plan is for a slightly
different plant. ``smooth_table`` passes through the same nodes with a
continuous gradient, so both can read the one table.
"""

import jax.numpy as jnp


def _node_slopes(grid, values):
    """The slope at each node of the parabola through it and its two
    neighbours (at an end, through the three nearest nodes), along axis 0."""
    h = jnp.diff(grid).reshape((-1,) + (1,) * (values.ndim - 1))
    secant = jnp.diff(values, axis=0) / h
    inner = (h[1:] * secant[:-1] + h[:-1] * secant[1:]) / (h[:-1] + h[1:])
    first = ((2.0 * h[0] + h[1]) * secant[0] - h[0] * secant[1]) / (h[0] + h[1])
    last = ((2.0 * h[-1] + h[-2]) * secant[-1] - h[-1] * secant[-2]) / (h[-1] + h[-2])
    return jnp.concatenate([first[None], inner, last[None]])


def _read_first_axis(grid, values, x):
    """``values`` at ``x`` along axis 0: a cubic Hermite piece per cell."""
    x = jnp.clip(x, grid[0], grid[-1])
    cell = jnp.clip(jnp.searchsorted(grid, x, side="right") - 1, 0, grid.shape[0] - 2)
    width = grid[cell + 1] - grid[cell]
    t = (x - grid[cell]) / width
    slopes = _node_slopes(grid, values)
    return (
        (1.0 + 2.0 * t) * (1.0 - t) ** 2 * values[cell]
        + t * (1.0 - t) ** 2 * width * slopes[cell]
        + t**2 * (3.0 - 2.0 * t) * values[cell + 1]
        + t**2 * (t - 1.0) * width * slopes[cell + 1]
    )


def smooth_table(grids, values, point):
    """``values`` on the grid ``grids`` read at ``point``, with a continuous
    gradient.

    ``grids`` is one increasing array of at least three nodes per dimension of
    ``values``, evenly spaced or not, and ``point`` one coordinate per
    dimension. The read returns the table's value at every node, is exact where
    the data are a polynomial of degree 2 or less in each coordinate, and holds
    the edge value past the grid, as ``jnp.interp`` does. Between nodes it may
    pass a little beyond its neighbours where the data bend sharply, which a
    straight line does not: look at a chart's read before trusting it.
    """
    values = jnp.asarray(values)
    for grid, x in zip(grids, point):
        values = _read_first_axis(jnp.asarray(grid, values.dtype), values, x)
    return values
