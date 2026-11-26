"""A table read smoothly, in one copy for a plant and for the planner that models it.

Contract (owner, 2026-11-24, "Smooth tables, one copy"): ``smooth_table(grids,
values, point)`` reads a table given on a grid, in one or more dimensions, so that

- it returns the table's value at every node;
- its gradient is continuous across the nodes, which a piecewise-linear read's is
  not, so a planner differentiating the plant sees no kink at a node;
- it is exact where the data are a polynomial of degree 2 or less in each
  coordinate, on a grid of any spacing;
- past the grid it holds the edge value, as ``jnp.interp`` does;
- in several dimensions it is the tensor product of the one-dimensional read;
- it works under ``jit``, ``vmap`` and ``grad``.

The read is a cubic Hermite piece per cell, whose node slopes are those of the
parabola through each node and its two neighbours. Expected values are the
functions the tables are sampled from, and the error bound of that construction.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.tables import smooth_table

try:
    from jax import enable_x64
except ImportError:  # JAX 0.6
    from jax.experimental import enable_x64

#: Uneven on purpose: charts are digitised where their curves bend.
GRID = np.array([0.0, 0.4, 1.0, 1.3, 2.1, 2.5, 3.2, 4.0])
INSIDE = np.linspace(0.0, 4.0, 401)


def _read(values, at, grid=GRID):
    return jax.vmap(lambda x: smooth_table((grid,), values, (x,)))(jnp.asarray(at))


def test_it_returns_the_tables_value_at_every_node():
    values = np.sin(GRID) + 0.3 * GRID
    np.testing.assert_allclose(_read(values, GRID), values, atol=1e-6)


def test_it_is_exact_on_a_parabola_whatever_the_spacing():
    parabola = lambda x: 2.0 * x**2 - 3.0 * x + 1.0
    np.testing.assert_allclose(
        _read(parabola(GRID), INSIDE), parabola(INSIDE), atol=2e-5
    )


def test_its_gradient_is_continuous_across_a_node():
    """Either side of each inner node the slopes agree, where a piecewise-linear
    read's differ by the change of secant, 0.05 to 0.5 on these data."""
    with enable_x64():
        values = np.sin(GRID)
        slope = jax.vmap(jax.grad(lambda x: smooth_table((GRID,), values, (x,))))
        nodes = jnp.asarray(GRID[1:-1])
        # 1e-9 either side: the cubic's own curvature moves a slope by 1e-8 there.
        np.testing.assert_allclose(slope(nodes - 1e-9), slope(nodes + 1e-9), atol=1e-7)
        secants = np.diff(values) / np.diff(GRID)
        assert np.min(np.abs(np.diff(secants))) > 0.05


def test_it_is_closer_than_the_bound_of_its_construction():
    """On an even grid of spacing h, away from the ends, a cubic Hermite piece
    with slopes off by ``d`` is within ``h^4 / 384 max|f''''| + h d / 4`` of f, and
    the parabola's slope is off by ``h^2 / 6 max|f'''|``. For sin and h = 0.5 that
    is 0.0054; the straight line between nodes is off by up to h^2 / 8 = 0.031."""
    h = 0.5
    grid = np.arange(0.0, 6.01, h)
    inner = np.linspace(grid[1], grid[-2], 501)
    with enable_x64():
        error = np.max(np.abs(_read(np.sin(grid), inner, grid) - np.sin(inner)))
    assert error < h**4 / 384 + h**3 / 24
    assert np.max(np.abs(np.interp(inner, grid, np.sin(grid)) - np.sin(inner))) > 0.02


def test_it_holds_the_edge_value_past_the_grid():
    values = np.cos(GRID)
    np.testing.assert_allclose(
        _read(values, [-5.0, -0.1, 4.2, 40.0]),
        [values[0], values[0], values[-1], values[-1]],
        atol=1e-6,
    )


def test_two_dimensions_are_the_product_of_the_one_dimensional_reads():
    """Data ``g(x) h(y)`` read in two dimensions is the read of g times the read
    of h: what a tensor product means, with the 1D read as the reference."""
    other = np.array([-1.0, 0.0, 0.5, 2.0, 2.5])
    g, h = np.sin(GRID), np.exp(-other)
    xs, ys = np.array([0.2, 1.7, 3.9]), np.array([-0.4, 0.25, 2.2])
    both = jax.vmap(lambda x, y: smooth_table((GRID, other), np.outer(g, h), (x, y)))(
        jnp.asarray(xs), jnp.asarray(ys)
    )
    np.testing.assert_allclose(both, _read(g, xs) * _read(h, ys, other), rtol=1e-5)


def test_two_dimensions_are_exact_on_a_surface_of_degree_two_each_way():
    other = np.array([-1.0, 0.0, 0.5, 2.0, 2.5])
    surface = lambda x, y: x**2 * y - 2.0 * x * y**2 + 3.0
    values = surface(GRID[:, None], other[None, :])
    xs, ys = np.array([0.2, 1.7, 3.9, 2.5]), np.array([-0.4, 0.25, 2.2, 1.0])
    read = jax.vmap(lambda x, y: smooth_table((GRID, other), values, (x, y)))(
        jnp.asarray(xs), jnp.asarray(ys)
    )
    np.testing.assert_allclose(read, surface(xs, ys), rtol=1e-5, atol=1e-5)


def test_it_reads_under_jit_and_its_gradient_is_the_functions():
    """On the parabola ``2 x^2 - 3 x + 1`` the read is exact, so its gradient is
    ``4 x - 3``: 1.8 at x = 1.2."""
    values = 2.0 * GRID**2 - 3.0 * GRID + 1.0
    slope = jax.jit(jax.grad(lambda x: smooth_table((GRID,), values, (x,))))
    assert float(slope(1.2)) == pytest.approx(1.8, abs=1e-4)
