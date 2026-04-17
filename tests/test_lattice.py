"""Tests for 3D lattice neighbor table."""

import numpy as np
import pytest
from src.lattice import build_neighbor_table, verify_neighbor_table, site_index


@pytest.mark.parametrize("m", [2, 3, 5, 10])
def test_neighbor_shape(m):
    nb = build_neighbor_table(m)
    assert nb.shape == (m ** 3, 6)


@pytest.mark.parametrize("m", [2, 3, 5, 10])
def test_no_self_loops(m):
    nb = build_neighbor_table(m)
    for i in range(m ** 3):
        assert i not in nb[i], f"Site {i} is its own neighbor for m={m}"


@pytest.mark.parametrize("m", [2, 3, 5, 10])
def test_symmetry(m):
    nb = build_neighbor_table(m)
    for i in range(m ** 3):
        for d_pos, d_neg in [(0, 1), (2, 3), (4, 5)]:
            j = nb[i, d_pos]
            assert nb[j, d_neg] == i, (
                f"m={m}: site {i} +dir({d_pos}) -> {j}, "
                f"but {j} -dir({d_neg}) -> {nb[j, d_neg]}"
            )


@pytest.mark.parametrize("m", [3, 5])
def test_verify_function(m):
    nb = build_neighbor_table(m)
    verify_neighbor_table(nb, m)  # should not raise


def test_periodic_wrapping_m3():
    """Corner site (0,0,0) in m=3 should wrap to (2,...) neighbors."""
    m = 3
    nb = build_neighbor_table(m)
    idx_000 = site_index(0, 0, 0, m)
    # -x neighbor should be (2, 0, 0)
    assert nb[idx_000, 1] == site_index(2, 0, 0, m)
    # -y neighbor should be (0, 2, 0)
    assert nb[idx_000, 3] == site_index(0, 2, 0, m)
    # -z neighbor should be (0, 0, 2)
    assert nb[idx_000, 5] == site_index(0, 0, 2, m)


def test_all_neighbors_in_range():
    """All neighbor indices should be in [0, N)."""
    m = 5
    nb = build_neighbor_table(m)
    N = m ** 3
    assert np.all(nb >= 0)
    assert np.all(nb < N)
