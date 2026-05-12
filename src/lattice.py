"""3D cubic lattice with periodic boundary conditions.

The lattice is an m × m × m cubic grid where each site has exactly 6
nearest neighbors (±x, ±y, ±z).  Periodic boundary conditions are
applied so the lattice wraps around in all three dimensions, removing
edge effects.

The flat site index follows row-major order: index = x*m² + y*m + z.
"""

import numpy as np


def site_index(x: int, y: int, z: int, m: int) -> int:
    """Convert 3D lattice coordinates to a flat (1D) index.

    Parameters
    ----------
    x, y, z : int
        Lattice coordinates in [0, m).
    m : int
        Linear lattice size.

    Returns
    -------
    int
        Flat index in [0, m³).
    """
    return x * m * m + y * m + z


def build_neighbor_table(m: int) -> np.ndarray:
    """Build the nearest-neighbor lookup table for an m × m × m cubic lattice
    with periodic boundary conditions.

    Each site has 6 neighbors corresponding to ±1 steps along the three
    Cartesian axes.  Periodic boundaries are enforced with modular arithmetic
    so that, e.g., site (m-1, y, z) wraps around to (0, y, z) in the +x
    direction.

    Parameters
    ----------
    m : int
        Linear lattice size.  The total number of sites is N = m³.

    Returns
    -------
    neighbors : ndarray of shape (N, 6), dtype int32
        ``neighbors[i, d]`` is the flat index of site *i*'s neighbor in
        direction *d*.  Column order: +x, -x, +y, -y, +z, -z.
    """
    N = m ** 3
    neighbors = np.empty((N, 6), dtype=np.int32)
    for x in range(m):
        for y in range(m):
            for z in range(m):
                i = site_index(x, y, z, m)
                neighbors[i, 0] = site_index((x + 1) % m, y, z, m)
                neighbors[i, 1] = site_index((x - 1) % m, y, z, m)
                neighbors[i, 2] = site_index(x, (y + 1) % m, z, m)
                neighbors[i, 3] = site_index(x, (y - 1) % m, z, m)
                neighbors[i, 4] = site_index(x, y, (z + 1) % m, m)
                neighbors[i, 5] = site_index(x, y, (z - 1) % m, m)
    return neighbors


def verify_neighbor_table(neighbors: np.ndarray, m: int) -> None:
    """Assert correctness of a pre-built neighbor table.

    Performs three sanity checks:
    * Shape — every site has exactly 6 neighbors.
    * No self-loops — no site is its own neighbor.
    * Symmetry — the +dir neighbor's -dir neighbor equals the original site,
      confirming that the periodic wrapping is mutually consistent.

    Parameters
    ----------
    neighbors : ndarray of shape (N, 6), dtype int32
        Neighbor table as returned by :func:`build_neighbor_table`.
    m : int
        Linear lattice size used to construct the table.

    Raises
    ------
    AssertionError
        If any check fails.
    """
    N = m ** 3
    assert neighbors.shape == (N, 6), f"Expected shape ({N}, 6), got {neighbors.shape}"

    for i in range(N):
        # No self-loops
        assert i not in neighbors[i], f"Site {i} is its own neighbor"

        # Symmetry checks: +dir and -dir are inverses
        for d_pos, d_neg in [(0, 1), (2, 3), (4, 5)]:
            j = neighbors[i, d_pos]
            assert neighbors[j, d_neg] == i, (
                f"Symmetry broken: site {i} +dir neighbor {j}, "
                f"but {j}'s -dir neighbor is {neighbors[j, d_neg]}"
            )
