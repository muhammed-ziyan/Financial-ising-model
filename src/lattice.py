"""3D cubic lattice with periodic boundary conditions."""

import numpy as np


def site_index(x: int, y: int, z: int, m: int) -> int:
    """Convert 3D coordinates to flat index."""
    return x * m * m + y * m + z


def build_neighbor_table(m: int) -> np.ndarray:
    """
    Build neighbor lookup table for m x m x m cubic lattice
    with periodic boundary conditions.

    Returns
    -------
    neighbors : ndarray of shape (N, 6), dtype int32
        Column order: +x, -x, +y, -y, +z, -z
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
    """Verify correctness of neighbor table.

    Checks:
    - Every site has exactly 6 neighbors
    - No self-loops
    - Symmetry: +x neighbor's -x neighbor is original site (etc.)
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
