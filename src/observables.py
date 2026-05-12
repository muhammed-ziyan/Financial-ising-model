"""Post-processing utilities for simulation outputs.

Provides return normalization (matching the paper's Eq. 4 convention),
a simple volatility proxy, and .npz serialization helpers used by
``run_simulation.py`` and the analysis scripts.
"""

import numpy as np


def normalize_returns(returns: np.ndarray, method: str = "paper") -> np.ndarray:
    """Normalized return used throughout the analysis layer.

    Parameters
    ----------
    returns : np.ndarray
        Raw return series ``r(t)``.
    method : {"paper", "std"}
        * ``"paper"`` (default) — matches Eq. (4) of Guimaraes & Lima
          (2021): ``g = (r - <r>) / <|r|>`` with ``<|r|>`` the mean of
          the absolute returns ("volatility average").
        * ``"std"`` — classic z-score ``g = (r - <r>) / std(r)``, useful
          as a cross-check.

    Memory/long-range statistics (ACF, R/S, DFA) are scale-invariant, so
    the choice only affects the *scale* of the histograms and tail
    fits.  We pick the paper convention by default so the reported
    quantities line up with Table II/III/IV of the paper.
    """
    mu = np.mean(returns)
    if method == "paper":
        scale = np.mean(np.abs(returns))
    elif method == "std":
        scale = np.std(returns)
    else:
        raise ValueError(f"Unknown normalization method: {method!r}")
    if scale == 0:
        return np.zeros_like(returns)
    return (returns - mu) / scale


def compute_volatility(returns: np.ndarray) -> np.ndarray:
    """Return absolute values of the return series as a volatility proxy.

    Parameters
    ----------
    returns : np.ndarray
        Raw or normalized return series.

    Returns
    -------
    np.ndarray
        Absolute returns ``|r(t)|``.
    """
    return np.abs(returns)


def save_results(filepath: str, **arrays) -> None:
    """Save named arrays to a compressed NumPy ``.npz`` archive.

    Parameters
    ----------
    filepath : str
        Destination path (the ``.npz`` extension is added by NumPy if
        omitted).
    **arrays
        Keyword arguments become the named arrays in the archive, e.g.
        ``save_results("out.npz", returns=r, price=p)``.
    """
    np.savez(filepath, **arrays)


def load_results(filepath: str) -> dict:
    """Load a ``.npz`` archive and return a plain dict of arrays.

    Parameters
    ----------
    filepath : str
        Path to the ``.npz`` file.

    Returns
    -------
    dict
        Mapping from array name to ``np.ndarray``.
    """
    data = np.load(filepath)
    return {k: data[k] for k in data.files}
