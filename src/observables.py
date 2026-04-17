"""Post-processing of simulation outputs."""

import numpy as np


def normalize_returns(returns: np.ndarray) -> np.ndarray:
    """Normalized return: g = (r - mean(r)) / std(r)."""
    mu = np.mean(returns)
    sigma = np.std(returns)
    if sigma == 0:
        return np.zeros_like(returns)
    return (returns - mu) / sigma


def compute_volatility(returns: np.ndarray) -> np.ndarray:
    """Absolute returns as volatility proxy."""
    return np.abs(returns)


def save_results(filepath: str, **arrays) -> None:
    """Save arrays to .npz file."""
    np.savez(filepath, **arrays)


def load_results(filepath: str) -> dict:
    """Load .npz file and return dict of arrays."""
    data = np.load(filepath)
    return {k: data[k] for k in data.files}
