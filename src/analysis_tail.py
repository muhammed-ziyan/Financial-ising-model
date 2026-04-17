"""Return distribution analysis: CCDF and tail exponent estimation."""

import numpy as np


def empirical_ccdf(data: np.ndarray):
    """Compute empirical complementary CDF.

    Returns (sorted_values, ccdf) where ccdf = P(X > x).
    """
    x = np.sort(data)
    n = len(x)
    ccdf = 1.0 - np.arange(1, n + 1) / n
    return x, ccdf


def tail_ccdf(data: np.ndarray, positive: bool = True):
    """CCDF of one tail.

    positive=True:  P(g > x) for x > 0
    positive=False: P(g < -x) for x > 0 (left tail as |g|)
    """
    if positive:
        tail = data[data > 0]
    else:
        tail = np.abs(data[data < 0])
    return empirical_ccdf(tail)


def hill_estimator(data: np.ndarray, k: int) -> float:
    """Hill estimator for tail index.

    Parameters
    ----------
    data : positive values from one tail
    k : number of upper order statistics

    Returns
    -------
    alpha : estimated tail exponent
    """
    sorted_data = np.sort(data)[::-1]  # descending
    if k >= len(sorted_data) or k < 1:
        raise ValueError(f"k={k} must be in [1, {len(sorted_data)-1})")
    log_ratios = np.log(sorted_data[:k] / sorted_data[k])
    if np.sum(log_ratios) == 0:
        return np.inf
    return k / np.sum(log_ratios)


def hill_plot(data: np.ndarray, k_range: np.ndarray = None):
    """Hill plot: alpha(k) vs k for different thresholds.

    Returns (k_values, alpha_values).
    """
    if k_range is None:
        k_max = max(10, len(data) // 5)
        k_range = np.arange(5, k_max)
    alphas = np.array([hill_estimator(data, int(k)) for k in k_range])
    return k_range, alphas


def fit_tail_ols(x: np.ndarray, ccdf: np.ndarray,
                 x_min: float, x_max: float):
    """OLS fit of log(CCDF) = -alpha * log(x) + const in [x_min, x_max].

    Returns (alpha, intercept).
    """
    mask = (x >= x_min) & (x <= x_max) & (ccdf > 0)
    if np.sum(mask) < 2:
        return np.nan, np.nan
    log_x = np.log(x[mask])
    log_ccdf = np.log(ccdf[mask])
    A = np.vstack([log_x, np.ones_like(log_x)]).T
    result = np.linalg.lstsq(A, log_ccdf, rcond=None)
    slope, intercept = result[0]
    return -slope, intercept
