"""Memory analysis: ACF, R/S Hurst exponent, and DFA."""

import numpy as np


def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocorrelation function for lags 0..max_lag."""
    x_centered = x - np.mean(x)
    var = np.var(x)
    if var == 0:
        return np.zeros(max_lag + 1)
    n = len(x)
    result = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        result[lag] = np.mean(x_centered[:n - lag] * x_centered[lag:]) / var
    return result


def rs_analysis(x: np.ndarray, n_values: np.ndarray = None):
    """Rescaled Range (R/S) analysis for Hurst exponent.

    Returns (n_values, rs_values, H).
    """
    T = len(x)
    if n_values is None:
        n_values = np.unique(np.geomspace(10, T // 4, num=30).astype(int))

    rs_values = np.empty(len(n_values))
    for idx, n in enumerate(n_values):
        n = int(n)
        num_blocks = T // n
        rs_list = []
        for b in range(num_blocks):
            block = x[b * n: (b + 1) * n]
            mean_b = np.mean(block)
            cumdev = np.cumsum(block - mean_b)
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(block, ddof=1)
            if S > 0:
                rs_list.append(R / S)
        rs_values[idx] = np.mean(rs_list) if rs_list else np.nan

    # OLS fit in log-log
    valid = ~np.isnan(rs_values) & (rs_values > 0)
    log_n = np.log(n_values[valid].astype(float))
    log_rs = np.log(rs_values[valid])
    H = np.polyfit(log_n, log_rs, 1)[0]

    return n_values, rs_values, H


def dfa(x: np.ndarray, order: int = 1, n_values: np.ndarray = None):
    """Detrended Fluctuation Analysis (DFA1/2/3).

    Parameters
    ----------
    x : 1D time series
    order : polynomial detrending order (1, 2, or 3)
    n_values : segment lengths

    Returns (n_values, F_values, alpha).
    """
    T = len(x)
    profile = np.cumsum(x - np.mean(x))

    if n_values is None:
        n_values = np.unique(np.geomspace(10, T // 4, num=30).astype(int))
        n_values = n_values[n_values >= order + 2]

    F_values = np.empty(len(n_values))
    for idx, n in enumerate(n_values):
        n = int(n)
        num_seg = T // n
        variances = []

        # Forward segments
        for s in range(num_seg):
            segment = profile[s * n: (s + 1) * n]
            t_local = np.arange(n, dtype=float)
            coeffs = np.polyfit(t_local, segment, order)
            trend = np.polyval(coeffs, t_local)
            variances.append(np.mean((segment - trend) ** 2))

        # Backward segments (use remaining data from the end)
        for s in range(num_seg):
            segment = profile[T - (s + 1) * n: T - s * n]
            t_local = np.arange(n, dtype=float)
            coeffs = np.polyfit(t_local, segment, order)
            trend = np.polyval(coeffs, t_local)
            variances.append(np.mean((segment - trend) ** 2))

        F_values[idx] = np.sqrt(np.mean(variances)) if variances else 0.0

    # OLS in log-log
    valid = F_values > 0
    log_n = np.log(n_values[valid].astype(float))
    log_F = np.log(F_values[valid])
    alpha = np.polyfit(log_n, log_F, 1)[0]

    return n_values, F_values, alpha
