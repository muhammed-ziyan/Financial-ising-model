"""Long-memory analysis: ACF, R/S Hurst exponent, and DFA.

Three complementary methods for quantifying long-range dependence in a
1D time series, reproducing Tables II–III of Guimaraes & Lima (2021):

* :func:`acf` — sample autocorrelation function up to a given maximum lag.
* :func:`rs_analysis` — Rescaled Range (R/S) analysis; returns the Hurst
  exponent *H* via OLS log-log regression.
* :func:`dfa` — Detrended Fluctuation Analysis of order 1, 2, or 3.

All three functions accept a 1D ``np.ndarray`` and return the fitted
scaling exponent as a float so that they can be used interchangeably in
the calibration and evaluation scripts.
"""

import numpy as np


def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute the sample autocorrelation function (ACF) up to ``max_lag``.

    Uses the biased (1/n) estimator consistent with standard signal-
    processing conventions.

    Parameters
    ----------
    x : np.ndarray
        1D time series.
    max_lag : int
        Maximum lag to compute (inclusive).

    Returns
    -------
    np.ndarray, shape (max_lag + 1,)
        ACF values for lags 0, 1, …, max_lag.  ``acf[0] == 1`` by
        construction.
    """
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
    """Rescaled Range (R/S) analysis for estimating the Hurst exponent.

    For each block size *n*, the series is divided into non-overlapping
    blocks.  For each block the rescaled range R/S is computed as the
    range of the cumulative deviation divided by the block standard
    deviation.  The Hurst exponent *H* is then the OLS slope of
    log(R/S) vs log(n).

    The minimum block size defaults to 50, following Weron (2002), to
    avoid the short-memory bias present at small *n*.

    Parameters
    ----------
    x : np.ndarray
        1D stationary time series (e.g. normalised returns).
    n_values : np.ndarray, optional
        Block sizes to evaluate.  If *None*, 30 geometrically spaced
        values from 50 to T//3 are used.

    Returns
    -------
    n_values : np.ndarray
        Block sizes used.
    rs_values : np.ndarray
        Mean R/S for each block size.
    H : float
        Hurst exponent from the OLS log-log fit.
    """
    T = len(x)
    if n_values is None:
        # Minimum box size n=50 per Weron (2002); upper limit T//3 gives a
        # denser log-log range for the long time series used after
        # calibration (standard practice once T >= 5000).
        n_values = np.unique(np.geomspace(50, T // 3, num=30).astype(int))

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
    """Detrended Fluctuation Analysis (DFA) of polynomial order 1, 2, or 3.

    The profile Y(t) = cumsum(x - mean(x)) is divided into non-overlapping
    segments of length *n*.  Each segment is detrended by fitting a
    polynomial of the specified ``order``, and the root-mean-squared
    residual F(n) is computed.  The DFA scaling exponent α is the OLS
    slope of log F(n) vs log(n).

    α ≈ 0.5 indicates uncorrelated noise; α > 0.5 indicates long-range
    positive correlations (persistence); α < 0.5 indicates anti-persistence.

    Parameters
    ----------
    x : np.ndarray
        1D stationary time series.
    order : int, optional
        Polynomial detrending order (1 = DFA1, 2 = DFA2, 3 = DFA3).
        Default is 1.
    n_values : np.ndarray, optional
        Segment lengths to evaluate.  If *None*, 40 geometrically spaced
        values between ``max(24, 4*(order+1))`` and T//4 are used.

    Returns
    -------
    n_values : np.ndarray
        Segment lengths used.
    F_values : np.ndarray
        Fluctuation function F(n) for each segment length.
    alpha : float
        DFA scaling exponent from the OLS log-log fit.
    """
    T = len(x)
    profile = np.cumsum(x - np.mean(x))

    if n_values is None:
        # Minimum segment size: enough points for a robust polynomial fit.
        # Floor raised to 24 so the smallest scale still has several
        # times the number of polynomial coefficients.
        n_min = max(24, 4 * (order + 1))
        n_values = np.unique(np.geomspace(n_min, T // 4, num=40).astype(int))

    F_values = np.empty(len(n_values))
    for idx, n in enumerate(n_values):
        n = int(n)
        num_seg = T // n
        variances = []

        # Forward non-overlapping segments only (standard DFA)
        for s in range(num_seg):
            segment = profile[s * n: (s + 1) * n]
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
