"""Return distribution analysis: CCDF and tail exponent estimation.

The paper reports a least-squares fit of the power-law cumulative
distribution ``P(|g| > x) ~ x^{-beta}`` in the region of *large* ``|r|``
(see Fig. 3 of Guimaraes & Lima 2021).  We implement a robust version
of the same procedure:

* ``empirical_ccdf`` / ``tail_ccdf`` build the empirical CCDF per side.
* ``fit_tail_ols`` does the log-log regression between two explicit
  thresholds.
* ``fit_tail_paper`` selects the fitting range automatically and returns
  the slope, the MSE reported by the paper, and the range used.
* ``hill_estimator`` / ``hill_adaptive`` keep the Hill estimator as a
  secondary cross-check (it is more biased for bounded magnetization
  tails but useful for sanity-checking).
"""

import numpy as np


def empirical_ccdf(data: np.ndarray):
    """Empirical complementary CDF.  Returns ``(sorted_x, ccdf)``."""
    x = np.sort(data)
    n = len(x)
    # Use (n-i)/n so that CCDF has the minimum value 1/n rather than 0
    # and avoids ``log(0)`` downstream.
    ccdf = (n - np.arange(n)) / n
    return x, ccdf


def tail_ccdf(data: np.ndarray, positive: bool = True):
    """CCDF of one tail.

    ``positive=True`` returns ``P(g > x)`` for ``x > 0`` and
    ``positive=False`` returns ``P(-g > x)`` for ``x > 0`` (i.e. the
    left tail expressed as ``|g|``).
    """
    if positive:
        tail = data[data > 0]
    else:
        tail = np.abs(data[data < 0])
    return empirical_ccdf(tail)


def hill_estimator(data: np.ndarray, k: int) -> float:
    """Hill estimator for the tail index.

    ``data`` should contain positive values from one tail.
    """
    sorted_data = np.sort(data)[::-1]
    if k >= len(sorted_data) or k < 1:
        raise ValueError(f"k={k} must be in [1, {len(sorted_data) - 1})")
    log_ratios = np.log(sorted_data[:k] / sorted_data[k])
    denom = float(np.sum(log_ratios))
    if denom == 0.0:
        return np.inf
    return k / denom


def hill_plot(data: np.ndarray, k_range: np.ndarray = None):
    if k_range is None:
        k_max = max(10, len(data) // 5)
        k_range = np.arange(5, k_max)
    alphas = np.array([hill_estimator(data, int(k)) for k in k_range])
    return k_range, alphas


def hill_adaptive(data: np.ndarray):
    """Hill estimator with plateau detection in the genuine tail (1%-10%)."""
    n = len(data)
    k_min = max(10, int(n * 0.01))
    k_max = max(k_min + 20, int(n * 0.10))
    k_range = np.arange(k_min, k_max)
    alphas = np.array([hill_estimator(data, int(k)) for k in k_range])

    window = max(5, len(k_range) // 5)
    best_var = np.inf
    best_start = 0
    for start in range(len(alphas) - window):
        local_var = np.var(alphas[start:start + window])
        if local_var < best_var:
            best_var = local_var
            best_start = start

    plateau = alphas[best_start:best_start + window]
    k_opt = int(k_range[best_start + window // 2])
    return float(np.median(plateau)), k_opt


def fit_tail_ols(x: np.ndarray, ccdf: np.ndarray,
                 x_min: float, x_max: float):
    """OLS fit of ``log CCDF = -beta log x + c`` on ``[x_min, x_max]``.

    Returns ``(beta, intercept, mse)`` with ``mse`` the mean-squared
    error of the residuals (in log-log space).  The MSE is the quantity
    reported alongside the slope in Fig. 3 of the paper.
    """
    mask = (x >= x_min) & (x <= x_max) & (ccdf > 0)
    if np.sum(mask) < 3:
        return np.nan, np.nan, np.nan
    log_x = np.log(x[mask])
    log_ccdf = np.log(ccdf[mask])
    A = np.vstack([log_x, np.ones_like(log_x)]).T
    coef, *_ = np.linalg.lstsq(A, log_ccdf, rcond=None)
    slope, intercept = coef
    residuals = log_ccdf - (slope * log_x + intercept)
    mse = float(np.mean(residuals ** 2))
    return float(-slope), float(intercept), mse


def fit_tail_paper(tail: np.ndarray,
                   lower_pct: float = 90.0,
                   upper_pct: float = 99.5):
    """Paper-style power-law fit in the large-|r| region.

    Parameters
    ----------
    tail : np.ndarray
        Positive values from one tail of the normalized return series.
    lower_pct, upper_pct : float
        Percentiles (in [0, 100]) that bracket the fitting region.
        ``lower_pct = 90`` matches the paper's "region of large |r|";
        ``upper_pct = 99.5`` excludes the two or three most extreme
        samples to stabilise the fit.

    Returns
    -------
    dict with keys ``beta`` (slope), ``intercept``, ``mse``,
    ``x_range`` (a 2-tuple of thresholds), ``n_points``.
    """
    x, ccdf = empirical_ccdf(tail)
    x_min = float(np.percentile(tail, lower_pct))
    x_max = float(np.percentile(tail, upper_pct))
    beta, intercept, mse = fit_tail_ols(x, ccdf, x_min, x_max)
    mask = (x >= x_min) & (x <= x_max) & (ccdf > 0)
    return {
        "beta": beta,
        "intercept": intercept,
        "mse": mse,
        "x_range": (x_min, x_max),
        "n_points": int(np.sum(mask)),
    }


def fit_tail_best(tail: np.ndarray,
                  lower_grid=(85.0, 88.0, 90.0, 92.0, 95.0),
                  upper_pct: float = 99.5):
    """Scan several lower thresholds and pick the fit with smallest MSE.

    This mirrors the paper's phrasing "a suitable cutoff" — we pick the
    cutoff that the log-log regression fits most tightly.
    """
    best = None
    for lp in lower_grid:
        res = fit_tail_paper(tail, lower_pct=lp, upper_pct=upper_pct)
        if not np.isfinite(res["beta"]) or res["n_points"] < 20:
            continue
        if best is None or res["mse"] < best["mse"]:
            best = res
            best["lower_pct"] = lp
    return best
