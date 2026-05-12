"""Centralised figure generation for the Ising market model.

Each public function in this module creates one Matplotlib figure and
optionally saves it to disk (when ``savepath`` is provided).  All
functions return the :class:`matplotlib.figure.Figure` object so the
caller can further customise or display it inline in a notebook.

Naming convention
-----------------
``plot_*`` functions accept the pre-computed analysis arrays (from
``src.analysis_*``) together with ``case_name`` (string label for the
title) and an optional ``savepath``.

Saving behaviour
----------------
If ``savepath`` is given the figure is saved at 150 dpi with
``bbox_inches='tight'`` and the axes are closed to free memory.  Parent
directories are created automatically.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def _savefig(fig, savepath):
    """Save *fig* to *savepath* (creating parent dirs) and close it."""
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_return_series(returns, case_name, savepath=None):
    """Plot the raw return time series r(t).

    Parameters
    ----------
    returns : np.ndarray
        1D array of raw (un-normalized) returns.
    case_name : str
        Case label used in the plot title (e.g. ``"A"``).
    savepath : str, optional
        File path to save the figure.  If *None* the figure is not saved.
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(returns, linewidth=0.5)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Return r(t)')
    ax.set_title(f'Return time series — Case {case_name}')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    _savefig(fig, savepath)
    return fig


def plot_price_series(price, case_name, savepath=None):
    """Plot the reconstructed log-price time series p(t) = exp(cumsum r).

    Parameters
    ----------
    price : np.ndarray
        1D array of price values.
    case_name : str
        Case label used in the plot title.
    savepath : str, optional
        File path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(price, linewidth=0.8)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Price p(t)')
    ax.set_title(f'Price time series — Case {case_name}')
    _savefig(fig, savepath)
    return fig


def plot_return_histogram(returns_norm, case_name, savepath=None):
    """Plot a log-scale histogram of normalized returns with a Gaussian overlay.

    Demonstrates the fat-tailed (leptokurtic) return distribution.

    Parameters
    ----------
    returns_norm : np.ndarray
        Normalized return series g (paper Eq. 4).
    case_name : str
        Case label used in the plot title.
    savepath : str, optional
        File path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(returns_norm, bins=80, density=True, alpha=0.7, label='Model')
    x_range = np.linspace(-6, 6, 300)
    ax.plot(x_range, norm.pdf(x_range), 'r-', linewidth=1.5, label='Gaussian')
    ax.set_xlabel('Normalized return g')
    ax.set_ylabel('PDF')
    ax.set_title(f'Return distribution — Case {case_name}')
    ax.set_yscale('log')
    ax.legend()
    _savefig(fig, savepath)
    return fig


def plot_ccdf(x, ccdf, case_name, tail_label='positive',
              fit_alpha=None, savepath=None):
    """Log-log CCDF scatter with an optional power-law reference line.

    Parameters
    ----------
    x : np.ndarray
        Sorted tail values (output of :func:`src.analysis_tail.tail_ccdf`).
    ccdf : np.ndarray
        Empirical complementary CDF values.
    case_name : str
        Case label used in the plot title.
    tail_label : str, optional
        ``"positive"`` or ``"negative"``; used in axis and title labels.
    fit_alpha : float, optional
        Power-law exponent β for the reference slope line.  If *None* or
        not finite, no line is drawn.
    savepath : str, optional
        File path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x, ccdf, '.', markersize=2, label='Empirical CCDF')
    if fit_alpha is not None and np.isfinite(fit_alpha):
        # Reference line
        x_fit = np.linspace(x[len(x) // 4], x[-10] if len(x) > 10 else x[-1], 50)
        y_fit = (x_fit / x_fit[0]) ** (-fit_alpha) * ccdf[len(x) // 4]
        ax.plot(x_fit, y_fit, 'r--', linewidth=1.5,
                label=f'$\\beta$ = {fit_alpha:.2f}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(f'|g| ({tail_label} tail)')
    ax.set_ylabel('P(|g| > x)')
    ax.set_title(f'CCDF {tail_label} tail — Case {case_name}')
    ax.legend()
    _savefig(fig, savepath)
    return fig


def plot_acf_comparison(acf_returns, acf_volatility, max_lag, case_name,
                        savepath=None):
    """Plot ACF of returns and ACF of |returns| on the same axes.

    The contrast between a quickly decaying return ACF and a slowly
    decaying |return| ACF is the signature of volatility clustering.

    Parameters
    ----------
    acf_returns : np.ndarray
        ACF of the normalized return series.
    acf_volatility : np.ndarray
        ACF of the absolute normalized returns.
    max_lag : int
        Maximum lag displayed.
    case_name : str
        Case label used in the plot title.
    savepath : str, optional
        File path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    lags = np.arange(max_lag + 1)
    ax.plot(lags, acf_returns, label='ACF of returns', alpha=0.8)
    ax.plot(lags, acf_volatility, label='ACF of |returns|', alpha=0.8)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(f'Volatility clustering — Case {case_name}')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.legend()
    _savefig(fig, savepath)
    return fig


def plot_rs_hurst(n_values, rs_values, H, case_name, savepath=None):
    """Log-log R/S scatter plot with the OLS fit line.

    Parameters
    ----------
    n_values : np.ndarray
        Block sizes (from :func:`src.analysis_memory.rs_analysis`).
    rs_values : np.ndarray
        Mean R/S for each block size.
    H : float
        Estimated Hurst exponent (displayed in the legend).
    case_name : str
        Case label used in the plot title.
    savepath : str, optional
        File path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    valid = (rs_values > 0) & ~np.isnan(rs_values)
    log_n = np.log10(n_values[valid].astype(float))
    log_rs = np.log10(rs_values[valid])
    ax.plot(log_n, log_rs, 'o', markersize=4, label='R/S data')
    # Fit line
    coeffs = np.polyfit(log_n, log_rs, 1)
    ax.plot(log_n, np.polyval(coeffs, log_n), 'r-', linewidth=1.5,
            label=f'H = {H:.4f}')
    ax.set_xlabel('log$_{10}$(n)')
    ax.set_ylabel('log$_{10}$(R/S)')
    ax.set_title(f'R/S analysis — Case {case_name}')
    ax.legend()
    _savefig(fig, savepath)
    return fig


def plot_dfa(n_values, F_values, alpha, order, case_name, savepath=None):
    """Log-log DFA fluctuation function F(n) with the OLS fit line.

    Parameters
    ----------
    n_values : np.ndarray
        Segment lengths (from :func:`src.analysis_memory.dfa`).
    F_values : np.ndarray
        Fluctuation function F(n).
    alpha : float
        DFA scaling exponent.
    order : int
        Polynomial detrending order (1, 2, or 3) — shown in the legend.
    case_name : str
        Case label used in the plot title.
    savepath : str, optional
        File path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    valid = F_values > 0
    log_n = np.log10(n_values[valid].astype(float))
    log_F = np.log10(F_values[valid])
    ax.plot(log_n, log_F, 'o', markersize=4, label='F(n)')
    coeffs = np.polyfit(log_n, log_F, 1)
    ax.plot(log_n, np.polyval(coeffs, log_n), 'r-', linewidth=1.5,
            label=f'DFA{order}: $\\alpha$ = {alpha:.4f}')
    ax.set_xlabel('log$_{10}$(n)')
    ax.set_ylabel('log$_{10}$(F(n))')
    ax.set_title(f'DFA{order} — Case {case_name}')
    ax.legend()
    _savefig(fig, savepath)
    return fig


def plot_multifractal_hq(q_values, hq, case_name, savepath=None):
    """Plot the generalised Hurst exponent h(q) vs q.

    A non-constant, monotonically decreasing h(q) curve signals genuine
    multifractality.  The dashed grey line at h = 0.5 marks the
    uncorrelated (Brownian) reference.

    Parameters
    ----------
    q_values : np.ndarray
        Moment orders.
    hq : np.ndarray
        Generalised Hurst exponents h(q).
    case_name : str
        Case label.
    savepath : str, optional
        File path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    valid = np.isfinite(hq)
    ax.plot(q_values[valid], hq[valid], 'o-', markersize=4)
    ax.set_xlabel('q')
    ax.set_ylabel('h(q)')
    ax.set_title(f'Generalized Hurst exponent — Case {case_name}')
    ax.axhline(0.5, color='gray', linewidth=0.5, linestyle='--')
    _savefig(fig, savepath)
    return fig


def plot_multifractal_tauq(q_values, tauq, case_name, savepath=None):
    """Plot the Rényi scaling exponent τ(q) vs q.

    τ(q) = q·h(q) − 1.  A linear τ(q) indicates monofractality;
    curvature is the fingerprint of multifractal behaviour.

    Parameters
    ----------
    q_values : np.ndarray
        Moment orders.
    tauq : np.ndarray
        Rényi exponents τ(q).
    case_name : str
        Case label.
    savepath : str, optional
        File path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    valid = np.isfinite(tauq)
    ax.plot(q_values[valid], tauq[valid], 'o-', markersize=4)
    ax.set_xlabel('q')
    ax.set_ylabel('$\\tau$(q)')
    ax.set_title(f'Renyi exponent — Case {case_name}')
    _savefig(fig, savepath)
    return fig


def plot_multifractal_spectrum(alpha_mf, f_alpha, case_name, savepath=None):
    """Plot the multifractal spectrum f(α) vs α.

    The width Δα = α_max − α_min of the parabolic arc measures the degree
    of multifractality.  A single point (monofractal) would appear as
    Δα ≈ 0.

    Parameters
    ----------
    alpha_mf : np.ndarray
        Singularity strength α = dτ/dq.
    f_alpha : np.ndarray
        Multifractal spectrum f(α) = q·α − τ(q).
    case_name : str
        Case label.
    savepath : str, optional
        File path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    valid = np.isfinite(alpha_mf) & np.isfinite(f_alpha)
    ax.plot(alpha_mf[valid], f_alpha[valid], 'o-', markersize=4)
    ax.set_xlabel('$\\alpha$')
    ax.set_ylabel('f($\\alpha$)')
    ax.set_title(f'Multifractal spectrum — Case {case_name}')
    _savefig(fig, savepath)
    return fig
