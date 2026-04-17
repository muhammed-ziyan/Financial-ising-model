"""Multifractal Detrended Fluctuation Analysis (MF-DFA)."""

import numpy as np


def mfdfa(x: np.ndarray, q_values: np.ndarray, order: int = 1,
          n_values: np.ndarray = None) -> dict:
    """Multifractal DFA.

    Parameters
    ----------
    x : 1D time series
    q_values : array of q values (should not include 0)
    order : polynomial detrending order
    n_values : segment sizes

    Returns
    -------
    dict with keys: n_values, q_values, Fq, hq, tauq, alpha, f_alpha
    """
    T = len(x)
    profile = np.cumsum(x - np.mean(x))

    if n_values is None:
        n_values = np.unique(np.geomspace(10, T // 4, num=30).astype(int))
        n_values = n_values[n_values >= order + 2]

    # Compute segment variances for all scales
    # var_segments[scale_idx] = list of F^2(v, n) for each segment
    var_segments = []
    for n in n_values:
        n = int(n)
        num_seg = T // n
        seg_vars = []

        # Forward segments
        for s in range(num_seg):
            segment = profile[s * n: (s + 1) * n]
            t_local = np.arange(n, dtype=float)
            coeffs = np.polyfit(t_local, segment, order)
            trend = np.polyval(coeffs, t_local)
            seg_vars.append(np.mean((segment - trend) ** 2))

        # Backward segments
        for s in range(num_seg):
            segment = profile[T - (s + 1) * n: T - s * n]
            t_local = np.arange(n, dtype=float)
            coeffs = np.polyfit(t_local, segment, order)
            trend = np.polyval(coeffs, t_local)
            seg_vars.append(np.mean((segment - trend) ** 2))

        var_segments.append(np.array(seg_vars))

    # Compute Fq(n) for each q and n
    n_scales = len(n_values)
    n_q = len(q_values)
    Fq = np.empty((n_q, n_scales))

    for qi, q in enumerate(q_values):
        for ni, seg_vars in enumerate(var_segments):
            # Filter out zero variances
            sv = seg_vars[seg_vars > 0]
            if len(sv) == 0:
                Fq[qi, ni] = np.nan
                continue

            if q == 0:
                # F_0(n) = exp(mean(ln(F^2)) / 2)
                Fq[qi, ni] = np.exp(0.5 * np.mean(np.log(sv)))
            else:
                # F_q(n) = [mean(F^2^{q/2})]^{1/q}
                Fq[qi, ni] = np.mean(sv ** (q / 2.0)) ** (1.0 / q)

    # Fit log Fq vs log n to get h(q) for each q
    hq = np.empty(n_q)
    log_n = np.log(n_values.astype(float))

    for qi in range(n_q):
        valid = np.isfinite(Fq[qi]) & (Fq[qi] > 0)
        if np.sum(valid) < 2:
            hq[qi] = np.nan
            continue
        hq[qi] = np.polyfit(log_n[valid], np.log(Fq[qi, valid]), 1)[0]

    # tau(q) = q * h(q) - 1
    tauq = q_values * hq - 1.0

    # Legendre transform: alpha = d(tau)/dq, f(alpha) = q*alpha - tau
    # Use central differences for interior points, forward/backward at edges
    alpha = np.gradient(tauq, q_values)
    f_alpha = q_values * alpha - tauq

    return {
        'n_values': n_values,
        'q_values': q_values,
        'Fq': Fq,
        'hq': hq,
        'tauq': tauq,
        'alpha': alpha,
        'f_alpha': f_alpha,
    }
