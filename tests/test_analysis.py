"""Tests for statistical analysis modules."""

import numpy as np
import pytest
from src.analysis_tail import (
    hill_estimator, hill_adaptive, empirical_ccdf,
    tail_ccdf, fit_tail_paper, fit_tail_best,
)
from src.analysis_memory import rs_analysis, dfa, acf
from src.analysis_multifractal import mfdfa


# ── tail analysis ─────────────────────────────────────────────────────────────

def test_hill_estimator_pareto():
    """Hill estimator on Pareto(alpha=3) data should recover ~3."""
    rng = np.random.default_rng(123)
    data = (1 - rng.random(10000)) ** (-1.0 / 3.0)  # Pareto, alpha=3
    alpha_hat = hill_estimator(data, 500)
    assert 2.0 < alpha_hat < 4.5, f"Hill estimate {alpha_hat} not near 3"


def test_hill_estimator_invalid_k():
    """hill_estimator raises ValueError for out-of-range k."""
    data = np.ones(50)
    with pytest.raises(ValueError):
        hill_estimator(data, k=0)
    with pytest.raises(ValueError):
        hill_estimator(data, k=50)


def test_hill_adaptive_pareto():
    """hill_adaptive on Pareto(alpha=3) data should return a value near 3."""
    rng = np.random.default_rng(7)
    data = (1 - rng.random(5000)) ** (-1.0 / 3.0)
    alpha_hat, k_opt = hill_adaptive(data)
    assert 1.5 < alpha_hat < 5.0, f"hill_adaptive estimate {alpha_hat} unreasonable"
    assert k_opt > 0


def test_hill_adaptive_too_small():
    """hill_adaptive raises ValueError for tiny input."""
    with pytest.raises(ValueError, match="30"):
        hill_adaptive(np.ones(10))


def test_ccdf_monotone():
    """CCDF should be non-increasing."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal(1000)
    x, ccdf = empirical_ccdf(data)
    assert np.all(np.diff(ccdf) <= 0)


def test_ccdf_length():
    """CCDF should have the same length as input data."""
    data = np.arange(1, 101, dtype=float)
    x, ccdf = empirical_ccdf(data)
    assert len(x) == len(data)
    assert len(ccdf) == len(data)


def test_tail_ccdf_positive_only():
    """tail_ccdf(positive=True) should return only positive values."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal(2000)
    x, ccdf = tail_ccdf(data, positive=True)
    assert np.all(x > 0)


def test_fit_tail_paper_pareto():
    """fit_tail_paper on Pareto data should estimate beta near 3."""
    rng = np.random.default_rng(42)
    data = (1 - rng.random(5000)) ** (-1.0 / 3.0)
    result = fit_tail_paper(data, lower_pct=90.0, upper_pct=99.5)
    assert result is not None
    assert np.isfinite(result["beta"])
    assert 1.5 < result["beta"] < 5.0


def test_fit_tail_best_returns_dict_or_none():
    """fit_tail_best returns a dict with 'beta' key or None."""
    rng = np.random.default_rng(99)
    data = (1 - rng.random(2000)) ** (-1.0 / 3.0)
    result = fit_tail_best(data)
    assert result is not None
    assert "beta" in result
    assert np.isfinite(result["beta"])


# ── memory analysis ────────────────────────────────────────────────────────────

def test_rs_white_noise():
    """R/S Hurst for white noise should be ~0.5."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal(5000)
    _, _, H = rs_analysis(data)
    assert 0.3 < H < 0.7, f"RS Hurst = {H}, expected ~0.5"


def test_dfa1_white_noise():
    """DFA1 exponent for white noise should be ~0.5."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal(5000)
    _, _, alpha = dfa(data, order=1)
    assert 0.3 < alpha < 0.7, f"DFA1 alpha = {alpha}, expected ~0.5"


def test_dfa1_random_walk():
    """DFA1 exponent for a random walk should be ~1.5."""
    rng = np.random.default_rng(42)
    data = np.cumsum(rng.standard_normal(5000))
    _, _, alpha = dfa(data, order=1)
    assert 1.2 < alpha < 1.8, f"DFA1 alpha = {alpha}, expected ~1.5"


def test_dfa_orders_return_arrays():
    """DFA of orders 1, 2, 3 should all return (n_vals, F_vals, alpha)."""
    rng = np.random.default_rng(5)
    data = rng.standard_normal(2000)
    for order in (1, 2, 3):
        n_vals, F_vals, alpha = dfa(data, order=order)
        assert len(n_vals) == len(F_vals)
        assert np.isfinite(alpha)


def test_acf_lag0_is_one():
    """ACF at lag 0 should equal 1."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal(5000)
    result = acf(data, 50)
    assert result[0] == pytest.approx(1.0, abs=0.01)


def test_acf_white_noise_small_at_high_lags():
    """ACF of white noise at lag > 5 should be near zero."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal(5000)
    result = acf(data, 50)
    assert np.all(np.abs(result[5:]) < 0.1)


# ── multifractal ───────────────────────────────────────────────────────────────

def test_mfdfa_smoke():
    """MF-DFA should run without error and return expected keys."""
    rng = np.random.default_rng(11)
    data = rng.standard_normal(2000)
    q_values = np.array([-2.0, -1.0, 1.0, 2.0])
    result = mfdfa(data, q_values, order=1)
    for key in ("n_values", "q_values", "Fq", "hq", "tauq", "alpha", "f_alpha"):
        assert key in result, f"Missing key: {key}"
    assert result["Fq"].shape == (len(q_values), len(result["n_values"]))


def test_mfdfa_white_noise_hq_near_half():
    """For white noise, h(q) should be near 0.5 for all q."""
    rng = np.random.default_rng(22)
    data = rng.standard_normal(3000)
    q_values = np.array([1.0, 2.0, 3.0])
    result = mfdfa(data, q_values, order=1)
    for qi, h in enumerate(result["hq"]):
        if np.isfinite(h):
            assert 0.2 < h < 0.8, f"h(q={q_values[qi]}) = {h}, expected ~0.5"
