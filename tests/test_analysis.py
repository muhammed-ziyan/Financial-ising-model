"""Tests for statistical analysis modules."""

import numpy as np
import pytest
from src.analysis_tail import hill_estimator, empirical_ccdf
from src.analysis_memory import rs_analysis, dfa, acf


def test_hill_estimator_pareto():
    """Hill estimator on Pareto(alpha=3) data should recover ~3."""
    rng = np.random.default_rng(123)
    # Pareto: P(X > x) = x^{-alpha} for x >= 1
    data = (1 - rng.random(10000)) ** (-1.0 / 3.0)  # alpha=3
    k = 500
    alpha_hat = hill_estimator(data, k)
    assert 2.0 < alpha_hat < 4.5, f"Hill estimate {alpha_hat} not near 3"


def test_ccdf_monotone():
    """CCDF should be non-increasing."""
    data = np.random.randn(1000)
    x, ccdf = empirical_ccdf(data)
    assert np.all(np.diff(ccdf) <= 0)


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
    """DFA1 exponent for random walk (cumsum of white noise) should be ~1.5."""
    rng = np.random.default_rng(42)
    data = np.cumsum(rng.standard_normal(5000))
    _, _, alpha = dfa(data, order=1)
    assert 1.2 < alpha < 1.8, f"DFA1 alpha = {alpha}, expected ~1.5"


def test_acf_white_noise():
    """ACF of white noise at lag > 0 should be near zero."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal(5000)
    result = acf(data, 50)
    assert result[0] == pytest.approx(1.0, abs=0.01)
    assert np.all(np.abs(result[5:]) < 0.1)
