"""Tests for the Ising market simulation engine."""

import numpy as np
import pytest
from src.config_loader import SimConfig
from src.model import IsingMarketModel


def make_config(**overrides):
    defaults = dict(
        case_name='test', m=5, lambda_=0.2, b_max=0.09,
        c_max=1.0, delta=1, beta=1.0, eta=30.0,
        noise_scale=0.5, burn_in=100, production=500, seed=42
    )
    defaults.update(overrides)
    return SimConfig(**defaults)


def test_reproducibility():
    """Same seed -> identical output."""
    cfg = make_config()
    r1 = IsingMarketModel(cfg).run()['returns']
    r2 = IsingMarketModel(cfg).run()['returns']
    np.testing.assert_array_equal(r1, r2)


def test_zero_beta_random_spins():
    """At beta=0 (infinite temperature), magnetization should be near 0."""
    cfg = make_config(beta=0.0, burn_in=0, production=1000)
    results = IsingMarketModel(cfg).run()
    mean_mag = np.mean(np.abs(results['magnetization']))
    # With N=125, random spins give |mag| ~ 1/sqrt(N) ~ 0.09
    assert mean_mag < 0.3, f"Mean |magnetization| = {mean_mag} too large for beta=0"


def test_return_scale():
    """Returns should be of order 1/eta."""
    cfg = make_config(eta=30.0, production=1000)
    results = IsingMarketModel(cfg).run()
    std_r = np.std(results['returns'])
    # eta=30, max possible |return| = 1/eta ~ 0.033
    assert std_r < 0.1, f"Return std = {std_r} too large"
    assert std_r > 1e-6, f"Return std = {std_r} too small"


def test_price_positive():
    """Price should remain positive and finite."""
    cfg = make_config(production=2000)
    results = IsingMarketModel(cfg).run()
    assert np.all(results['price'] > 0)
    assert np.all(np.isfinite(results['price']))


def test_coupling_bounded():
    """J_coupling should not diverge over a simulation."""
    cfg = make_config(burn_in=0, production=2000)
    model = IsingMarketModel(cfg)
    max_J = 0
    for _ in range(cfg.production):
        model.step()
        max_J = max(max_J, np.max(np.abs(model.J_coupling)))
    assert max_J < 100, f"Max |J| = {max_J}, couplings may be diverging"


def test_output_lengths():
    """Output arrays should have correct length."""
    cfg = make_config(production=500)
    results = IsingMarketModel(cfg).run()
    for key in ['magnetization', 'returns', 'news', 'price']:
        assert len(results[key]) == 500, f"{key} has wrong length"
