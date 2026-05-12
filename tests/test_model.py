"""Tests for the Ising market simulation engine."""

import numpy as np
import pytest
from src.config_loader import SimConfig
from src.model import IsingMarketModel, run_ensemble


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


def test_run_ensemble_count():
    """run_ensemble should return exactly n_seeds result dicts."""
    cfg = make_config(production=100, burn_in=50)
    runs = run_ensemble(cfg, n_seeds=3, base_seed=0)
    assert len(runs) == 3
    for r in runs:
        assert "returns" in r
        assert len(r["returns"]) == 100


def test_run_ensemble_seeds_differ():
    """Different seeds in an ensemble should produce different returns."""
    cfg = make_config(production=200, burn_in=50)
    runs = run_ensemble(cfg, n_seeds=2, base_seed=10)
    assert not np.array_equal(runs[0]["returns"], runs[1]["returns"]), (
        "Two different seeds produced identical returns"
    )


def test_explicit_seed_override():
    """Passing an explicit seed to IsingMarketModel should override cfg.seed."""
    cfg = make_config(seed=42)
    r1 = IsingMarketModel(cfg, seed=100).run()["returns"]
    r2 = IsingMarketModel(cfg, seed=100).run()["returns"]
    r3 = IsingMarketModel(cfg, seed=200).run()["returns"]
    np.testing.assert_array_equal(r1, r2)
    assert not np.array_equal(r1, r3)


def test_magnetization_bounded():
    """Magnetization must always lie in [-1, 1]."""
    cfg = make_config(production=500)
    results = IsingMarketModel(cfg).run()
    assert np.all(results['magnetization'] >= -1.0)
    assert np.all(results['magnetization'] <= 1.0)


def test_sweeps_per_step():
    """sweeps_per_step > 1 should run without error."""
    cfg = make_config(sweeps_per_step=3, production=100, burn_in=20)
    results = IsingMarketModel(cfg).run()
    assert len(results['returns']) == 100
