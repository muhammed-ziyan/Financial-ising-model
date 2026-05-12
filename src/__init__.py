"""FinIising — 3D Ising Financial Market Model.

This package implements the self-organizing 3D Ising model of financial markets
described in Guimaraes & Lima, *Phys. Rev. E* 103, 062130 (2021).

Modules
-------
lattice
    3D cubic lattice construction with periodic boundary conditions.
model
    Monte Carlo simulation engine (Glauber heat-bath updates, optional
    Numba JIT acceleration).
config_loader
    YAML configuration loader with case/default merging.
observables
    Return normalization, volatility proxy, and .npz I/O helpers.
analysis_memory
    Long-memory statistics: ACF, R/S Hurst exponent, DFA (orders 1–3).
analysis_multifractal
    Multifractal DFA (MF-DFA): h(q), τ(q), f(α) spectrum.
analysis_tail
    Return-tail analysis: empirical CCDF, OLS power-law fits, Hill estimator.
plotting
    Matplotlib figure generators for all standard diagnostic plots.
"""
