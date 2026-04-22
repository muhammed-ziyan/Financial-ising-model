"""CLI entry point: run simulation, analysis, and plotting.

Usage:
    python run_simulation.py config/case_A.yaml --analyze --plot
"""

import argparse
import os
import numpy as np

from src.config_loader import load_config
from src.model import IsingMarketModel
from src.observables import normalize_returns, compute_volatility, save_results
from src.analysis_tail import tail_ccdf, hill_estimator, hill_adaptive, fit_tail_ols
from src.analysis_memory import acf, rs_analysis, dfa
from src.analysis_multifractal import mfdfa
from src import plotting as plt_mod


def main():
    parser = argparse.ArgumentParser(description='3D Ising Market Model')
    parser.add_argument('config', help='Path to YAML config file')
    parser.add_argument('--analyze', action='store_true', help='Run statistical analysis')
    parser.add_argument('--plot', action='store_true', help='Generate figures')
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"=== Case {cfg.case_name}: m={cfg.m}, N={cfg.m**3}, "
          f"lambda={cfg.lambda_}, b_max={cfg.b_max} ===")

    # --- Simulation (Layers A + B) ---
    print("Running simulation...")
    model = IsingMarketModel(cfg)
    results = model.run()

    outdir = f"data/outputs/{cfg.case_name}"
    os.makedirs(outdir, exist_ok=True)
    save_results(f"{outdir}/timeseries.npz", **results)
    print(f"  Saved time series to {outdir}/timeseries.npz")

    returns = results['returns']
    price = results['price']
    g = normalize_returns(returns)

    print(f"  Return: mean={np.mean(returns):.6f}, std={np.std(returns):.6f}")
    print(f"  Price: final={price[-1]:.4f}, min={price.min():.4f}, max={price.max():.4f}")

    if not args.analyze:
        print("Done. Use --analyze to run statistical analysis.")
        return

    # --- Layer C: Statistical Validation ---
    print("\n--- Tail Analysis ---")
    pos_x, pos_ccdf = tail_ccdf(g, positive=True)
    neg_x, neg_ccdf = tail_ccdf(g, positive=False)

    pos_tail = g[g > 0]
    neg_tail = np.abs(g[g < 0])

    # OLS regression on CCDF tail (primary — matches paper's method)
    # Fit in the tail region: large |g| where power-law behavior holds
    ols_pos = ols_neg = np.nan
    if len(pos_x) > 20:
        x_min = np.percentile(pos_tail, 80)
        x_max = pos_tail.max()
        ols_pos, _ = fit_tail_ols(pos_x, pos_ccdf, x_min, x_max)
        print(f"  OLS tail fit:  beta+ = {ols_pos:.4f}")
    if len(neg_x) > 20:
        x_min = np.percentile(neg_tail, 80)
        x_max = neg_tail.max()
        ols_neg, _ = fit_tail_ols(neg_x, neg_ccdf, x_min, x_max)
        print(f"  OLS tail fit:  beta- = {ols_neg:.4f}")

    # Hill estimator (secondary cross-check)
    alpha_pos, k_pos = hill_adaptive(pos_tail)
    alpha_neg, k_neg = hill_adaptive(neg_tail)
    print(f"  Hill adaptive: beta+ = {alpha_pos:.4f} (k={k_pos})")
    print(f"  Hill adaptive: beta- = {alpha_neg:.4f} (k={k_neg})")

    # Use OLS as primary if available
    alpha_pos = ols_pos if np.isfinite(ols_pos) else alpha_pos
    alpha_neg = ols_neg if np.isfinite(ols_neg) else alpha_neg

    print("\n--- Volatility Clustering ---")
    max_lag = 100
    acf_r = acf(g, max_lag)
    acf_v = acf(np.abs(g), max_lag)
    print(f"  ACF returns at lag 1: {acf_r[1]:.4f}, lag 10: {acf_r[10]:.4f}")
    print(f"  ACF |returns| at lag 1: {acf_v[1]:.4f}, lag 10: {acf_v[10]:.4f}")

    print("\n--- R/S Hurst Analysis ---")
    n_rs, rs_vals, H_rs = rs_analysis(g)
    print(f"  Hurst exponent (R/S): H = {H_rs:.4f}")

    print("\n--- DFA Analysis ---")
    dfa_results = {}
    for order in [1, 2, 3]:
        n_d, F_d, alpha_d = dfa(g, order=order)
        dfa_results[order] = (n_d, F_d, alpha_d)
        print(f"  DFA{order} exponent: alpha = {alpha_d:.4f}")

    print("\n--- Multifractal DFA ---")
    q_vals = np.concatenate([np.arange(-5, 0, 0.5), np.arange(0.5, 5.5, 0.5)])
    mf = mfdfa(g, q_vals, order=1)
    valid_hq = np.isfinite(mf['hq'])
    if np.any(valid_hq):
        print(f"  h(q=-5) = {mf['hq'][0]:.4f}, h(q=2) = {mf['hq'][np.argmin(np.abs(q_vals - 2))]:.4f}, "
              f"h(q=5) = {mf['hq'][-1]:.4f}")
        delta_h = np.max(mf['hq'][valid_hq]) - np.min(mf['hq'][valid_hq])
        print(f"  Delta h = {delta_h:.4f} (multifractal width)")

    # --- Summary Table ---
    print(f"\n{'='*50}")
    print(f"SUMMARY — Case {cfg.case_name}")
    print(f"{'='*50}")
    print(f"  Tail exponent (Hill): beta+ = {alpha_pos:.2f}, beta- = {alpha_neg:.2f}")
    print(f"  Hurst (R/S):          H = {H_rs:.4f}")
    for order in [1, 2, 3]:
        print(f"  DFA{order}:                alpha = {dfa_results[order][2]:.4f}")
    print(f"{'='*50}")

    # --- Plotting ---
    if args.plot:
        print("\nGenerating figures...")
        figdir = f"figures/{cfg.case_name}"
        os.makedirs(figdir, exist_ok=True)

        plt_mod.plot_return_series(returns, cfg.case_name, f"{figdir}/returns.png")
        plt_mod.plot_price_series(price, cfg.case_name, f"{figdir}/price.png")
        plt_mod.plot_return_histogram(g, cfg.case_name, f"{figdir}/hist_returns.png")
        plt_mod.plot_ccdf(pos_x, pos_ccdf, cfg.case_name, 'positive',
                          alpha_pos, f"{figdir}/ccdf_positive.png")
        plt_mod.plot_ccdf(neg_x, neg_ccdf, cfg.case_name, 'negative',
                          alpha_neg, f"{figdir}/ccdf_negative.png")
        plt_mod.plot_acf_comparison(acf_r, acf_v, max_lag, cfg.case_name,
                                    f"{figdir}/acf_comparison.png")
        plt_mod.plot_rs_hurst(n_rs, rs_vals, H_rs, cfg.case_name,
                              f"{figdir}/rs_hurst.png")
        for order in [1, 2, 3]:
            n_d, F_d, alpha_d = dfa_results[order]
            plt_mod.plot_dfa(n_d, F_d, alpha_d, order, cfg.case_name,
                             f"{figdir}/dfa{order}.png")
        plt_mod.plot_multifractal_hq(mf['q_values'], mf['hq'], cfg.case_name,
                                     f"{figdir}/mf_hq.png")
        plt_mod.plot_multifractal_tauq(mf['q_values'], mf['tauq'], cfg.case_name,
                                       f"{figdir}/mf_tauq.png")
        plt_mod.plot_multifractal_spectrum(mf['alpha'], mf['f_alpha'], cfg.case_name,
                                           f"{figdir}/mf_spectrum.png")
        print(f"  Figures saved to {figdir}/")

    print("\nDone.")


if __name__ == '__main__':
    main()
