"""CLI entry point: run simulation, analysis, and plotting.

Usage
-----
    python run_simulation.py config/case_A.yaml --analyze --plot
    python run_simulation.py config/case_A.yaml --analyze --seeds 5
    python run_simulation.py config/my_case.yaml --analyze --plot \\
        --set model.lambda_=0.30 --set simulation.production=12000

Options
-------
``--seeds N``
    Run ``N`` independent realisations and ensemble-average the reported
    statistics.  ``N`` overrides the ``simulation.n_seeds`` entry of the
    YAML config.

``--set section.key=value``
    Override any YAML parameter on the command line without editing a
    config file.  The section prefix must match the YAML structure:
    ``lattice.*``, ``model.*``, or ``simulation.*``.  Values are parsed
    as float if possible, otherwise as int, otherwise as string.

    Examples::

        --set model.lambda_=0.30
        --set model.eta=2.5
        --set simulation.production=15000
        --set model.beta=1.2
"""

import argparse
import json
import os

import numpy as np

from src.analysis_memory import acf, dfa, rs_analysis
from src.analysis_multifractal import mfdfa
from src.analysis_tail import (
    fit_tail_best,
    hill_adaptive,
    tail_ccdf,
)
from src.config_loader import load_config
from src.model import run_ensemble
from src.observables import normalize_returns, save_results
from src import plotting as plt_mod


# ---------------------------------------------------------------------------
# Per-realisation analysis bundle
# ---------------------------------------------------------------------------

def _analyze_single(returns: np.ndarray, q_values: np.ndarray, max_lag: int,
                    verbose: bool = False) -> dict:
    """Compute the full statistics panel for a single realisation."""
    g = normalize_returns(returns, method="paper")

    # Tail fits (paper-style: large |r|, least-squares on CCDF).
    pos_tail = g[g > 0]
    neg_tail = np.abs(g[g < 0])
    pos_fit = fit_tail_best(pos_tail) if len(pos_tail) > 50 else None
    neg_fit = fit_tail_best(neg_tail) if len(neg_tail) > 50 else None

    # Hill estimator (secondary) — requires at least 20 tail points.
    _hill_min = 20
    if len(pos_tail) > _hill_min:
        hill_pos, k_pos = hill_adaptive(pos_tail)
    else:
        hill_pos, k_pos = float("nan"), 0
    if len(neg_tail) > _hill_min:
        hill_neg, k_neg = hill_adaptive(neg_tail)
    else:
        hill_neg, k_neg = float("nan"), 0

    # ACF.
    acf_r = acf(g, max_lag)
    acf_v = acf(np.abs(g), max_lag)

    # Long-range memory.
    n_rs, rs_vals, H_rs = rs_analysis(g)
    dfa_out = {order: dfa(g, order=order) for order in (1, 2, 3)}

    # Multifractal.
    mf = mfdfa(g, q_values, order=1)

    if verbose:
        print(f"    Hurst R/S = {H_rs:.4f}, DFA1/2/3 = "
              f"{dfa_out[1][2]:.4f}/{dfa_out[2][2]:.4f}/{dfa_out[3][2]:.4f}")

    return {
        "g": g,
        "pos_fit": pos_fit,
        "neg_fit": neg_fit,
        "hill_pos": (hill_pos, k_pos),
        "hill_neg": (hill_neg, k_neg),
        "acf_r": acf_r,
        "acf_v": acf_v,
        "rs": (n_rs, rs_vals, H_rs),
        "dfa": dfa_out,
        "mf": mf,
    }


def _aggregate(results: list[dict]) -> dict:
    """Ensemble-average the scalar metrics across realisations."""

    def _median(values):
        vals = [v for v in values if v is not None and np.isfinite(v)]
        return float(np.median(vals)) if vals else float("nan")

    H_rs = [r["rs"][2] for r in results]
    dfa_alpha = {order: [r["dfa"][order][2] for r in results] for order in (1, 2, 3)}
    beta_pos = [r["pos_fit"]["beta"] if r["pos_fit"] else np.nan for r in results]
    beta_neg = [r["neg_fit"]["beta"] if r["neg_fit"] else np.nan for r in results]
    hill_pos = [r["hill_pos"][0] for r in results]
    hill_neg = [r["hill_neg"][0] for r in results]

    return {
        "H_rs_median": _median(H_rs),
        "H_rs_values": H_rs,
        "DFA1_median": _median(dfa_alpha[1]),
        "DFA2_median": _median(dfa_alpha[2]),
        "DFA3_median": _median(dfa_alpha[3]),
        "DFA_values": dfa_alpha,
        "beta_pos_median": _median(beta_pos),
        "beta_neg_median": _median(beta_neg),
        "hill_pos_median": _median(hill_pos),
        "hill_neg_median": _median(hill_neg),
    }


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_set_value(raw: str):
    """Coerce a ``--set`` value string to int, float, or str (in that order).

    Integer strings like ``"100"`` are returned as ``int``; decimal strings
    like ``"0.30"`` are returned as ``float``; anything else is left as
    ``str``.
    """
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _build_overrides(set_args: list[str]) -> dict:
    """Convert ``["model.lambda_=0.30", "simulation.production=5000"]``
    into a nested dict suitable for :func:`load_config` overrides.

    Supported prefixes: ``lattice``, ``model``, ``simulation``.

    Raises
    ------
    SystemExit
        For malformed ``key=value`` pairs or unrecognised section prefixes.
    """
    allowed_sections = {"lattice", "model", "simulation"}
    overrides: dict = {}
    for item in set_args:
        if "=" not in item:
            print(f"ERROR: --set argument must be in 'section.key=value' form; got: {item!r}")
            raise SystemExit(1)
        key_path, _, raw_val = item.partition("=")
        parts = key_path.strip().split(".", 1)
        if len(parts) != 2:
            print(
                f"ERROR: --set key must be 'section.key' (e.g. model.lambda_=0.3); "
                f"got: {key_path!r}"
            )
            raise SystemExit(1)
        section, key = parts
        if section not in allowed_sections:
            print(
                f"ERROR: unknown section {section!r} in --set {item!r}. "
                f"Allowed: {', '.join(sorted(allowed_sections))}"
            )
            raise SystemExit(1)
        overrides.setdefault(section, {})[key] = _parse_set_value(raw_val)
    return overrides


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="3D Ising financial-market model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", help="Path to a case YAML file")
    parser.add_argument("--analyze", action="store_true",
                        help="Run statistical analysis after simulation")
    parser.add_argument("--plot", action="store_true",
                        help="Generate figures (first realisation only)")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Ensemble size; overrides simulation.n_seeds")
    parser.add_argument(
        "--set", metavar="section.key=value", action="append", default=[],
        dest="set_args",
        help=(
            "Override any YAML parameter inline, e.g. "
            "--set model.lambda_=0.30 --set simulation.production=12000"
        ),
    )
    args = parser.parse_args()

    cli_overrides = _build_overrides(args.set_args)
    cfg = load_config(args.config, overrides=cli_overrides if cli_overrides else None)
    n_seeds = args.seeds if args.seeds is not None else cfg.n_seeds

    print(f"=== Case {cfg.case_name}: m={cfg.m}, N={cfg.m ** 3}, "
          f"lambda={cfg.lambda_}, b_max={cfg.b_max}, "
          f"eta={cfg.eta:.4f}, c_max={cfg.c_max:.4f}, "
          f"noise={cfg.noise_scale:.4f}, beta={cfg.beta:.4f} ===")
    if cli_overrides:
        flat = [f"{s}.{k}={v}" for s, kv in cli_overrides.items() for k, v in kv.items()]
        print(f"    CLI overrides applied: {', '.join(flat)}")
    if n_seeds > 1:
        print(f"    Ensemble averaging over {n_seeds} seed(s)")

    # ---- simulation (may be multi-seed) ----
    runs = run_ensemble(cfg, n_seeds=n_seeds)

    # First realisation is the one we keep for time-series artefacts.
    first = runs[0]
    returns0 = first["returns"]
    price0 = first["price"]

    outdir = f"data/outputs/{cfg.case_name}"
    os.makedirs(outdir, exist_ok=True)
    save_results(f"{outdir}/timeseries.npz", **first)
    print(f"  Saved time series to {outdir}/timeseries.npz")
    print(f"  Return: mean={np.mean(returns0):.6g}, std={np.std(returns0):.6g}, "
          f"mean|r|={np.mean(np.abs(returns0)):.6g}")
    print(f"  Price:  final={price0[-1]:.4f}, min={price0.min():.4f}, "
          f"max={price0.max():.4f}")

    if not args.analyze:
        print("Done. Use --analyze for statistics.")
        return

    q_values = np.concatenate([np.arange(-5, 0, 0.5), np.arange(0.5, 5.5, 0.5)])
    max_lag = 100

    print("\n--- Per-seed statistics ---")
    all_res = []
    for k, r in enumerate(runs):
        print(f"  [seed {k + 1}/{len(runs)}]", end=" ")
        res = _analyze_single(r["returns"], q_values, max_lag, verbose=True)
        all_res.append(res)

    agg = _aggregate(all_res)

    # ---- report ----
    res0 = all_res[0]
    print(f"\n{'=' * 60}")
    print(f"SUMMARY — Case {cfg.case_name}  (ensemble median of {len(runs)} seeds)")
    print(f"{'=' * 60}")
    print(f"  Hurst (R/S):   {agg['H_rs_median']:.4f}   "
          f"values={[f'{v:.4f}' for v in agg['H_rs_values']]}")
    print(f"  DFA1:          {agg['DFA1_median']:.4f}")
    print(f"  DFA2:          {agg['DFA2_median']:.4f}")
    print(f"  DFA3:          {agg['DFA3_median']:.4f}")
    if res0["pos_fit"] and res0["neg_fit"]:
        print(f"  Tail beta+ (OLS): {agg['beta_pos_median']:.3f}   "
              f"range={res0['pos_fit']['x_range']}")
        print(f"  Tail beta- (OLS): {agg['beta_neg_median']:.3f}   "
              f"range={res0['neg_fit']['x_range']}")
    print(f"  Hill adaptive:    beta+={agg['hill_pos_median']:.3f}   "
          f"beta-={agg['hill_neg_median']:.3f}")
    print(f"  ACF lag-1:        returns={res0['acf_r'][1]:.3f}   "
          f"|returns|={res0['acf_v'][1]:.3f}")

    # Persist aggregate metrics as JSON.
    with open(f"{outdir}/metrics.json", "w") as fh:
        json.dump({
            "case": cfg.case_name,
            "params": {
                "m": cfg.m, "lambda": cfg.lambda_, "b_max": cfg.b_max,
                "c_max": cfg.c_max, "eta": cfg.eta,
                "noise_scale": cfg.noise_scale, "beta": cfg.beta,
                "delta": cfg.delta, "burn_in": cfg.burn_in,
                "production": cfg.production, "n_seeds": len(runs),
                "sweeps_per_step": cfg.sweeps_per_step,
            },
            "metrics": {k: v for k, v in agg.items() if not isinstance(v, dict)},
        }, fh, indent=2)

    # ---- plotting (first realisation only, for reproducibility) ----
    if args.plot:
        print("\nGenerating figures (first realisation)...")
        figdir = f"figures/{cfg.case_name}"
        os.makedirs(figdir, exist_ok=True)

        g = res0["g"]
        pos_x, pos_ccdf = tail_ccdf(g, positive=True)
        neg_x, neg_ccdf = tail_ccdf(g, positive=False)

        plt_mod.plot_return_series(returns0, cfg.case_name, f"{figdir}/returns.png")
        plt_mod.plot_price_series(price0, cfg.case_name, f"{figdir}/price.png")
        plt_mod.plot_return_histogram(g, cfg.case_name, f"{figdir}/hist_returns.png")
        beta_pos = res0["pos_fit"]["beta"] if res0["pos_fit"] else np.nan
        beta_neg = res0["neg_fit"]["beta"] if res0["neg_fit"] else np.nan
        plt_mod.plot_ccdf(pos_x, pos_ccdf, cfg.case_name, "positive",
                          beta_pos, f"{figdir}/ccdf_positive.png")
        plt_mod.plot_ccdf(neg_x, neg_ccdf, cfg.case_name, "negative",
                          beta_neg, f"{figdir}/ccdf_negative.png")
        plt_mod.plot_acf_comparison(res0["acf_r"], res0["acf_v"], max_lag,
                                    cfg.case_name, f"{figdir}/acf_comparison.png")
        n_rs, rs_vals, H_rs = res0["rs"]
        plt_mod.plot_rs_hurst(n_rs, rs_vals, H_rs, cfg.case_name,
                              f"{figdir}/rs_hurst.png")
        for order in (1, 2, 3):
            n_d, F_d, alpha_d = res0["dfa"][order]
            plt_mod.plot_dfa(n_d, F_d, alpha_d, order, cfg.case_name,
                             f"{figdir}/dfa{order}.png")
        mf = res0["mf"]
        plt_mod.plot_multifractal_hq(mf["q_values"], mf["hq"], cfg.case_name,
                                     f"{figdir}/mf_hq.png")
        plt_mod.plot_multifractal_tauq(mf["q_values"], mf["tauq"], cfg.case_name,
                                       f"{figdir}/mf_tauq.png")
        plt_mod.plot_multifractal_spectrum(mf["alpha"], mf["f_alpha"],
                                           cfg.case_name,
                                           f"{figdir}/mf_spectrum.png")
        print(f"  Figures saved to {figdir}/")

    print("\nDone.")


if __name__ == "__main__":
    main()
