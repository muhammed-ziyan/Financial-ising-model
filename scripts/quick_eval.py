"""Quick evaluation of the currently calibrated ``config/default.yaml`` parameters.

Runs all three cases (A, B, C) with the shared parameters currently stored in
``config/default.yaml`` and prints a side-by-side comparison table:

    Paper target  |  Before (pre-calibration baseline)  |  Now (current params)  |  Δ

Usage
-----
    python scripts/quick_eval.py

The script uses ``N_SEEDS = 3`` realisations and ``PRODUCTION = 8000`` steps per
case to obtain stable medians without the full calibration runtime.  No files
are written — results are printed to stdout only.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.analysis_memory import dfa, rs_analysis
from src.analysis_tail import fit_tail_best
from src.config_loader import load_config
from src.model import IsingMarketModel
from src.observables import normalize_returns

PAPER_TARGETS = {
    "A": {"H_rs": 0.5533, "DFA1": 0.5409, "DFA2": 0.5535, "DFA3": 0.5493,
          "beta_pos": 3.1262, "beta_neg": 3.6018},
    "B": {"H_rs": 0.4842, "DFA1": 0.4707, "DFA2": 0.4752, "DFA3": 0.4839,
          "beta_pos": 3.1262, "beta_neg": 3.6018},
    "C": {"H_rs": 0.5090, "DFA1": 0.5141, "DFA2": 0.4989, "DFA3": 0.4840,
          "beta_pos": 3.1262, "beta_neg": 3.6018},
}

BEFORE = {
    "A": {"H_rs": 0.6103, "DFA1": 0.5516, "DFA2": 0.5861, "DFA3": 0.6102,
          "beta_pos": 2.433, "beta_neg": 2.589},
    "B": {"H_rs": 0.5620, "DFA1": 0.5331, "DFA2": 0.5456, "DFA3": 0.5894,
          "beta_pos": 2.782, "beta_neg": 2.941},
    "C": {"H_rs": 0.5554, "DFA1": 0.5293, "DFA2": 0.5419, "DFA3": 0.5784,
          "beta_pos": 2.899, "beta_neg": 2.935},
}

CASE_FILES = {
    "A": "config/case_A.yaml",
    "B": "config/case_B.yaml",
    "C": "config/case_C.yaml",
}

N_SEEDS = 3
PRODUCTION = 8000
BURN_IN = 3000


def measure(case_path: str) -> dict:
    cfg = load_config(case_path, overrides={
        "simulation": {"burn_in": BURN_IN, "production": PRODUCTION, "n_seeds": N_SEEDS}
    })
    agg = {k: [] for k in ("H_rs", "DFA1", "DFA2", "DFA3", "beta_pos", "beta_neg")}
    for k in range(N_SEEDS):
        returns = IsingMarketModel(cfg, seed=cfg.seed + 1000 * k).run()["returns"]
        g = normalize_returns(returns, method="paper")
        try:
            _, _, h = rs_analysis(g)
            agg["H_rs"].append(h)
        except Exception:
            pass
        for order in (1, 2, 3):
            try:
                _, _, a = dfa(g, order=order)
                agg[f"DFA{order}"].append(a)
            except Exception:
                pass
        for sign, key in ((1, "beta_pos"), (-1, "beta_neg")):
            tail = g[g > 0] if sign == 1 else np.abs(g[g < 0])
            if len(tail) > 80:
                fit = fit_tail_best(tail)
                if fit is not None and np.isfinite(fit["beta"]):
                    agg[key].append(fit["beta"])
    return {k: float(np.median(v)) if v else float("nan") for k, v in agg.items()}


def fmt(v):
    return f"{v:.4f}" if np.isfinite(v) else "  nan "


def delta_str(now, target):
    d = now - target
    sign = "+" if d >= 0 else ""
    return f"({sign}{d:.4f})"


def main():
    print(f"\nEvaluating current default.yaml parameters "
          f"(T={PRODUCTION}, burn={BURN_IN}, {N_SEEDS} seeds each case)…\n")

    now_results = {}
    for case, path in CASE_FILES.items():
        print(f"  Running Case {case}…", flush=True)
        now_results[case] = measure(path)

    metrics = ["H_rs", "DFA1", "DFA2", "DFA3", "beta_pos", "beta_neg"]

    for case in ("A", "B", "C"):
        tgt = PAPER_TARGETS[case]
        bef = BEFORE[case]
        now = now_results[case]

        print(f"\n{'-'*80}")
        print(f"  Case {case}")
        print(f"{'-'*80}")
        header = f"  {'Metric':<12}{'Paper':>10}{'Before':>14}{'Now':>14}{'dBefore':>13}{'dNow':>13}"
        print(header)
        print(f"  {'-'*77}")
        total_before = 0.0
        total_now = 0.0
        for m in metrics:
            t = tgt[m]
            b = bef[m]
            n = now[m]
            db = abs(b - t)
            dn = abs(n - t) if np.isfinite(n) else float("nan")
            total_before += db
            if np.isfinite(dn):
                total_now += dn
            print(f"  {m:<12}{fmt(t):>10}{fmt(b):>14}{fmt(now[m]):>14}"
                  f"  {delta_str(b, t):>12}{delta_str(n, t):>12}")
        print(f"  {'-'*77}")
        print(f"  {'L1 sum':<12}{'':>10}{total_before:>13.4f} {total_now:>13.4f}")

    print(f"\n{'='*80}")
    print("  WEIGHTS: H_rs x3, DFA1 x2.5, DFA2 x1.5, DFA3 x1, beta+ x2, beta- x2")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
