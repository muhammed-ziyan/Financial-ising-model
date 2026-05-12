"""Automatic calibration of the four undisclosed paper parameters.

The paper only specifies ``(m, lambda, b_max)`` per case in Table I.
The four remaining knobs — ``eta`` (market depth), ``c_max`` (news
sensitivity), ``noise_scale`` (private-noise std), and ``beta`` (inverse
temperature of the heat-bath rule) — are not reported.

This script searches for the single quadruple
``(eta, c_max, noise_scale, beta)`` that minimises the joint L1 distance
between model and paper metrics across all three cases.  Those four
parameters are kept **identical across Cases A/B/C** because the paper
treats them as shared (only Table I values vary).

Target metrics (from Guimaraes & Lima 2021, PRE 103, 062130)
-------------------------------------------------------------
  Case A :  H_RS = 0.5533, DFA1 = 0.5409, DFA2 = 0.5535, DFA3 = 0.5493
  Case B :  H_RS = 0.4842, DFA1 = 0.4707, DFA2 = 0.4752, DFA3 = 0.4839
  Case C :  H_RS = 0.5090, DFA1 = 0.5141, DFA2 = 0.4989, DFA3 = 0.4840
  Tail   :  beta+ = 3.13, beta- = 3.60

The scoring function is a weighted sum of absolute deviations; the
weights emphasise H_RS and DFA1 (the paper's headline numbers) over
DFA2/DFA3 and the tail exponents (which are intrinsically noisier).

Procedure
---------
Stage 1 — coarse random / Latin-hypercube search over a wide box with
  short (burn-in + production) and a single seed.  Cheap; prunes.
Stage 2 — refine the top-K with a longer simulation and 2-3 seeds.
Stage 3 — report the winner, and also write it into
  ``config/default.yaml`` if ``--write`` is given.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import yaml
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis_memory import dfa, rs_analysis
from src.analysis_tail import fit_tail_best
from src.config_loader import load_config, override_config
from src.model import IsingMarketModel
from src.observables import normalize_returns


PAPER_TARGETS = {
    "A": {"H_rs": 0.5533, "DFA1": 0.5409, "DFA2": 0.5535, "DFA3": 0.5493},
    "B": {"H_rs": 0.4842, "DFA1": 0.4707, "DFA2": 0.4752, "DFA3": 0.4839},
    "C": {"H_rs": 0.5090, "DFA1": 0.5141, "DFA2": 0.4989, "DFA3": 0.4840},
}
TAIL_TARGETS = {"beta_pos": 3.1262, "beta_neg": 3.6018}

# Weight vector — emphasise H/DFA1 (reported as primary in paper), then
# DFA2/DFA3, and give the tails (large measurement error) the least weight.
WEIGHTS = {
    "H_rs": 3.0,
    "DFA1": 2.5,
    "DFA2": 1.5,
    "DFA3": 1.0,
    "beta_pos": 2.0,
    "beta_neg": 2.0,
}

CASE_FILES = {
    "A": "config/case_A.yaml",
    "B": "config/case_B.yaml",
    "C": "config/case_C.yaml",
}


@dataclass
class Metrics:
    H_rs: float = float("nan")
    DFA1: float = float("nan")
    DFA2: float = float("nan")
    DFA3: float = float("nan")
    beta_pos: float = float("nan")
    beta_neg: float = float("nan")


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _simulate(case_path: str, shared: dict, burn_in: int, production: int,
              seed: int) -> Metrics:
    """Run one simulation and extract the scalar metrics."""
    model_overrides = {
        "eta": shared["eta"],
        "c_max": shared["c_max"],
        "noise_scale": shared["noise_scale"],
        "beta": shared["beta"],
    }
    if "delta" in shared:
        model_overrides["delta"] = shared["delta"]
    overrides = {
        "model": model_overrides,
        "simulation": {
            "burn_in": burn_in,
            "production": production,
            "seed": seed,
            "sweeps_per_step": shared.get("sweeps_per_step", 1),
        },
    }
    cfg = load_config(case_path, overrides=overrides)
    returns = IsingMarketModel(cfg).run()["returns"]

    g = normalize_returns(returns, method="paper")
    m = Metrics()
    try:
        _, _, m.H_rs = rs_analysis(g)
    except Exception:
        pass
    for order in (1, 2, 3):
        try:
            _, _, a = dfa(g, order=order)
            setattr(m, f"DFA{order}", a)
        except Exception:
            pass

    pos = g[g > 0]
    neg = np.abs(g[g < 0])
    if len(pos) > 80:
        fit = fit_tail_best(pos)
        if fit is not None:
            m.beta_pos = fit["beta"]
    if len(neg) > 80:
        fit = fit_tail_best(neg)
        if fit is not None:
            m.beta_neg = fit["beta"]
    return m


def _eval_shared(shared: dict, burn_in: int, production: int,
                 seeds: list[int]) -> tuple[float, dict]:
    """Average metrics across seeds and score them against paper targets."""
    per_case: dict[str, dict] = {}
    total = 0.0
    for case, path in CASE_FILES.items():
        agg = {k: [] for k in ("H_rs", "DFA1", "DFA2", "DFA3",
                                "beta_pos", "beta_neg")}
        for sd in seeds:
            try:
                m = _simulate(path, shared, burn_in, production, sd)
            except Exception:
                continue
            for k in agg:
                val = getattr(m, k)
                if np.isfinite(val):
                    agg[k].append(val)
        med = {k: (float(np.median(v)) if v else float("nan")) for k, v in agg.items()}
        per_case[case] = med

        tgt = PAPER_TARGETS[case]
        for key, target_val in tgt.items():
            if np.isfinite(med[key]):
                total += WEIGHTS[key] * abs(med[key] - target_val)
            else:
                total += WEIGHTS[key] * 1.0  # penalise missing

        for key in ("beta_pos", "beta_neg"):
            target_val = TAIL_TARGETS[key]
            if np.isfinite(med[key]):
                total += WEIGHTS[key] * abs(med[key] - target_val)
            else:
                total += WEIGHTS[key] * 1.0

    return total, per_case


# ---------------------------------------------------------------------------
# Parallel worker + checkpointing
# ---------------------------------------------------------------------------

def _worker(task: tuple) -> tuple:
    """Module-level worker so it can be pickled for ProcessPoolExecutor."""
    shared, burn_in, production, seeds = task
    score, per_case = _eval_shared(shared, burn_in, production, seeds)
    return shared, score, per_case


def _checkpoint(path: str, **parts) -> None:
    """Write the current calibration state to ``path`` atomically-ish."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(parts, fh, indent=2, default=float)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Sampling the parameter space
# ---------------------------------------------------------------------------

def _latin_hypercube(n_samples: int, bounds: list[tuple[float, float]],
                     rng: np.random.Generator,
                     scales: list[str] | None = None) -> np.ndarray:
    """Draw ``n_samples`` Latin-hypercube samples on the product of ranges.

    Parameters
    ----------
    scales : list of {"linear", "log"}
        Per-dimension sampling scale.  ``"log"`` samples uniformly in
        ``log(lo), log(hi)`` then exponentiates back to the original
        range (requires ``lo > 0``).  Defaults to linear for every dim.
    """
    d = len(bounds)
    if scales is None:
        scales = ["linear"] * d
    if len(scales) != d:
        raise ValueError("scales must have same length as bounds")
    result = np.empty((n_samples, d))
    for j, ((lo, hi), scale) in enumerate(zip(bounds, scales)):
        cuts = np.linspace(0.0, 1.0, n_samples + 1)
        u = rng.uniform(cuts[:-1], cuts[1:])
        rng.shuffle(u)
        if scale == "log":
            if lo <= 0 or hi <= 0:
                raise ValueError(f"log scale requires positive bounds, got ({lo}, {hi})")
            log_lo, log_hi = np.log(lo), np.log(hi)
            result[:, j] = np.exp(log_lo + u * (log_hi - log_lo))
        else:
            result[:, j] = lo + u * (hi - lo)
    return result


# ---------------------------------------------------------------------------
# Main calibration loop
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--coarse-samples", type=int, default=120,
                   help="Number of Latin-hypercube samples in stage 1")
    p.add_argument("--refine-top", type=int, default=15,
                   help="Number of best stage-1 points to refine")
    p.add_argument("--local-samples", type=int, default=200,
                   help="Max iterations of the Nelder-Mead local search "
                        "(stage 3).  Set 0 to skip stage 3.")
    p.add_argument("--coarse-production", type=int, default=3500)
    p.add_argument("--coarse-burnin", type=int, default=2000)
    p.add_argument("--refine-production", type=int, default=8000)
    p.add_argument("--refine-burnin", type=int, default=3000)
    p.add_argument("--coarse-seeds", type=int, default=2)
    p.add_argument("--refine-seeds", type=int, default=5)
    p.add_argument("--include-sweeps", action="store_true",
                   help="Also search over sweeps_per_step (integer 1..4)")
    p.add_argument("--include-delta", action="store_true",
                   help="Also search over delta in [0.2, 4.0] (log scale)")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of parallel worker processes for stage 1 "
                        "(1 = serial).")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--write", action="store_true",
                   help="Write best parameters into config/default.yaml")
    p.add_argument("--out", default="data/outputs/calibration.json")
    return p.parse_args()


def _format_case(metrics: dict, target: dict) -> str:
    bits = []
    for k, tv in target.items():
        mv = metrics.get(k, float("nan"))
        bits.append(f"{k}={mv:.4f}(T={tv:.4f})")
    return "  ".join(bits)


def main():
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    bounds = [
        (0.3, 8.0),      # eta
        (0.1, 2.0),      # c_max
        (0.1, 2.0),      # noise_scale
        (0.1, 2.0),      # beta
    ]
    names = ["eta", "c_max", "noise_scale", "beta"]
    scales = ["log", "log", "log", "log"]
    if args.include_delta:
        bounds.append((0.2, 4.0))
        names.append("delta")
        scales.append("log")
    if args.include_sweeps:
        bounds.append((1.0, 4.0))  # sweeps_per_step, rounded to int
        names.append("sweeps_per_step")
        scales.append("linear")

    # Stage 1 — coarse Latin-hypercube search.
    print(f"\n[Stage 1] Coarse LHS with {args.coarse_samples} samples "
          f"(production={args.coarse_production}, burn_in={args.coarse_burnin}, "
          f"seeds={args.coarse_seeds})")
    t0 = time.time()
    lhs = _latin_hypercube(args.coarse_samples, bounds, rng, scales=scales)
    coarse_seeds = list(range(100, 100 + args.coarse_seeds))

    def _row_to_shared(row):
        shared = {names[j]: float(row[j]) for j in range(len(names))}
        if "sweeps_per_step" in shared:
            shared["sweeps_per_step"] = int(round(max(1.0, shared["sweeps_per_step"])))
        else:
            shared["sweeps_per_step"] = 1
        return shared

    stage1: list[dict] = []
    tasks = [(
        _row_to_shared(row),
        args.coarse_burnin, args.coarse_production, coarse_seeds,
    ) for row in lhs]
    progress_every = max(1, args.coarse_samples // 10)

    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"  (using {args.workers} worker processes)")
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_worker, t) for t in tasks]
            for n_done, fut in enumerate(as_completed(futures), 1):
                shared, score, per_case = fut.result()
                stage1.append({"shared": shared, "score": score,
                               "per_case": per_case})
                if n_done % progress_every == 0:
                    print(f"  progress {n_done}/{args.coarse_samples}  "
                          f"score={score:.3f}  eta={shared['eta']:.3f}  "
                          f"c_max={shared['c_max']:.3f}  "
                          f"noise={shared['noise_scale']:.3f}  "
                          f"beta={shared['beta']:.3f}")
                    _checkpoint(args.out, stage1=stage1)
    else:
        for i, task in enumerate(tasks):
            shared, score, per_case = _worker(task)
            stage1.append({"shared": shared, "score": score,
                           "per_case": per_case})
            if (i + 1) % progress_every == 0:
                print(f"  progress {i + 1}/{args.coarse_samples}  "
                      f"score={score:.3f}  eta={shared['eta']:.3f}  "
                      f"c_max={shared['c_max']:.3f}  "
                      f"noise={shared['noise_scale']:.3f}  "
                      f"beta={shared['beta']:.3f}")
                _checkpoint(args.out, stage1=stage1)
    print(f"[Stage 1] done in {time.time() - t0:.1f} s")
    _checkpoint(args.out, stage1=stage1)

    stage1.sort(key=lambda d: d["score"])
    print("\nTop 10 after stage 1:")
    for k, entry in enumerate(stage1[:10]):
        s = entry["shared"]
        print(f"  #{k + 1}  score={entry['score']:.3f}  "
              f"eta={s['eta']:.2f} c_max={s['c_max']:.2f} "
              f"noise={s['noise_scale']:.2f} beta={s['beta']:.2f}")

    # Stage 2 — refine top-K with longer sims and more seeds.
    top = stage1[: args.refine_top]
    print(f"\n[Stage 2] Refining top {len(top)} with "
          f"production={args.refine_production}, burn_in={args.refine_burnin}, "
          f"seeds={args.refine_seeds}")
    refine_seeds = list(range(200, 200 + args.refine_seeds))
    t1 = time.time()

    stage2 = []
    for k, entry in enumerate(top):
        shared = dict(entry["shared"])
        score, per_case = _eval_shared(shared, args.refine_burnin,
                                       args.refine_production, refine_seeds)
        stage2.append({"shared": shared, "score": score, "per_case": per_case})
        print(f"  candidate {k + 1}/{len(top)}  refined score={score:.3f}  "
              f"(was {entry['score']:.3f})")
        _checkpoint(args.out, stage1=stage1, stage2=stage2)
    print(f"[Stage 2] done in {time.time() - t1:.1f} s")

    stage2.sort(key=lambda d: d["score"])
    best = stage2[0]

    # Stage 2 best is the current best before local search.
    _checkpoint(args.out, stage1=stage1, stage2=stage2, stage3=[], best=best)

    # Stage 3 — Nelder-Mead local search around the stage-2 best.
    stage3: list[dict] = []
    if args.local_samples > 0:
        print(f"\n[Stage 3] Nelder-Mead local search (max "
              f"{args.local_samples} iterations) around stage-2 best")
        t2 = time.time()
        center = best["shared"]

        # Build an internal vector representation that respects each
        # parameter's sampling scale.  Log-scaled dims use x = log(value)
        # so simplex steps are scale-appropriate.
        def _to_internal(shared_dict: dict) -> np.ndarray:
            xs = []
            for nm, scale in zip(names, scales):
                v = float(shared_dict[nm])
                xs.append(np.log(v) if scale == "log" else v)
            return np.asarray(xs, dtype=float)

        def _from_internal(x: np.ndarray) -> dict:
            trial = {}
            for j, (nm, scale, (lo, hi)) in enumerate(zip(names, scales, bounds)):
                if scale == "log":
                    lo_i, hi_i = np.log(lo), np.log(hi)
                    val = float(np.exp(np.clip(x[j], lo_i, hi_i)))
                else:
                    val = float(np.clip(x[j], lo, hi))
                trial[nm] = val
            if "sweeps_per_step" in trial:
                trial["sweeps_per_step"] = int(round(max(1.0, trial["sweeps_per_step"])))
            else:
                trial["sweeps_per_step"] = center.get("sweeps_per_step", 1)
            return trial

        def _objective(x: np.ndarray) -> float:
            trial = _from_internal(x)
            score, per_case = _eval_shared(
                trial, args.refine_burnin, args.refine_production, refine_seeds
            )
            stage3.append({"shared": trial, "score": score, "per_case": per_case})
            print(f"    NM iter {len(stage3):3d}  score={score:.4f}  "
                  + "  ".join(f"{k}={trial[k]:.4f}" for k in names
                              if k != "sweeps_per_step"))
            # Periodic checkpointing so partial results survive interrupts.
            if len(stage3) % 10 == 0:
                running_best = min([best] + stage3, key=lambda d: d["score"])
                _checkpoint(args.out, stage1=stage1, stage2=stage2,
                            stage3=stage3, best=running_best)
            return score

        x0 = _to_internal(center)
        # Simplex scale: larger step for log-dims (~0.3 in log ≈ ×1.35),
        # smaller absolute step for the single linear dim sweeps_per_step.
        initial_simplex = np.vstack([x0] + [
            x0 + 0.3 * np.eye(len(x0))[i] for i in range(len(x0))
        ])
        minimize(
            _objective,
            x0,
            method="Nelder-Mead",
            options={
                "xatol": 1e-3,
                "fatol": 1e-3,
                "maxiter": int(args.local_samples),
                "initial_simplex": initial_simplex,
                "adaptive": True,
            },
        )
        stage3.sort(key=lambda d: d["score"])
        if stage3:
            print(f"[Stage 3] done in {time.time() - t2:.1f} s, "
                  f"best local score={stage3[0]['score']:.4f}")
            if stage3[0]["score"] < best["score"]:
                best = stage3[0]
                print("  Stage 3 winner beats stage 2 — using it.")
        else:
            print(f"[Stage 3] done in {time.time() - t2:.1f} s (no evaluations)")

    # ------ report ------
    print(f"\n{'=' * 72}")
    print("BEST PARAMETERS")
    print(f"{'=' * 72}")
    s = best["shared"]
    print(f"  eta         = {s['eta']:.4f}")
    print(f"  c_max       = {s['c_max']:.4f}")
    print(f"  noise_scale = {s['noise_scale']:.4f}")
    print(f"  beta        = {s['beta']:.4f}")
    print(f"  total score = {best['score']:.4f}")
    print()
    for case, met in best["per_case"].items():
        tgt = dict(PAPER_TARGETS[case], **TAIL_TARGETS)
        print(f"  Case {case}: {_format_case(met, tgt)}")

    _checkpoint(args.out, stage1=stage1, stage2=stage2, stage3=stage3,
                best=best)
    print(f"\nCalibration log -> {args.out}")

    s = best["shared"]
    print(f"\n  (Best parameters:  eta={s['eta']:.4f}  c_max={s['c_max']:.4f}  "
          f"noise={s['noise_scale']:.4f}  beta={s['beta']:.4f}  "
          f"sweeps={s.get('sweeps_per_step', 1)})")
    if args.write:
        default_path = "config/default.yaml"
        with open(default_path, "r") as fh:
            cfg = yaml.safe_load(fh)
        cfg["model"]["eta"] = float(s["eta"])
        cfg["model"]["c_max"] = float(s["c_max"])
        cfg["model"]["noise_scale"] = float(s["noise_scale"])
        cfg["model"]["beta"] = float(s["beta"])
        if "sweeps_per_step" in s:
            cfg["simulation"]["sweeps_per_step"] = int(s["sweeps_per_step"])
        with open(default_path, "w") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)
        print(f"Wrote best parameters into {default_path}")


if __name__ == "__main__":
    main()
