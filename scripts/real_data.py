"""Real market data analysis — compare stylized facts against paper targets
and current model output.

Downloads daily close prices for a configurable list of tickers using
yfinance, computes the same metrics as the simulation pipeline
(H_rs, DFA1/2/3, tail exponents), and prints a side-by-side table.

Usage
-----
    python scripts/real_data.py
    python scripts/real_data.py --tickers SPY QQQ GLD --start 2000-01-01
    python scripts/real_data.py --tickers SPY --start 2010-01-01 --end 2023-12-31

The script saves the metric results to data/outputs/real_data_metrics.json
so you can compare them without re-downloading.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date

import numpy as np

try:
    import yfinance as yf
    _HAS_YFINANCE = True
except ImportError:
    _HAS_YFINANCE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis_memory import dfa, rs_analysis
from src.analysis_tail import fit_tail_best, hill_adaptive
from src.observables import normalize_returns

# ---------------------------------------------------------------------------
# Paper targets (Guimaraes & Lima 2021, PRE 103, 062130)
# ---------------------------------------------------------------------------

PAPER_TARGETS = {
    "A": {"H_rs": 0.5533, "DFA1": 0.5409, "DFA2": 0.5535, "DFA3": 0.5493,
          "beta_pos": 3.1262, "beta_neg": 3.6018},
    "B": {"H_rs": 0.4842, "DFA1": 0.4707, "DFA2": 0.4752, "DFA3": 0.4839,
          "beta_pos": 3.1262, "beta_neg": 3.6018},
    "C": {"H_rs": 0.5090, "DFA1": 0.5141, "DFA2": 0.4989, "DFA3": 0.4840,
          "beta_pos": 3.1262, "beta_neg": 3.6018},
}

# Average across A/B/C for a single-row paper reference in the ticker table
PAPER_AVG = {
    k: float(np.mean([PAPER_TARGETS[c][k] for c in ("A", "B", "C")]))
    for k in ("H_rs", "DFA1", "DFA2", "DFA3", "beta_pos", "beta_neg")
}

# Current model results (from run_simulation.py, n_seeds=5, T=8000, burn-in=3000)
# with calibrated defaults eta=3.225, c_max=0.636, noise_scale=1.333, beta=1.378.
MODEL_NOW = {
    "A": {"H_rs": 0.5663, "DFA1": 0.5195, "DFA2": 0.5292, "DFA3": 0.5337,
          "beta_pos": 3.3924, "beta_neg": 3.9344},
    "B": {"H_rs": 0.5682, "DFA1": 0.5403, "DFA2": 0.5466, "DFA3": 0.5706,
          "beta_pos": 3.2090, "beta_neg": 3.3970},
    "C": {"H_rs": 0.5619, "DFA1": 0.5396, "DFA2": 0.5427, "DFA3": 0.5632,
          "beta_pos": 3.7333, "beta_neg": 3.5495},
}

MODEL_AVG = {
    k: float(np.mean([MODEL_NOW[c][k] for c in ("A", "B", "C")]))
    for k in ("H_rs", "DFA1", "DFA2", "DFA3", "beta_pos", "beta_neg")
}

METRICS = ["H_rs", "DFA1", "DFA2", "DFA3", "beta_pos", "beta_neg"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_log_returns(ticker: str, start: str, end: str) -> np.ndarray | None:
    """Download adjusted closes and compute daily log-returns.

    Requires the optional ``yfinance`` package.  Returns *None* if the
    download fails or yields insufficient data.
    """
    if not _HAS_YFINANCE:
        raise ImportError(
            "yfinance is required for real-data download.\n"
            "Install it with:  pip install yfinance"
        )
    try:
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)
    except Exception as e:
        print(f"  [WARNING] Download failed for {ticker}: {e}")
        return None
    if df is None or len(df) < 200:
        print(f"  [WARNING] Insufficient data for {ticker} "
              f"({0 if df is None else len(df)} rows)")
        return None
    closes = df["Close"].dropna().values.flatten().astype(float)
    if len(closes) < 200:
        print(f"  [WARNING] Too few non-NaN closes for {ticker}")
        return None
    returns = np.diff(np.log(closes))
    return returns


def analyse(returns: np.ndarray) -> dict:
    """Run the full stylized-facts pipeline on a return series."""
    g = normalize_returns(returns, method="paper")
    result: dict[str, float] = {"T": len(g)}

    try:
        _, _, h = rs_analysis(g)
        result["H_rs"] = float(h)
    except Exception:
        result["H_rs"] = float("nan")

    for order in (1, 2, 3):
        try:
            _, _, a = dfa(g, order=order)
            result[f"DFA{order}"] = float(a)
        except Exception:
            result[f"DFA{order}"] = float("nan")

    for label, mask in (("pos", g > 0), ("neg", g < 0)):
        tail = g[mask] if label == "pos" else np.abs(g[mask])
        key = f"beta_{label}"
        if len(tail) > 80:
            fit = fit_tail_best(tail)
            result[key] = float(fit["beta"]) if fit is not None else float("nan")
            hill_val, _ = hill_adaptive(tail)
            result[f"hill_{label}"] = float(hill_val)
        else:
            result[key] = float("nan")
            result[f"hill_{label}"] = float("nan")

    # Extra descriptive stats
    result["mean_r"] = float(np.mean(returns))
    result["std_r"] = float(np.std(returns))
    result["skew"] = float(
        np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 3)
    )
    result["kurt"] = float(
        np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 4) - 3.0
    )
    return result


def fmt(v, decimals=4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "   n/a  "
    return f"{v:.{decimals}f}"


def delta_str(v, target) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "  n/a "
    d = v - target
    return f"({'+' if d >= 0 else ''}{d:.3f})"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser(
        description="Compare real market stylized facts with model output")
    p.add_argument("--tickers", nargs="+",
                   default=["^GSPC", "^DJI", "^IXIC", "^FTSE", "^N225"],
                   help="Yahoo Finance tickers to download")
    p.add_argument("--start", default="1990-01-01",
                   help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default=str(date.today()),
                   help="End date (YYYY-MM-DD)")
    p.add_argument("--out", default="data/outputs/real_data_metrics.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not _HAS_YFINANCE:
        print(
            "ERROR: yfinance is not installed.\n"
            "Install it with:  pip install yfinance\n"
            "Then re-run this script."
        )
        raise SystemExit(1)

    args = _parse()

    print(f"\n{'='*72}")
    print("  Real Market Data — Stylized Facts Analysis")
    print(f"  Tickers : {', '.join(args.tickers)}")
    print(f"  Period  : {args.start}  to  {args.end}")
    print(f"{'='*72}\n")

    results: dict[str, dict] = {}
    for ticker in args.tickers:
        print(f"  Downloading {ticker}...", end=" ", flush=True)
        ret = fetch_log_returns(ticker, args.start, args.end)
        if ret is None:
            print("skipped.")
            continue
        print(f"{len(ret)} daily returns. Analysing...", end=" ", flush=True)
        metrics = analyse(ret)
        results[ticker] = metrics
        print(f"H_rs={metrics['H_rs']:.4f}  DFA1={metrics['DFA1']:.4f}  "
              f"beta+={metrics['beta_pos']:.3f}  beta-={metrics['beta_neg']:.3f}")

    if not results:
        print("No data could be retrieved. Check ticker symbols and dates.")
        return

    # -----------------------------------------------------------------------
    # Table 1 — Per-ticker metrics vs paper avg and model avg
    # -----------------------------------------------------------------------
    print(f"\n{'-'*100}")
    print(f"  {'Ticker':<10} {'T':>6} "
          f"{'H_rs':>8} {'DFA1':>8} {'DFA2':>8} {'DFA3':>8} "
          f"{'beta+':>8} {'beta-':>8} "
          f"{'skew':>7} {'kurt':>7}")
    print(f"  {'-'*97}")

    # Reference rows first
    print(f"  {'PAPER(avg)':<10} {'~5700':>6} "
          + "  ".join(fmt(PAPER_AVG[m]) for m in METRICS)
          + f"  {'':>7} {'':>7}")
    print(f"  {'MODEL(avg)':<10} {'8000':>6} "
          + "  ".join(fmt(MODEL_AVG[m]) for m in METRICS)
          + f"  {'':>7} {'':>7}")
    print(f"  {'-'*97}")

    for ticker, m in results.items():
        row = (
            f"  {ticker:<10} {m['T']:>6} "
            + "  ".join(fmt(m.get(met)) for met in METRICS)
            + f"  {fmt(m.get('skew'), 3):>7} {fmt(m.get('kurt'), 2):>7}"
        )
        print(row)

    # -----------------------------------------------------------------------
    # Table 2 — Deviations from paper targets (per metric)
    # -----------------------------------------------------------------------
    print(f"\n{'-'*100}")
    print("  DEVIATIONS from paper average targets (value - target):")
    print(f"  {'Ticker':<10} "
          + " ".join(f"{'d'+m:>10}" for m in METRICS))
    print(f"  {'-'*97}")

    for ticker, m in results.items():
        deltas = "  ".join(
            f"{delta_str(m.get(met), PAPER_AVG[met]):>10}"
            for met in METRICS
        )
        print(f"  {ticker:<10}  {deltas}")

    # -----------------------------------------------------------------------
    # Table 3 — Summary: real data range vs model vs paper
    # -----------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("  SUMMARY — Real market ranges vs paper targets vs model")
    print(f"  {'Metric':<12} {'RealMin':>9} {'RealMax':>9} "
          f"{'Paper(avg)':>11} {'Model(avg)':>11} {'In range?':>10}")
    print(f"  {'-'*67}")

    for m in METRICS:
        vals = [r[m] for r in results.values()
                if np.isfinite(r.get(m, float("nan")))]
        if not vals:
            continue
        lo, hi = min(vals), max(vals)
        paper_v = PAPER_AVG[m]
        model_v = MODEL_AVG[m]
        in_range = lo <= paper_v <= hi
        model_in = lo <= model_v <= hi
        range_str = "YES" if in_range else "NO "
        model_str = "YES" if model_in else "NO "
        print(f"  {m:<12} {fmt(lo):>9} {fmt(hi):>9} "
              f"{fmt(paper_v):>11} {fmt(model_v):>11} "
              f"  paper:{range_str}  model:{model_str}")

    print(f"{'='*72}\n")

    # Save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {
        "tickers": results,
        "paper_avg": PAPER_AVG,
        "model_avg": MODEL_AVG,
        "period": {"start": args.start, "end": args.end},
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"  Metrics saved -> {args.out}\n")


if __name__ == "__main__":
    main()
