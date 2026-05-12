"""Generate cross-case and real-data summary figures.

Must be run **after** ``run_simulation.py`` and ``scripts/real_data.py``
have been executed for all three cases (A, B, C) because this script
reads the pre-computed artefacts:

    data/outputs/{A,B,C}/timeseries.npz
    data/outputs/{A,B,C}/metrics.json
    data/outputs/mf_hq_summary.json
    data/outputs/real_data_metrics.json

Usage
-----
    python scripts/generate_summary_figures.py

All figures are written to ``figures/summary/``:

    01_metrics_comparison.png   — grouped bar: H_RS, DFA1-3 for Paper / Model / Real
    02_tail_exponents.png       — grouped bar: β⁺, β⁻ vs paper targets and real range
    03_hq_comparison.png        — h(q) curves for model A/B/C vs paper values
    04_returns_overlay.png      — normalised return PDFs for all three cases
    05_acf_overlay.png          — ACF of returns and |returns| for all three cases
    06_real_vs_model_scatter.png — H_RS vs DFA1/DFA3 scatter: real indices + model
    07_timeseries_panel.png     — 3×2 panel: returns and price for all cases
    08_ccdf_positive.png        — positive-tail CCDF all cases on one log-log plot
    09_ccdf_negative.png        — negative-tail CCDF all cases on one log-log plot
    10_dfa1_overlay.png         — DFA1 fluctuation function F(n) all cases
    11_mf_spectrum_overlay.png  — multifractal f(α) spectrum all cases
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm

from src.observables import normalize_returns
from src.analysis_memory import acf as compute_acf, dfa as compute_dfa
from src.analysis_tail import tail_ccdf, fit_tail_best
from src.analysis_multifractal import mfdfa

OUTDIR = "figures/summary"
os.makedirs(OUTDIR, exist_ok=True)

# ── load per-case simulation artefacts ─────────────────────────────────────────

def _load_metrics(case):
    with open(f"data/outputs/{case}/metrics.json") as f:
        return json.load(f)["metrics"]

metrics = {c: _load_metrics(c) for c in ("A", "B", "C")}

# ── compute MF-DFA h(q=1..4) on-the-fly from timeseries.npz ───────────────────
# (avoids requiring a separately-generated mf_hq_summary.json artefact)

_MF_Q_VALS = np.concatenate([np.arange(-5, 0, 0.5), np.arange(0.5, 5.5, 0.5)])

def _compute_mf_hq(case: str) -> dict:
    """Return h(q=1,2,3,4) and Δh for *case* from its timeseries.npz."""
    d = np.load(f"data/outputs/{case}/timeseries.npz")
    g = normalize_returns(d["returns"], method="paper")
    result = mfdfa(g, _MF_Q_VALS, order=1)
    hq_all = result["hq"]
    q_all  = result["q_values"]
    out = {}
    for qi in (1, 2, 3, 4):
        idx = np.argmin(np.abs(q_all - qi))
        out[f"h_q{qi}"] = float(hq_all[idx]) if np.isfinite(hq_all[idx]) else float("nan")
    valid = hq_all[np.isfinite(hq_all)]
    out["delta_h"] = float(valid.max() - valid.min()) if len(valid) >= 2 else float("nan")
    return out

print("Computing MF-DFA h(q) for each case (this takes a moment)…")
mf_hq = {c: _compute_mf_hq(c) for c in ("A", "B", "C")}

# ── load real-market metrics ───────────────────────────────────────────────────

with open("data/outputs/real_data_metrics.json") as f:
    real = json.load(f)

real_tickers = real["tickers"]
paper_avg    = real["paper_avg"]
model_avg    = real["model_avg"]

# ── colour and label maps with safe fallback for custom tickers ────────────────

TICKER_COLORS = {
    "^GSPC": "#E53935", "^GDAXI": "#8E24AA",
    "^N225": "#00897B", "^NSEI": "#FB8C00", "^BSESN": "#6D4C41",
}
TICKER_LABELS = {
    "^GSPC": "S&P 500", "^GDAXI": "DAX",
    "^N225": "NIKKEI", "^NSEI": "NIFTY", "^BSESN": "SENSEX",
}
_FALLBACK_COLORS = [
    "#1565C0", "#2E7D32", "#6A1B9A", "#AD1457", "#00695C",
    "#E65100", "#37474F", "#558B2F", "#283593", "#880E4F",
]

def _ticker_color(ticker: str, idx: int) -> str:
    return TICKER_COLORS.get(ticker, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])

def _ticker_label(ticker: str) -> str:
    return TICKER_LABELS.get(ticker, ticker)

PAPER_CASE = {
    "A": {"H_rs": 0.5533, "DFA1": 0.5409, "DFA2": 0.5535, "DFA3": 0.5493,
          "beta_pos": 3.1262, "beta_neg": 3.6018,
          "h_q1": 0.5507, "h_q2": 0.5493, "h_q3": 0.5477, "h_q4": 0.5459},
    "B": {"H_rs": 0.4842, "DFA1": 0.4707, "DFA2": 0.4752, "DFA3": 0.4839,
          "beta_pos": 3.1262, "beta_neg": 3.6018,
          "h_q1": 0.4864, "h_q2": 0.4839, "h_q3": 0.4809, "h_q4": 0.4772},
    "C": {"H_rs": 0.5090, "DFA1": 0.5141, "DFA2": 0.4989, "DFA3": 0.4840,
          "beta_pos": 3.1262, "beta_neg": 3.6018,
          "h_q1": 0.4874, "h_q2": 0.4840, "h_q3": 0.4801, "h_q4": 0.4757},
}

# ── helpers ────────────────────────────────────────────────────────────────────

COLORS = {"A": "#2196F3", "B": "#4CAF50", "C": "#FF9800"}

def savefig(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 01 — Memory metrics comparison (H_RS, DFA1-3)
# ══════════════════════════════════════════════════════════════════════════════

def fig_metrics_comparison():
    metric_labels = ["H_RS", "DFA1", "DFA2", "DFA3"]
    metric_keys   = ["H_rs_median", "DFA1_median", "DFA2_median", "DFA3_median"]
    paper_keys    = ["H_rs", "DFA1", "DFA2", "DFA3"]

    n_metrics = len(metric_labels)
    cases = ["A", "B", "C"]

    fig, axes = plt.subplots(1, n_metrics, figsize=(14, 4), sharey=False)
    fig.suptitle("Memory metrics: Paper vs Model vs Real markets", fontsize=12, fontweight="bold")

    real_vals_all = {
        "H_rs": [real_tickers[t]["H_rs"] for t in real_tickers],
        "DFA1": [real_tickers[t]["DFA1"] for t in real_tickers],
        "DFA2": [real_tickers[t]["DFA2"] for t in real_tickers],
        "DFA3": [real_tickers[t]["DFA3"] for t in real_tickers],
    }

    for ax, mlabel, mkey, pkey in zip(axes, metric_labels, metric_keys, paper_keys):
        # Real market scatter (jittered)
        rvals = real_vals_all[pkey]
        xs = np.random.default_rng(0).uniform(-0.15, 0.15, len(rvals)) + 0
        ax.scatter(xs, rvals, color="gray", alpha=0.7, zorder=3, label="Real" if mlabel == "H_RS" else None, s=40)
        ax.errorbar(0, np.mean(rvals), yerr=np.std(rvals), fmt="D",
                    color="black", capsize=4, markersize=6, zorder=4)

        for i, case in enumerate(cases):
            xpos = i + 1
            paper_v = PAPER_CASE[case][pkey]
            model_v = metrics[case][mkey]
            ax.bar(xpos - 0.18, paper_v, width=0.32, color=COLORS[case], alpha=0.4,
                   label=f"Paper {case}" if mlabel == "H_RS" else None)
            ax.bar(xpos + 0.18, model_v, width=0.32, color=COLORS[case], alpha=0.9,
                   label=f"Model {case}" if mlabel == "H_RS" else None)

        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["Real\nmarkets", "Case A", "Case B", "Case C"], fontsize=8)
        ax.set_title(mlabel, fontsize=10)
        ax.set_ylabel("Exponent value" if mlabel == "H_RS" else "")
        ax.axhline(0.5, color="gray", linewidth=0.7, linestyle="--")
        ax.set_ylim(0.38, 0.72)

    axes[0].legend(fontsize=7, ncol=2, loc="upper right")
    fig.text(0.5, -0.01,
             "Light bars = paper targets, dark bars = model medians (5 seeds). "
             "Grey dots = five real indices; ◆ = mean ± 1 std.", ha="center", fontsize=8)
    fig.tight_layout()
    savefig(fig, "01_metrics_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 02 — Tail exponents
# ══════════════════════════════════════════════════════════════════════════════

def fig_tail_exponents():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Power-law tail exponents β⁺ and β⁻", fontsize=12, fontweight="bold")

    cases = ["A", "B", "C"]
    x = np.arange(3)
    width = 0.35

    for ax, sign, title in [(ax1, "pos", "β⁺  (positive tail)"), (ax2, "neg", "β⁻  (negative tail)")]:
        paper_vals = [PAPER_CASE[c][f"beta_{sign}"] for c in cases]
        model_vals = [metrics[c][f"beta_{sign}_median"] for c in cases]

        bars1 = ax.bar(x - width/2, paper_vals, width, alpha=0.4, color=[COLORS[c] for c in cases],
                       label="Paper")
        bars2 = ax.bar(x + width/2, model_vals, width, alpha=0.9, color=[COLORS[c] for c in cases],
                       label="Model")

        # Real market range
        real_vals = [real_tickers[t][f"beta_{sign}"] for t in real_tickers]
        ax.axhspan(min(real_vals), max(real_vals), alpha=0.08, color="gray",
                   label=f"Real range [{min(real_vals):.2f}, {max(real_vals):.2f}]")
        ax.axhline(np.mean(real_vals), color="gray", linewidth=1.2, linestyle="--", label="Real mean")
        ax.axhline(3.0, color="red", linewidth=0.8, linestyle=":", label="Cubic law β=3")

        ax.set_xticks(x)
        ax.set_xticklabels(["Case A", "Case B", "Case C"])
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("β")
        ax.set_ylim(1.8, 5.0)
        ax.legend(fontsize=7)

    fig.tight_layout()
    savefig(fig, "02_tail_exponents.png")


# ══════════════════════════════════════════════════════════════════════════════
# 03 — h(q) generalised Hurst
# ══════════════════════════════════════════════════════════════════════════════

def fig_hq_comparison():
    q_vals = np.array([1, 2, 3, 4])
    fig, ax = plt.subplots(figsize=(7, 5))

    for case in ("A", "B", "C"):
        model_hq = [mf_hq[case][f"h_q{q}"] for q in q_vals]
        paper_hq = [PAPER_CASE[case][f"h_q{q}"] for q in q_vals]
        ax.plot(q_vals, model_hq, "o-", color=COLORS[case], linewidth=1.8,
                label=f"Model {case} (Δh={mf_hq[case]['delta_h']:.3f})")
        ax.plot(q_vals, paper_hq, "s--", color=COLORS[case], linewidth=1.0, alpha=0.5,
                label=f"Paper {case}")

    ax.axhline(0.5, color="gray", linewidth=0.7, linestyle="--")
    ax.set_xlabel("q")
    ax.set_ylabel("h(q)")
    ax.set_title("Generalised Hurst exponent h(q) — Model vs Paper", fontsize=11, fontweight="bold")
    ax.set_xticks(q_vals)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    savefig(fig, "03_hq_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 04 — Normalised return PDFs overlay
# ══════════════════════════════════════════════════════════════════════════════

def fig_returns_overlay():
    fig, ax = plt.subplots(figsize=(8, 5))

    for case in ("A", "B", "C"):
        d = np.load(f"data/outputs/{case}/timeseries.npz")
        g = normalize_returns(d["returns"], method="paper")
        ax.hist(g, bins=100, density=True, alpha=0.45, color=COLORS[case],
                label=f"Case {case}")

    x = np.linspace(-8, 8, 300)
    ax.plot(x, norm.pdf(x), "k-", linewidth=1.5, label="Gaussian N(0,1)")
    ax.set_yscale("log")
    ax.set_xlabel("Normalised return g")
    ax.set_ylabel("PDF")
    ax.set_title("Return distributions (normalised) — all cases", fontsize=11, fontweight="bold")
    ax.set_xlim(-8, 8)
    ax.legend()
    fig.tight_layout()
    savefig(fig, "04_returns_overlay.png")


# ══════════════════════════════════════════════════════════════════════════════
# 05 — ACF of |returns| overlay
# ══════════════════════════════════════════════════════════════════════════════

def fig_acf_overlay():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Autocorrelation functions — volatility clustering", fontsize=11, fontweight="bold")
    MAX_LAG = 50

    for case in ("A", "B", "C"):
        d = np.load(f"data/outputs/{case}/timeseries.npz")
        g = normalize_returns(d["returns"], method="paper")
        acf_r  = compute_acf(g, MAX_LAG)
        acf_v  = compute_acf(np.abs(g), MAX_LAG)
        lags = np.arange(MAX_LAG + 1)
        ax1.plot(lags[1:], acf_r[1:],  color=COLORS[case], linewidth=1.5, label=f"Case {case}")
        ax2.plot(lags[1:], acf_v[1:],  color=COLORS[case], linewidth=1.5, label=f"Case {case}")

    for ax, title in [(ax1, "ACF of returns r(t)"), (ax2, "ACF of |returns|  (volatility clustering)")]:
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)
        ax.set_xlim(1, MAX_LAG)

    fig.tight_layout()
    savefig(fig, "05_acf_overlay.png")


# ══════════════════════════════════════════════════════════════════════════════
# 06 — Real vs Model scatter
# ══════════════════════════════════════════════════════════════════════════════

def fig_real_vs_model_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Model vs Real markets — Hurst (R/S) and DFA1", fontsize=11, fontweight="bold")

    pairs = [("H_rs", "DFA1", axes[0]), ("H_rs", "DFA3", axes[1])]

    for xkey, ykey, ax in pairs:
        # Real market points
        for idx, (ticker, tdata) in enumerate(real_tickers.items()):
            ax.scatter(tdata[xkey], tdata[ykey],
                       color=_ticker_color(ticker, idx), s=80, zorder=4,
                       label=_ticker_label(ticker))

        # Model cases
        for case in ("A", "B", "C"):
            mxkey = "H_rs_median" if xkey == "H_rs" else f"{xkey}_median"
            mykey = "DFA1_median" if ykey == "DFA1" else f"{ykey}_median"
            ax.scatter(metrics[case][mxkey], metrics[case][mykey],
                       marker="*", s=200, color=COLORS[case], zorder=5,
                       label=f"Model {case}")
            # Paper targets
            ax.scatter(PAPER_CASE[case][xkey], PAPER_CASE[case][ykey],
                       marker="^", s=80, color=COLORS[case], alpha=0.5, zorder=4,
                       label=f"Paper {case}" if case == "A" else None)

        ax.plot([0.4, 0.65], [0.4, 0.65], "k--", linewidth=0.8, alpha=0.3)
        ax.axhline(0.5, color="gray", linewidth=0.5, linestyle=":")
        ax.axvline(0.5, color="gray", linewidth=0.5, linestyle=":")
        ax.set_xlabel(f"H_RS", fontsize=10)
        ax.set_ylabel(ykey, fontsize=10)
        ax.set_title(f"H_RS vs {ykey}", fontsize=10)
        ax.set_xlim(0.44, 0.64)
        ax.set_ylim(0.38, 0.62)
        ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    savefig(fig, "06_real_vs_model_scatter.png")


# ══════════════════════════════════════════════════════════════════════════════
# 07 — Per-case return + price side by side
# ══════════════════════════════════════════════════════════════════════════════

def fig_timeseries_panel():
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.3)
    fig.suptitle("Return and price time series — all cases", fontsize=12, fontweight="bold")

    for row, case in enumerate(("A", "B", "C")):
        d = np.load(f"data/outputs/{case}/timeseries.npz")
        returns = d["returns"]
        price   = d["price"]
        t = np.arange(len(returns))

        ax_r = fig.add_subplot(gs[row, 0])
        ax_r.plot(t, returns, linewidth=0.5, color=COLORS[case])
        ax_r.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax_r.set_ylabel("r(t)")
        m_map = {"A": 10, "B": 5, "C": 5}
        lam_map = {"A": 0.25, "B": 0.20, "C": 0.15}
        ax_r.set_title(f"Case {case} \u2014 Returns (m={m_map[case]}, \u03bb={lam_map[case]})",
                       fontsize=9)
        if row == 2:
            ax_r.set_xlabel("Time step")

        ax_p = fig.add_subplot(gs[row, 1])
        ax_p.plot(t, price, linewidth=0.8, color=COLORS[case])
        ax_p.set_ylabel("p(t)  (log)" if price.max() / price.min() > 100 else "p(t)")
        if price.max() / price.min() > 100:
            ax_p.set_yscale("log")
        ax_p.set_title(f"Case {case} — Log-price", fontsize=9)
        if row == 2:
            ax_p.set_xlabel("Time step")

    savefig(fig, "07_timeseries_panel.png")


# ══════════════════════════════════════════════════════════════════════════════
# 08 — CCDF positive tail overlay
# ══════════════════════════════════════════════════════════════════════════════

def fig_ccdf_positive():
    fig, ax = plt.subplots(figsize=(7, 5))

    for case in ("A", "B", "C"):
        d = np.load(f"data/outputs/{case}/timeseries.npz")
        g = normalize_returns(d["returns"], method="paper")
        x, ccdf = tail_ccdf(g, positive=True)
        fit = fit_tail_best(g[g > 0])
        ax.plot(x, ccdf, ".", markersize=2, color=COLORS[case], alpha=0.7,
                label=f"Case {case}")
        if fit is not None and np.isfinite(fit["beta"]):
            x0_idx = len(x) // 4
            x_fit = np.linspace(x[x0_idx], x[-10] if len(x) > 10 else x[-1], 50)
            y_fit = (x_fit / x_fit[0]) ** (-fit["beta"]) * ccdf[x0_idx]
            ax.plot(x_fit, y_fit, "--", linewidth=1.5, color=COLORS[case],
                    label=f"\u03b2\u207a={fit['beta']:.2f} ({case})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|g|  (positive tail)")
    ax.set_ylabel("P(g > x)")
    ax.set_title("Positive-tail CCDF — all cases", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "08_ccdf_positive.png")


# ══════════════════════════════════════════════════════════════════════════════
# 09 — CCDF negative tail overlay
# ══════════════════════════════════════════════════════════════════════════════

def fig_ccdf_negative():
    fig, ax = plt.subplots(figsize=(7, 5))

    for case in ("A", "B", "C"):
        d = np.load(f"data/outputs/{case}/timeseries.npz")
        g = normalize_returns(d["returns"], method="paper")
        x, ccdf = tail_ccdf(g, positive=False)
        fit = fit_tail_best(np.abs(g[g < 0]))
        ax.plot(x, ccdf, ".", markersize=2, color=COLORS[case], alpha=0.7,
                label=f"Case {case}")
        if fit is not None and np.isfinite(fit["beta"]):
            x0_idx = len(x) // 4
            x_fit = np.linspace(x[x0_idx], x[-10] if len(x) > 10 else x[-1], 50)
            y_fit = (x_fit / x_fit[0]) ** (-fit["beta"]) * ccdf[x0_idx]
            ax.plot(x_fit, y_fit, "--", linewidth=1.5, color=COLORS[case],
                    label=f"\u03b2\u207b={fit['beta']:.2f} ({case})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|g|  (negative tail)")
    ax.set_ylabel("P(-g > x)")
    ax.set_title("Negative-tail CCDF — all cases", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "09_ccdf_negative.png")


# ══════════════════════════════════════════════════════════════════════════════
# 10 — DFA1 log-log fluctuation function overlay
# ══════════════════════════════════════════════════════════════════════════════

def fig_dfa1_overlay():
    fig, ax = plt.subplots(figsize=(7, 5))

    for case in ("A", "B", "C"):
        d = np.load(f"data/outputs/{case}/timeseries.npz")
        g = normalize_returns(d["returns"], method="paper")
        n_vals, F_vals, alpha = compute_dfa(g, order=1)
        valid = F_vals > 0
        log_n = np.log10(n_vals[valid].astype(float))
        log_F = np.log10(F_vals[valid])
        ax.plot(log_n, log_F, "o", markersize=4, color=COLORS[case], alpha=0.7)
        coeffs = np.polyfit(log_n, log_F, 1)
        ax.plot(log_n, np.polyval(coeffs, log_n), "-", linewidth=1.8,
                color=COLORS[case], label=f"Case {case}  \u03b1={alpha:.4f}")

    ax.set_xlabel("log\u2081\u2080(n)")
    ax.set_ylabel("log\u2081\u2080(F(n))")
    ax.set_title("DFA1 fluctuation function — all cases", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, "10_dfa1_overlay.png")


# ══════════════════════════════════════════════════════════════════════════════
# 11 — Multifractal spectrum f(alpha) overlay
# ══════════════════════════════════════════════════════════════════════════════

def fig_mf_spectrum_overlay():
    fig, ax = plt.subplots(figsize=(7, 5))

    q_values = np.concatenate([np.arange(-5, 0, 0.5), np.arange(0.5, 5.5, 0.5)])

    for case in ("A", "B", "C"):
        d = np.load(f"data/outputs/{case}/timeseries.npz")
        g = normalize_returns(d["returns"], method="paper")
        mf = mfdfa(g, q_values, order=1)
        alpha_mf = mf["alpha"]
        f_alpha  = mf["f_alpha"]
        valid = np.isfinite(alpha_mf) & np.isfinite(f_alpha)
        dh = mf_hq[case]["delta_h"]
        ax.plot(alpha_mf[valid], f_alpha[valid], "o-", markersize=4,
                color=COLORS[case], linewidth=1.8,
                label=f"Case {case}  (\u0394h={dh:.3f})")

    ax.set_xlabel("\u03b1")
    ax.set_ylabel("f(\u03b1)")
    ax.set_title("Multifractal spectrum f(\u03b1) — all cases", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, "11_mf_spectrum_overlay.png")


# ══════════════════════════════════════════════════════════════════════════════
# run all
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating summary figures …")
    np.random.seed(0)

    fig_metrics_comparison()
    fig_tail_exponents()
    fig_hq_comparison()
    fig_returns_overlay()
    fig_acf_overlay()
    fig_real_vs_model_scatter()
    fig_timeseries_panel()
    fig_ccdf_positive()
    fig_ccdf_negative()
    fig_dfa1_overlay()
    fig_mf_spectrum_overlay()

    print(f"\nAll summary figures written to  {OUTDIR}/")
