# FinIising — Self-Organizing 3D Ising Model of Financial Markets

A Python implementation and full reproduction of the agent-based financial
market model introduced in:

> W. R. S. Guimaraes and L. S. Lima, *"Self-organizing three-dimensional Ising
> model of financial markets"*, **Phys. Rev. E 103, 062130 (2021)**.
> DOI: [10.1103/PhysRevE.103.062130](https://doi.org/10.1103/PhysRevE.103.062130)

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Requirements and Installation](#requirements-and-installation)
4. [Configuration](#configuration)
5. [Usage](#usage)
   - [Run a simulation](#1-run-a-simulation)
   - [Calibrate hidden parameters](#2-calibrate-hidden-parameters)
   - [Compare with real market data](#3-compare-with-real-market-data)
   - [Generate summary figures](#4-generate-summary-figures)
   - [Quick evaluation](#5-quick-evaluation)
6. [Reproducing Results from Scratch](#reproducing-results-from-scratch)
7. [Output Files](#output-files)
8. [Model Details](#model-details)
9. [Key Results](#key-results)

---

## Overview

Each "trader" is an Ising spin S_i ∈ {+1 (buy), −1 (sell)} on an
m × m × m cubic lattice with periodic boundary conditions.  Spins interact
through **dynamic couplings** J_i(t) that encode market memory and news
feedback.  The aggregate magnetisation maps directly to the market return:

```
r(t) = ⟨S(t)⟩ / (N · η)
```

The model reproduces five stylised facts of real financial returns:

| Stylised fact | Status |
|---|---|
| Fat-tailed return distribution (inverse cubic law, β ≈ 3) | ✔ reproduced |
| Vanishing long-lag autocorrelation of returns | ✔ reproduced |
| Long-range memory: Hurst H > 0.5, DFA exponent > 0.5 | ✔ reproduced |
| Genuine multifractality h(q) non-constant | ✔ reproduced |
| Volatility clustering (ACF of \|r\|) | partial (strong at lag 1, decays by lag 10) |

---

## Project Structure

```
FinIising/
├── config/
│   ├── default.yaml          # Shared calibrated parameters (η, c_max, σ, β)
│   ├── case_A.yaml           # Paper Table I: m=10, λ=0.25, b_max=0.05
│   ├── case_B.yaml           # Paper Table I: m=5,  λ=0.20, b_max=0.09
│   └── case_C.yaml           # Paper Table I: m=5,  λ=0.15, b_max=0.09
├── src/
│   ├── __init__.py           # Package docstring
│   ├── lattice.py            # 3D PBC neighbor table
│   ├── model.py              # Monte Carlo engine (Glauber + Numba JIT)
│   ├── config_loader.py      # YAML config loader with default merging
│   ├── observables.py        # Return normalization, .npz I/O
│   ├── analysis_memory.py    # ACF, R/S Hurst, DFA (orders 1-3)
│   ├── analysis_multifractal.py  # MF-DFA: h(q), τ(q), f(α)
│   ├── analysis_tail.py      # CCDF, OLS power-law fits, Hill estimator
│   └── plotting.py           # Matplotlib figure generators
├── scripts/
│   ├── calibrate.py          # LHS + Nelder-Mead calibration of 4 params
│   ├── quick_eval.py         # Fast comparison vs paper targets
│   ├── real_data.py          # Download real indices, compute stylized facts
│   └── generate_summary_figures.py  # Cross-case comparison figures
├── tests/
│   ├── test_lattice.py
│   ├── test_model.py
│   └── test_analysis.py
├── reports/
│   └── final_report.md       # Full reproduction report with tables and figures
├── run_simulation.py         # Main CLI entry point
└── requirements.txt
```

---

## Requirements and Installation

**Python 3.11+** is required.

### Core dependencies (required)

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:
- `numpy >= 1.24`
- `scipy >= 1.10`
- `matplotlib >= 3.7`
- `pyyaml >= 6.0`

### Optional but recommended

| Package | Purpose |
|---|---|
| `numba` | ~100× JIT speed-up of the Monte Carlo sweep |
| `yfinance` | Download real market data in `scripts/real_data.py` |

```bash
pip install numba yfinance
```

> **Without Numba** the simulation falls back to a pure-Python inner loop.
> A full Case A run (m=10, T=8000, 5 seeds) takes a few minutes with Numba
> and tens of minutes without.

### Recommended virtual-environment setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install numba yfinance      # optional
```

---

## Configuration

Configuration uses a two-level YAML hierarchy:

1. **`config/default.yaml`** — shared parameters common to all cases:
   - Calibrated hidden parameters: `eta`, `c_max`, `noise_scale`, `beta`, `delta`
   - Simulation defaults: `burn_in`, `production`, `seed`, `sweeps_per_step`, `n_seeds`

2. **`config/case_*.yaml`** — per-case paper parameters (override defaults):
   - `case_name`, `lattice.m`, `model.lambda_`, `model.b_max`

Example `config/case_A.yaml`:

```yaml
case_name: "A"
lattice:
  m: 10
model:
  lambda_: 0.25
  b_max: 0.05
```

Example `config/default.yaml` (auto-written by `scripts/calibrate.py --write`):

```yaml
model:
  eta: 3.2249
  c_max: 0.6364
  noise_scale: 1.3328
  beta: 1.3783
  delta: 1
simulation:
  burn_in: 3000
  production: 8000
  seed: 42
  sweeps_per_step: 1
  n_seeds: 5
```

Any key set in a case file overrides the corresponding default.

---

## Usage

All commands should be run from the repository root.

### 1. Run a simulation

```bash
# Run case A — simulation only
python run_simulation.py config/case_A.yaml

# Run case A with statistical analysis
python run_simulation.py config/case_A.yaml --analyze

# Run case A with analysis and figures (first realisation)
python run_simulation.py config/case_A.yaml --analyze --plot

# Override ensemble size on the command line
python run_simulation.py config/case_A.yaml --analyze --seeds 10
```

This saves:
- `data/outputs/A/timeseries.npz` — returns, price, magnetisation, news
- `data/outputs/A/metrics.json` — ensemble-median statistics
- `figures/A/*.png` (13 figures, if `--plot` is given)

Repeat with `case_B.yaml` and `case_C.yaml` for the other two cases.

### 2. Calibrate hidden parameters

The paper does not disclose the four shared parameters
(η, c_max, noise_scale, β).  `scripts/calibrate.py` recovers them via a
three-stage search: Latin-Hypercube sampling → refinement → Nelder-Mead.

```bash
# Fast test run (~5 min, single worker)
python scripts/calibrate.py \
    --coarse-samples 40 --coarse-seeds 1 \
    --coarse-burnin 1000 --coarse-production 2000 \
    --refine-top 5 --refine-seeds 2 \
    --local-samples 0

# Full calibration (matches paper targets, ~16 min, 8 workers)
python scripts/calibrate.py \
    --coarse-samples 80 --coarse-seeds 2 \
    --coarse-burnin 2000 --coarse-production 3500 \
    --refine-top 12 --refine-seeds 3 \
    --refine-burnin 3000 --refine-production 6000 \
    --local-samples 60 --workers 8 \
    --write --out data/outputs/calibration.json
```

The `--write` flag automatically updates `config/default.yaml` with the
best parameters found.

Key options:

| Option | Default | Description |
|---|---|---|
| `--coarse-samples` | 120 | LHS candidates in stage 1 |
| `--refine-top` | 15 | Best stage-1 points to refine |
| `--local-samples` | 200 | Max Nelder-Mead iterations (0 = skip) |
| `--workers` | 1 | Parallel worker processes |
| `--write` | false | Write best params to `config/default.yaml` |
| `--out` | `data/outputs/calibration.json` | Checkpoint file |

### 3. Compare with real market data

Downloads daily closing prices via `yfinance` and runs the full analysis
pipeline on the downloaded returns.

```bash
# Default: S&P 500, DJI, NASDAQ, FTSE, Nikkei 225 from 1990 to today
python scripts/real_data.py

# Paper's window and indices
python scripts/real_data.py \
    --tickers "^GSPC" "^GDAXI" "^N225" "^NSEI" "^BSESN" \
    --start 1999-01-01 --end 2019-08-31 \
    --out data/outputs/real_data_metrics.json

# Custom indices
python scripts/real_data.py --tickers AAPL MSFT --start 2010-01-01
```

Outputs a formatted table of H_RS, DFA1–3, β⁺, β⁻ per ticker vs. the
paper targets, and saves metrics to `data/outputs/real_data_metrics.json`.

### 4. Generate summary figures

After running all three cases and `scripts/real_data.py`:

```bash
python scripts/generate_summary_figures.py
```

Writes 11 comparison figures to `figures/summary/`.

### 5. Quick evaluation

Compare the current `config/default.yaml` parameters against the paper
targets without re-running calibration:

```bash
python scripts/quick_eval.py
```

Prints a table with per-metric deviations for all three cases.

---

## Reproducing Results from Scratch

```bash
# Step 0 (optional): clean previous outputs
rm -rf data/outputs/* figures/A/* figures/B/* figures/C/* figures/summary/*

# Step 1: calibrate the four shared hidden parameters
python scripts/calibrate.py \
    --coarse-samples 80 --coarse-seeds 2 \
    --coarse-burnin 2000 --coarse-production 3500 \
    --refine-top 12 --refine-seeds 3 \
    --refine-burnin 3000 --refine-production 6000 \
    --local-samples 60 --workers 8 \
    --write --out data/outputs/calibration.json

# Step 2: production simulation for all three cases
python run_simulation.py config/case_A.yaml --analyze --plot
python run_simulation.py config/case_B.yaml --analyze --plot
python run_simulation.py config/case_C.yaml --analyze --plot

# Step 3: real-market comparison
python scripts/real_data.py \
    --tickers "^GSPC" "^GDAXI" "^N225" "^NSEI" "^BSESN" \
    --start 1999-01-01 --end 2019-08-31

# Step 4: summary figures
python scripts/generate_summary_figures.py
```

> **Seeds**: Production simulations use seeds
> `cfg.seed + 1000·k` for k = 0 … n_seeds−1 (defaults: 42, 1042, 2042, …).
> Calibration LHS uses `numpy.random.default_rng(123)`.

---

## Output Files

| Path | Description |
|---|---|
| `config/default.yaml` | Calibrated shared parameters |
| `data/outputs/calibration.json` | Full calibration history (stage 1/2/3) |
| `data/outputs/{A,B,C}/timeseries.npz` | Returns, price, magnetisation, news (seed 0) |
| `data/outputs/{A,B,C}/metrics.json` | Ensemble-median statistics (all seeds) |
| `data/outputs/real_data_metrics.json` | Stylized-fact metrics for all downloaded tickers |
| `figures/{A,B,C}/*.png` | 13 per-case diagnostic figures |
| `figures/summary/*.png` | 11 cross-case and real-data comparison figures |

---

## Model Details

### Lattice

An m × m × m cubic lattice with periodic boundary conditions.
Each site *i* has exactly 6 nearest neighbors (±x, ±y, ±z).
See `src/lattice.py`.

### Coupling recursion (paper Eq. 2)

```
J_i(t) = b_i + λ · J_i(t-1) + δ · r(t-1) · G(t-1)
```

- `b_i ~ U[0, b_max]` — frozen heterogeneous base coupling
- `λ` — persistence parameter (0 < λ < 1)
- `δ = 1` — market-feedback amplitude (fixed by paper)
- `r(t-1)` — previous-step return; `G(t-1)` — previous-step news draw

### Glauber heat-bath update

Local field: `h_i = J_i · Σ_{j ∈ NN} S_j + c_i · G_t + ε_i`

Flip probability: `P(S_i = +1) = 1 / (1 + exp(-2β h_i))`

Spins are updated in **random sequential** order (one full sweep = N
update attempts in a random permutation), avoiding the spurious
correlations of synchronous updates.

### Return and price

```
r(t) = ⟨S(t)⟩ / (N · η)      # log-return proxy
p(t) = exp(Σ_{s≤t} r(s))       # log-price
g(t) = (r(t) - ⟨r⟩) / ⟨|r|⟩   # normalised return (paper Eq. 4)
```

### Numba acceleration

When `numba` is installed, the inner sweep loop is JIT-compiled with
`@njit(cache=True, fastmath=True)`, giving roughly 100× speed-up over the
pure-Python fallback.  The API is identical in both cases.

### Paper parameter sets (Table I)

| Case | m | N = m³ | λ | b_max |
|------|---|--------|------|-------|
| A | 10 | 1000 | 0.25 | 0.05 |
| B | 5  | 125  | 0.20 | 0.09 |
| C | 5  | 125  | 0.15 | 0.09 |

---

## Key Results

Calibrated shared parameters: η = 3.225, c_max = 0.636, noise_scale = 1.333, β = 1.378

| Metric | Paper (Case A) | Model (Case A, median 5 seeds) |
|---|---|---|
| Hurst R/S | 0.5533 | **0.5663** |
| DFA1 | 0.5409 | **0.5195** |
| DFA2 | 0.5535 | **0.5292** |
| DFA3 | 0.5493 | **0.5337** |
| β⁺ (positive tail) | 3.13 | **3.39** |
| β⁻ (negative tail) | 3.60 | **3.93** |

All three model Hurst values fall inside the paper's quoted confidence
interval [0.42, 0.58].  The "inverse cubic law" (β ≈ 3) is reproduced
across all cases.  See `reports/final_report.md` for a full per-case
analysis and comparison with five real stock-market indices.

---

## Running Tests

```bash
pytest tests/
```

Tests cover the lattice neighbor table, a smoke-test of a short simulation
run, and basic sanity checks on the analysis functions.
