# FinIising — Self-Organizing 3D Ising Model of Financial Markets

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

A Python implementation and reproduction of the agent-based financial market
model introduced in:

> W. R. S. Guimaraes and L. S. Lima, *"Self-organizing three-dimensional Ising
> model of financial markets"*, **Phys. Rev. E 103, 062130 (2021)**.
> DOI: [10.1103/PhysRevE.103.062130](https://doi.org/10.1103/PhysRevE.103.062130)

Traders are modeled as Ising spins on a 3D lattice whose couplings evolve
with market feedback.  Running the simulation generates price series and
return statistics that reproduce key stylised facts of real financial markets.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Requirements and Installation](#requirements-and-installation)
4. [Configuration](#configuration)
5. [Usage](#usage)
   - [Run a simulation](#1-run-a-simulation)
   - [Custom simulations](#2-custom-simulations)
   - [Calibrate hidden parameters](#3-calibrate-hidden-parameters)
   - [Compare with real market data](#4-compare-with-real-market-data)
   - [Generate summary figures](#5-generate-summary-figures)
   - [Quick evaluation](#6-quick-evaluation)
6. [What to Expect](#what-to-expect)
7. [Reproducing the Paper Cases](#reproducing-the-paper-cases)
8. [Output Files](#output-files)
9. [Model Details](#model-details)
10. [Running Tests](#running-tests)
11. [License](#license)

---

## Overview

Each "trader" is an Ising spin S_i ∈ {+1 (buy), −1 (sell)} on an
m × m × m cubic lattice with periodic boundary conditions.  Spins interact
through **dynamic couplings** J_i(t) that encode market memory and news
feedback.  The aggregate magnetisation maps directly to the market return:

```
r(t) = ⟨S(t)⟩ / (N · η)
```

The model is designed to reproduce these stylised facts of real financial
returns:

| Stylised fact | Expected behaviour |
|---|---|
| Fat-tailed return distribution | Power-law CCDF with exponent β ≈ 3 ("inverse cubic law") |
| Return autocorrelation | Near-zero beyond a few lags (no spurious momentum) |
| Long-range memory | Hurst exponent H > 0.5, DFA scaling exponent α > 0.5 |
| Multifractality | h(q) decreasing in q — richer structure than Brownian motion |
| Volatility clustering | Strong ACF of \|r\| at lag 1; decays within ~10 lags |

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
│   ├── test_analysis.py
│   └── test_config_loader.py
├── run_simulation.py         # Main CLI entry point
├── pyproject.toml
└── requirements.txt
```

---

## Requirements and Installation

**Python 3.11+** is required.

### Quick setup (recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Editable install — makes 'from src import ...' work from anywhere
pip install -e .

# With optional extras
pip install -e ".[numba]"       # JIT acceleration (~100× faster Monte Carlo)
pip install -e ".[data]"        # yfinance for real market data download
pip install -e ".[full]"        # numba + yfinance
pip install -e ".[dev]"         # full + pytest + ruff (for development)
```

### Alternative (requirements file only)

```bash
pip install -r requirements.txt
pip install numba yfinance      # optional
```

### Optional packages explained

| Package | Why you want it |
|---|---|
| `numba` | JIT-compiles the inner MC sweep; a full Case A run (m=10, T=8000, 5 seeds) takes **minutes** with Numba vs **tens of minutes** without |
| `yfinance` | Required only for `scripts/real_data.py` to download historical index prices |

---

## Configuration

Configuration is layered: a **default file** holds shared calibrated
parameters, and each **case file** adds or overrides the values it needs.

### Layer 1 — `config/default.yaml` (shared calibrated parameters)

```yaml
model:
  eta: 3.2249          # market depth: r = ⟨S⟩ / (N·η)
  c_max: 0.6364        # max news-sensitivity U[0, c_max]
  noise_scale: 1.3328  # private noise std ε_i ~ N(0, σ)
  beta: 1.3783         # inverse temperature of Glauber rule
  delta: 1             # feedback amplitude in coupling recursion
simulation:
  burn_in: 3000        # timesteps discarded before recording
  production: 8000     # timesteps recorded and analysed
  seed: 42             # base RNG seed
  sweeps_per_step: 1   # MC sweeps per timestep
  n_seeds: 5           # ensemble size
```

These four parameters (`eta`, `c_max`, `noise_scale`, `beta`) are not
disclosed in the paper and were recovered by `scripts/calibrate.py`.

### Layer 2 — `config/case_*.yaml` (per-case paper parameters)

Each case file only needs the values from paper Table I:

```yaml
case_name: "A"
lattice:
  m: 10          # lattice size; N = m³ agents
model:
  lambda_: 0.25  # coupling persistence
  b_max: 0.05    # max base coupling U[0, b_max]
```

Any key in the case file overrides the corresponding default.

---

## Usage

All commands must be run from the **repository root**.

### 1. Run a simulation

```bash
# Simulation only (saves timeseries.npz)
python run_simulation.py config/case_A.yaml

# With statistical analysis (also saves metrics.json)
python run_simulation.py config/case_A.yaml --analyze

# With analysis + all diagnostic figures
python run_simulation.py config/case_A.yaml --analyze --plot

# Override ensemble size
python run_simulation.py config/case_A.yaml --analyze --seeds 10
```

Outputs written to `data/outputs/A/` and `figures/A/`.

---

### 2. Custom Simulations

You can run the model with any parameter combination in two ways.

#### Option A — Custom YAML file (recommended for reusable configs)

Create `config/my_experiment.yaml`:

```yaml
case_name: "my_exp"
lattice:
  m: 8               # 8×8×8 = 512 agents
model:
  lambda_: 0.30      # higher persistence than Case A
  b_max: 0.07
  beta: 1.5          # stronger ordering tendency
  eta: 2.0           # shallower market depth → larger returns
simulation:
  production: 15000  # longer run for better statistics
  n_seeds: 3
```

Then run it:

```bash
python run_simulation.py config/my_experiment.yaml --analyze --plot
```

Only the keys you specify are overridden; everything else inherits from
`config/default.yaml`.

#### Option B — Inline `--set` overrides (quick one-off experiments)

Override any parameter directly on the command line without editing a file:

```bash
python run_simulation.py config/case_A.yaml --analyze --plot \
    --set model.lambda_=0.35 \
    --set model.eta=2.0 \
    --set simulation.production=12000
```

The format is `--set section.key=value` where `section` is one of
`lattice`, `model`, or `simulation`.  You can chain as many `--set`
flags as needed.  Active overrides are printed at startup.

#### Full parameter reference

| YAML key | Section | Type | Default | Description |
|---|---|---|---|---|
| `m` | `lattice` | int | — | Linear lattice size; N = m³ agents |
| `lambda_` | `model` | float | — | Coupling persistence (0 < λ < 1) |
| `b_max` | `model` | float | — | Upper bound of base coupling U[0, b_max] |
| `c_max` | `model` | float | 0.6364 | Upper bound of news-sensitivity U[0, c_max] |
| `eta` | `model` | float | 3.2249 | Market depth: r = ⟨S⟩ / (N·η) |
| `beta` | `model` | float | 1.3783 | Inverse temperature of the Glauber heat-bath |
| `noise_scale` | `model` | float | 1.3328 | Std of per-agent private noise ε_i ~ N(0, σ) |
| `delta` | `model` | float | 1 | Feedback amplitude in coupling recursion |
| `burn_in` | `simulation` | int | 3000 | Timesteps discarded before recording |
| `production` | `simulation` | int | 8000 | Timesteps recorded and analysed |
| `seed` | `simulation` | int | 42 | Base RNG seed |
| `sweeps_per_step` | `simulation` | int | 1 | Full MC sweeps per timestep |
| `n_seeds` | `simulation` | int | 5 | Ensemble size (overridden by `--seeds`) |

**Parameter intuition**

- **`m`** — larger lattice means more agents and smoother statistics but
  slower runs (runtime scales as m³).  m=5 (125 agents) is fast; m=10
  (1000 agents) is the most realistic.
- **`lambda_`** — higher λ increases coupling memory, leading to stronger
  long-range persistence (higher Hurst exponent).
- **`b_max`** — higher b_max increases heterogeneity; agents with larger b
  are more sensitive to their neighbors.
- **`beta`** — higher β → stronger ordering → larger magnetisation swings
  and more volatile returns.
- **`eta`** — lower η amplifies the return magnitude; very low η can cause
  extreme price excursions.

---

### 3. Calibrate hidden parameters

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

# Full calibration (~16 min on 8 cores)
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

| Option | Default | Description |
|---|---|---|
| `--coarse-samples` | 120 | LHS candidates in stage 1 |
| `--refine-top` | 15 | Best stage-1 points to refine |
| `--local-samples` | 200 | Max Nelder-Mead iterations (0 = skip) |
| `--workers` | 1 | Parallel worker processes |
| `--write` | false | Write best params to `config/default.yaml` |
| `--out` | `data/outputs/calibration.json` | Checkpoint file |

---

### 4. Compare with real market data

Downloads daily closing prices via `yfinance` and computes the same
stylised-fact statistics on real return series.  Requires `pip install yfinance`.

```bash
# Default: S&P 500, DJI, NASDAQ, FTSE, Nikkei 225 (1990 to today)
python scripts/real_data.py

# Paper's window and indices
python scripts/real_data.py \
    --tickers "^GSPC" "^GDAXI" "^N225" "^NSEI" "^BSESN" \
    --start 1999-01-01 --end 2019-08-31

# Custom tickers
python scripts/real_data.py --tickers AAPL MSFT --start 2010-01-01
```

Prints a comparison table and saves metrics to
`data/outputs/real_data_metrics.json`.

---

### 5. Generate summary figures

After running all three paper cases **and** `scripts/real_data.py`:

```bash
python scripts/generate_summary_figures.py
```

Writes 11 cross-case comparison figures to `figures/summary/`:

| Figure | Contents |
|---|---|
| `01_metrics_comparison.png` | H_RS, DFA1–3 grouped bars: Paper / Model / Real |
| `02_tail_exponents.png` | β⁺, β⁻ vs paper targets and real-market range |
| `03_hq_comparison.png` | h(q) curves for model vs paper (multifractality) |
| `04_returns_overlay.png` | Normalised return PDFs for all three cases |
| `05_acf_overlay.png` | ACF of returns and \|returns\| (volatility clustering) |
| `06_real_vs_model_scatter.png` | H_RS vs DFA1/DFA3: real indices + model |
| `07_timeseries_panel.png` | 3×2 panel: returns and price for all cases |
| `08_ccdf_positive.png` | Positive-tail CCDF all cases (log-log) |
| `09_ccdf_negative.png` | Negative-tail CCDF all cases (log-log) |
| `10_dfa1_overlay.png` | DFA1 fluctuation function F(n) all cases |
| `11_mf_spectrum_overlay.png` | Multifractal f(α) spectrum all cases |

---

### 6. Quick evaluation

Compare the current `config/default.yaml` parameters against the paper
targets without re-running calibration:

```bash
python scripts/quick_eval.py
```

Prints a side-by-side table of Paper target / Before calibration / Now for
all three cases.

---

## What to Expect

Running the simulation generates time series and figures.  Here is what
each diagnostic should look like for a well-calibrated run:

### Return time series `figures/{case}/returns.png`
Irregular, zero-mean fluctuations with occasional large spikes ("volatility
clusters").  The series should not look like pure white noise — there will
be episodes of elevated variance.

### Price series `figures/{case}/price.png`
A random-walk-like trajectory built from cumulative log-returns.  Expect
realistic-looking price paths with occasional trend-reversals, not a
monotone ramp.

### Return histogram `figures/{case}/hist_returns.png`
The distribution should be **leptokurtic** — taller peak and heavier tails
than the overlaid Gaussian.  The log-scale y-axis makes this clear: the
model distribution sits noticeably above the Gaussian bell at large |g|.

### Tail CCDF `figures/{case}/ccdf_positive.png` / `ccdf_negative.png`
A log-log plot.  The empirical CCDF should follow a power-law (straight line)
in the tail region.  The fitted slope β should be in the range **2.5–5**,
with values near 3 matching the "inverse cubic law" of real markets.

### ACF comparison `figures/{case}/acf_comparison.png`
The ACF of raw returns should drop to near zero within a few lags (no
predictability).  The ACF of |returns| should show a stronger positive
value at lag 1 (volatility clustering), decaying within ~10 lags.

### R/S Hurst `figures/{case}/rs_hurst.png`
A log-log scatter of R/S vs block size n.  The slope (Hurst exponent H)
should be slightly above 0.5, indicating mild long-range dependence.
H ≈ 0.5–0.6 is the expected range.

### DFA1/2/3 `figures/{case}/dfa1.png` etc.
Log-log fluctuation function F(n) vs n.  The scaling exponent α should
also be above 0.5.  DFA2 and DFA3 should give similar results to DFA1
(different polynomial detrending does not change the conclusion for these
series).

### Multifractal h(q) `figures/{case}/mf_hq.png`
A decreasing curve of h(q) vs q confirms genuine multifractality.
Δh = h(q_min) − h(q_max) in the range **0.05–0.20** is typical.
A flat line at h = 0.5 would indicate a simple random walk.

### Multifractal spectrum `figures/{case}/mf_spectrum.png`
A concave-downward parabolic arc in the f(α) vs α plane.  The wider the
arc, the stronger the multifractality.

---

### Effect of changing parameters

| Parameter change | Expected effect on output |
|---|---|
| Larger `m` (e.g. 10 → 15) | Smoother statistics; longer runtime |
| Higher `lambda_` (e.g. 0.25 → 0.40) | Stronger memory → higher Hurst H, higher DFA α |
| Lower `lambda_` (e.g. 0.25 → 0.10) | Weaker memory → H closer to 0.5 |
| Higher `beta` | Stronger ordering → larger magnetisation swings → fatter tails |
| Lower `eta` | Larger return magnitude; same qualitative shape |
| Lower `b_max` | Weaker coupling heterogeneity → more uniform agent behaviour |
| Longer `production` | More stable tail and memory estimates; figures smoother |

---

## Reproducing the Paper Cases

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

# Step 2: production simulation for all three paper cases
python run_simulation.py config/case_A.yaml --analyze --plot
python run_simulation.py config/case_B.yaml --analyze --plot
python run_simulation.py config/case_C.yaml --analyze --plot

# Step 3: real-market comparison (requires yfinance)
python scripts/real_data.py \
    --tickers "^GSPC" "^GDAXI" "^N225" "^NSEI" "^BSESN" \
    --start 1999-01-01 --end 2019-08-31

# Step 4: cross-case summary figures
python scripts/generate_summary_figures.py
```

> **Seeds**: production runs use `seed + 1000·k` for k = 0 … n_seeds−1
> (default: 42, 1042, 2042, …).  Calibration LHS uses
> `numpy.random.default_rng(123)`.

---

## Output Files

| Path | Description |
|---|---|
| `config/default.yaml` | Calibrated shared parameters |
| `data/outputs/calibration.json` | Full calibration history (stage 1/2/3 + winner) |
| `data/outputs/{case}/timeseries.npz` | Returns, price, magnetisation, news (first seed) |
| `data/outputs/{case}/metrics.json` | Ensemble-median statistics across all seeds |
| `data/outputs/real_data_metrics.json` | Stylized-fact metrics for downloaded tickers |
| `figures/{case}/*.png` | 13 per-case diagnostic figures |
| `figures/summary/*.png` | 11 cross-case and real-data comparison figures |

All output directories are created automatically. `data/outputs/` and
`figures/` are in `.gitignore` and will never be committed.

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
- `λ` — persistence (0 < λ < 1)
- `δ = 1` — market-feedback amplitude (fixed by paper)
- `r(t-1)` — previous-step return; `G(t-1)` — previous-step news draw

### Glauber heat-bath update

Local field: `h_i = J_i · Σ_{j ∈ NN} S_j + c_i · G_t + ε_i`

Flip probability: `P(S_i = +1) = 1 / (1 + exp(-2β h_i))`

Spins are updated in **random sequential** order (one full sweep = N
update attempts in a random permutation), which avoids the spurious
correlations of synchronous updates.

### Return and price

```
r(t) = ⟨S(t)⟩ / (N · η)        # log-return proxy
p(t) = exp(Σ_{s≤t} r(s))         # log-price
g(t) = (r(t) − ⟨r⟩) / ⟨|r|⟩    # normalised return (paper Eq. 4)
```

### Numba acceleration

When `numba` is installed, the inner sweep loop is JIT-compiled with
`@njit(cache=True, fastmath=True)`, giving roughly 100× speed-up over the
pure-Python fallback.  The simulation API is identical in both cases.

### Paper parameter sets (Table I)

| Case | m | N = m³ | λ | b_max |
|------|---|--------|------|-------|
| A | 10 | 1000 | 0.25 | 0.05 |
| B | 5  | 125  | 0.20 | 0.09 |
| C | 5  | 125  | 0.15 | 0.09 |

---

## Running Tests

```bash
pytest tests/
```

The suite contains 51 tests across four files:

| File | Coverage |
|---|---|
| `tests/test_lattice.py` | Neighbor table shape, symmetry, periodic wrapping |
| `tests/test_model.py` | Reproducibility, physics sanity, ensemble, seed overrides |
| `tests/test_analysis.py` | ACF, R/S, DFA, Hill estimator, CCDF, tail fits, MF-DFA |
| `tests/test_config_loader.py` | YAML merging, inline overrides, SimConfig field defaults |

---

## License

This project is released under the [MIT License](LICENSE).

The reference paper is © American Physical Society (2021).  The PDF is
not redistributed in this repository.
