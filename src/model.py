"""Core 3D Ising financial market simulation engine.

The model places N = m³ heterogeneous "traders" (spins S_i ∈ {+1, -1})
on a cubic lattice with periodic boundary conditions.  At each timestep the
spins are updated sequentially via a Glauber heat-bath rule driven by a
dynamic, feedback-aware coupling J_i(t).

Physics summary
---------------
*   **Coupling recursion** (paper Eq. 2):

        J_i(t) = b_i + λ J_i(t-1) + δ · r(t-1) · G(t-1)

    where ``b_i ~ U[0, b_max]`` is frozen heterogeneity, ``λ`` controls
    persistence, and ``δ·r·G`` injects a market-feedback term.

*   **Local field** for site *i*:

        h_i = J_i · Σ_{j ∈ NN(i)} S_j + c_i · G + ε_i

    where ``c_i ~ U[0, c_max]`` is the news-sensitivity coupling,
    G ~ N(0,1) is exogenous news, and ``ε_i ~ N(0, σ)`` is private noise.

*   **Glauber heat-bath update**:

        P(S_i = +1) = 1 / (1 + exp(-2β h_i))

*   **Market return** (paper Eq. 5):

        r(t) = ⟨S(t)⟩ / (N · η)

*   **Log-price**: p(t) = exp(Σ r(t))

Numba acceleration
------------------
If `numba` is installed the inner Monte Carlo sweep (`_sweep_numba`) is JIT
compiled with ``@njit(cache=True, fastmath=True)``, giving roughly two orders
of magnitude speed-up over the pure-Python fallback.  The public API is
identical in both cases.
"""

import numpy as np

from .config_loader import SimConfig
from .lattice import build_neighbor_table

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False


def _sweep_python(spins, neighbors, J_coupling, c, G_t, noise,
                  beta, order, u):
    """Random sequential (Glauber) Monte Carlo sweep — Python fallback."""
    N = len(spins)
    for idx in range(N):
        i = order[idx]
        nn_sum = 0.0
        for d in range(6):
            nn_sum += spins[neighbors[i, d]]
        h_i = J_coupling[i] * nn_sum + c[i] * G_t + noise[i]
        prob_up = 1.0 / (1.0 + np.exp(-2.0 * beta * h_i))
        spins[i] = 1 if u[idx] < prob_up else -1


if _HAS_NUMBA:
    @njit(cache=True, fastmath=True)
    def _sweep_numba(spins, neighbors, J_coupling, c, G_t, noise,
                     beta, order, u):
        N = len(spins)
        for idx in range(N):
            i = order[idx]
            nn_sum = 0.0
            for d in range(6):
                nn_sum += spins[neighbors[i, d]]
            h_i = J_coupling[i] * nn_sum + c[i] * G_t + noise[i]
            prob_up = 1.0 / (1.0 + np.exp(-2.0 * beta * h_i))
            spins[i] = 1 if u[idx] < prob_up else -1

    _sweep = _sweep_numba
else:  # pragma: no cover
    _sweep = _sweep_python


class IsingMarketModel:
    """Self-organizing 3D Ising model of financial markets.

    Based on Guimaraes & Lima, Phys. Rev. E 103, 062130 (2021).
    """

    def __init__(self, config: SimConfig, seed: int | None = None):
        """Initialize the model from a configuration object.

        Parameters
        ----------
        config : SimConfig
            Fully merged simulation configuration (see
            :func:`src.config_loader.load_config`).
        seed : int, optional
            Random seed for this realisation.  If *None* the seed stored
            in ``config.seed`` is used.  Passing an explicit seed allows
            multiple independent realisations without editing the config
            (see :func:`run_ensemble`).
        """
        self.cfg = config
        self.m = config.m
        self.N = config.m ** 3
        self.neighbors = build_neighbor_table(config.m)
        self.rng = np.random.default_rng(config.seed if seed is None else seed)

        # Spin state: +1 (buy) or -1 (sell)
        self.spins = self.rng.choice(
            np.array([1, -1], dtype=np.int8), size=self.N
        )

        # Frozen heterogeneous agent parameters drawn once at construction
        self.b = self.rng.uniform(0.0, config.b_max, size=self.N)
        self.c = self.rng.uniform(0.0, config.c_max, size=self.N)

        # Dynamic coupling per agent (initialized to b_i)
        self.J_coupling = self.b.copy()

        # Previous-step market state used in the coupling recursion
        self.r_prev = 0.0
        self.G_prev = 0.0

    def step(self) -> dict:
        """Execute one simulation timestep and return observables.

        The sequence follows the paper's algorithm:
        1. Draw exogenous news G_t ~ N(0, 1).
        2. Update all couplings J_i(t) via the recursion (paper Eq. 2).
        3. Run ``sweeps_per_step`` random-sequential Glauber sweeps over
           all N sites.
        4. Compute magnetization and return r(t) = ⟨S⟩ / (N·η).

        Returns
        -------
        dict with keys:
            ``magnetization`` : float
                Mean spin ⟨S⟩ ∈ [-1, 1].
            ``raw_return`` : float
                Log-return proxy r(t) = magnetization / η.
            ``G_news`` : float
                The exogenous news draw G_t used in this step.
        """
        cfg = self.cfg

        # 1. External news ~ N(0, 1)
        G_t = self.rng.standard_normal()

        # 2. Coupling recursion (paper Eq. 2):
        #    J_i(t) = b_i + lambda J_i(t-1) + delta * r(t-1) * G(t-1)
        self.J_coupling = (
            self.b
            + cfg.lambda_ * self.J_coupling
            + cfg.delta * self.r_prev * self.G_prev
        )

        # 3-4. Random sequential (Glauber) Monte Carlo sweeps.
        #       Synchronous updates over-correlate neighboring sites.
        spins_f = self.spins.astype(np.int64)
        for _ in range(cfg.sweeps_per_step):
            sweep_order = self.rng.permutation(self.N).astype(np.int32)
            private_noise = self.rng.normal(0.0, cfg.noise_scale, size=self.N)
            u = self.rng.random(self.N)
            _sweep(spins_f, self.neighbors, self.J_coupling, self.c,
                   G_t, private_noise, cfg.beta, sweep_order, u)
        self.spins = spins_f.astype(np.int8)

        # 5. Magnetization and return (paper Eq. 5 with <S> = sum S_i):
        #    r(t) = <S(t)> / (N * eta) = magnetization / eta
        magnetization = float(np.mean(self.spins.astype(np.float64)))
        r_t = magnetization / cfg.eta

        self.r_prev = r_t
        self.G_prev = G_t

        return {
            "magnetization": magnetization,
            "raw_return": r_t,
            "G_news": G_t,
        }

    def run(self) -> dict:
        """Run the full simulation: burn-in phase followed by production phase.

        ``burn_in`` steps are discarded to let the system reach a
        stationary state before recording.  ``production`` steps are then
        recorded and returned.

        Returns
        -------
        dict with keys:
            ``magnetization`` : ndarray, shape (production,)
                Mean spin per timestep.
            ``returns`` : ndarray, shape (production,)
                Log-return proxy r(t) = ⟨S⟩ / (N·η).
            ``news`` : ndarray, shape (production,)
                Exogenous news draws G_t.
            ``price`` : ndarray, shape (production,)
                Reconstructed log-price ``p(t) = exp(cumsum(r))``.
        """
        for _ in range(self.cfg.burn_in):
            self.step()

        T = self.cfg.production
        mag = np.empty(T)
        returns = np.empty(T)
        news = np.empty(T)

        for t in range(T):
            out = self.step()
            mag[t] = out["magnetization"]
            returns[t] = out["raw_return"]
            news[t] = out["G_news"]

        # Log-price process: p(t) = p(0) * exp(cumsum r).  Returns here
        # are already the log-return proxy r(t) = x_t / eta, so this
        # cumulation is equivalent to the paper's "small price changes"
        # log-return convention.
        price = np.exp(np.cumsum(returns))

        return {
            "magnetization": mag,
            "returns": returns,
            "news": news,
            "price": price,
        }


def run_ensemble(cfg: SimConfig, n_seeds: int | None = None,
                 base_seed: int | None = None) -> list[dict]:
    """Run multiple independent realisations and return a list of result dicts.

    Seeds are assigned deterministically as ``base_seed + 1000 * k``
    (k = 0, 1, …, n_seeds-1) so that ensemble results are exactly
    reproducible.  When ``n_seeds == 1`` this is equivalent to a single
    ``IsingMarketModel(cfg).run()``.

    Parameters
    ----------
    cfg : SimConfig
        Simulation configuration shared by all realisations.
    n_seeds : int, optional
        Ensemble size.  Falls back to ``cfg.n_seeds`` when not given.
    base_seed : int, optional
        Seed offset for the first realisation.  Falls back to ``cfg.seed``
        when not given.

    Returns
    -------
    list of dict
        One element per realisation, each with the keys returned by
        :meth:`IsingMarketModel.run`.
    """
    n = int(n_seeds if n_seeds is not None else cfg.n_seeds)
    base = int(base_seed if base_seed is not None else cfg.seed)
    out: list[dict] = []
    for k in range(n):
        model = IsingMarketModel(cfg, seed=base + 1000 * k)
        out.append(model.run())
    return out
