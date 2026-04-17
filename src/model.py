"""Core 3D Ising financial market simulation engine."""

import numpy as np
from .lattice import build_neighbor_table
from .config_loader import SimConfig


class IsingMarketModel:
    """Self-organizing 3D Ising model of financial markets.

    Based on Guimaraes & Lima, Phys. Rev. E 103, 062130 (2021).
    """

    def __init__(self, config: SimConfig):
        self.cfg = config
        self.m = config.m
        self.N = config.m ** 3
        self.neighbors = build_neighbor_table(config.m)
        self.rng = np.random.default_rng(config.seed)

        # Spin state: +1 (buy) or -1 (sell)
        self.spins = self.rng.choice(
            np.array([1, -1], dtype=np.int8), size=self.N
        )

        # Frozen heterogeneous agent parameters
        self.b = self.rng.uniform(0, config.b_max, size=self.N)
        self.c = self.rng.uniform(0, config.c_max, size=self.N)

        # Dynamic coupling per agent (initialized to b_i)
        self.J_coupling = self.b.copy()

        # Previous-step market state
        self.r_prev = 0.0
        self.G_prev = 0.0

    def step(self) -> dict:
        """Execute one simulation timestep.

        Returns dict with magnetization, raw_return, G_news.
        """
        cfg = self.cfg

        # 1. External news ~ N(0,1)
        G_t = self.rng.standard_normal()

        # 2. Update couplings: J_i(t) = b_i + lambda * J_i(t-1) + delta * r(t-1) * G(t-1)
        self.J_coupling = (
            self.b
            + cfg.lambda_ * self.J_coupling
            + cfg.delta * self.r_prev * self.G_prev
        )

        # 3. Local field: h_i = J_i * sum_j S_j(neighbors) + c_i * G(t) + eps_i
        neighbor_spins = self.spins[self.neighbors]  # (N, 6)
        sum_neighbor_spins = np.sum(neighbor_spins, axis=1).astype(np.float64)  # (N,)
        interaction = self.J_coupling * sum_neighbor_spins
        news_field = self.c * G_t
        private_noise = self.rng.normal(0, cfg.noise_scale, size=self.N)
        h = interaction + news_field + private_noise

        # 4. Heat-bath synchronous update
        prob_up = 1.0 / (1.0 + np.exp(-2.0 * cfg.beta * h))
        u = self.rng.random(self.N)
        self.spins = np.where(u < prob_up, np.int8(1), np.int8(-1))

        # 5. Magnetization and return
        magnetization = np.mean(self.spins.astype(np.float64))
        r_t = magnetization / cfg.eta

        # 6. Store for next step
        self.r_prev = r_t
        self.G_prev = G_t

        return {
            'magnetization': magnetization,
            'raw_return': r_t,
            'G_news': G_t,
        }

    def run(self) -> dict:
        """Full simulation: burn-in + production.

        Returns dict of arrays: magnetization, returns, news, price.
        """
        # Burn-in (discard)
        for _ in range(self.cfg.burn_in):
            self.step()

        # Production
        T = self.cfg.production
        mag = np.empty(T)
        returns = np.empty(T)
        news = np.empty(T)

        for t in range(T):
            result = self.step()
            mag[t] = result['magnetization']
            returns[t] = result['raw_return']
            news[t] = result['G_news']

        # Price from cumulative returns: p(t) = exp(sum r(0..t))
        price = np.exp(np.cumsum(returns))

        return {
            'magnetization': mag,
            'returns': returns,
            'news': news,
            'price': price,
        }
