"""YAML configuration loader.

A case configuration layers on top of ``config/default.yaml``.  The default
file holds the four undisclosed-in-the-paper parameters (``eta``,
``c_max``, ``beta``, ``noise_scale``) plus simulation settings.  Each
``case_*.yaml`` file only needs the paper-specified triple
``(m, lambda_, b_max)`` and the case label.  Any key set in the case
file overrides the corresponding default.
"""

import os
from copy import deepcopy
from dataclasses import dataclass, replace

import yaml


@dataclass
class SimConfig:
    """Flat parameter container for one simulation run.

    Fields published in paper Table I (vary per case)
    --------------------------------------------------
    case_name : str
        Human-readable label, e.g. ``"A"``.
    m : int
        Linear lattice size; the lattice has N = m³ sites.
    lambda_ : float
        Coupling persistence (0 < λ < 1).
    b_max : float
        Upper bound of the frozen coupling distribution U[0, b_max].

    Shared model parameters (calibrated, not disclosed in the paper)
    ----------------------------------------------------------------
    c_max : float
        Upper bound of the news-sensitivity distribution U[0, c_max].
    delta : float
        Feedback amplitude in the coupling recursion (paper fixes δ = 1).
    beta : float
        Inverse temperature of the Glauber heat-bath update.
    eta : float
        Market-depth parameter; return r = ⟨S⟩ / (N·η).
    noise_scale : float
        Standard deviation of the per-site private noise ε_i ~ N(0, σ).

    Simulation control
    ------------------
    burn_in : int
        Number of timesteps discarded before recording starts.
    production : int
        Number of timesteps recorded and returned.
    seed : int
        Base random seed (seed for a single run; ensemble uses
        seed + 1000·k).
    sweeps_per_step : int, optional
        Number of full Glauber sweeps per timestep (default 1).
    n_seeds : int, optional
        Default ensemble size when no explicit value is given to
        :func:`run_ensemble` (default 1).
    """

    case_name: str
    m: int
    lambda_: float
    b_max: float
    c_max: float
    delta: float
    beta: float
    eta: float
    noise_scale: float
    burn_in: int
    production: int
    seed: int
    sweeps_per_step: int = 1
    n_seeds: int = 1


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (non-destructive)."""
    out = deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _default_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "config", "default.yaml"))


def load_config(path: str, overrides: dict | None = None) -> SimConfig:
    """Load a case YAML file, merging with ``config/default.yaml``.

    Parameters
    ----------
    path : str
        Path to a case YAML file (e.g. ``config/case_A.yaml``).
    overrides : dict, optional
        Optional nested dict of last-mile overrides, same schema as the
        YAML files.  Useful for calibration sweeps.
    """
    default_path = _default_path()
    with open(default_path, "r") as f:
        merged = yaml.safe_load(f) or {}

    with open(path, "r") as f:
        case_raw = yaml.safe_load(f) or {}
    merged = _deep_merge(merged, case_raw)

    if overrides:
        merged = _deep_merge(merged, overrides)

    lat = merged.get("lattice", {})
    mdl = merged.get("model", {})
    sim = merged.get("simulation", {})

    return SimConfig(
        case_name=merged["case_name"],
        m=lat["m"],
        lambda_=mdl["lambda_"],
        b_max=mdl["b_max"],
        c_max=mdl["c_max"],
        delta=mdl["delta"],
        beta=mdl["beta"],
        eta=mdl["eta"],
        noise_scale=mdl["noise_scale"],
        burn_in=sim["burn_in"],
        production=sim["production"],
        seed=sim["seed"],
        sweeps_per_step=sim.get("sweeps_per_step", 1),
        n_seeds=sim.get("n_seeds", 1),
    )


def override_config(cfg: SimConfig, **kwargs) -> SimConfig:
    """Return a copy of ``cfg`` with selected fields replaced."""
    return replace(cfg, **kwargs)
