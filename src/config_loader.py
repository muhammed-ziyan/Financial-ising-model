"""YAML configuration loader."""

import yaml
from dataclasses import dataclass


@dataclass
class SimConfig:
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


def load_config(path: str) -> SimConfig:
    """Load YAML config file and return SimConfig."""
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)

    return SimConfig(
        case_name=raw['case_name'],
        m=raw['lattice']['m'],
        lambda_=raw['model']['lambda_'],
        b_max=raw['model']['b_max'],
        c_max=raw['model']['c_max'],
        delta=raw['model']['delta'],
        beta=raw['model']['beta'],
        eta=raw['model']['eta'],
        noise_scale=raw['model']['noise_scale'],
        burn_in=raw['simulation']['burn_in'],
        production=raw['simulation']['production'],
        seed=raw['simulation']['seed'],
        sweeps_per_step=raw['simulation'].get('sweeps_per_step', 1),
    )
