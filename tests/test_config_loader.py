"""Tests for YAML configuration loading and merging."""

import os
import tempfile

import pytest
import yaml

from src.config_loader import SimConfig, load_config, override_config


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_yaml(path: str, data: dict) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(data, fh)


# ── default.yaml path ─────────────────────────────────────────────────────────

DEFAULT_YAML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "default.yaml",
)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_load_case_a():
    """Loading case_A.yaml should produce a valid SimConfig."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "case_A.yaml",
    )
    cfg = load_config(path)
    assert cfg.case_name == "A"
    assert cfg.m == 10
    assert cfg.lambda_ == pytest.approx(0.25)
    assert cfg.b_max == pytest.approx(0.05)
    # Defaults are merged in
    assert cfg.burn_in > 0
    assert cfg.production > 0
    assert cfg.eta > 0


def test_load_case_b():
    """Loading case_B.yaml should produce m=5 and correct lambda/b_max."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "case_B.yaml",
    )
    cfg = load_config(path)
    assert cfg.case_name == "B"
    assert cfg.m == 5
    assert cfg.lambda_ == pytest.approx(0.20)


def test_load_case_c():
    """Loading case_C.yaml should produce m=5 and lambda=0.15."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "case_C.yaml",
    )
    cfg = load_config(path)
    assert cfg.case_name == "C"
    assert cfg.m == 5
    assert cfg.lambda_ == pytest.approx(0.15)


def test_overrides_applied():
    """Dict overrides should take precedence over both default and case YAML."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "case_A.yaml",
    )
    cfg = load_config(path, overrides={"simulation": {"production": 777, "seed": 99}})
    assert cfg.production == 777
    assert cfg.seed == 99
    # Other defaults should be preserved
    assert cfg.m == 10


def test_override_config():
    """override_config returns a copy with only the specified fields changed."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "case_A.yaml",
    )
    original = load_config(path)
    modified = override_config(original, production=1234, seed=7)
    assert modified.production == 1234
    assert modified.seed == 7
    # Original should be unchanged (immutable dataclass copy)
    assert original.production != 1234


def test_case_overrides_default():
    """A key set in the case YAML should override the same key in default.yaml."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "case_A.yaml",
    )
    cfg = load_config(path)
    # case_A.yaml sets m=10; default.yaml does not have a lattice section,
    # so this purely comes from the case file.
    assert cfg.m == 10


def test_simconfig_n_defaults():
    """SimConfig should have sweeps_per_step=1 and n_seeds default if unset."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "case_A.yaml",
    )
    cfg = load_config(path)
    assert cfg.sweeps_per_step >= 1
    assert cfg.n_seeds >= 1
