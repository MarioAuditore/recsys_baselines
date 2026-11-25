"""Locate the Hydra configs and compose them for Python callers."""

from pathlib import Path

import hydra
from omegaconf import OmegaConf


def config_directory():
    """Find configs in the source checkout or in the installed package."""
    checkout = Path(__file__).resolve().parents[2] / "configs"
    if (checkout / "config.yaml").exists():
        return str(checkout)
    return str(Path(__file__).parent / "configs")


def compose(overrides=None, *, config_name="config"):
    """Return the resolved YAML configuration with the given Hydra overrides."""
    with hydra.initialize_config_dir(config_dir=config_directory(), version_base="1.3"):
        return OmegaConf.to_container(
            hydra.compose(config_name=config_name, overrides=overrides or []), resolve=True
        )
