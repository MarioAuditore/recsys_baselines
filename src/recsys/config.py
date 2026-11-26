"""Locate and resolve the Hydra configs for CLI and Python callers."""

import sys
from functools import wraps
from pathlib import Path

import hydra
from omegaconf import OmegaConf


def config_directory():
    """Find configs in the source checkout or in the installed package."""
    checkout = Path(__file__).resolve().parents[2] / "configs"
    if (checkout / "config.yaml").exists():
        return str(checkout)
    return str(Path(__file__).parent / "configs")


def _config_name(name):
    """Resolve bundled presets by path or by their short dataset names."""
    if not name:
        return name
    stem = name.removesuffix(".yaml")
    parts = stem.split("/" if "/" in stem else ".")
    if any(part in {"", ".", ".."} for part in parts):
        return name
    configs = Path(config_directory())
    nested = "/".join(parts)
    dataset_preset = nested.removeprefix("datasets/")
    if dataset_preset == "csv/dataset":
        dataset_preset = "dataset"
    for candidate in (stem, nested, f"datasets/{dataset_preset}"):
        if (configs / f"{candidate}.yaml").is_file():
            return candidate + name[len(stem) :]
    return name


def _config_args(arguments):
    boundary = arguments.index("--") if "--" in arguments else len(arguments)
    options = arguments[1:boundary]
    if any(arg.partition("=")[0] in ("-cp", "--config-path") for arg in options):
        return arguments

    resolved = arguments.copy()
    for index, option in enumerate(options, start=1):
        flag, separator, value = option.partition("=")
        if flag in ("-cn", "--config-name"):
            if separator:
                resolved[index] = f"{flag}={_config_name(value)}"
            elif index + 1 < boundary:
                resolved[index + 1] = _config_name(arguments[index + 1])
    return resolved


def hydra_cli(function):
    """Wrap a Hydra command with support for bundled dotted preset names."""
    command = hydra.main(config_path=config_directory(), config_name="config", version_base="1.3")(
        function
    )

    @wraps(command)
    def main(*args, **kwargs):
        arguments = sys.argv
        try:
            sys.argv = _config_args(arguments)
            return command(*args, **kwargs)
        finally:
            sys.argv = arguments

    return main


def compose(overrides=None, *, config_name="config"):
    """Return the resolved YAML configuration with the given Hydra overrides."""
    with hydra.initialize_config_dir(config_dir=config_directory(), version_base="1.3"):
        return OmegaConf.to_container(
            hydra.compose(config_name=_config_name(config_name), overrides=overrides or []),
            resolve=True,
        )
