"""Aggregate saved evaluation results with python -m recsys.report."""

import hydra
from omegaconf import DictConfig, OmegaConf

from recsys.config import config_directory


@hydra.main(config_path=config_directory(), config_name="config", version_base="1.3")
def main(config: DictConfig) -> None:
    from recsys.reporting import report

    options = OmegaConf.to_container(config.reporting, resolve=True)
    print(report(**options, overwrite=config.output.overwrite))


if __name__ == "__main__":
    main()
