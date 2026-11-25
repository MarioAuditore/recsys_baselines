"""Train a model from scratch with python -m recsys.train."""

import hydra
from omegaconf import DictConfig, OmegaConf

from recsys.config import config_directory


@hydra.main(config_path=config_directory(), config_name="config", version_base="1.3")
def main(config: DictConfig) -> None:
    from recsys.training import run_training

    print(run_training(OmegaConf.to_container(config, resolve=True)))


if __name__ == "__main__":
    main()
