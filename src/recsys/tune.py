"""Run an Optuna study with python -m recsys.tune."""

import hydra
from omegaconf import DictConfig, OmegaConf

from recsys.config import config_directory


@hydra.main(config_path=config_directory(), config_name="config", version_base="1.3")
def main(config: DictConfig) -> None:
    from recsys.tuning import tune

    print(tune(OmegaConf.to_container(config, resolve=True)))


if __name__ == "__main__":
    main()
