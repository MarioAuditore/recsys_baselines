"""Prepare a configured dataset with python -m recsys.prepare."""

import hydra
from omegaconf import DictConfig, OmegaConf

from recsys.config import config_directory


@hydra.main(config_path=config_directory(), config_name="config", version_base="1.3")
def main(config: DictConfig) -> None:
    from recsys.data import prepare

    dataset = OmegaConf.to_container(config.dataset, resolve=True)
    preparation = OmegaConf.to_container(config.preparation, resolve=True)
    print(prepare(dataset, preparation, config.output.overwrite))


if __name__ == "__main__":
    main()
