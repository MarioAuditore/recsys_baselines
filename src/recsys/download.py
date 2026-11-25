"""Download a configured dataset with python -m recsys.download."""

import hydra
from omegaconf import DictConfig, OmegaConf

from recsys.config import config_directory


@hydra.main(config_path=config_directory(), config_name="config", version_base="1.3")
def main(config: DictConfig) -> None:
    from recsys.data.sources import download

    dataset = OmegaConf.to_container(config.dataset, resolve=True)
    print(download(dataset, config.output.overwrite))


if __name__ == "__main__":
    main()
