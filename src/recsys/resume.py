"""Resume an interrupted run with python -m recsys.resume."""

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from recsys.checkpoint_config import load_checkpoint_config
from recsys.config import config_directory


@hydra.main(config_path=config_directory(), config_name="config", version_base="1.3")
def main(config: DictConfig) -> None:
    from recsys.training import run_training

    config, _ = load_checkpoint_config(config, HydraConfig.get().overrides.task)
    print(run_training(OmegaConf.to_container(config, resolve=True), mode="resume"))


if __name__ == "__main__":
    main()
