"""Start a new run from trained weights with python -m recsys.finetune."""

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from recsys.checkpoint_config import load_checkpoint_config
from recsys.config import config_directory


@hydra.main(config_path=config_directory(), config_name="config", version_base="1.3")
def main(config: DictConfig) -> None:
    from recsys.training import run_training

    config, explicit = load_checkpoint_config(config, HydraConfig.get().overrides.task)
    # Fine-tuning targets validation events in the combined train and validation history.
    if "fit_split" not in explicit:
        config.fit_split = "train_val"
    if "target_split" not in explicit:
        config.target_split = "val"
    if "training.validate" not in explicit:
        config.training.validate = False
    if "run_name" not in explicit:
        config.run_name += "_finetune"
    if "output.dir" not in explicit:
        config.output.dir = f"outputs/{config.run_name}"
    print(run_training(OmegaConf.to_container(config, resolve=True), mode="finetune"))


if __name__ == "__main__":
    main()
