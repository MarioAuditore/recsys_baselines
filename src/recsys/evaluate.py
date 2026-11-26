"""Evaluate a checkpoint with python -m recsys.evaluate."""

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from recsys.checkpoint_config import load_checkpoint_config
from recsys.config import hydra_cli


@hydra_cli
def main(config: DictConfig) -> None:
    from recsys.training import evaluate_checkpoint

    config, explicit = load_checkpoint_config(
        config, HydraConfig.get().overrides.task, apply_runtime_training=False
    )
    if "output.path" not in explicit:
        config.output.path = f"results/{config.dataset.name}/{config.run_name}_{config.split}.json"
    result = evaluate_checkpoint(OmegaConf.to_container(config, resolve=True))
    print({"metrics": result["metrics"], "support": result["support"], "path": config.output.path})


if __name__ == "__main__":
    main()
