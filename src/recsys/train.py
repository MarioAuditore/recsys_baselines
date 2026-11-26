"""Train a model from scratch with python -m recsys.train."""

from omegaconf import DictConfig, OmegaConf

from recsys.config import hydra_cli


@hydra_cli
def main(config: DictConfig) -> None:
    from recsys.training import run_training

    print(run_training(OmegaConf.to_container(config, resolve=True)))


if __name__ == "__main__":
    main()
