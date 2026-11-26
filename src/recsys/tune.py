"""Run an Optuna study with python -m recsys.tune."""

from omegaconf import DictConfig, OmegaConf

from recsys.config import hydra_cli


@hydra_cli
def main(config: DictConfig) -> None:
    from recsys.tuning import tune

    print(tune(OmegaConf.to_container(config, resolve=True)))


if __name__ == "__main__":
    main()
