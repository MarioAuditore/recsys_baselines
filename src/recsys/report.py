"""Aggregate saved evaluation results with python -m recsys.report."""

from omegaconf import DictConfig, OmegaConf

from recsys.config import hydra_cli


@hydra_cli
def main(config: DictConfig) -> None:
    from recsys.reporting import report

    options = OmegaConf.to_container(config.reporting, resolve=True)
    print(report(**options, overwrite=config.output.overwrite))


if __name__ == "__main__":
    main()
