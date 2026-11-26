"""Download a configured dataset with python -m recsys.download."""

from omegaconf import DictConfig, OmegaConf

from recsys.config import hydra_cli


@hydra_cli
def main(config: DictConfig) -> None:
    from recsys.data.sources import download

    dataset = OmegaConf.to_container(config.dataset, resolve=True)
    print(download(dataset, config.output.overwrite))


if __name__ == "__main__":
    main()
