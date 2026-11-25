"""Explicit model names shared by training, tuning and evaluation."""

from recsys.models.baselines import Baseline, ItemKNN, MostPopular, PureSVD, Random
from recsys.models.graph import LightGCN, UltraGCN
from recsys.models.sasrec import SASRec

_MODEL_CLASSES = {
    "sasrec": SASRec,
    "lightgcn": LightGCN,
    "ultragcn": UltraGCN,
    "random": Random,
    "mostpopular": MostPopular,
    "itemknn": ItemKNN,
    "puresvd": PureSVD,
}


def build_model(history, config, seed=42, device="cpu"):
    """Construct the recommender named by the model config."""
    name = config["name"]
    model_class = _MODEL_CLASSES.get(name)
    if model_class is None:
        raise ValueError(f"Unknown model {name}")
    if issubclass(model_class, Baseline):
        return model_class(history, config, seed)
    return model_class(history, config).to(device)


__all__ = [
    "SASRec",
    "LightGCN",
    "UltraGCN",
    "Random",
    "MostPopular",
    "PureSVD",
    "ItemKNN",
    "build_model",
]
