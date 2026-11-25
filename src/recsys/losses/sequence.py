"""Select a next-item loss over valid positions and active item embeddings."""

from functools import partial

from torch.nn.functional import cross_entropy

from recsys.losses.sce import scalable_cross_entropy


def full_cross_entropy(hidden, targets, item_embeddings):
    """Score the entire active item catalog for every supervised position."""
    return cross_entropy(hidden @ item_embeddings.T, targets)


def build_sequence_loss(config):
    """Return a callable with inputs (hidden, targets, item_embeddings)."""
    name = config["name"]
    if name == "ce":
        return full_cross_entropy
    if name == "sce":
        options = {key: value for key, value in config.items() if key != "name"}
        return partial(scalable_cross_entropy, **options)
    raise ValueError(f"Unknown sequence loss {name}")
