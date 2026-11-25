"""Apply explicit Hydra overrides to the config saved in a checkpoint."""

from omegaconf import OmegaConf


def load_checkpoint_config(config, overrides, *, apply_runtime_training=True):
    """Return the merged config and explicit CLI keys, with optional runtime training limits."""
    from recsys.training.checkpoints import load_checkpoint

    if not config.checkpoint:
        raise ValueError("checkpoint=<path> is required")
    original = load_checkpoint(config.checkpoint)["config"]
    merged = OmegaConf.create(original)
    # Resolve interpolations against the command's config before copying selected values.
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    explicit = set()
    for override in overrides:
        key = override.lstrip("+").split("=", 1)[0]
        explicit.add(key)
        value = OmegaConf.select(config, key)
        OmegaConf.update(merged, key, value, merge=False, force_add=True)
        if apply_runtime_training and key == "runtime" and override.endswith("=cpu_smoke"):
            # The smoke preset puts its training limits outside the runtime section.
            for field in ("epochs", "batch_size", "max_batches"):
                merged.training[field] = config.training[field]
    merged.checkpoint = config.checkpoint
    return merged, explicit
