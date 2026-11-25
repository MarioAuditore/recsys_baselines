"""Load a trained model, score one split, and write its evaluation record."""

import importlib.metadata
import platform
import subprocess
from pathlib import Path

from recsys.artifacts import read_json, write_json
from recsys.data import PreparedDataset
from recsys.evaluation import evaluate_model
from recsys.models import build_model
from recsys.models.baselines import Baseline
from recsys.training.checkpoints import load_checkpoint, seed_everything
from recsys.training.runtime import configure_device


def _environment(device):
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    versions = {
        name: importlib.metadata.version(name)
        for name in ["torch", "numpy", "scipy", "hydra-core", "optuna"]
    }
    return {
        "device": device,
        "python": platform.python_version(),
        "git_commit": commit,
        "versions": versions,
    }


def _result_record(config, dataset, history, checkpoint, model, result):
    training_runtime = checkpoint["config"]["runtime"]
    device = config["runtime"]["device"]
    if isinstance(model, Baseline):
        training_runtime = training_runtime | {"device": "cpu"}
        device = "cpu"

    return {
        "schema_version": 1,
        "run_id": config["run_name"],
        "seed": config["seed"],
        "model": config["model"]["name"],
        "loss": config["model"]["loss"]["name"] if config["model"]["loss"] else None,
        "stage": checkpoint["stage"],
        "fit_split": config["fit_split"],
        "target_split": config["target_split"],
        "checkpoint": str(Path(config["checkpoint"]).resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "training_runtime": training_runtime,
        "diagnostic_only": config["split"] == "val" and config["fit_split"] == "train_val",
        "dataset": {
            key: dataset.metadata[key] for key in ["name", "fingerprint", "mapping_fingerprint"]
        },
        "fit_fingerprint": history.fingerprint,
        "config": config,
        "environment": _environment(device),
        **result,
    }


def evaluate_checkpoint(config):
    """Reconstruct a saved model and evaluate the split requested in config."""
    device = configure_device(config["runtime"])
    checkpoint = load_checkpoint(config["checkpoint"])
    dataset = PreparedDataset(config["dataset"]["path"])
    history = dataset.history(config["fit_split"], config["max_seq_length"], config["min_history"])
    if (
        checkpoint["mapping_fingerprint"] != dataset.metadata["mapping_fingerprint"]
        or checkpoint["dataset_fingerprint"] != dataset.metadata["fingerprint"]
        or checkpoint["fit_fingerprint"] != history.fingerprint
    ):
        raise ValueError("Checkpoint data/history mismatch; evaluate on its original prepared data")
    for key in ["model", "seed", "fit_split", "target_split", "training"]:
        if config[key] != checkpoint["config"][key]:
            raise ValueError(f"Cannot change checkpoint {key} during evaluation")
    seed_everything(config["seed"])
    model = build_model(history, config["model"], config["seed"], device)
    model.load_state_dict(checkpoint["model_state"])
    result = evaluate_model(model, dataset, history, config["split"], config["evaluation"])
    record = _result_record(config, dataset, history, checkpoint, model, result)
    path = config["output"]["path"]
    try:
        write_json(path, record, overwrite=config["output"]["overwrite"])
    except FileExistsError:
        try:
            saved = read_json(path)
        except (OSError, ValueError):
            saved = None
        # The elapsed time changes even when the evaluation produces the same result.
        if isinstance(saved, dict) and (saved | {"timing": record["timing"]}) == record:
            return saved
        raise
    return record
