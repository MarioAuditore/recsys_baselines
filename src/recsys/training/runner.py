"""Train or resume one model and record its validation and checkpoints."""

import json
import time
from copy import deepcopy
from pathlib import Path

from omegaconf import OmegaConf
from tqdm import tqdm

from recsys.artifacts import fingerprint, write_json
from recsys.data import PreparedDataset
from recsys.evaluation import evaluate_model
from recsys.models import build_model
from recsys.models.baselines import Baseline
from recsys.training.checkpoints import (
    load_checkpoint,
    restore_rng,
    rng_state,
    save_checkpoint,
    seed_everything,
)
from recsys.training.optimization import make_optimizer, optimize_epoch
from recsys.training.runtime import configure_device


def _validate_config(config, mode):
    if mode not in ("train", "resume", "finetune"):
        raise ValueError("Invalid training mode")
    if mode == "resume" and Path(config["checkpoint"]).name != "last.pt":
        raise ValueError("Resume requires last.pt; use finetune to branch from best.pt")
    target = config["target_split"]
    if mode == "train" and target != "all":
        raise ValueError("train uses all fit events; use finetune for selected target events")
    if target not in ("all", "val"):
        raise ValueError("target_split must be all or val")
    if target == "val" and config["fit_split"] != "train_val":
        raise ValueError("val targets need fit_split=train_val to supply past context")
    if config["training"]["epochs"] < 1 or config["training"]["batch_size"] < 1:
        raise ValueError("epochs and batch_size must be positive")

    validate = config["training"]["validate"]
    if validate is None:
        validate = config["fit_split"] == "train"
    if validate and config["fit_split"] != "train":
        raise ValueError("Validation is in fit data; use training.validate=false and fixed epochs")
    return validate


def training_signature(config):
    return fingerprint(
        {
            k: config[k]
            for k in ["model", "seed", "fit_split", "target_split", "max_seq_length", "min_history"]
        }
        | {
            "training": {k: v for k, v in config["training"].items() if k != "epochs"},
            "evaluation": config["evaluation"],
            "runtime": config["runtime"],
        }
    )


def _restore_model(model, checkpoint, config, dataset, position_length):
    if checkpoint["mapping_fingerprint"] != dataset.metadata["mapping_fingerprint"]:
        raise ValueError("Checkpoint mapping differs from dataset mapping")
    if checkpoint["config"]["model"] != config["model"]:
        raise ValueError("Checkpoint model/loss configuration differs; use matching model config")
    if isinstance(model, Baseline):
        raise ValueError("Analytic baselines support train/evaluate, not resume/finetune")
    if position_length is not None and checkpoint["max_position_length"] != position_length:
        raise ValueError("Positional embedding length must match checkpoint")
    model.load_state_dict(checkpoint["model_state"])


def _discard_uncommitted_epochs(path, committed_epoch):
    if not path.exists():
        return
    with path.open("r+b") as stream:
        end = 0
        while line := stream.readline():
            # The row is appended before last.pt is replaced. A failed save leaves an extra row.
            if not line.endswith(b"\n") or json.loads(line)["epoch"] > committed_epoch:
                stream.truncate(end)
                break
            end = stream.tell()


def _validate_epoch(model, dataset, history, evaluation):
    result = evaluate_model(model, dataset, history, "val", evaluation)
    objective = result["metrics"].get(evaluation["selection_metric"])
    if objective is None:
        raise ValueError(
            "Validation objective is undefined; inspect split support and selection_metric"
        )
    return result, objective


def run_training(config, mode="train", epoch_callback=None):
    """Run one fit with optional validation. The callback receives each validation objective."""
    config = deepcopy(config)
    validate = _validate_config(config, mode)
    options = config["training"]
    evaluation = config["evaluation"]
    seed_everything(config["seed"])
    device = configure_device(config["runtime"])
    if options["amp"] and not device.startswith("cuda"):
        raise ValueError("AMP currently supported only on CUDA")

    dataset = PreparedDataset(config["dataset"]["path"])
    history = dataset.history(config["fit_split"], config["max_seq_length"], config["min_history"])
    model = build_model(history, config["model"], config["seed"], device)
    analytic = isinstance(model, Baseline)
    position_length = getattr(model, "max_position_length", None)
    previous = None
    if mode != "train":
        previous = load_checkpoint(config["checkpoint"])
        _restore_model(model, previous, config, dataset, position_length)

    output = Path(config["output"]["dir"])
    last_path, best_path = output / "last.pt", output / "best.pt"
    signature = training_signature(config)
    if mode == "resume":
        if (
            previous["signature"] != signature
            or previous["fit_fingerprint"] != history.fingerprint
            or previous["dataset_fingerprint"] != dataset.metadata["fingerprint"]
        ):
            raise ValueError(
                "Resume requires identical data/training/protocol/runtime; only epochs/output paths may change"
            )
        if output.resolve() != Path(config["checkpoint"]).resolve().parent:
            raise ValueError(
                "Resume writes to the original run directory; use finetune for a new run"
            )
    elif (last_path.exists() or best_path.exists()) and not config["output"]["overwrite"]:
        raise FileExistsError(
            f"{output} contains checkpoints. Choose a new run_name or output.dir. "
            "To replace this run, set output.overwrite=true."
        )

    optimizer = scheduler = scaler = None
    if not analytic:
        optimizer, scheduler, scaler = make_optimizer(model, options)
    start_epoch, best_value, best_epoch, bad_epochs, global_step = 0, None, 0, 0, 0
    if mode == "resume":
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        scaler.load_state_dict(previous["scaler"])
        start_epoch, best_value, best_epoch, bad_epochs, global_step = [
            previous[key]
            for key in ["epoch", "best_value", "best_epoch", "bad_epochs", "global_step"]
        ]
        restore_rng(previous["rng"])
    epochs = 1 if analytic else options["epochs"]
    if start_epoch >= epochs:
        raise ValueError("Checkpoint already reached requested epochs; increase training.epochs")

    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(config), output / "config.yaml")
    history_path = output / "validation.jsonl"
    if mode == "resume":
        _discard_uncommitted_epochs(history_path, start_epoch)
        # An interruption after saving last.pt can leave best.pt missing or stale.
        if start_epoch == best_epoch:
            save_checkpoint(best_path, previous)
    else:
        history_path.write_text("")

    target_start = dataset.metadata["boundaries"][0] if config["target_split"] == "val" else None
    checkpoint_metadata = {
        "schema_version": 1,
        "config": config,
        "stage": previous["stage"] if mode == "resume" else mode,
        "mapping_fingerprint": dataset.metadata["mapping_fingerprint"],
        "dataset_fingerprint": dataset.metadata["fingerprint"],
        "fit_fingerprint": history.fingerprint,
        "signature": signature,
        "max_position_length": position_length,
        "parent_checkpoint": config["checkpoint"]
        if mode == "finetune"
        else (previous.get("parent_checkpoint") if previous else None),
    }
    started = time.perf_counter()
    with tqdm(
        total=epochs,
        initial=start_epoch,
        desc="Epochs",
        bar_format="{desc} {percentage:3.0f}% |{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}",
    ) as progress:
        for epoch in range(start_epoch + 1, epochs + 1):
            if analytic:
                model.fit(history)
                loss = None
            else:
                model.train()
                batches = model.training_batches(history, options["batch_size"], target_start)
                loss, steps = optimize_epoch(
                    model, batches, optimizer, scheduler, scaler, options, device
                )
                global_step += steps

            validation = objective = None
            if validate:
                validation, objective = _validate_epoch(model, dataset, history, evaluation)
            improved = not validate or best_value is None or objective > best_value
            if improved:
                best_value, best_epoch, bad_epochs = objective, epoch, 0
            else:
                bad_epochs += 1
            log = {
                "seed": config["seed"],
                "epoch": epoch,
                "global_step": global_step,
                "loss": loss,
                "validation": validation,
            }
            with history_path.open("a") as stream:
                stream.write(json.dumps(log, allow_nan=False) + "\n")
            payload = {
                **checkpoint_metadata,
                "model_state": model.state_dict(),
                "epoch": epoch,
                "best_value": best_value,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
                "global_step": global_step,
                "optimizer": optimizer.state_dict() if optimizer else None,
                "scheduler": scheduler.state_dict() if scheduler else None,
                "scaler": scaler.state_dict() if scaler else None,
                "rng": rng_state(),
            }
            save_checkpoint(last_path, payload)
            if improved:
                save_checkpoint(best_path, payload)
            metrics = {"loss": f"{loss:.5f}"} if loss is not None else {}
            if validation is not None:
                cutoff = evaluation["selection_metric"].rsplit("@", 1)[-1]
                ndcg_metric = f"ndcg@{cutoff}"
                ndcg = validation["metrics"].get(ndcg_metric)
                if ndcg is not None:
                    metrics[f"val_{ndcg_metric}"] = f"{ndcg:.5f}"
            progress.set_postfix(metrics, refresh=False)
            progress.update()
            if epoch_callback is not None and validate:
                epoch_callback(epoch, objective)
            patience = options["patience"]
            if validate and patience is not None and bad_epochs >= patience:
                break

    summary = {
        "best_checkpoint": str(best_path.resolve()),
        "last_checkpoint": str(last_path.resolve()),
        "best_value": best_value,
        "best_epoch": best_epoch,
        "seed": config["seed"],
        "seconds": time.perf_counter() - started,
    }
    write_json(output / "summary.json", summary, overwrite=True)
    return summary
