"""Independent Optuna studies using the same training runner as normal experiments."""

import csv
from copy import deepcopy
from pathlib import Path
from tempfile import mkdtemp

import optuna
from omegaconf import OmegaConf

from recsys.artifacts import atomic_path, fingerprint, write_json
from recsys.data import PreparedDataset
from recsys.training import run_training


def _write_trial_index(path, trials, records, parameters):
    columns = ["number", "state", "value", "failure", "config_path", *sorted(parameters)]
    parent = path.parent.resolve()
    with (
        atomic_path(path, overwrite=True) as temporary,
        temporary.open("w", encoding="utf-8", newline="") as stream,
    ):
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for trial, record in zip(trials, records, strict=True):
            directory = trial.user_attrs.get("config", {}).get("output", {}).get("dir")
            config_path = None
            if directory:
                artifact = (Path(directory) / "config.yaml").resolve()
                if artifact.is_file():
                    config_path = (
                        artifact.relative_to(parent)
                        if artifact.is_relative_to(parent)
                        else artifact
                    )
            writer.writerow({**record, **record["params"], "config_path": config_path})


def _search_specs(space, prefix=""):
    for name, spec in space.items():
        key = f"{prefix}.{name}" if prefix else name
        if not isinstance(spec, dict) or not spec:
            raise ValueError(f"Expected a search specification at tuning.space.{key}")
        if "type" in spec:
            yield key, spec
        else:
            yield from _search_specs(spec, key)


def tune(config):
    if (
        config["fit_split"] != "train"
        or config["target_split"] != "all"
        or config["training"]["validate"] is False
    ):
        raise ValueError("Tuning requires train-only fit and independent validation")
    options = config["tuning"]
    space = options["space"]
    if not space:
        raise ValueError("Empty tuning.space; select an experiment with tunable parameters")
    if not isinstance(space, dict):
        raise ValueError("tuning.space must be a mapping")
    search = {}
    missing = object()
    for key, spec in _search_specs(space):
        if key in search:
            raise ValueError(f"Duplicate tuning parameter: {key}")
        if not key.startswith(("model.", "training.")) or key in {
            "model.name",
            "model.loss.name",
            "training.validate",
        }:
            raise ValueError(f"Tuning cannot vary this field: {key}")
        value = config
        for part in key.split("."):
            value = value.get(part, missing) if isinstance(value, dict) else missing
        if value is missing or isinstance(value, (dict, list)):
            raise ValueError(f"Tuning parameter must be an existing scalar field: {key}")
        if key.startswith("training.") and config["model"]["loss"] is None:
            raise ValueError(f"Analytic models do not use training hyperparameters: {key}")
        search[key] = spec
    if options["sampler"] == "grid":
        if any(spec["type"] != "categorical" for spec in search.values()):
            raise ValueError("Grid spaces require categorical choices for every parameter")
        sampler = optuna.samplers.GridSampler(
            {key: spec["choices"] for key, spec in search.items()}, seed=config["seed"]
        )
    elif options["sampler"] == "tpe":
        sampler = optuna.samplers.TPESampler(seed=config["seed"])
    else:
        raise ValueError("sampler must be grid or tpe")
    dataset = PreparedDataset(config["dataset"]["path"])
    identity = fingerprint(
        {
            "dataset": dataset.metadata["fingerprint"],
            "model": config["model"],
            "history": [config["max_seq_length"], config["min_history"]],
            "protocol": config["evaluation"],
            "training": config["training"],
            "runtime": config["runtime"],
            "space": search,
            "seed": config["seed"],
            "sampler": options["sampler"],
            "pruning": [options["pruning"], options["warmup_epochs"]],
        }
    )
    name = (
        options["study_name"]
        or f"{config['model']['name']}_{config['dataset']['name']}_{identity[:16]}"
    )
    storage = options["storage"]
    if storage and storage.startswith("sqlite:///"):
        Path(storage.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    pruner = (
        optuna.pruners.MedianPruner(n_warmup_steps=options["warmup_epochs"])
        if options["pruning"]
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(
        study_name=name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    if study.user_attrs.get("identity", identity) != identity:
        raise ValueError("Existing study belongs to another experiment/protocol")
    study.set_user_attr("identity", identity)
    if isinstance(sampler, optuna.samplers.TPESampler):
        previous_trials = len(study.get_trials(deepcopy=False))
        if previous_trials:
            # Restarting with the original seed repeats TPE's startup samples.
            seed = (config["seed"] + previous_trials) % (2**32)
            study.sampler = optuna.samplers.TPESampler(seed=seed)
    output = Path(config["output"]["dir"]) / "tuning" / name
    if storage is None:
        # Each in-memory study starts again at trial 0, so reserve its own directory.
        try:
            output.mkdir(parents=True)
        except FileExistsError:
            output = Path(mkdtemp(prefix=f"{output.name}_", dir=output.parent))
    else:
        output.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        cfg = OmegaConf.create(config)
        for key, spec in search.items():
            kind = spec["type"]
            if kind == "categorical":
                value = trial.suggest_categorical(key, spec["choices"])
            elif kind in ("int", "float"):
                method = trial.suggest_int if kind == "int" else trial.suggest_float
                kwargs = {k: v for k, v in spec.items() if k != "type"}
                value = method(key, **kwargs)
            else:
                raise ValueError(f"Unknown search type {kind}")
            OmegaConf.update(cfg, key, value, merge=False)
        cfg.output.dir = str(output / f"trial_{trial.number}")
        cfg.output.overwrite = False
        cfg.run_name = f"{config['run_name']}_trial{trial.number}"
        trial_config = OmegaConf.to_container(cfg, resolve=True)
        trial.set_user_attr("config", trial_config)

        def on_epoch(epoch, metric):
            trial.report(metric, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            summary = run_training(trial_config, epoch_callback=on_epoch)
        except (RuntimeError, FloatingPointError) as error:
            trial.set_user_attr("failure", str(error))
            raise
        return summary["best_value"]

    if not isinstance(sampler, optuna.samplers.GridSampler) or not sampler.is_exhausted(study):
        study.optimize(
            objective,
            n_trials=options["trials"],
            timeout=options["timeout"],
            catch=(RuntimeError, FloatingPointError),
        )
    trials = study.get_trials(deepcopy=False)
    records = [
        {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "params": trial.params,
            "failure": trial.user_attrs.get("failure"),
        }
        for trial in trials
    ]
    _write_trial_index(output / "trials.csv", trials, records, search)
    if not any(trial.state == optuna.trial.TrialState.COMPLETE for trial in trials):
        raise ValueError("Study has no completed trials")
    best = deepcopy(study.best_trial.user_attrs["config"])
    best["run_name"] = config["run_name"] + "_tuned"
    best["output"]["dir"] = "outputs/" + best["run_name"]
    best["output"]["path"] = f"results/{best['dataset']['name']}/{best['run_name']}_test.json"
    best["checkpoint"] = None
    OmegaConf.save(OmegaConf.create(best), output / "best.yaml")
    summary = {
        "study": name,
        "identity": identity,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_config": str((output / "best.yaml").resolve()),
        "trials": records,
    }
    write_json(output / "study.json", summary, overwrite=True)
    return summary
