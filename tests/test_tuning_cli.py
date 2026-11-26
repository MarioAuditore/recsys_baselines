import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from re import escape

import pytest
from omegaconf import OmegaConf

from recsys.config import compose
from recsys.tuning import tune


def _cli(action, *arguments, env=None):
    proc = subprocess.run(
        [sys.executable, "-m", f"recsys.{action}", *arguments],
        capture_output=True,
        text=True,
        env=env,
        timeout=90,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.parametrize(
    "dataset",
    [
        "ml1m",
        "amazon_beauty",
        "amazon_clothing",
        "amazon_sports",
        "yambda50m_listens",
        "yambda50m_likes",
    ],
)
def test_every_experiment_composes(dataset):
    assert compose(config_name=f"{dataset}.dataset")["dataset"]["name"] == dataset
    models = (
        "random",
        "mostpopular",
        "itemknn",
        "puresvd",
        "lightgcn",
        "ultragcn",
        "sasrec",
        "sasrec_sce",
    )
    for model in models:
        cfg = compose(config_name=f"{dataset}.{model}")
        assert cfg["dataset"]["name"] == dataset
        assert cfg["model"]["name"] == model.removesuffix("_sce")
        assert cfg["training"]["batch_size"] > 0
        assert not cfg.get("tuning", {}).get("space")
        if model not in {"random", "mostpopular"}:
            search = compose(config_name=f"{dataset}.tuning.{model}")
            assert search["dataset"] == cfg["dataset"] and search["model"] == cfg["model"]
            assert search["tuning"]["sampler"] == "tpe" and search["tuning"]["space"]


def test_sasrec_training_values_are_independent_of_the_search_space():
    preset = OmegaConf.load(Path(__file__).parents[1] / "configs/datasets/ml1m/sasrec.yaml")
    training = compose(config_name="ml1m.sasrec")
    assert compose(config_name="datasets.ml1m.sasrec") == training
    assert training["model"]["embedding_dim"] == preset.model.embedding_dim
    assert training["model"]["dropout"] == preset.model.dropout
    assert training["model"]["loss"]["name"] == "ce"
    assert training["training"]["learning_rate"] == preset.training.learning_rate
    assert training["training"]["batch_size"] == preset.training.batch_size
    with pytest.raises(ValueError, match="select a search preset"):
        tune(training)
    large = compose(["training.batch_size=1024"], config_name="ml1m.sasrec")
    assert large["training"]["batch_size"] == 1024
    smoke = compose(["runtime=cpu_smoke"], config_name="ml1m.sasrec")
    assert smoke["training"]["batch_size"] == 8 and smoke["max_seq_length"] == 16
    for dataset, choices in [("amazon_beauty", [4, 8]), ("ml1m", [64, 128])]:
        search = compose(config_name=f"{dataset}.tuning.sasrec")
        assert search["tuning"]["space"]["training"]["batch_size"]["choices"] == choices
    sce = compose(config_name="ml1m.sasrec_sce")
    assert sce["model"]["loss"]["name"] == "sce"
    assert sce["model"]["loss"]["n_buckets"] > 0


def test_optuna_resume_and_export(make_config, tmp_path):
    cfg = make_config()
    cfg["training"]["epochs"] = 1
    preset = compose(
        ["tuning.space.model.dropout.high=0.4"], config_name="amazon_beauty.tuning.sasrec"
    )
    space = preset["tuning"]["space"]
    assert space["model"]["dropout"]["high"] == 0.4
    legacy_space = {
        f"{section}.{name}": spec
        for section, group in space.items()
        for name, spec in group.items()
    }
    cfg["tuning"].update(
        trials=1,
        study_name="test_study",
        storage=f"sqlite:///{tmp_path}/study.db",
        pruning=False,
        space=legacy_space,
    )
    first = tune(cfg)
    cfg["tuning"]["space"] = space
    second = tune(cfg)
    assert first["identity"] == second["identity"]
    assert len(first["trials"]) == 1 and len(second["trials"]) == 2
    assert first["trials"][0]["params"] != second["trials"][1]["params"]
    assert set(second["best_params"]) == set(legacy_space)
    exported = OmegaConf.load(second["best_config"])
    assert exported.model.loss.name == cfg["model"]["loss"]["name"]
    for key, value in second["best_params"].items():
        assert OmegaConf.select(exported, key) == value
    mismatch = deepcopy(cfg)
    mismatch["max_seq_length"] = 4

    with pytest.raises(ValueError, match="another experiment"):
        tune(mismatch)


@pytest.mark.parametrize(
    "key", ["model.dropuot", "model.loss[name]", "training.validate", "max_seq_length"]
)
def test_tuning_rejects_invalid_parameters_before_loading_data(key):
    cfg = compose(["dataset.path=unused"], config_name="amazon_beauty.tuning.sasrec")
    cfg["tuning"]["space"] = {key: {"type": "categorical", "choices": [0.1]}}
    with pytest.raises(ValueError, match=escape(key)):
        tune(cfg)


def test_independent_cli_checkpoint_modes(prepared, tmp_path):
    dataset, _ = prepared
    run = tmp_path / "cli"

    _cli(
        "train",
        "-cn",
        "ml1m.sasrec",
        "runtime=cpu_smoke",
        f"dataset.path={dataset.path}",
        f"output.dir={run}",
        "training.epochs=1",
        "training.validate=false",
        "model.embedding_dim=8",
        "model.heads=2",
        "training.max_batches=1",
    )
    _cli("resume", f"checkpoint={run}/last.pt", "training.epochs=2")
    _cli(
        "evaluate",
        f"checkpoint={run}/last.pt",
        "runtime=cpu_smoke",
        "split=val",
        f"output.path={tmp_path}/val.json",
    )
    _cli(
        "finetune", f"checkpoint={run}/last.pt", f"output.dir={tmp_path}/fine", "training.epochs=1"
    )
    evaluated = json.loads((tmp_path / "val.json").read_text())
    assert evaluated["config"]["training"]["max_batches"] == 1
    assert evaluated["config"]["runtime"] == {"device": "cpu", "threads": 2}
    assert (tmp_path / "fine/last.pt").exists()
    _cli(
        "report",
        f"reporting.input_dir={tmp_path}",
        f"reporting.output_dir={tmp_path}/report",
        env={**os.environ, "MPLCONFIGDIR": str(tmp_path / "matplotlib")},
    )
    assert (tmp_path / "report/metrics.csv").exists()
    assert list((tmp_path / "report").glob("*.png"))


def test_dotted_grid_cli_exports_a_trainable_config(prepared, tmp_path):
    dataset, _ = prepared
    study_output = tmp_path / "search"
    study_path = study_output / "tuning/sasrec_grid"

    _cli(
        "tune",
        "--config-name=ml1m.tuning.sasrec_grid",
        "runtime=cpu_smoke",
        f"dataset.path={dataset.path}",
        f"output.dir={study_output}",
        "training.epochs=1",
        "training.max_batches=1",
        "model.embedding_dim=8",
        "model.heads=2",
        "tuning.study_name=sasrec_grid",
        "tuning.storage=null",
    )
    index = json.loads((study_path / "study.json").read_text())
    assert len(index["trials"]) == 4
    best = OmegaConf.load(study_path / "best.yaml")
    assert best.model.name == "sasrec"
    assert best.model.dropout == index["best_params"]["model.dropout"]
    assert best.training.learning_rate == index["best_params"]["training.learning_rate"]

    run = tmp_path / "retuned"
    _cli(
        "train",
        "--config-path",
        str(study_path),
        "--config-name",
        "best",
        "seed=43",
        f"output.dir={run}",
    )
    assert (run / "best.pt").is_file()
    config = OmegaConf.load(run / "config.yaml")
    assert config.seed == 43 and config.model.dropout == best.model.dropout


def test_grid_search_visits_configured_values(make_config, tmp_path):
    profile = compose(config_name="ml1m.tuning.sasrec_grid")["tuning"]
    assert profile["space"] == {
        "training": {"learning_rate": {"type": "categorical", "choices": [0.001, 0.003]}},
        "model": {"dropout": {"type": "categorical", "choices": [0.1, 0.2]}},
    }

    cfg = make_config("itemknn", suffix="grid")
    cfg["tuning"].update(
        sampler="grid",
        trials=2,
        storage=f"sqlite:///{tmp_path}/grid.db",
        pruning=False,
        space={"model.neighbors": {"type": "categorical", "choices": [1, 2]}},
    )
    result = tune(cfg)
    assert len(result["trials"]) == 2
    assert {trial["params"]["model.neighbors"] for trial in result["trials"]} == {1, 2}
    assert all(trial["state"] == "COMPLETE" for trial in result["trials"])
    assert tune(cfg)["trials"] == result["trials"]
