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


def test_every_experiment_composes():
    directory = Path(__file__).parents[1] / "configs"
    presets = {}
    for path in directory.glob("*_*.yaml"):
        cfg = compose(config_name=path.stem)
        dataset = cfg["dataset"]["name"]
        model = cfg["model"]["name"]
        canonical = f"{dataset.removeprefix('amazon_')}_{model}"
        assert path.stem == canonical
        presets[canonical] = cfg
        assert cfg["training"]["batch_size"] > 0
        assert bool(cfg["tuning"]["space"]) == (model not in {"random", "mostpopular"})
    for name, choices in [("beauty_sasrec", [4, 8]), ("ml1m_sasrec", [64, 128])]:
        space = presets[name]["tuning"]["space"]
        assert space["training"]["batch_size"]["choices"] == choices
    sce = compose(["model.loss.name=sce"], config_name="ml1m_sasrec")
    assert sce["model"]["loss"]["name"] == "sce"
    assert sce["model"]["loss"]["n_buckets"] > 0


def test_optuna_resume_and_export(make_config, tmp_path):
    cfg = make_config()
    cfg["training"]["epochs"] = 1
    preset = compose(["tuning.space.model.dropout.high=0.4"], config_name="beauty_sasrec")
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
    cfg = compose(["dataset.path=unused"], config_name="beauty_sasrec")
    cfg["tuning"]["space"] = {key: {"type": "categorical", "choices": [0.1]}}
    with pytest.raises(ValueError, match=escape(key)):
        tune(cfg)


def test_independent_cli_checkpoint_modes(prepared, tmp_path):
    dataset, _ = prepared
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    run = tmp_path / "cli"

    def cli(action, *arguments):
        proc = subprocess.run(
            [sys.executable, "-m", f"recsys.{action}", *arguments],
            capture_output=True,
            text=True,
            env=env,
            timeout=90,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        return proc

    cli(
        "train",
        "-cn",
        "ml1m_sasrec",
        "runtime=cpu_smoke",
        f"dataset.path={dataset.path}",
        f"output.dir={run}",
        "training.epochs=1",
        "training.validate=false",
        "model.embedding_dim=8",
        "training.max_batches=1",
    )
    cli("resume", f"checkpoint={run}/last.pt", "training.epochs=2")
    cli(
        "evaluate",
        f"checkpoint={run}/last.pt",
        "runtime=cpu_smoke",
        "split=val",
        f"output.path={tmp_path}/val.json",
    )
    cli("finetune", f"checkpoint={run}/last.pt", f"output.dir={tmp_path}/fine", "training.epochs=1")
    evaluated = json.loads((tmp_path / "val.json").read_text())
    assert evaluated["config"]["training"]["max_batches"] == 1
    assert evaluated["config"]["runtime"] == {"device": "cpu", "threads": 2}
    assert (tmp_path / "fine/last.pt").exists()
    cli("report", f"reporting.input_dir={tmp_path}", f"reporting.output_dir={tmp_path}/report")
    assert (tmp_path / "report/metrics.csv").exists()
    assert list((tmp_path / "report").glob("*.png"))


def test_grid_search_visits_configured_values(make_config, tmp_path):
    profile = compose(["tuning=sasrec_grid"], config_name="ml1m_sasrec")["tuning"]
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
