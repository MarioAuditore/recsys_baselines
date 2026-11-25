import csv
from pathlib import Path

import pytest

from recsys.tuning import tune


def test_trial_index_links_to_trials_across_output_directories(make_config, tmp_path):
    config = make_config("itemknn")
    config["tuning"].update(
        sampler="grid",
        trials=1,
        storage=f"sqlite:///{tmp_path}/study.db",
        pruning=False,
        space={"model.neighbors": {"type": "categorical", "choices": [1, 2]}},
    )

    first = tune(config)
    original = Path(first["best_config"]).parent
    with (original / "trials.csv").open(newline="") as stream:
        assert [Path(row["config_path"]) for row in csv.DictReader(stream)] == [
            Path("trial_0/config.yaml")
        ]

    config["output"]["dir"] = str(tmp_path / "resumed")
    second = tune(config)
    resumed = Path(second["best_config"]).parent
    with (resumed / "trials.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))

    assert len(records) == 2
    for record, trial in zip(records, second["trials"], strict=True):
        assert record["number"] == str(trial["number"])
        assert record["state"] == trial["state"] == "COMPLETE"
        assert float(record["value"]) == trial["value"]
        assert int(record["model.neighbors"]) == trial["params"]["model.neighbors"]
    assert Path(records[0]["config_path"]) == original / "trial_0/config.yaml"
    assert Path(records[1]["config_path"]) == Path("trial_1/config.yaml")


def test_repeated_in_memory_study_preserves_previous_run(make_config):
    config = make_config("itemknn")
    config["tuning"].update(
        sampler="grid",
        trials=1,
        study_name="repeated",
        storage=None,
        pruning=False,
        space={"model.neighbors": {"type": "categorical", "choices": [1]}},
    )

    original = Path(tune(config)["best_config"]).parent
    saved_index = (original / "trials.csv").read_bytes()
    repeated = Path(tune(config)["best_config"]).parent

    assert original.name == "repeated"
    assert repeated.parent == original.parent
    assert repeated != original
    assert (original / "trials.csv").read_bytes() == saved_index
    assert (original / "trial_0/best.pt").is_file()
    assert (repeated / "trial_0/best.pt").is_file()


def test_trial_index_records_failures_without_artifacts(make_config, monkeypatch):
    config = make_config("itemknn")
    config["tuning"].update(
        trials=1,
        study_name="failed",
        storage=None,
        space={"model.neighbors": {"type": "categorical", "choices": [1]}},
    )

    def fail_training(config, epoch_callback=None):
        raise RuntimeError("test failure before writing artifacts")

    monkeypatch.setattr("recsys.tuning.run_training", fail_training)
    with pytest.raises(ValueError, match="Study has no completed trials"):
        tune(config)

    index = Path(config["output"]["dir"]) / "tuning/failed/trials.csv"
    with index.open(newline="") as stream:
        assert list(csv.DictReader(stream)) == [
            {
                "number": "0",
                "state": "FAIL",
                "value": "",
                "failure": "test failure before writing artifacts",
                "config_path": "",
                "model.neighbors": "1",
            }
        ]
