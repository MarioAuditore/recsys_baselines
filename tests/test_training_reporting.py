import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from recsys.artifacts import read_json, write_json
from recsys.reporting import aggregate, confidence_interval, report
from recsys.training import evaluate_checkpoint, run_training
from recsys.training.checkpoints import load_checkpoint, save_checkpoint


def test_training_progress_shows_loss_and_validation_ndcg(make_config, capsys):
    config = make_config()
    config["training"]["epochs"] = 1
    config["evaluation"]["selection_metric"] = "recall@3"
    summary = run_training(config)
    journal = Path(summary["last_checkpoint"]).with_name("validation.jsonl")
    epoch = json.loads(journal.read_text())

    progress = capsys.readouterr().err
    assert "1/1" in progress
    assert f"loss={epoch['loss']:.5f}" in progress
    assert f"val_ndcg@3={epoch['validation']['metrics']['ndcg@3']:.5f}" in progress


def test_resume_matches_uninterrupted(make_config):
    full = make_config(suffix="full")
    full["training"]["validate"] = False
    run_training(full)
    partial = make_config(suffix="partial")
    partial["training"].update(epochs=1, validate=False)
    first = run_training(partial)
    partial["checkpoint"] = first["best_checkpoint"]
    with pytest.raises(ValueError, match="Resume requires last.pt"):
        run_training(partial, "resume")
    partial["checkpoint"] = first["last_checkpoint"]
    journal = Path(first["last_checkpoint"]).with_name("validation.jsonl")
    # Simulate an interruption after logging epoch 2 but before publishing the checkpoint.
    with journal.open("a") as stream:
        stream.write('{"epoch": 2}\n')
    partial["training"]["epochs"] = 2
    run_training(partial, "resume")
    assert [json.loads(line)["epoch"] for line in journal.read_text().splitlines()] == [1, 2]
    a = load_checkpoint(full["output"]["dir"] + "/last.pt")
    b = load_checkpoint(partial["checkpoint"])
    for key in a["model_state"]:
        torch.testing.assert_close(a["model_state"][key], b["model_state"][key], rtol=0, atol=0)
    assert a["global_step"] == b["global_step"]


def test_resume_restores_best_after_interrupted_publication(make_config, monkeypatch):
    config = make_config()
    config["training"]["epochs"] = 1
    objectives = iter([0.8, 0.4])

    def validate(model, dataset, history, evaluation):
        objective = next(objectives)
        return {"metrics": {evaluation["selection_metric"]: objective}}, objective

    monkeypatch.setattr("recsys.training.runner._validate_epoch", validate)
    first = run_training(config)
    best = Path(first["best_checkpoint"])
    best.unlink()
    config["checkpoint"] = first["last_checkpoint"]
    config["training"]["epochs"] = 2

    resumed = run_training(config, "resume")

    assert load_checkpoint(best)["epoch"] == 1
    assert load_checkpoint(resumed["last_checkpoint"])["epoch"] == 2


def test_train_evaluate_finetune_and_refit(make_config):
    cfg = make_config()
    result = run_training(cfg)
    cfg["checkpoint"] = result["best_checkpoint"]
    evaluated = evaluate_checkpoint(cfg)
    assert read_json(cfg["output"]["path"])["seed"] == 42
    assert evaluated["metrics"]["recall@3"] is not None
    fine = deepcopy(cfg)
    fine.update(fit_split="train_val", target_split="val", run_name="fine")
    fine["training"].update(epochs=1, validate=False)
    fine["output"]["dir"] += "_fine"
    result_fine = run_training(fine, "finetune")
    checkpoint = load_checkpoint(result_fine["last_checkpoint"])
    assert checkpoint["epoch"] == 1 and checkpoint["stage"] == "finetune"
    assert checkpoint["parent_checkpoint"] == cfg["checkpoint"]
    refit = make_config(suffix="refit")
    refit["fit_split"] = "train_val"
    run_training(refit)
    refit["training"]["validate"] = True
    with pytest.raises(ValueError, match="Validation is in fit"):
        run_training(refit)


@pytest.mark.parametrize(
    "name", ["random", "mostpopular", "itemknn", "puresvd", "lightgcn", "ultragcn", "sasrec_sce"]
)
def test_all_models_runner(make_config, name):
    cfg = make_config(name, suffix=name)
    cfg["training"]["epochs"] = 1
    summary = run_training(cfg)
    cfg["checkpoint"] = summary["best_checkpoint"]
    result = evaluate_checkpoint(cfg)
    assert result["support"]["n_queries"] > 0
    assert all(np.isfinite(value) for value in result["metrics"].values())


def test_baseline_evaluation_records_actual_cpu_device(make_config, monkeypatch):
    config = make_config("puresvd")
    config["checkpoint"] = run_training(config)["best_checkpoint"]
    checkpoint = load_checkpoint(config["checkpoint"])
    checkpoint["config"]["runtime"]["device"] = "mps"
    save_checkpoint(config["checkpoint"], checkpoint)
    config["runtime"]["device"] = "mps"
    monkeypatch.setattr(
        "recsys.training.checkpoint_evaluation.configure_device", lambda runtime: runtime["device"]
    )

    result = evaluate_checkpoint(config)

    assert result["config"]["runtime"]["device"] == "mps"
    assert result["training_runtime"]["device"] == "cpu"
    assert result["environment"]["device"] == "cpu"


def test_ci_and_duplicate_detection(make_config):
    interval = confidence_interval([1, 2, 3])
    assert interval["mean"] == 2
    assert interval["upper"] == pytest.approx(4.4841377)
    assert confidence_interval([1])["upper"] is None
    cfg = make_config("mostpopular")
    cfg["checkpoint"] = run_training(cfg)["best_checkpoint"]
    result = evaluate_checkpoint(cfg)
    repeated = deepcopy(result)
    repeated.pop("training_runtime")
    repeated["config"]["checkpoint"] = "relative/best.pt"
    repeated["config"]["output"]["path"] = "another_result.json"
    repeated["timing"]["evaluation_seconds"] += 1
    assert aggregate([result, repeated]) == aggregate([result])

    conflict = deepcopy(result)
    conflict["metrics"]["recall@3"] += 0.1
    with pytest.raises(ValueError, match="Duplicate seed"):
        aggregate([result, conflict])
    conflict = deepcopy(result)
    conflict["checkpoint"] += ".other"
    with pytest.raises(ValueError, match="Duplicate seed"):
        aggregate([result, conflict])
    other = deepcopy(result)
    other["seed"] = 43
    other["config"]["seed"] = 43
    groups = aggregate([result, other])
    assert len(groups) == 1 and groups[0]["metrics"]["recall@3"]["n"] == 2
    other["fit_fingerprint"] = "different"
    assert len(aggregate([result, other])) == 2

    extended = deepcopy(result)
    extended["metrics"]["mrr@3"] = 0.5
    for bucket in extended["buckets"]["groups"].values():
        bucket["metrics"]["mrr@3"] = 0.5
    groups = aggregate([extended, result])
    assert len({group["comparison_id"] for group in groups}) == 2
    assert ["mrr@3" in group["metrics"] for group in groups] == [True, False]


def test_report_refresh_includes_new_model_in_existing_chart(make_config, tmp_path):
    results = tmp_path / "results"
    output = tmp_path / "report"

    def evaluate(model):
        config = make_config(model, suffix=model)
        config["training"]["epochs"] = 1
        config["evaluation"]["ks"] = [3]
        config["output"]["path"] = str(results / f"{model}.json")
        config["checkpoint"] = run_training(config)["best_checkpoint"]
        record = evaluate_checkpoint(config)
        record["buckets"]["groups"]["tail"]["metrics"]["tail_only@3"] = 0.25
        write_json(config["output"]["path"], record, overwrite=True)

    evaluate("sasrec")
    assert report(results, output)["groups"] == 1
    (chart,) = output.glob("*_ndcg_at_3.svg")
    assert "sasrec/ce" in chart.read_text()
    assert "puresvd" not in chart.read_text()
    notes = output / "notes.txt"
    notes.write_text("experiment notes")

    evaluate("puresvd")
    assert report(results, output)["groups"] == 2
    assert {group["model"] for group in read_json(output / "aggregate.json")} == {
        "puresvd",
        "sasrec",
    }
    with (output / "metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["model"] for row in rows} == {"puresvd", "sasrec"}
    assert {row["seeds"] for row in rows} == {"42"}
    assert len({row["comparison"] for row in rows}) == 1
    for row in rows:
        record = read_json(results / f"{row['model']}.json")
        assert row["n"] == "1"
        assert float(row["mean"]) == record["metrics"][row["metric"]]

    markdown = (output / "metrics.md").read_text()
    assert "puresvd" in markdown and "sasrec" in markdown
    assert "next_item" in markdown and "42" in markdown
    assert "Popularity buckets" in chart.read_text()
    assert "puresvd/analytic" in chart.read_text() and "sasrec/ce" in chart.read_text()
    (bucket_only,) = output.glob("*_tail_only_at_3.svg")
    assert "NA" in bucket_only.read_text()
    assert notes.read_text() == "experiment notes"
