from pathlib import Path

import numpy as np
import pytest

from recsys.evaluation import build_queries, compute_metrics, exact_topk, popularity_buckets
from recsys.models import build_model
from recsys.training import evaluate_checkpoint, run_training


def test_hand_computed_metrics():
    popularity = np.array([6, 3, 1])
    recs = np.array([[0, 2], [1, 0]])
    targets = np.array([2, 1])
    overall, buckets = compute_metrics(recs, targets, popularity, [1, 2])
    assert overall["recall@1"] == 0.5
    assert overall["recall@2"] == 1
    assert overall["ndcg@2"] == pytest.approx((1 / np.log2(3) + 1) / 2)
    assert overall["coverage@2"] == 1
    assert overall["arp@2"] == 4
    tail = buckets["groups"]["tail"]
    assert tail["metrics"]["recall@1"] == 0
    assert tail["metrics"]["ndcg@2"] == pytest.approx(1 / np.log2(3))
    assert tail["metrics"]["exposure_share@2"] == 0.25
    assert tail["metrics"]["arp@1"] is None


def test_empty_queries_and_buckets():
    overall, buckets = compute_metrics(
        np.empty((0, 1), dtype=int), np.array([], dtype=int), np.array([5]), [1]
    )
    assert all(value is None for value in overall.values())
    assert buckets["groups"]["mid"]["item_count"] == 0
    assert buckets["groups"]["mid"]["metrics"]["coverage@1"] is None


@pytest.mark.parametrize("model", ["random", "mostpopular", "itemknn", "puresvd"])
def test_block_ranking_equal_to_dense(prepared, make_config, model):
    dataset, _ = prepared
    history = dataset.history("train", 6)
    cfg = make_config(model)
    m = build_model(history, cfg["model"], 42)
    m.fit(history)
    full = exact_topk(m, history, history.users, 5, 100, 100)
    blocked = exact_topk(m, history, history.users, 5, 2, 3)
    np.testing.assert_array_equal(full, blocked)
    assert all(len(set(row)) == 5 for row in full)
    assert (full < history.n_items).all()


def test_repeat_target_is_not_masked(prepared):
    dataset, _ = prepared
    history = dataset.history("train", 6)
    users, targets, _ = build_queries(dataset, history, "test")
    assert len(users) > 0
    assert all(target in history.seen[user] for user, target in zip(users, targets))
    new_users, _, _ = build_queries(dataset, history, "test", "next_new_item")
    assert len(new_users) == 0


def test_bucket_ties_are_deterministic():
    groups, _ = popularity_buckets(np.ones(6, dtype=int), 3)
    np.testing.assert_array_equal(groups, [0, 0, 1, 1, 2, 2])


def test_repeated_checkpoint_evaluation_preserves_saved_result(make_config):
    config = make_config("mostpopular")
    config["checkpoint"] = run_training(config)["best_checkpoint"]
    path = Path(config["output"]["path"])

    first = evaluate_checkpoint(config)
    saved = path.read_bytes()
    assert evaluate_checkpoint(config) == first
    assert path.read_bytes() == saved

    config["split"] = "val"
    with pytest.raises(FileExistsError):
        evaluate_checkpoint(config)
    assert path.read_bytes() == saved

    config["output"]["overwrite"] = True
    assert evaluate_checkpoint(config)["eval_split"] == "val"
    assert path.read_bytes() != saved
