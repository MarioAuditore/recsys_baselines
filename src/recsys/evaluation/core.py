"""Evaluate a model by composing query selection, ranking, and metric calculation."""

import time

from recsys.artifacts import fingerprint
from recsys.evaluation.metrics import compute_metrics
from recsys.evaluation.queries import build_queries
from recsys.evaluation.ranking import exact_topk


def evaluate_model(model, dataset, history, split, config):
    """Build evaluation queries, rank candidates, and return metrics with provenance."""
    start = time.perf_counter()
    ks = sorted(set(config["ks"]))
    if not ks or min(ks) < 1:
        raise ValueError("ks must contain positive integers")
    protocol = config["protocol"]
    filter_seen = protocol == "next_new_item"
    users, targets, support = build_queries(dataset, history, split, protocol)
    recommendations = exact_topk(
        model,
        history,
        users,
        max(ks),
        config["user_batch_size"],
        config["item_block_size"],
        filter_seen,
    )
    metrics, buckets = compute_metrics(
        recommendations, targets, history.popularity, ks, config["buckets"]
    )
    return {
        "eval_split": split,
        "protocol": {"name": protocol, "filter_seen": filter_seen, "ks": ks},
        "evaluation_fingerprint": fingerprint(
            {
                "users": users.tolist(),
                "targets": targets.tolist(),
                "protocol": protocol,
                "fit": history.fingerprint,
            }
        ),
        "support": support,
        "metrics": metrics,
        "buckets": buckets,
        "catalog": {
            "n_items": history.n_items,
            "active_count": len(history.items),
            "fingerprint": fingerprint(history.items.tolist()),
        },
        "timing": {"evaluation_seconds": time.perf_counter() - start},
    }
