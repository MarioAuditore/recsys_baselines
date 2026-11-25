"""Ranking and popularity metrics for single-target recommendation queries."""

import numpy as np

from recsys.artifacts import fingerprint


def target_ranks(recommendations: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return one-based target ranks, using 0 for a missing target."""
    if not recommendations.shape[1]:
        return np.zeros(len(targets), dtype=np.int64)
    matches = (recommendations == targets[:, None]) & (recommendations >= 0)
    return np.where(matches.any(axis=1), matches.argmax(axis=1) + 1, 0)


def recall(ranks: np.ndarray) -> float | None:
    """Fraction of queries whose single target was recommended."""
    return float((ranks > 0).mean()) if ranks.size else None


def ndcg(ranks: np.ndarray) -> float | None:
    """Average logarithmic discount of the target rank, with zero gain for misses."""
    if not ranks.size:
        return None
    hits = ranks > 0
    gains = np.zeros(len(ranks), dtype=float)
    gains[hits] = 1 / np.log2(ranks[hits] + 1)
    return float(gains.mean())


def catalog_coverage(shown: np.ndarray, item_count: int) -> float | None:
    """Fraction of the corresponding catalog shown at least once."""
    return len(np.unique(shown)) / item_count if item_count else None


def arp_by_query(recommendations: np.ndarray, popularity: np.ndarray) -> float | None:
    """Average item popularity per query, weighted equally across nonempty queries."""
    valid = recommendations >= 0
    lengths = valid.sum(axis=1)
    nonempty = lengths > 0
    if not nonempty.any():
        return None
    item_popularity = np.where(valid, popularity[np.maximum(recommendations, 0)], 0)
    return float((item_popularity.sum(axis=1)[nonempty] / lengths[nonempty]).mean())


def arp_by_exposure(shown: np.ndarray, popularity: np.ndarray) -> float | None:
    """Average popularity across individual impressions within an item bucket."""
    return float(popularity[shown].mean()) if shown.size else None


def exposure_share(bucket_exposures: int, total_exposures: int) -> float | None:
    """Fraction of all impressions attributable to one item bucket."""
    return bucket_exposures / total_exposures if total_exposures else None


def popularity_buckets(popularity, count=3):
    """Partition active items by cumulative interaction mass in popularity order."""
    if count < 1:
        raise ValueError("bucket count must be positive")
    active = np.flatnonzero(popularity > 0)
    order = active[np.lexsort((active, -popularity[active]))]
    cumulative = np.cumsum(popularity[order])
    # Each boundary is after the item that reaches the corresponding mass quantile.
    cuts = (
        np.searchsorted(cumulative, cumulative[-1] * np.arange(1, count) / count, side="left") + 1
        if len(active)
        else np.zeros(count - 1, dtype=int)
    )
    membership = np.full(len(popularity), -1, dtype=np.int64)
    for group, items in enumerate(np.split(order, cuts)):
        membership[items] = group
    names = ["head", "mid", "tail"] if count == 3 else [f"bucket_{i}" for i in range(count)]
    return membership, names


def _target_metrics(ranks):
    # The same rank-based definitions apply overall and within each target bucket.
    return {"recall": recall(ranks), "ndcg": ndcg(ranks)}


def _at_k(k, metrics):
    return {f"{name}@{k}": value for name, value in metrics.items()}


def compute_metrics(recommendations, targets, popularity, ks, bucket_count=3):
    """Compute overall metrics and popularity-bucket metrics at each cutoff.

    Recommendation rows are in rank order with -1 for unfilled slots. Rank-based
    metrics group queries by the target's bucket. Catalog metrics group impressions
    by the recommended item's bucket.
    """
    has_queries = len(targets) > 0
    membership, names = popularity_buckets(popularity, bucket_count)
    target_buckets = membership[targets] if has_queries else np.empty(0, dtype=np.int64)
    active_count = int((popularity > 0).sum())
    total_interactions = popularity.sum()
    groups = {}
    for index, name in enumerate(names):
        items = membership == index
        interactions = popularity[items].sum()
        groups[name] = {
            "item_count": int(items.sum()),
            "interaction_count": int(interactions),
            "interaction_share": float(interactions / total_interactions)
            if total_interactions
            else None,
            "target_count": int((target_buckets == index).sum()),
            "metrics": {},
            "exposure_count": {},
        }

    overall = {}
    for k in ks:
        recs = recommendations[:, :k]
        ranks = target_ranks(recs, targets)
        shown = recs[recs >= 0]
        shown_buckets = membership[shown]
        metrics = _target_metrics(ranks) | {
            "coverage": catalog_coverage(shown, active_count) if has_queries else None,
            "arp": arp_by_query(recs, popularity),
        }
        overall.update(_at_k(k, metrics))
        for index, name in enumerate(names):
            group = groups[name]
            group_shown = shown[shown_buckets == index]
            group["exposure_count"][str(k)] = len(group_shown)
            metrics = _target_metrics(ranks[target_buckets == index]) | {
                "coverage": catalog_coverage(group_shown, group["item_count"])
                if has_queries
                else None,
                "arp": arp_by_exposure(group_shown, popularity),
                "exposure_share": exposure_share(len(group_shown), len(shown)),
            }
            group["metrics"].update(_at_k(k, metrics))

    return overall, {
        "fingerprint": fingerprint(
            {"membership": membership.tolist(), "popularity": popularity.tolist()}
        ),
        "groups": groups,
    }
