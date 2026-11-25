"""Evaluation query selection from prepared validation or test events."""

import numpy as np


def build_queries(dataset, history, split, protocol="next_item"):
    """Select the first eligible held-out event per user and summarize exclusions."""
    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test")
    if protocol not in ("next_item", "next_new_item"):
        raise ValueError("Unknown evaluation protocol")
    events = dataset.split(split)
    user_ok = events.user_id.isin(history.users)
    item_ok = events.item_id.isin(history.items)
    valid = events[user_ok & item_ok]
    removed_seen = 0
    if protocol == "next_new_item":
        mask = np.array(
            [
                item not in history.seen.get(user, set())
                for user, item in zip(valid.user_id, valid.item_id)
            ],
            dtype=bool,
        )
        removed_seen = int((~mask).sum())
        valid = valid[mask]
    queries = valid.groupby("user_id", sort=False).head(1)
    support = {
        "n_queries": len(queries),
        "events_before_effective_filter": len(events),
        "inactive_user_events": int((~user_ok).sum()),
        "inactive_item_events": int((~item_ok).sum()),
        "excluded_union_events": int((~(user_ok & item_ok)).sum()),
        "seen_target_events_removed": removed_seen,
        "preparation": dataset.metadata["splits"][split],
    }
    return queries.user_id.to_numpy(), queries.item_id.to_numpy(), support
