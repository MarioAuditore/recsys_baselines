"""Exact ranking over the active catalog using bounded score blocks."""

import numpy as np
import torch


def exact_topk(
    model, history, users, k, user_batch_size=128, item_block_size=4096, filter_seen=False
):
    """Rank the active catalog by score, breaking ties by item ID.

    The output has one row per query with -1 in slots left empty by filtering.
    """
    if min(k, user_batch_size, item_block_size) < 1:
        raise ValueError("K and evaluation batch sizes must be positive")
    width = min(k, len(history.items))
    ranked = np.full((len(users), width), -1, dtype=np.int64)
    with torch.inference_mode():
        model.prepare_scoring(history)
        for start in range(0, len(users), user_batch_size):
            batch_users = users[start : start + user_batch_size]
            encoded = model.encode(batch_users, history)
            best_ids = np.empty((len(batch_users), 0), dtype=np.int64)
            best_scores = np.empty((len(batch_users), 0), dtype=np.float64)
            for offset in range(0, len(history.items), item_block_size):
                candidates = history.items[offset : offset + item_block_size]
                values = model.score(encoded, candidates)
                if isinstance(values, torch.Tensor):
                    values = values.detach().cpu().numpy()
                values = np.array(values, dtype=np.float64, copy=True)
                if (
                    values.shape != (len(batch_users), len(candidates))
                    or not np.isfinite(values).all()
                ):
                    raise ValueError("Model must return finite [users,candidates] scores")
                if filter_seen:
                    for row, user in enumerate(batch_users):
                        values[
                            row, np.isin(candidates, list(history.seen.get(int(user), set())))
                        ] = -np.inf
                ids = np.concatenate([best_ids, np.broadcast_to(candidates, values.shape)], axis=1)
                scores = np.concatenate([best_scores, values], axis=1)
                order = np.lexsort((ids, -scores), axis=1)[:, :width]
                best_ids = np.take_along_axis(ids, order, axis=1)
                best_scores = np.take_along_axis(scores, order, axis=1)
            best_ids[~np.isfinite(best_scores)] = -1
            ranked[start : start + len(batch_users)] = best_ids
    return ranked
