"""Turn an effective history into sequential or pairwise training batches."""

import numpy as np
import torch


def sequence_batches(history, batch_size, device, target_start=None):
    """Yield padded sequences and shifted targets, masking older targets during finetune."""
    users = history.users.copy()
    np.random.shuffle(users)
    timestamps = history.events.timestamp.to_numpy()
    for start in range(0, len(users), batch_size):
        batch_users = users[start : start + batch_size]
        length = max(len(history.sequence(u)) - 1 for u in batch_users)
        inputs = np.full((len(batch_users), length), history.n_items, dtype=np.int64)
        targets = inputs.copy()
        for row, user in enumerate(batch_users):
            seq = history.sequence(user)
            size = len(seq) - 1
            inputs[row, :size], targets[row, :size] = seq[:-1], seq[1:]
            if target_start is not None:
                times = timestamps[history.offsets[user] + 1 : history.offsets[user + 1]]
                targets[row, :size][times < target_start] = history.n_items
        if (targets != history.n_items).any():
            yield torch.as_tensor(inputs, device=device), torch.as_tensor(targets, device=device)


def sample_negatives(users, history, count, distribution="uniform", alpha=0.75):
    """Draw unseen active items with replacement for each user."""
    if count < 1 or distribution not in ("uniform", "popularity"):
        raise ValueError("Invalid negative sampling configuration")
    probabilities = None
    if distribution == "popularity":
        probabilities = history.popularity[history.items].astype(float) ** alpha
        probabilities /= probabilities.sum()
    result = np.empty((len(users), count), dtype=np.int64)
    for row, user in enumerate(users):
        observed = history.matrix[int(user)].indices
        if len(observed) == len(history.items):
            raise ValueError(f"User {user} has no unobserved negative candidates")
        sample = np.random.choice(history.items, size=count, p=probabilities)
        invalid = np.isin(sample, observed)
        for _ in range(10):
            if not invalid.any():
                break
            sample[invalid] = np.random.choice(
                history.items, size=int(invalid.sum()), p=probabilities
            )
            invalid = np.isin(sample, observed)
        if invalid.any():
            allowed = ~np.isin(history.items, observed)
            weights = probabilities[allowed] if probabilities is not None else None
            if weights is not None:
                weights = weights / weights.sum()
            sample[invalid] = np.random.choice(
                history.items[allowed], int(invalid.sum()), p=weights
            )
        result[row] = sample
    return result


def graph_batches(history, batch_size, device, sampling, target_start=None):
    """Yield (users, positives, negatives) for eligible events in the target window."""
    events = history.events
    if target_start is not None:
        events = events[events.timestamp >= target_start]
    # Saturated users cannot yield a valid pairwise training example.
    degrees = np.diff(history.matrix.indptr)
    events = events[degrees[events.user_id.to_numpy()] < len(history.items)]
    pairs = events[["user_id", "item_id"]].to_numpy()
    np.random.shuffle(pairs)
    for start in range(0, len(pairs), batch_size):
        users, positive = pairs[start : start + batch_size].T
        negative = sample_negatives(users, history, **sampling)
        yield tuple(torch.as_tensor(value, device=device) for value in (users, positive, negative))
