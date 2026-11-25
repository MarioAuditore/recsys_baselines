"""Bounded sparse item co-occurrence blocks; never materialize the full I x I matrix."""

import numpy as np
from scipy import sparse


def item_neighbors(matrix, neighbors, block_size=128, mode="cosine", shrinkage=0.0, diagonal=False):
    if neighbors < 1 or block_size < 1:
        raise ValueError("neighbors and block_size must be positive")
    n_items = matrix.shape[1]
    k = min(neighbors, n_items)
    indices = np.zeros((n_items, k), dtype=np.int64)
    weights = np.zeros((n_items, k), dtype=np.float32)
    degree = np.asarray(matrix.sum(0)).ravel()
    if mode == "ultra":
        # row sums of M.T @ M without constructing that product
        co_degree = np.asarray(matrix.T @ np.asarray(matrix.sum(1)).ravel()).ravel()
        if not diagonal:
            co_degree -= degree
    for start in range(0, n_items, block_size):
        block = (matrix[:, start : start + block_size].T @ matrix).tocsr()
        for local in range(block.shape[0]):
            item = start + local
            lo, hi = block.indptr[local : local + 2]
            ids, values = block.indices[lo:hi], block.data[lo:hi].copy()
            if not diagonal:
                keep = ids != item
                ids, values = ids[keep], values[keep]
            if mode == "cosine":
                values /= np.sqrt(degree[item] * degree[ids]) + shrinkage
            elif mode == "ultra":
                if co_degree[item] > 0:
                    values *= (
                        np.sqrt(co_degree[item] + 1) / co_degree[item] / np.sqrt(co_degree[ids] + 1)
                    )
            else:
                raise ValueError(f"Unknown similarity {mode}")
            order = np.lexsort((ids, -values))[:k]
            indices[item, : len(order)] = ids[order]
            weights[item, : len(order)] = values[order]
    return indices, weights


def neighbor_matrix(indices, weights):
    rows = np.repeat(np.arange(len(indices)), indices.shape[1])
    keep = weights.ravel() != 0
    return sparse.csr_matrix(
        (weights.ravel()[keep], (rows[keep], indices.ravel()[keep])),
        shape=(len(indices), len(indices)),
    )
