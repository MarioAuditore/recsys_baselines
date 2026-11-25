"""Non-neural baselines sharing the same encode/score interface as PyTorch models."""

import numpy as np
from scipy.sparse.linalg import svds

from recsys.models.neighbors import item_neighbors, neighbor_matrix


class Baseline:
    def __init__(self, history, config, seed=42):
        self.config, self.seed = config, seed
        self.state = {}
        if config["loss"] is not None:
            raise ValueError("Analytic baselines have loss=null")

    def state_dict(self):
        return self.state

    def load_state_dict(self, state):
        self.state = state

    def prepare_scoring(self, history):
        pass

    def encode(self, users, history):
        return users


class Random(Baseline):
    def fit(self, history):
        pass

    def score(self, users, items):
        # SplitMix64: scores are a deterministic function of (seed,user,item).
        with np.errstate(over="ignore"):
            x = np.asarray(users, dtype=np.uint64)[:, None] * np.uint64(0x9E3779B97F4A7C15)
            x = x ^ (np.asarray(items, dtype=np.uint64)[None, :] + np.uint64(self.seed % 2**64))
            x = (x ^ (x >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
            x = (x ^ (x >> 27)) * np.uint64(0x94D049BB133111EB)
            x = x ^ (x >> 31)
        return (x >> 11).astype(np.float64) / 2**53


class MostPopular(Baseline):
    def fit(self, history):
        self.state["popularity"] = history.popularity

    def score(self, users, items):
        return np.broadcast_to(self.state["popularity"][items], (len(users), len(items)))


class PureSVD(Baseline):
    def fit(self, history):
        matrix = history.matrix
        rank = self.config["rank"]
        if not 0 < rank < min(matrix.shape):
            raise ValueError(f"SVD rank must be < min(matrix.shape)={min(matrix.shape)}")
        u, s, vt = svds(
            matrix, k=rank, which="LM", random_state=self.seed, tol=self.config["tolerance"]
        )
        self.state = {"users": u * s, "items": vt.T}

    def encode(self, users, history):
        return self.state["users"][users]

    def score(self, encoded, items):
        return encoded @ self.state["items"][items].T


class ItemKNN(Baseline):
    def fit(self, history):
        indices, weights = item_neighbors(
            history.matrix,
            self.config["neighbors"],
            self.config["block_size"],
            shrinkage=self.config["shrinkage"],
        )
        self.state["similarity"] = neighbor_matrix(indices, weights)

    def encode(self, users, history):
        return history.matrix[users] @ self.state["similarity"]

    def score(self, encoded, items):
        return encoded[:, items].toarray()
