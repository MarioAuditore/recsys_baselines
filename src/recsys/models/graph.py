"""LightGCN propagation and UltraGCN constraints over the common training history."""

import numpy as np
import torch
from scipy import sparse
from torch import nn
from torch.nn import functional as F

from recsys.data.batches import graph_batches
from recsys.losses.bpr import bpr_loss
from recsys.models.neighbors import item_neighbors


def torch_sparse(matrix):
    coo = matrix.tocoo()
    return torch.sparse_coo_tensor(
        np.vstack([coo.row, coo.col]),
        coo.data,
        coo.shape,
        dtype=torch.float32,
        check_invariants=True,
    ).coalesce()


class GraphModel(nn.Module):
    def __init__(self, history, config):
        super().__init__()
        self.config = config
        self.users = nn.Embedding(history.n_users, config["embedding_dim"])
        self.items = nn.Embedding(history.n_items, config["embedding_dim"])
        nn.init.normal_(self.users.weight, std=config["init_std"])
        nn.init.normal_(self.items.weight, std=config["init_std"])
        self._cached = None

    def train(self, mode=True):
        if mode:
            self._cached = None
        return super().train(mode)

    def representations(self):
        return self.users.weight, self.items.weight

    def training_batches(self, history, batch_size, target_start=None):
        return graph_batches(
            history, batch_size, self.users.weight.device, self.config["sampling"], target_start
        )

    def prepare_scoring(self, history):
        self.eval()
        with torch.no_grad():
            self._cached = tuple(value.detach() for value in self.representations())

    def encode(self, users, history):
        return self._cached[0][torch.as_tensor(users, device=self.users.weight.device)]

    def score(self, encoded, items):
        return encoded @ self._cached[1][torch.as_tensor(items, device=encoded.device)].T


class LightGCN(GraphModel):
    def __init__(self, history, config):
        super().__init__(history, config)
        if config["loss"]["name"] != "bpr":
            raise ValueError("LightGCN supports bpr")
        matrix = history.matrix
        adjacency = sparse.bmat([[None, matrix], [matrix.T, None]], format="csr")
        degree = np.asarray(adjacency.sum(1)).ravel()
        inv = np.zeros_like(degree)
        inv[degree > 0] = degree[degree > 0] ** -0.5
        normalized = sparse.diags(inv) @ adjacency @ sparse.diags(inv)
        self.register_buffer("adjacency", torch_sparse(normalized), persistent=False)
        self.n_users = history.n_users

    def representations(self):
        x = torch.cat([self.users.weight, self.items.weight])
        total = x
        for _ in range(self.config["layers"]):
            x = torch.sparse.mm(self.adjacency, x)
            total = total + x
        total = total / (self.config["layers"] + 1)
        return total[: self.n_users], total[self.n_users :]

    def loss(self, users, positive, negative):
        user_emb, item_emb = self.representations()
        u, p, n = user_emb[users], item_emb[positive], item_emb[negative]
        ranking = bpr_loss((u * p).sum(-1), (u[:, None, :] * n).sum(-1))
        regularizer = (
            self.users(users).square().sum()
            + self.items(positive).square().sum()
            + self.items(negative).square().sum() / negative.shape[1]
        ) / (2 * len(users))
        return ranking + self.config["loss"]["regularization"] * regularizer


class UltraGCN(GraphModel):
    def __init__(self, history, config):
        super().__init__(history, config)
        if config["loss"]["name"] != "ultragcn":
            raise ValueError("UltraGCN requires its constraint objective")
        matrix = history.matrix
        ud, id_ = np.asarray(matrix.sum(1)).ravel(), np.asarray(matrix.sum(0)).ravel()
        beta_u = np.zeros_like(ud)
        beta_u[ud > 0] = np.sqrt(ud[ud > 0] + 1) / ud[ud > 0]
        self.register_buffer("beta_u", torch.from_numpy(beta_u), persistent=False)
        self.register_buffer("beta_i", torch.from_numpy(1 / np.sqrt(id_ + 1)), persistent=False)
        indices, weights = item_neighbors(
            matrix,
            config["neighbors"],
            config["block_size"],
            mode="ultra",
            diagonal=config["include_self"],
        )
        self.register_buffer("neighbors", torch.from_numpy(indices), persistent=False)
        self.register_buffer("neighbor_weights", torch.from_numpy(weights), persistent=False)

    def loss(self, users, positive, negative):
        c = self.config["loss"]
        u, p, n = self.users(users), self.items(positive), self.items(negative)
        pos_score = (u * p).sum(-1)
        neg_score = (u[:, None, :] * n).sum(-1)
        pos_weight = c["w1"] + c["w2"] * self.beta_u[users] * self.beta_i[positive]
        neg_weight = c["w3"] + c["w4"] * self.beta_u[users, None] * self.beta_i[negative]
        ui = (
            pos_weight * F.softplus(-pos_score)
            + c["negative_weight"] * (neg_weight * F.softplus(neg_score)).mean(-1)
        ).sum()
        neighbors = self.items(self.neighbors[positive])
        ii = (
            self.neighbor_weights[positive] * F.softplus(-(u[:, None, :] * neighbors).sum(-1))
        ).sum()
        regularizer = (self.users.weight.square().sum() + self.items.weight.square().sum()) / 2
        # Reference objective uses SUM over positive pairs and mean over negatives.
        return ui + c["item_weight"] * ii + c["regularization"] * regularizer
