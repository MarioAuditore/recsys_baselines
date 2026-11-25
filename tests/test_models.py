from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from recsys.losses import scalable_cross_entropy
from recsys.models import _MODEL_CLASSES, build_model
from recsys.models.neighbors import item_neighbors
from recsys.training import evaluate_checkpoint, run_training


@pytest.mark.parametrize("name", ["sasrec", "sasrec_sce", "lightgcn", "ultragcn"])
def test_neural_losses_and_checkpoint(prepared, make_config, name):
    dataset, _ = prepared
    history = dataset.history("train", 6)
    cfg = make_config(name)
    model = build_model(history, cfg["model"])
    batch = next(model.training_batches(history, 3))
    loss = model.loss(*batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    restored = build_model(history, cfg["model"])
    restored.load_state_dict(model.state_dict())
    with torch.no_grad():
        for m in (model, restored):
            m.prepare_scoring(history)
        a = model.score(model.encode(history.users, history), history.items)
        b = restored.score(restored.encode(history.users, history), history.items)
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize("name", ["lightgcn", "ultragcn"])
def test_graph_model_factory_accepts_mps(prepared, make_config, name):
    dataset, _ = prepared
    history = dataset.history("train", 6)
    config = make_config(name)["model"]
    model_class = _MODEL_CLASSES[name]
    with patch.object(
        model_class, "to", autospec=True, side_effect=lambda model, device: model
    ) as transfer:
        model = build_model(history, config, device="mps")
    transfer.assert_called_once_with(model, "mps")


def test_new_neural_model_supplies_its_own_batches(make_config, monkeypatch):
    class TrainablePopularity(torch.nn.Module):
        def __init__(self, history, config):
            super().__init__()
            self.item_scores = torch.nn.Parameter(torch.zeros(history.n_items))

        def training_batches(self, history, batch_size, target_start=None):
            items = history.events.item_id.to_numpy()[:batch_size]
            yield (torch.as_tensor(items, device=self.item_scores.device),)

        def loss(self, items):
            return -self.item_scores.log_softmax(dim=0)[items].mean()

        def prepare_scoring(self, history):
            self.eval()

        def encode(self, users, history):
            return users

        def score(self, users, items):
            return self.item_scores[items].expand(len(users), -1)

    monkeypatch.setitem(_MODEL_CLASSES, "trainable_popularity", TrainablePopularity)
    config = make_config()
    config["model"] = {"name": "trainable_popularity", "loss": {"name": "ce"}}
    config["training"]["epochs"] = 1
    summary = run_training(config)
    config["checkpoint"] = summary["best_checkpoint"]
    result = evaluate_checkpoint(config)
    assert result["model"] == "trainable_popularity"
    assert result["support"]["n_queries"] > 0
    assert result["metrics"]["ndcg@3"] is not None


def test_item_zero_and_padding(prepared, make_config):
    dataset, _ = prepared
    history = dataset.history("train", 6)
    model = build_model(history, make_config()["model"])
    n = history.n_items
    inputs = torch.tensor([[0, 1, n], [2, n, n]])
    targets = torch.tensor([[1, 0, n], [0, n, n]])
    loss = model.loss(inputs, targets)
    loss.backward()
    assert model.item_embedding.weight.grad[0].abs().sum() > 0
    assert model.item_embedding.weight.grad[n].abs().sum() == 0
    assert torch.isfinite(model(inputs)).all()
    model.eval()
    changed = inputs.clone()
    changed[0, 1] = 3
    torch.testing.assert_close(model(inputs)[0, 0], model(changed)[0, 0])


def test_sce_all_classes_matches_ce():
    torch.manual_seed(1)
    x = torch.randn(5, 4, requires_grad=True)
    w = torch.randn(7, 4, requires_grad=True)
    targets = torch.tensor([0, 2, 3, 4, 6])
    projections = torch.randn(2, 4)
    actual = scalable_cross_entropy(
        x, targets, w, bucket_size_x=5, bucket_size_y=7, projections=projections
    )
    expected = F.cross_entropy(x @ w.T, targets)
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual, (x, w), retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (x, w))
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b)


def test_sce_subset_matches_manual_buckets():
    torch.manual_seed(2)
    x = torch.randn(6, 3, requires_grad=True)
    w = torch.randn(8, 3, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3, 4, 5])
    projections = torch.randn(3, 3)
    collected = {}
    for p in projections:
        xs = (x.detach() @ p).topk(3).indices
        ys = (w.detach() @ p).topk(4).indices
        for row in xs.tolist():
            candidates = ys[ys != targets[row]]
            logits = torch.cat([x[row] @ w[candidates].T, (x[row] * w[targets[row]]).sum().view(1)])
            ce = F.cross_entropy(logits[None], torch.tensor([len(candidates)]))
            collected.setdefault(row, []).append(ce)
    expected = torch.stack([torch.stack(losses).max() for losses in collected.values()]).mean()
    actual = scalable_cross_entropy(
        x, targets, w, bucket_size_x=3, bucket_size_y=4, projections=projections
    )
    torch.testing.assert_close(actual, expected)


def test_ultra_sparse_weights_match_dense(prepared):
    dataset, _ = prepared
    history = dataset.history("train", 6)
    ids, weights = item_neighbors(history.matrix, 3, 2, "ultra", diagonal=True)
    a = (history.matrix.T @ history.matrix).toarray()
    d = a.sum(1)
    expected = a * np.sqrt(d[:, None] + 1) / d[:, None] / np.sqrt(d[None, :] + 1)
    for item in range(len(a)):
        selected = np.lexsort((np.arange(len(a)), -expected[item]))[:3]
        np.testing.assert_allclose(weights[item], expected[item, selected], rtol=1e-6)
        np.testing.assert_array_equal(ids[item], selected)
