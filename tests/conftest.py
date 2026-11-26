import csv
from copy import deepcopy

import pytest
import torch

from recsys.config import compose
from recsys.data import PreparedDataset, prepare


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


@pytest.fixture
def prepared(tmp_path):
    raw = tmp_path / "raw.csv"
    with raw.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp", "item_id", "user_id", "rating"])
        for t in range(12):
            for u in range(6):
                item = (2 * u + (2 if t == 3 else t % 4)) % 12
                writer.writerow([t * 10 + u, f"i{item:02}", f"u{u}", 5])
        writer.writerow([105, "cold", "u0", 5])
        writer.writerow([106, "i00", "cold_user", 5])
        writer.writerow([1, "i00", "u0", 1])
        writer.writerow([20, "i02", "u0", 5])  # retain this exact duplicate
    cfg = compose(config_name="csv.dataset")
    cfg["dataset"].update(name="fixture", input_path=str(raw), path=str(tmp_path / "prepared"))
    cfg["preparation"]["boundaries"] = [80, 100]
    prepare(cfg["dataset"], cfg["preparation"])
    return PreparedDataset(cfg["dataset"]["path"]), cfg


@pytest.fixture
def make_config(prepared, tmp_path):
    dataset, base = prepared

    def make(model="sasrec", suffix="run"):
        cfg = compose([f"model={model}", "runtime=cpu_smoke"])
        cfg["dataset"] = deepcopy(base["dataset"])
        cfg["run_name"] = suffix
        cfg["output"] = {
            "dir": str(tmp_path / suffix),
            "path": str(tmp_path / f"{suffix}.json"),
            "overwrite": False,
        }
        cfg["max_seq_length"] = 6
        cfg["evaluation"].update(
            ks=[1, 3], selection_metric="ndcg@3", user_batch_size=2, item_block_size=3
        )
        cfg["training"].update(epochs=2, batch_size=3, max_batches=2, patience=None)
        if "embedding_dim" in cfg["model"]:
            cfg["model"]["embedding_dim"] = 8
        if model == "puresvd":
            cfg["model"]["rank"] = 2
        if model in ("itemknn", "ultragcn"):
            cfg["model"].update(neighbors=2, block_size=3)
        return cfg

    return make
