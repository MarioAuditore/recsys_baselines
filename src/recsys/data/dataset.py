"""Canonical CSV loading and a single effective history shared by all models."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from recsys.artifacts import file_hash, fingerprint, read_json

COLUMNS = ["timestamp", "item_id", "user_id"]


class PreparedDataset:
    def __init__(self, path, verify=True):
        self.path = Path(path)
        self.metadata = read_json(self.path / "metadata.json")
        self.n_users = self.metadata["n_users"]
        self.n_items = self.metadata["n_items"]
        self._splits = {}
        self.verify = verify
        if verify:
            for name in ("user_mapping.csv", "item_mapping.csv"):
                if file_hash(self.path / name) != self.metadata["files"][name]:
                    raise ValueError(f"Mapping checksum mismatch: {name}")

    def split(self, name):
        if name not in ("train", "val", "test"):
            raise ValueError(f"Invalid split {name}")
        if name not in self._splits:
            path = self.path / f"{name}.csv"
            if self.verify and file_hash(path) != self.metadata["files"][path.name]:
                raise ValueError(f"CSV checksum mismatch: {path}")
            data = pd.read_csv(path, dtype={col: "int64" for col in COLUMNS})
            if list(data.columns) != COLUMNS or data.isna().any().any():
                raise ValueError(f"Expected exactly {COLUMNS}, without nulls")
            if (
                not data.item_id.between(0, self.n_items - 1).all()
                or not data.user_id.between(0, self.n_users - 1).all()
            ):
                raise ValueError("ID outside canonical mapping")
            self._splits[name] = data.sort_values(
                ["user_id", "timestamp"], kind="stable"
            ).reset_index(drop=True)
        return self._splits[name]

    def history(self, fit_split="train", max_seq_length=50, min_history=2):
        if min_history < 2:
            raise ValueError("min_history must be >= 2 for the common sequential benchmark")
        if fit_split not in ("train", "train_val"):
            raise ValueError("fit_split must be train or train_val")
        if max_seq_length is not None and max_seq_length < 2:
            raise ValueError("max_seq_length must be >= 2 or null")
        parts = [self.split("train")]
        if fit_split == "train_val":
            parts.append(self.split("val"))
        full = (
            pd.concat(parts)
            .sort_values(["user_id", "timestamp"], kind="stable")
            .reset_index(drop=True)
        )
        events = (
            full
            if max_seq_length is None
            else full.groupby("user_id", sort=False).tail(max_seq_length)
        )
        sizes = events.groupby("user_id").size()
        events = events[events.user_id.isin(sizes[sizes >= min_history].index)].reset_index(
            drop=True
        )
        if events.empty:
            raise ValueError("No users have enough effective history")
        return History(
            events, full, self.n_users, self.n_items, fit_split, max_seq_length, min_history
        )


@dataclass
class History:
    events: pd.DataFrame
    full: pd.DataFrame
    n_users: int
    n_items: int
    fit_split: str
    max_seq_length: int | None
    min_history: int

    def __post_init__(self):
        self.users = np.sort(self.events.user_id.unique())
        self.items = np.sort(self.events.item_id.unique())
        self.popularity = np.bincount(self.events.item_id.to_numpy(), minlength=self.n_items)
        counts = np.bincount(self.events.user_id.to_numpy(), minlength=self.n_users)
        self.offsets = np.r_[0, counts.cumsum()]
        self.flat_items = self.events.item_id.to_numpy(copy=True)
        self.fingerprint = fingerprint(
            {
                "events": hashlib.sha256(
                    pd.util.hash_pandas_object(self.events, index=False).to_numpy().tobytes()
                ).hexdigest(),
                "fit_split": self.fit_split,
                "max_seq_length": self.max_seq_length,
                "min_history": self.min_history,
            }
        )
        self._matrix = None
        self._seen = None

    def sequence(self, user):
        return self.flat_items[self.offsets[user] : self.offsets[user + 1]]

    @property
    def matrix(self):
        if self._matrix is None:
            self._matrix = sparse.csr_matrix(
                (
                    np.ones(len(self.events), dtype=np.float32),
                    (self.events.user_id, self.events.item_id),
                ),
                shape=(self.n_users, self.n_items),
            )
            self._matrix.data[:] = 1
        return self._matrix

    @property
    def seen(self):
        if self._seen is None:
            self._seen = {
                int(u): set(rows.item_id) for u, rows in self.full.groupby("user_id", sort=False)
            }
        return self._seen

    def padded(self, users, length=None):
        sequences = [self.sequence(int(user)) for user in users]
        length = length or max(map(len, sequences))
        result = np.full((len(users), length), self.n_items, dtype=np.int64)
        # Right padding avoids fully masked attention rows before the first real event.
        for row, seq in enumerate(sequences):
            seq = seq[-length:]
            result[row, : len(seq)] = seq
        return result
