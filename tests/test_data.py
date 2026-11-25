import csv
import gzip
import json
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from recsys.data.sources import normalize, raw_events


def test_split_mapping_repeats_and_threshold(prepared):
    dataset, _ = prepared
    train, val, test = [dataset.split(name) for name in ["train", "val", "test"]]
    assert train.timestamp.max() < val.timestamp.min() < test.timestamp.min()
    assert list(train.columns) == ["timestamp", "item_id", "user_id"]
    assert set(train.item_id) == set(range(dataset.n_items))
    assert set(train.user_id) == set(range(dataset.n_users))
    assert train.duplicated().sum() == 1
    stats = dataset.metadata["train_statistics"]
    assert stats["duplicate_timestamp_user_item_events"] == 1
    assert stats["repeated_pair_events"] == len(train) - len(
        train.drop_duplicates(["user_id", "item_id"])
    )
    assert stats["history_length"]["max"] == 9
    assert dataset.metadata["counts"]["rating_removed"] == 1
    assert dataset.metadata["splits"]["test"]["unknown_items"] == 1
    assert dataset.metadata["splits"]["test"]["unknown_users"] == 1
    assert dataset.metadata["splits"]["test"]["excluded_union"] == 2


def test_history_shared_truncation_and_refit(prepared):
    dataset, _ = prepared
    hist = dataset.history("train", 3)
    refit = dataset.history("train_val", 3)
    for user in hist.users:
        np.testing.assert_array_equal(
            hist.sequence(user),
            dataset.split("train").query("user_id==@user").item_id.to_numpy()[-3:],
        )
        both = (
            pd.concat([dataset.split("train"), dataset.split("val")])
            .query("user_id==@user")
            .sort_values("timestamp", kind="stable")
        )
        np.testing.assert_array_equal(refit.sequence(user), both.item_id.to_numpy()[-3:])
    assert hist.fingerprint != refit.fingerprint
    assert hist.popularity.sum() == len(hist.events)
    assert np.all(hist.matrix.data == 1)


def test_checksum_catches_modified_data(prepared):
    dataset, _ = prepared
    with (dataset.path / "train.csv").open("a") as stream:
        stream.write("1,0,0\n")
    with pytest.raises(ValueError, match="checksum"):
        dataset.split("train")


def test_adapters_and_thresholds(tmp_path):
    ml = tmp_path / "ml.zip"
    with zipfile.ZipFile(ml, "w") as archive:
        archive.writestr("ml-1m/ratings.dat", "1::2::4::100\n")
    assert list(raw_events({"source": "ml1m", "input_path": str(ml)})) == [
        (100, "2", "1", 4.0, None)
    ]
    amazon = tmp_path / "reviews.jsonl.gz"
    with gzip.open(amazon, "wt") as stream:
        stream.write(
            json.dumps(
                {
                    "sort_timestamp": 123,
                    "parent_asin": "parent",
                    "asin": "variant",
                    "user_id": "u",
                    "rating": 4,
                }
            )
            + "\n"
        )
    assert list(raw_events({"source": "amazon", "input_path": str(amazon)}))[0][:4] == (
        123,
        "parent",
        "u",
        4,
    )
    yambda = tmp_path / "listens.parquet"
    pq.write_table(
        pa.table(
            {
                "uid": [1, 1, 1],
                "item_id": [2, 2, 3],
                "timestamp": [5, 5, 6],
                "played_ratio_pct": [50, 51, 99],
            }
        ),
        yambda,
    )
    cfg = {
        "source": "yambda",
        "event": "listens",
        "input_path": str(yambda),
        "timestamp_scale": 1000,
    }
    output = tmp_path / "normalized.csv"
    counts = normalize(
        cfg, output, {"min_rating": 4, "listen_threshold": 50, "listen_operator": "gt"}
    )
    assert counts["accepted"] == 2
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["timestamp"] == "5000"
    counts = normalize(
        cfg, output, {"min_rating": 4, "listen_threshold": 50, "listen_operator": "ge"}
    )
    assert counts["accepted"] == 3


def test_same_timestamp_never_crosses_splits(prepared):
    dataset, _ = prepared
    sets = [set(dataset.split(s).timestamp) for s in ["train", "val", "test"]]
    assert not (sets[0] & sets[1] or sets[1] & sets[2] or sets[0] & sets[2])
