"""Download original files and stream only the fields needed for recommendation."""

import csv
import gzip
import io
import json
import zipfile
from pathlib import Path

import requests

from recsys.artifacts import atomic_path, file_hash, read_json, write_json


def source_url(dataset):
    if dataset.get("url"):
        return dataset["url"]
    source = dataset["source"]
    if source == "ml1m":
        return "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    if source == "amazon":
        category = dataset["category"]
        return f"https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/{category}.jsonl.gz"
    if source == "yambda":
        revision = dataset.get("revision", "main")
        return f"https://huggingface.co/datasets/yandex/yambda/resolve/{revision}/flat/{dataset['size']}/{dataset['event']}.parquet"
    raise ValueError(f"No download source for {source}; set dataset.input_path")


def download(dataset, overwrite=False):
    path = Path(dataset["input_path"])
    url = source_url(dataset)
    manifest_path = path.with_suffix(path.suffix + ".source.json")
    if path.exists() and not overwrite:
        if not manifest_path.exists():
            raise FileExistsError(
                f"{path} has no source manifest; use prepare directly or overwrite=true"
            )
        manifest = read_json(manifest_path)
        if manifest["url"] != url or manifest["sha256"] != file_hash(path):
            raise ValueError("Cached source does not match URL/checksum")
        return manifest
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with atomic_path(path, overwrite) as temporary:
            with temporary.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
        manifest = {
            "url": url,
            "sha256": file_hash(path),
            "bytes": path.stat().st_size,
            "revision": dataset.get("revision"),
            "resolved_revision": response.headers.get("x-repo-commit"),
            "etag": response.headers.get("ETag"),
        }
    write_json(manifest_path, manifest, overwrite=True)
    return manifest


def raw_events(dataset):
    """Yield (timestamp, original_item, original_user, rating, played_percent)."""
    path = Path(dataset["input_path"])
    source = dataset["source"]
    if source == "ml1m":
        with zipfile.ZipFile(path) as archive:
            with archive.open("ml-1m/ratings.dat") as data:
                for line in io.TextIOWrapper(data, encoding="utf-8"):
                    user, item, rating, timestamp = line.strip().split("::")
                    yield int(timestamp), item, user, float(rating), None
    elif source == "amazon":
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as data:
            for line in data:
                record = json.loads(line)
                timestamp = record.get("timestamp", record.get("sort_timestamp"))
                yield timestamp, record["parent_asin"], record["user_id"], record["rating"], None
    elif source == "yambda":
        import pyarrow.parquet as pq

        columns = ["uid", "item_id", "timestamp"]
        if dataset["event"] == "listens":
            columns.append("played_ratio_pct")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=columns):
            rows = batch.to_pydict()
            for index in range(batch.num_rows):
                played = rows["played_ratio_pct"][index] if "played_ratio_pct" in rows else None
                yield (
                    rows["timestamp"][index],
                    str(rows["item_id"][index]),
                    str(rows["uid"][index]),
                    None,
                    played,
                )
    elif source == "csv":
        with path.open(newline="") as data:
            for row in csv.DictReader(data):
                rating = float(row["rating"]) if row.get("rating") else None
                played = float(row["played_ratio_pct"]) if row.get("played_ratio_pct") else None
                yield int(row["timestamp"]), row["item_id"], row["user_id"], rating, played
    else:
        raise ValueError(f"Unsupported source: {source}")


def normalize(dataset, path, options):
    """Write a temporary normalized stream; preserve every accepted event and its order."""
    counts = {"raw": 0, "rating_removed": 0, "listen_removed": 0, "accepted": 0}
    operator = options["listen_operator"]
    if operator not in ("gt", "ge"):
        raise ValueError("listen_operator must be gt or ge")
    with open(path, "w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["timestamp", "item_key", "user_key", "event_order"])
        for order, (timestamp, item, user, rating, played) in enumerate(raw_events(dataset)):
            counts["raw"] += 1
            if (
                timestamp is None
                or item is None
                or user is None
                or str(item) == ""
                or str(user) == ""
            ):
                raise ValueError(f"Missing required field in source event {order}")
            if (
                rating is not None
                and options["min_rating"] is not None
                and rating < options["min_rating"]
            ):
                counts["rating_removed"] += 1
                continue
            threshold = options["listen_threshold"]
            if dataset.get("event") == "listens" and threshold is not None:
                accepted = played is not None and (
                    played > threshold if operator == "gt" else played >= threshold
                )
                if not accepted:
                    counts["listen_removed"] += 1
                    continue
            value = int(timestamp) * int(dataset.get("timestamp_scale", 1))
            writer.writerow([value, str(item), str(user), order])
            counts["accepted"] += 1
    return counts
