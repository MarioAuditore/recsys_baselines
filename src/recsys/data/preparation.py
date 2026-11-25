"""Out-of-core preparation with DuckDB. CSV is the only persistent event format."""

import tempfile
from pathlib import Path

import duckdb

from recsys.artifacts import file_hash, fingerprint, read_json, write_json
from recsys.data.sources import normalize


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def prepare(dataset, options, overwrite=False):
    destination = Path(dataset["path"])
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{destination} is not empty; set overwrite=true or choose another path"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".prepare-") as temporary:
        work = Path(temporary)
        counts = normalize(dataset, work / "events.csv", options)
        if not counts["accepted"]:
            raise ValueError("No events remain after signal filtering")
        con = duckdb.connect(str(work / "work.duckdb"))
        try:
            con.execute(f"SET memory_limit={sql_string(options['memory_limit'])}")
            con.execute("SET threads=2")
            con.execute(f"SET temp_directory={sql_string(work / 'spill')}")
            con.execute(
                f"CREATE TABLE events AS SELECT * FROM read_csv({sql_string(work / 'events.csv')}, columns={{'timestamp':'BIGINT','item_key':'VARCHAR','user_key':'VARCHAR','event_order':'BIGINT'}}, header=true)"
            )
            if options.get("sample_users") is not None:
                n = int(options["sample_users"])
                if n < 1:
                    raise ValueError("sample_users must be positive")
                con.execute(
                    f"DELETE FROM events WHERE user_key NOT IN (SELECT DISTINCT user_key FROM events ORDER BY md5(user_key || {sql_string(str(options['sample_seed']))}), user_key LIMIT {n})"
                )
            counts["after_user_sample"] = con.execute("SELECT count(*) FROM events").fetchone()[0]
            boundaries = options.get("boundaries")
            if boundaries is None:
                ratios = options["ratios"]
                if len(ratios) != 3 or min(ratios) <= 0 or abs(sum(ratios) - 1) > 1e-8:
                    raise ValueError("split ratios must be three positive numbers summing to one")
                boundaries = con.execute(
                    "SELECT quantile_disc(timestamp, ?), quantile_disc(timestamp, ?) FROM events",
                    [ratios[0], sum(ratios[:2])],
                ).fetchone()
            t_val, t_test = map(int, boundaries)
            if t_val >= t_test:
                raise ValueError("Need two distinct increasing temporal boundaries")
            con.execute("CREATE TABLE train AS SELECT * FROM events WHERE timestamp < ?", [t_val])
            while True:
                before = con.execute("SELECT count(*) FROM train").fetchone()[0]
                con.execute(
                    "DELETE FROM train WHERE user_key IN (SELECT user_key FROM train GROUP BY user_key HAVING count(*) < ?) OR item_key IN (SELECT item_key FROM train GROUP BY item_key HAVING count(*) < ?)",
                    [options["min_user_events"], options["min_item_events"]],
                )
                after = con.execute("SELECT count(*) FROM train").fetchone()[0]
                if not options["iterative_filter"] or after == before:
                    break
            if after == 0:
                raise ValueError("Training part is empty after temporal/degree filtering")
            con.execute(
                "CREATE TABLE users AS SELECT user_key, row_number() OVER (ORDER BY user_key)-1 AS user_id FROM (SELECT DISTINCT user_key FROM train)"
            )
            con.execute(
                "CREATE TABLE items AS SELECT item_key, row_number() OVER (ORDER BY item_key)-1 AS item_id FROM (SELECT DISTINCT item_key FROM train)"
            )
            manifest = {
                "schema_version": 1,
                "name": dataset["name"],
                "source": dict(dataset),
                "preparation": dict(options),
                "boundaries": [t_val, t_test],
                "timestamp_unit": "ms",
                "timestamp_origin": dataset.get("time_origin", "unix_utc"),
                "source_sha256": file_hash(dataset["input_path"]),
                "counts": counts,
                "splits": {},
            }
            source_manifest = Path(str(dataset["input_path"]) + ".source.json")
            if source_manifest.exists():
                manifest["download"] = read_json(source_manifest)
            lengths = con.execute(
                "SELECT min(n), max(n), avg(n), quantile_disc(n, 0.5), quantile_disc(n, 0.9) FROM (SELECT count(*) n FROM train GROUP BY user_key)"
            ).fetchone()
            unique_pairs, unique_events = con.execute(
                "SELECT count(DISTINCT (user_key, item_key)), count(DISTINCT (user_key, item_key, timestamp)) FROM train"
            ).fetchone()
            manifest["train_statistics"] = {
                "history_length": dict(zip(["min", "max", "mean", "median", "p90"], lengths)),
                "unique_user_item_pairs": unique_pairs,
                "repeated_pair_events": after - unique_pairs,
                "duplicate_timestamp_user_item_events": after - unique_events,
            }
            for split, source in [
                ("train", "train"),
                (
                    "val",
                    f"(SELECT * FROM events WHERE timestamp >= {t_val} AND timestamp < {t_test})",
                ),
                ("test", f"(SELECT * FROM events WHERE timestamp >= {t_test})"),
            ]:
                total, unknown_users, unknown_items, retained = con.execute(
                    f"SELECT count(*), count(*) FILTER (WHERE u.user_id IS NULL), count(*) FILTER (WHERE i.item_id IS NULL), count(*) FILTER (WHERE u.user_id IS NOT NULL AND i.item_id IS NOT NULL) FROM {source} e LEFT JOIN users u USING(user_key) LEFT JOIN items i USING(item_key)"
                ).fetchone()
                query = f"SELECT e.timestamp, i.item_id, u.user_id FROM {source} e JOIN users u USING(user_key) JOIN items i USING(item_key) ORDER BY u.user_id,e.timestamp,e.event_order"
                con.execute(
                    f"COPY ({query}) TO {sql_string(work / (split + '.csv'))} (HEADER, DELIMITER ',')"
                )
                manifest["splits"][split] = {
                    "before_catalog_filter": total,
                    "unknown_users": unknown_users,
                    "unknown_items": unknown_items,
                    "retained": retained,
                    "excluded_union": total - retained,
                }
            for table, key, ident in [
                ("users", "user_key", "user_id"),
                ("items", "item_key", "item_id"),
            ]:
                filename = ident.replace("_id", "_mapping") + ".csv"
                con.execute(
                    f"COPY (SELECT {key} AS original_id,{ident} FROM {table} ORDER BY {ident}) TO {sql_string(work / filename)} (HEADER)"
                )
                manifest["n_" + table] = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            files = ["train.csv", "val.csv", "test.csv", "user_mapping.csv", "item_mapping.csv"]
            manifest["files"] = {name: file_hash(work / name) for name in files}
            manifest["mapping_fingerprint"] = fingerprint(
                {name: manifest["files"][name] for name in files if "mapping" in name}
            )
            manifest["fingerprint"] = fingerprint(
                {
                    key: manifest[key]
                    for key in ["files", "preparation", "boundaries", "source_sha256"]
                }
            )
            destination.mkdir(parents=True, exist_ok=True)
            # Metadata is the completion marker and is written last.
            (destination / "metadata.json").unlink(missing_ok=True)
            for name in files:
                (work / name).replace(destination / name)
            write_json(destination / "metadata.json", manifest, overwrite=True)
            return manifest
        finally:
            con.close()
