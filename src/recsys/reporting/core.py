"""Aggregate independently launched evaluations; confidence intervals are over seeds."""

from collections import defaultdict
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np
from scipy.stats import t

from recsys.artifacts import atomic_path, fingerprint, read_json, write_json
from recsys.reporting.tables import write_tables


def confidence_interval(values, confidence=0.95):
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie between 0 and 1")
    values = np.asarray([x for x in values if x is not None], dtype=float)
    n = len(values)
    if not n:
        return {"mean": None, "lower": None, "upper": None, "n": 0}
    mean = float(values.mean())
    if n == 1:
        return {"mean": mean, "lower": None, "upper": None, "n": 1}
    radius = float(t.ppf((1 + confidence) / 2, n - 1) * values.std(ddof=1) / np.sqrt(n))
    return {"mean": mean, "lower": mean - radius, "upper": mean + radius, "n": n}


def comparison_key(record):
    return {
        key: record[key]
        for key in [
            "dataset",
            "fit_fingerprint",
            "evaluation_fingerprint",
            "protocol",
            "fit_split",
            "target_split",
            "eval_split",
            "stage",
        ]
    } | {
        "catalog": record["catalog"]["fingerprint"],
        "buckets": record["buckets"]["fingerprint"],
        # A metric change yields a separate comparison even when the catalog and protocol match.
        "metric_names": sorted(record["metrics"]),
        "bucket_metric_names": {
            name: sorted(bucket["metrics"]) for name, bucket in record["buckets"]["groups"].items()
        },
        "diagnostic_only": record.get("diagnostic_only", False),
    }


def group_key(record):
    return comparison_key(record) | {
        "model": record["config"]["model"],
        "training": record["config"]["training"],
        "runtime": record.get("training_runtime", record["config"]["runtime"]),
    }


def _evaluation_content(record):
    # The resolved checkpoint path is stored separately. Training runtime is part of the group.
    config = {key: value for key, value in record["config"].items() if key != "output"}
    if "checkpoint" in record:
        config.pop("checkpoint", None)
    return {
        key: value for key, value in record.items() if key not in {"timing", "training_runtime"}
    } | {"config": config}


def aggregate(records, confidence=0.95):
    grouped = defaultdict(dict)
    for record in records:
        if record.get("schema_version") != 1 or "metrics" not in record or "buckets" not in record:
            raise ValueError("Input is not a version 1 evaluation JSON")
        identity = fingerprint(group_key(record))
        runs_by_seed = grouped[identity]
        seed = record["seed"]
        if seed in runs_by_seed:
            if _evaluation_content(runs_by_seed[seed]) == _evaluation_content(record):
                continue
            raise ValueError(
                f"Duplicate seed in result group {identity[:10]}; select one checkpoint per seed"
            )
        runs_by_seed[seed] = record
    output = []
    for identity, runs_by_seed in grouped.items():
        runs = runs_by_seed.values()
        first = next(iter(runs))
        metrics = {
            metric: confidence_interval([run["metrics"][metric] for run in runs], confidence)
            for metric in first["metrics"]
        }
        buckets = {}
        for name, group in first["buckets"]["groups"].items():
            buckets[name] = {
                "item_count": group["item_count"],
                "target_count": group["target_count"],
                "metrics": {
                    metric: confidence_interval(
                        [run["buckets"]["groups"][name]["metrics"][metric] for run in runs],
                        confidence,
                    )
                    for metric in group["metrics"]
                },
            }
        output.append(
            {
                "id": identity,
                "comparison_id": fingerprint(comparison_key(first)),
                "model": first["model"],
                "loss": first["loss"],
                "dataset": first["dataset"]["name"],
                "split": first["eval_split"],
                "stage": first["stage"],
                "fit_split": first["fit_split"],
                "seeds": sorted(runs_by_seed),
                "runs": sorted(
                    (
                        {
                            "seed": run["seed"],
                            "run_id": run.get("run_id"),
                            "checkpoint_epoch": run.get("checkpoint_epoch"),
                        }
                        for run in runs
                    ),
                    key=lambda run: run["seed"],
                ),
                "n_queries": first.get("support", {}).get("n_queries"),
                "max_seq_length": first["config"].get("max_seq_length"),
                "confidence": confidence,
                "definition": group_key(first),
                "metrics": metrics,
                "buckets": buckets,
            }
        )
    return output


def _plot_name(comparison_id, metric, suffix):
    name = f"{comparison_id[:10]}_{metric.replace('@', '_at_')}.{suffix}"
    if Path(name).name != name:
        raise ValueError(f"Invalid report plot filename: {name}")
    return name


def _bucket_metric_names(group):
    # A metric can be defined for only a subset of the popularity buckets.
    return sorted({metric for bucket in group["buckets"].values() for metric in bucket["metrics"]})


def _previous_report_files(output):
    """Use the saved aggregate to identify files generated by the previous report."""
    try:
        groups = read_json(output / "aggregate.json")
        if not isinstance(groups, list) or not groups:
            return set()
        plots = {
            _plot_name(group["comparison_id"], metric, suffix)
            for group in groups
            for metric in _bucket_metric_names(group)
            for suffix in ("png", "svg")
        }
    except (FileNotFoundError, ValueError, AttributeError, KeyError, TypeError):
        return set()
    return {"aggregate.json", "metrics.csv", "metrics.md"} | plots


def _render_report(output, groups, confidence):
    write_json(output / "aggregate.json", groups)
    panels = defaultdict(list)
    for group in groups:
        panels[group["comparison_id"]].append(group)
    write_tables(output, panels, confidence)
    # Matplotlib only imported when a user requests a report.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for comparison, models in panels.items():
        first = models[0]
        support = first["buckets"]
        names = list(support)
        for metric in _bucket_metric_names(first):
            fig, axis = plt.subplots(figsize=(max(7, len(models) * 1.5), 4.5))
            width = 0.8 / len(models)
            has_intervals = False
            missing = {"mean": None, "upper": None}
            for index, group in enumerate(models):
                stats = [group["buckets"][name]["metrics"].get(metric, missing) for name in names]
                values = [s["mean"] if s["mean"] is not None else np.nan for s in stats]
                errors = [s["upper"] - s["mean"] if s["upper"] is not None else 0 for s in stats]
                positions = np.arange(len(names)) - 0.4 + width * (index + 0.5)
                label = f"{group['model']}/{group['loss'] or 'analytic'} {group['id'][:6]} (n={len(group['seeds'])})"
                axis.bar(positions, values, width, label=label)
                known = [i for i, stat in enumerate(stats) if stat["upper"] is not None]
                if known:
                    has_intervals = True
                    axis.errorbar(
                        positions[known],
                        np.asarray(values)[known],
                        yerr=np.asarray(errors)[known],
                        fmt="none",
                        color="black",
                        capsize=3,
                    )
                for x, stat in zip(positions, stats):
                    if stat["mean"] is None:
                        axis.text(x, 0, "NA", ha="center", va="bottom", fontsize=8)
            axis.set_xticks(
                np.arange(len(names)),
                [
                    f"{name}\n{support[name]['item_count']} items\n{support[name]['target_count']} targets"
                    for name in names
                ],
            )
            axis.set_ylabel(metric)
            protocol = first["definition"]["protocol"]["name"]
            interval = f" · {confidence:.0%} CI" if has_intervals else ""
            axis.set_title(
                f"{first['dataset']} · {protocol} · {first['fit_split']} → {first['split']}\n"
                f"Popularity buckets{interval}"
            )
            axis.legend(fontsize=7)
            fig.tight_layout()
            try:
                for suffix in ("png", "svg"):
                    fig.savefig(output / _plot_name(comparison, metric, suffix), dpi=160)
            finally:
                plt.close(fig)


def report(input_dir, output_dir, confidence=0.95, overwrite=False):
    """Rebuild an aggregate report from the evaluation JSON currently in input_dir."""
    records = []
    for path in sorted(Path(input_dir).rglob("*.json")):
        value = read_json(path)
        if isinstance(value, dict) and "eval_split" in value and "metrics" in value:
            records.append(value)
    if not records:
        raise ValueError(f"No evaluation JSON in {input_dir}")
    groups = aggregate(records, confidence)
    output = Path(output_dir)
    previous = _previous_report_files(output)
    output.mkdir(parents=True, exist_ok=True)
    # Finish rendering before replacing any part of the report. Publish the index last.
    with TemporaryDirectory(dir=output, prefix=".report-") as directory:
        staging = Path(directory)
        _render_report(staging, groups, confidence)
        names = sorted(path.name for path in staging.iterdir() if path.name != "aggregate.json")
        names.append("aggregate.json")
        for name in names:
            path = output / name
            if path.exists() and not (overwrite or name in previous):
                raise FileExistsError(f"{path} exists; choose another path or set overwrite=true")
        for name in names:
            with atomic_path(output / name, overwrite or name in previous) as temporary:
                copyfile(staging / name, temporary)
    for name in previous.difference(names):
        path = output / name
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)
    return {
        "groups": len(groups),
        "evaluations": sum(len(group["runs"]) for group in groups),
        "output_dir": str(output.resolve()),
    }
