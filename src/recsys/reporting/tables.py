"""Write machine-readable and human-readable tables of overall metrics."""

import csv
from pathlib import Path

_CSV_COLUMNS = [
    "group",
    "dataset",
    "model",
    "loss",
    "split",
    "fit_split",
    "metric",
    "mean",
    "lower",
    "upper",
    "n",
    "comparison",
    "protocol",
    "stage",
    "target_split",
    "diagnostic_only",
    "seeds",
    "n_queries",
]

_METRIC_LABELS = {"recall": "Recall", "ndcg": "NDCG", "coverage": "Coverage", "arp": "ARP"}


def _write_csv(path, panels):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for comparison_id, groups in panels.items():
            for group in groups:
                definition = group["definition"]
                context = {
                    "group": group["id"][:10],
                    "dataset": group["dataset"],
                    "model": group["model"],
                    "loss": group["loss"],
                    "split": group["split"],
                    "fit_split": group["fit_split"],
                    "comparison": comparison_id[:10],
                    "protocol": definition["protocol"]["name"],
                    "stage": group["stage"],
                    "target_split": definition["target_split"],
                    "diagnostic_only": definition.get("diagnostic_only", False),
                    "seeds": ",".join(map(str, group["seeds"])),
                    "n_queries": group.get("n_queries"),
                }
                for metric, stats in group["metrics"].items():
                    writer.writerow({**context, "metric": metric, **stats})


def _markdown_row(cells):
    values = [str(value).replace("|", r"\|").replace("\n", " ") for value in cells]
    return "| " + " | ".join(values) + " |"


def _metrics_by_cutoff(groups):
    # Comparison groups normally share a metric schema. Preserve its column order.
    metrics = dict.fromkeys(metric for group in groups for metric in group["metrics"])
    by_cutoff = {}
    for metric in metrics:
        _, separator, suffix = metric.rpartition("@")
        cutoff = int(suffix) if separator and suffix.isdecimal() else None
        by_cutoff.setdefault(cutoff, []).append(metric)
    return by_cutoff


def _format_stats(stats, seed_count):
    if stats is None or stats["mean"] is None:
        return "—"
    value = f"{stats['mean']:.5f}"
    if stats["lower"] is not None:
        value += f" [{stats['lower']:.5f}, {stats['upper']:.5f}]"
    if stats["n"] != seed_count:
        value += f" (n={stats['n']})"
    return value


def _comparison_lines(comparison_id, groups):
    first = groups[0]
    definition = first["definition"]
    protocol = definition["protocol"]["name"]
    context = [
        f"Comparison `{comparison_id[:10]}`",
        f"stage `{first['stage']}`",
        f"training targets `{definition['target_split']}`",
    ]
    if first.get("n_queries") is not None:
        context.append(f"{first['n_queries']} queries")
    if first.get("max_seq_length") is not None:
        context.append(f"max history length `{first['max_seq_length']}`")
    if definition.get("diagnostic_only"):
        context.append("diagnostic evaluation")
    lines = [
        f"## {first['dataset']} · {first['fit_split']} → {first['split']} · {protocol}",
        "",
        " · ".join(context),
    ]
    by_cutoff = _metrics_by_cutoff(groups)
    for cutoff in sorted(by_cutoff, key=lambda value: (value is None, value)):
        metrics = by_cutoff[cutoff]
        names = [metric.rpartition("@")[0] if cutoff is not None else metric for metric in metrics]
        labels = [_METRIC_LABELS.get(name, name) for name in names]
        columns = ["Model", "Loss", "Group", "Seeds", *labels]
        lines.extend(
            [
                "",
                f"### K = {cutoff}" if cutoff is not None else "### Metrics without a cutoff",
                "",
                _markdown_row(columns),
                _markdown_row(["---"] * 4 + ["---:"] * len(metrics)),
            ]
        )
        for group in groups:
            values = [
                _format_stats(group["metrics"].get(metric), len(group["seeds"]))
                for metric in metrics
            ]
            lines.append(
                _markdown_row(
                    [
                        group["model"],
                        group["loss"] or "—",
                        group["id"][:10],
                        ", ".join(map(str, group["seeds"])),
                        *values,
                    ]
                )
            )
    runs = [(group, run) for group in groups for run in group.get("runs", [])]
    if runs:
        lines.extend(
            [
                "",
                "### Runs",
                "",
                "| Model | Group | Seed | Run | Checkpoint epoch |",
                "|---|---|---:|---|---:|",
            ]
        )
        for group, run in runs:
            epoch = run.get("checkpoint_epoch")
            lines.append(
                _markdown_row(
                    [
                        group["model"],
                        group["id"][:10],
                        run["seed"],
                        run.get("run_id") or "—",
                        epoch if epoch is not None else "—",
                    ]
                )
            )
    return lines


def write_tables(output: Path, panels, confidence):
    """Write long CSV and per-comparison Markdown from the same aggregates."""
    _write_csv(output / "metrics.csv", panels)
    lines = [
        "# Overall metrics",
        "",
        "Overall metrics use the full evaluation set. Each cell averages a run-level metric "
        "over the available seeds.",
        f"Brackets show the {confidence * 100:g}% confidence interval when at least two seeds "
        "have a value. PNG and SVG charts show metrics by popularity bucket.",
    ]
    for comparison_id, groups in panels.items():
        lines.append("")
        lines.extend(_comparison_lines(comparison_id, groups))
    (output / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
