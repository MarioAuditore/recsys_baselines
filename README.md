# Recommender System Baselines

`recsys_baselines` provides a shared pipeline for next-item recommendation experiments. It prepares chronological data splits, fits models on the same interaction histories, ranks the active item catalog, and summarizes results across seeds. The experiment presets provide starting hyperparameters for each dataset and model.

| Model | Training preset suffix | Method |
| --- | --- | --- |
| SASRec | `sasrec` or `sasrec_sce` | Causal self-attention with full-catalog cross-entropy (CE) or Scalable Cross-Entropy (SCE) |
| LightGCN | `lightgcn` | Graph propagation with Bayesian Personalized Ranking (BPR) |
| UltraGCN | `ultragcn` | Graph-based recommendation with user–item and item–item constraints |
| Random | `random` | Seeded random item scores |
| MostPopular | `mostpopular` | Item scores based on interaction counts |
| ItemKNN | `itemknn` | Item–item nearest neighbors |
| PureSVD | `puresvd` | Truncated SVD of the user–item matrix |

## Installation

Use Python 3.11 or newer. Run the examples from the repository root. For an installation using the committed lockfile, use `uv`.

```bash
uv sync --locked --extra dev
source .venv/bin/activate
```

Alternatively, create a virtual environment and install the project with pip.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The `dev` extra installs pytest and Ruff. For the experiment CLI alone, omit `--extra dev` from the uv command or use `python -m pip install -e .`.

| Runtime | Use |
| --- | --- |
| `runtime=cpu` | Default runtime for training and evaluation |
| `runtime=cpu_smoke` | Short CPU runs with two epochs, up to three batches per epoch, batch size 8, and a maximum history of 16 interactions |
| `runtime=mps` | Apple Silicon GPU. SASRec CE/SCE, LightGCN, and UltraGCN were smoke-tested on an M1 Pro with PyTorch 2.15. LightGCN needs sparse matrix multiplication and backward support on MPS. |
| `runtime=cuda` | NVIDIA GPU with a compatible PyTorch build and driver. Set `training.amp=true` to enable CUDA mixed precision. |

Random, MostPopular, ItemKNN, and PureSVD fit and score on CPU. Their evaluation records identify the effective device. Reports are also generated on CPU.

## Quickstart on MovieLens 1M

This run fits SASRec on a deterministic sample of 512 users, evaluates its best checkpoint on validation and test, and generates a report. The short run is suitable for checking installation and the pipeline.

```bash
python -m recsys.download -cn ml1m.dataset

python -m recsys.prepare -cn ml1m.dataset \
  dataset.path=data/global_temporal_split/ml1m_quickstart \
  preparation.sample_users=512

python -m recsys.train -cn ml1m.sasrec runtime=cpu_smoke \
  dataset.path=data/global_temporal_split/ml1m_quickstart \
  run_name=ml1m_sasrec_quickstart_seed42

python -m recsys.evaluate \
  checkpoint=outputs/ml1m_sasrec_quickstart_seed42/best.pt \
  split=val output.path=results/ml1m_quickstart/sasrec_val_seed42.json

python -m recsys.evaluate \
  checkpoint=outputs/ml1m_sasrec_quickstart_seed42/best.pt \
  split=test output.path=results/ml1m_quickstart/sasrec_test_seed42.json

python -m recsys.report \
  reporting.input_dir=results/ml1m_quickstart \
  reporting.output_dir=reports/ml1m_quickstart
```

The report starts at `reports/ml1m_quickstart/metrics.md`. If you already have `ml-1m.zip`, start at `prepare` and set `dataset.input_path=/absolute/path/to/ml-1m.zip` on that command. The default archive path is `data/raw/ml1m/ml-1m.zip`.

## Datasets and preparation

| Dataset prefix | Source |
| --- | --- |
| `ml1m` | MovieLens 1M |
| `amazon_beauty` | Amazon Reviews 2023 Beauty and Personal Care |
| `amazon_clothing` | Amazon Reviews 2023 Clothing, Shoes and Jewelry |
| `amazon_sports` | Amazon Reviews 2023 Sports and Outdoors |
| `yambda50m_listens` | Yambda 50M listens |
| `yambda50m_likes` | Yambda 50M likes |

Select `<dataset>.dataset` for `download` and `prepare`. `download` retrieves an upstream source file and records its checksum. `prepare` reads the local source, applies signal and activity filters, and writes `train.csv`, `val.csv`, `test.csv`, `metadata.json`, and user and item mappings under `dataset.path`. For full MovieLens 1M, use the default path and omit user sampling. If the full split already exists, proceed directly to training. Use a new `dataset.path` to try a different preparation recipe.

```bash
python -m recsys.prepare -cn ml1m.dataset
```

By default, rated datasets keep ratings of at least 4 and Yambda listens keep events with more than 50% played. Adjust `preparation.min_rating` or `preparation.listen_threshold`, or set a threshold to `null` to disable it. Use `preparation.listen_operator=ge` for an inclusive listen threshold. Existing prepared splits can be reused by training and evaluation.

For a custom CSV, use the [generic dataset preset](configs/datasets/dataset.yaml). The CSV requires `timestamp,item_id,user_id` and can include `rating`. Set `dataset.timestamp_scale=1000` for seconds or `1` for milliseconds.

```bash
python -m recsys.prepare -cn datasets.dataset dataset.name=my_dataset \
  dataset.input_path=/absolute/path/to/events.csv \
  dataset.timestamp_scale=1000 preparation.min_rating=null

python -m recsys.train -cn datasets.dataset dataset.name=my_dataset model=sasrec
```

The default **global temporal split** uses 80/10/10 event quantiles after filtering and keeps equal timestamps in the same split. Set `preparation.boundaries` to two explicit timestamps in milliseconds to fix the cutoffs. The mappings come from the training split. Validation and test events with unknown users or items are excluded and counted in metadata. The prepared CSVs preserve repeated events and store integer millisecond timestamps and zero-based IDs.

## Training and checkpoint selection

Select a dataset and model with `-cn <dataset>.<model>`. For example, `ml1m.lightgcn` loads `configs/datasets/ml1m/lightgcn.yaml`, and `amazon_beauty.sasrec` loads `configs/datasets/amazon_beauty/sasrec.yaml`. With no preset, the CLI loads [the default Hydra config](configs/config.yaml).

```bash
python -m recsys.train -cn ml1m.sasrec

python -m recsys.train -cn ml1m.sasrec_sce runtime=cpu \
  seed=42 run_name=sasrec_sce_ml1m_seed42

python -m recsys.train -cn ml1m.lightgcn runtime=cpu seed=42
```

`--cfg job --resolve` prints the fully resolved config before launching a run. Add `--package training` to print only the active training settings.

```bash
python -m recsys.train -cn ml1m.sasrec runtime=mps --cfg job --resolve
```

### Where to edit hyperparameters

The dataset folders are grouped under [`configs/datasets/`](configs/datasets/). Here are the MovieLens 1M source, training, and SASRec search presets.

```text
configs/datasets/ml1m/
  dataset.yaml             # Source and prepared dataset paths
  sasrec.yaml              # Fixed model and training settings
  sasrec_sce.yaml          # Fixed settings for the SCE variant
  tuning/
    sasrec.yaml            # TPE search space
    sasrec_grid.yaml       # Finite grid search space
```

Edit `model` and `training` in the [fixed SASRec preset](configs/datasets/ml1m/sasrec.yaml) to save the values you want for future runs. For an individual run, pass `key=value` overrides to the CLI. A CUDA training run can set the batch size to 1024 or higher, subject to available memory. `runtime=cpu_smoke` sets a smaller batch size for local checks.

```bash
python -m recsys.train -cn ml1m.sasrec runtime=cuda \
  training.batch_size=1024 run_name=sasrec_ml1m_cuda_bs1024_seed42
```

### Training budget and outputs

The training runner builds an effective history from the last `max_seq_length` fit interactions per user and keeps users with at least `min_history` interactions. All models use this history to define the active catalog. For a full training run, use a regular runtime and leave `training.max_batches=null` so each epoch can use all batches. The base config allows up to 100 epochs. Analytic baselines fit once.

With the default `fit_split=train` and `training.validate=null`, training evaluates validation after each epoch. It selects the checkpoint with the highest `evaluation.selection_metric`, which defaults to `ndcg@20`. The default patience is 10 epochs without improvement. The validation journal records the loss, optimizer step count, metrics, and query counts for each epoch.

| File in `output.dir` | Contents |
| --- | --- |
| `config.yaml` | Resolved configuration |
| `best.pt` | Checkpoint selected by validation, or the latest epoch when validation is disabled |
| `last.pt` | Latest checkpoint, including state for resuming neural training |
| `validation.jsonl` | Per-epoch journal |
| `summary.json` | Selected epoch, validation score, and checkpoint paths |

The default run directory is `outputs/<run_name>`. The default name includes the model, dataset, and seed. Use a distinct `run_name` or `output.dir` for changed settings. Existing checkpoints are protected unless `output.overwrite=true` is set. Load checkpoints only from sources you trust.

### Resume, fine-tune, or refit

Resume a neural run from its latest checkpoint. `training.epochs` is the new total epoch budget. Resume uses the original directory, data, training settings, and runtime.

```bash
python -m recsys.resume checkpoint=outputs/sasrec_ml1m_seed42/last.pt \
  training.epochs=150
```

Fine-tuning starts a new neural run from saved weights with a new optimizer. By default it uses train and validation events for history and trains on validation targets. The model configuration and item/user mappings must match the checkpoint.

```bash
python -m recsys.finetune checkpoint=outputs/sasrec_ml1m_seed42/best.pt \
  run_name=sasrec_ml1m_finetune_seed42 \
  training.learning_rate=0.0001 training.epochs=10
```

To fit a fresh model on the combined train and validation split, use `fit_split=train_val` and a fixed epoch budget. Training-time validation is disabled because those events are now part of the fit data. A later evaluation of that same validation split is marked diagnostic. Keep the test split held out.

```bash
python -m recsys.train -cn ml1m.sasrec runtime=cpu fit_split=train_val \
  training.epochs=30 run_name=sasrec_ml1m_refit_seed42
```

## Evaluation and metrics

`evaluate` reloads a checkpoint and writes one JSON result per split. It inherits the saved model, dataset, history, and runtime settings. You can override the evaluation runtime with `runtime=cuda`, `runtime=mps`, or `runtime=cpu`. The default split is `test` and the default output path is `results/<dataset>/<run_name>_<split>.json`.

```bash
python -m recsys.evaluate checkpoint=outputs/sasrec_ml1m_seed42/best.pt split=val
python -m recsys.evaluate checkpoint=outputs/sasrec_ml1m_seed42/best.pt split=test
```

On CUDA or MPS, neural models encode users and score candidates on the GPU. The evaluator transfers scores back to CPU for exact top-K ranking and metric computation. For larger evaluations, adjust `evaluation.user_batch_size` and `evaluation.item_block_size` to balance scoring throughput and memory use. These settings apply independently of the checkpoint’s training batch size.

```bash
python -m recsys.evaluate checkpoint=outputs/sasrec_ml1m_cuda_bs1024_seed42/best.pt \
  runtime=cuda split=test evaluation.user_batch_size=256 evaluation.item_block_size=8192
```

The default `next_item` protocol selects the first eligible positive event per user in the chosen holdout window and ranks the entire active catalog. Previously seen items remain eligible. History stays fixed to the fit split, so train-only test evaluation skips the intervening validation events. `evaluation.protocol=next_new_item` instead excludes previously seen targets and masks seen items using the full fit history. Check the query counts and exclusions in the JSON before comparing results.

Ranking cutoffs come from `evaluation.ks` in the [default evaluation config](configs/evaluation/default.yaml). To override them on the CLI, use an argument such as `'evaluation.ks=[10,20,50]'`. During training and tuning, include the cutoff used by `evaluation.selection_metric`.

| Metric | Interpretation |
| --- | --- |
| Recall@K | Fraction of queries whose single target appears in the top K, equivalent to hit rate |
| NDCG@K | Average rank-discounted gain, with zero for a miss and `1/log2(rank+1)` for a hit |
| Coverage@K | Fraction of the active catalog recommended at least once |
| ARP@K | Average Recommendation Popularity, measured as mean training interaction count of recommended items per query |

Popularity breakdowns use head, mid, and tail buckets with approximately equal training interaction mass. Bucket Recall and NDCG group queries by the target item. Bucket coverage, ARP, and exposure share describe the items actually recommended. Undefined metrics appear as `null`.

## Reporting

`report` aggregates the evaluation JSON files under `reporting.input_dir`. Point it at the directory for the experiment you want to compare.

```bash
python -m recsys.report reporting.input_dir=results/ml1m \
  reporting.output_dir=reports/ml1m
```

The report writes `metrics.md`, `metrics.csv`, `aggregate.json`, and PNG and SVG plots of the popularity breakdowns. It keeps incompatible datasets, histories, evaluation protocols, and splits in separate comparisons. Within a model configuration, compatible runs with different seeds contribute to the mean and, with at least two values, a Student’s t confidence interval. The interval describes variation across seeds and defaults to 95%. Set `reporting.confidence` to change it. Identical evaluations of the same checkpoint and seed count once. Conflicting results for the same seed in a group require selecting one result.

## Hyperparameter tuning with Optuna

Select a search preset with `-cn <dataset>.tuning.<search>`. Each search preset starts from a fixed training preset and defines the parameters a trial may vary. `tune` fits each trial on the training split and maximizes its best validation score for `evaluation.selection_metric`. Reserve the test split for evaluating configurations selected on validation.

Shared sampler and study defaults live in [`configs/optuna/`](configs/optuna/). Search spaces live in each dataset’s `tuning/` folder. Use `tuning.*` for CLI overrides such as `tuning.trials=30`.

The MovieLens 1M SASRec grid tries four combinations of learning rate and dropout.

```bash
python -m recsys.tune -cn ml1m.tuning.sasrec_grid
```

Use the TPE preset for a search over broader ranges.

```bash
python -m recsys.tune -cn ml1m.tuning.sasrec runtime=cpu \
  training.epochs=30 tuning.trials=20 \
  tuning.space.model.dropout.high=0.4 \
  tuning.study_name=ml1m_sasrec_tpe output.dir=outputs/ml1m_sasrec_search
```

- **TPE** (Tree-structured Parzen Estimator) uses previous validation scores to propose new hyperparameters after its startup trials. The bundled TPE settings enable median pruning and allow 20 trials per invocation.
- **Search spaces** live under `tuning.space.model` and `tuning.space.training` in the search presets. A `categorical` field takes `choices`. An `int` or `float` field takes `low` and `high`, with `log: true` for logarithmic sampling. See the [MovieLens 1M SASRec search](configs/datasets/ml1m/tuning/sasrec.yaml) for an example.
- **Grid search** uses explicit categorical choices for every parameter. Random and MostPopular have no hyperparameter search presets. See the [SASRec grid](configs/datasets/ml1m/tuning/sasrec_grid.yaml).

Override the search space when tuning on larger GPUs. To hold batch size fixed during the study, give it a single choice such as `[1024]`.

```bash
python -m recsys.tune -cn ml1m.tuning.sasrec runtime=cuda \
  'tuning.space.training.batch_size.choices=[512,1024,2048]'
```

Studies use `sqlite:///outputs/optuna.db` by default. Repeating the same study with the same settings and storage continues it, with `tuning.trials` controlling the budget of the new invocation. Use a distinct `tuning.study_name` and output path for another experiment, or `tuning.storage=null` for a fresh in-memory study.

Each study writes trial directories, `trials.csv`, `study.json`, and a resolved `best.yaml` under `<output.dir>/tuning/<study_name>`. The CSV lists each trial’s parameters, score, status, and any recorded failure. Copy the winning `model` and `training` values you want from `best.yaml` into your fixed training preset. The CLI also prints the absolute path to `best.yaml`, which can be used directly to train a fresh checkpoint.

```bash
python -m recsys.train \
  --config-path "$PWD/outputs/ml1m_sasrec_search/tuning/ml1m_sasrec_tpe" \
  --config-name best seed=43 run_name=sasrec_ml1m_tuned_seed43 \
  output.dir=outputs/sasrec_ml1m_tuned_seed43

python -m recsys.evaluate \
  checkpoint=outputs/sasrec_ml1m_tuned_seed43/best.pt split=test
```

In the resolved `best.yaml`, override fields directly. For example, use `runtime.device=mps` to change the device.

## Interpreting results

Compare models on the same split, effective history, ranking protocol, and candidate catalog. The quickstart and batch-capped runs check that the pipeline works. For quality comparisons, run full epochs and report results across independent seeds. Full CE can require substantial memory as batch size, sequence length, and catalog size grow. With finite history length `L`, the current SASRec training batches use at most `L-1` input positions, while evaluation can use all `L`. On histories of length `L`, evaluation can therefore use a positional embedding that received no training updates.

## Development and license

Run the same CPU checks used in CI.

```bash
python -m pytest -q
ruff check src tests
ruff format --check src tests
```

The tests run on CPU with synthetic fixtures. [GitHub Actions](.github/workflows/tests.yml) also builds and checks the installed wheel outside the checkout.

The CLI uses [Hydra configs](configs/) and delegates to the Python APIs in [`src/recsys`](src/recsys/). See the [project license](LICENSE) and [third-party attribution](THIRD_PARTY.md).
