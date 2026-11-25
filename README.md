# recsys_baselines

This repository contains baselines and pipelines, which may form as basis for research projects in recommendation systems.

Небольшой исследовательский репозиторий для **next-item recommendation**: SASRec (CE/SCE), LightGCN, UltraGCN, Random, MostPopular, ItemKNN и PureSVD. Общие CSV, история взаимодействий, evaluator и результаты позволяют сравнивать модели без отдельных пайплайнов под каждый алгоритм.

Реализованы независимые команды `download`, `prepare`, `train`, `resume`, `finetune`, `evaluate`, `tune`, `report`. Первые проверки выполнены на CPU и MovieLens-1M. Конфиги — стартовые, **не результаты полноценного тюнинга**. См. [проверки и ограничения](docs/VALIDATION.md).

## Установка

Python 3.11 или новее. Из корня репозитория:

```bash
uv sync --extra dev
source .venv/bin/activate
```

`uv.lock` фиксирует зависимости. Альтернатива без uv:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Для CUDA нужна совместимая с GPU версия PyTorch. `runtime=cuda` выбирает устройство. На Apple Silicon `runtime=mps` проверен для SASRec (CE/SCE), LightGCN и UltraGCN на Apple M1 Pro с PyTorch 2.15. LightGCN требует поддержки разреженного матричного умножения и обратного прохода на MPS в установленной версии PyTorch. Для быстрой проверки на CPU используйте `runtime=cpu_smoke`.

## Первый запуск: ML-1M

Команды выполняются отдельно. Никакая из них не запускает автоматически следующий этап.

```bash
python -m recsys.download dataset=ml1m

python -m recsys.prepare dataset=ml1m \
  dataset.path=data/global_temporal_split/ml1m_smoke \
  preparation.sample_users=512

python -m recsys.train -cn ml1m_sasrec runtime=cpu_smoke \
  dataset.path=data/global_temporal_split/ml1m_smoke \
  run_name=sasrec_ml1m_quickstart_seed42

python -m recsys.evaluate \
  checkpoint=outputs/sasrec_ml1m_quickstart_seed42/best.pt \
  split=val output.path=results/ml1m_smoke/sasrec_val_seed42.json

python -m recsys.evaluate \
  checkpoint=outputs/sasrec_ml1m_quickstart_seed42/best.pt \
  split=test output.path=results/ml1m_smoke/sasrec_test_seed42.json

python -m recsys.report reporting.input_dir=results/ml1m_smoke \
  reporting.output_dir=reports/ml1m_smoke
```

`cpu_smoke` — две эпохи, максимум три batch на эпоху, batch size 8, история 16 событий. Такой запуск проверяет код, а не качество модели. `sample_users=512` выбирает пользователей детерминированно по хешу исходного ID; это отдельный срез, явно записанный в metadata.

Для полного ML-1M подготовить данные без `preparation.sample_users` в отдельный каталог и выбрать обычный `runtime=cpu` или `runtime=cuda`.

## Данные

```text
data/
  raw/<dataset>/...
  global_temporal_split/<dataset>/
    train.csv
    val.csv
    test.csv
    metadata.json
    user_mapping.csv
    item_mapping.csv
```

Формат событий строго:

```csv
timestamp,item_id,user_id
978300760000,0,0
978302109000,3,0
```

- ID от нуля, один mapping на все части; padding SASRec равен `n_items` и в CSV не записывается.
- Все повторы сохраняются, включая одинаковые строки. Дополнительной дедупликации нет.
- Время — целое число миллисекунд. MovieLens/Amazon используют Unix time, Yambda — относительное время источника. Исходная точность не повышается.
- CSV отсортирован по пользователю и времени, для ties сохраняется исходный порядок.
- `metadata.json` хранит пороги, границы, источник, checksum, размеры частей, исключения, длины train-историй и счётчики повторов.
- Split CSV содержат полные временные части. Последние L событий выбираются общим слоем при запуске модели. SASRec получает из них до L−1 обучающих переходов; графы и остальные модели используют тот же набор событий.
- Популярность считается по событиям этого набора с учётом повторов. Бинарный граф/матрица объединяет события в рёбра уже на уровне представления модели.

Доступные `dataset`:

| Имя | Источник |
|---|---|
| `ml1m` | MovieLens-1M |
| `amazon_beauty` | Amazon Reviews 2023 / Beauty_and_Personal_Care |
| `amazon_clothing` | Amazon Reviews 2023 / Clothing_Shoes_and_Jewelry |
| `amazon_sports` | Amazon Reviews 2023 / Sports_and_Outdoors |
| `yambda50m_listens` | Yambda 50M / listens |
| `yambda50m_likes` | Yambda 50M / likes |
| `csv` | Собственный источник с `timestamp,item_id,user_id` и необязательными `rating`, `played_ratio_pct` |

```bash
python -m recsys.download dataset=amazon_beauty
python -m recsys.prepare dataset=amazon_beauty preparation.min_rating=4

python -m recsys.download dataset=yambda50m_listens
python -m recsys.prepare dataset=yambda50m_listens \
  preparation.listen_threshold=50 preparation.listen_operator=gt
```

Amazon скачивается как raw JSONL.gz, используется `parent_asin`; для Yambda исходный Parquet только читается адаптером, итоговые файлы — CSV. Смена порога не требует нового скачивания. Для другого размера Yambda переопределить `dataset.size`, `dataset.name` и путь исходника/результата, например имя `yambda500m_listens`, чтобы не смешивать разные варианты.

По умолчанию рейтинги `>=4`, прослушивания `>50%`. Можно отключить порог значением `null`. `listen_operator=ge` означает включённую границу. В лайках реконструкция состояния через unlike не выполняется. Пороги активности задаются `preparation.min_user_events` и `preparation.min_item_events`, итеративная фильтрация включается явно через `preparation.iterative_filter=true`.

Собственный CSV:

```bash
python -m recsys.prepare dataset=csv dataset.name=my_dataset \
  dataset.input_path=/absolute/path/events.csv \
  dataset.timestamp_scale=1000 preparation.min_rating=null
```

`timestamp_scale` — множитель перевода исходного целого времени в миллисекунды. Для уже готовых миллисекунд он равен 1.

## Временной протокол

По умолчанию GTS 80/10/10 по глобальным квантилям событий **после порога сигнала**. Равные timestamps не разбиваются. Для фиксированного эксперимента задайте две границы в миллисекундах:

```bash
python -m recsys.prepare dataset=ml1m \
  'preparation.boundaries=[975000000000,978000000000]'
```

Mapping строится по train. Неизвестные train пользователи и объекты исключаются из val/test, счётчики сохраняются. Минимальная эффективная train-история для общего сравнения — два события. После ограничения L объекты/пользователи могут стать неактивными: они также исключаются из кандидатов и оценки, без перенумерации внешнего mapping.

Основной `evaluation.protocol=next_item`:

- один запрос на пользователя в выбранном val/test-окне;
- цель — первое допустимое положительное событие в этом окне;
- повторные объекты разрешены, seen-каталог не маскируется;
- история фиксирована выбранной fit-частью; val не добавляется к test-контексту автоматически;
- при train-only это прогноз первого события в test-окне по train-истории, с пропущенным промежуточным val-окном;
- после отдельного train на train+val используется история объединённой части;
- точный top-K по всему активному каталогу, scores считаются блоками; ties разрешаются по item ID.

`evaluation.protocol=next_new_item` — отдельная задача: повторные цели исключаются, ранее встречавшиеся объекты маскируются по всей доступной fit-истории, включая события за пределами L. Результаты разных протоколов не объединяются.

## Конфигурации моделей

```text
configs/
  beauty_sasrec.yaml        # конфиг пары датасет × модель
  ml1m_lightgcn.yaml
  base.yaml                 # общие параметры, датасет, модель, оценка
  execution.yaml            # профили среды запуска и Optuna
  config.yaml               # конфиг по умолчанию
  dataset/
  model/
  evaluation/
  runtime/
  tuning/
```

Файл `configs/<dataset>_<model>.yaml` наследует общие настройки, выбирает датасет и модель и задаёт переопределения пары вместе с её пространством поиска. Выберите файл через `-cn` или `--config-name`, указав имя без `.yaml`. Без этого аргумента загружается `configs/config.yaml`. Префиксы `beauty`, `clothing` и `sports` соответствуют датасетам `amazon_beauty`, `amazon_clothing` и `amazon_sports`.

Полностью разрешённый конфиг сохраняется в каталоге запуска. Его можно посмотреть до обучения.

```bash
python -m recsys.train -cn beauty_sasrec training.batch_size=4 --cfg job --resolve
```

Параметры можно переопределить через `ключ=значение` после имени конфига.

```bash
python -m recsys.train -cn ml1m_lightgcn runtime=cuda seed=42
python -m recsys.train -cn ml1m_ultragcn runtime=cuda seed=42
python -m recsys.train -cn ml1m_puresvd seed=42
python -m recsys.train -cn ml1m_itemknn model.neighbors=50

python -m recsys.train -cn ml1m_sasrec runtime=cuda \
  model.loss.name=sce model.loss.n_buckets=8 \
  model.loss.bucket_size_x=64 model.loss.bucket_size_y=256 \
  run_name=sasrec_sce_ml1m_seed42
```

SASRec: `ce|sce`; LightGCN: `bpr`; UltraGCN: `ultragcn` с собственными коэффициентами. У аналитических моделей `loss=null`. Случайные отрицательные объекты графовых моделей берутся из активного каталога и исключают известные положительные пары. Настройки `model.sampling`: число negatives, `uniform|popularity`, степень популярности. Пользователи, заполнившие весь активный каталог, не дают pairwise train-примеров.

Полная CE может требовать много памяти: batch size, длина истории и каталог перемножаются. Для больших датасетов стартовые batch size уменьшены, но ещё не профилировались на целевой GPU. SCE использует алгоритм проекционных групп AIRI; это не случайный sampled softmax. [Источники и лицензии](THIRD_PARTY.md).

## Отдельные режимы

**Продолжение того же запуска:** восстанавливаются optimizer, scheduler, RNG и счётчики. `training.epochs` — новое общее число эпох.

```bash
python -m recsys.resume checkpoint=outputs/my_run/last.pt training.epochs=150
```

Resume требует `last.pt`, прежние данные, модель, протокол и runtime; записи продолжаются в исходном каталоге. Это предотвращает добавление повторных эпох в журнал при возврате к более раннему `best.pt`. Для новой ветки обучения используйте finetune.

**Дообучение:** новый optimizer и новый каталог. По умолчанию история строится по train+val, а цели loss выбираются только из val; test не используется.

```bash
python -m recsys.finetune checkpoint=outputs/my_run/best.pt \
  training.learning_rate=0.0001 training.epochs=10 \
  output.dir=outputs/my_finetune run_name=my_finetune
```

Для дообучения по всем событиям fit-части — `target_split=all`. Граф при finetune тоже перестраивается по явно выбранному `fit_split`. Поддерживаются нейросетевые модели; аналитическим бейзлайнам нужен новый `train`. Mapping и конфигурация модели/loss должны совпадать с checkpoint; расширение словаря автоматически не выполняется.

**Обучение с нуля на train+val:** отдельный вызов, без early stopping по уже включённой val.

```bash
python -m recsys.train -cn ml1m_sasrec runtime=cuda \
  fit_split=train_val training.epochs=30 run_name=sasrec_ml1m_refit_seed42
```

При `training.validate=null` validation включена только для train-only. Явное `training.validate=true` при train_val отклоняется. Mapping подготовленного датасета остаётся прежним; отфильтрованные при prepare холодные события не восстанавливаются.

## Результаты и метрики

`output.dir` определяет каталог обучения, `output.path` — файл самостоятельной оценки. `evaluate` по умолчанию использует `split=test`. Если повторный расчёт даёт тот же результат, команда возвращает сохранённый JSON. Для записи отличающегося результата в существующий файл задайте `output.overwrite=true` или выберите другой `output.path`.

Checkpoints `best.pt` и `last.pt` предназначены соответственно для оценки и resume. Загружайте только свои доверенные checkpoints: они содержат сериализованное состояние Python/PyTorch.

`evaluate` сохраняет в JSON seed, checkpoint и эпоху, stage, fit/eval split, полный конфиг, fingerprints данных, истории и каталога, support, общие и bucket-метрики. При обучении короткая полоса прогресса показывает время, средний loss и validation NDCG на текущей эпохе с точностью до пяти знаков после точки. `validation.jsonl` хранит значения по эпохам с полной точностью. Диагностическая оценка val после обучения на train_val явно помечается.

Метрики при одной цели:

- Recall@K = HitRate@K;
- NDCG@K = `1/log2(rank+1)` при попадании;
- Coverage@K — доля активного каталога, встретившаяся в рекомендациях;
- ARP@K — средняя популярность рекомендаций, сначала по запросу, затем по запросам;
- buckets head/mid/tail делятся примерно по трети суммы train-взаимодействий, а не по числу объектов.

Bucket Recall/NDCG группируются по **целевому объекту**, позиции сохраняются из общего top-K. Bucket Coverage нормируется на размер группы. Bucket ARP усредняется по показам группы. ExposureShare — доля рекомендательных позиций группы. Отсутствующие значения — `null`, не ноль.

`report` строит CSV/Markdown-таблицы, PNG/SVG-графики и `aggregate.json`. В Markdown общие метрики собраны по протоколу и K с одной строкой на конфигурацию модели. CSV хранит по строке на метрику и конфигурацию. Графики показывают разбивку по бакетам популярности. Повторный запуск в тот же `reporting.output_dir` пересобирает отчёт по текущим результатам и обновляет файлы предыдущего отчёта.

Совпадающие результаты повторных оценок одного checkpoint и seed учитываются один раз. При конфликте между результатами одного seed в одной группе отчёт требует выбрать один результат.

Совместимые независимые запуски объединяются в среднее и 95% t-интервал. В отчёте указаны значения seed, а в CSV `n` — число доступных значений для конкретной метрики. Для одного значения CI отсутствует. Дубли одного seed отклоняются. Разные данные, L, протоколы, fit-части, loss и гиперпараметры не смешиваются. Интервалы отражают вариативность обучения, не bootstrap пользователей.

## Optuna

После подготовки Amazon Beauty запустите подбор для SASRec.

```bash
python -m recsys.tune -cn beauty_sasrec runtime=cuda \
  tuning.trials=30 tuning.space.model.dropout.high=0.4
```

Редактируйте `tuning.space.model` и `tuning.space.training` в конфиге пары. Тип `categorical` принимает список `choices`, а `int` и `float` принимают `low` и `high` с опциональным `log: true`. Например, `tuning.space.model.dropout.high=0.4` меняет верхнюю границу через CLI. Диапазоны стартовые. Для Random и MostPopular пространство поиска пустое. Для подбора с `config.yaml` задайте своё пространство поиска. [Конфиг Beauty × SASRec](configs/beauty_sasrec.yaml).

В истории Optuna имена параметров записываются как `model.dropout` и `training.learning_rate`. Сохранённые конфиги с такими плоскими ключами в `tuning.space` поддерживаются.

По умолчанию используется TPE. Для `tuning.sampler=grid` задайте все параметры как `categorical` с явными списками значений. Профиль `tuning=sasrec_grid` по-прежнему запускает двухпараметрическую решётку для SASRec.

Study сохраняется в SQLite и продолжается при повторном вызове с теми же настройками исследования. Для независимых исследований одного конфига в разных SQLite-базах задавайте разные `output.dir`. При `tuning.storage=null` каждый запуск создаёт новое исследование и хранит его файлы в отдельном каталоге. `tuning.trials` задаёт бюджет текущего вызова. История, протокол оценки, модель и тип loss фиксируются на весь поиск. Optuna выбирает параметры по метрике validation через общий код обучения.

Таблица `trials.csv` в каталоге исследования содержит состояние, метрику, гиперпараметры, ошибки и путь к сохранённому конфигу каждого trial. Там же лежит сводка `study.json`. Лучший конфиг экспортируется в самостоятельный `best.yaml` для обычного train.

```bash
python -m recsys.train --config-path /absolute/path/to/study --config-name best \
  seed=43 run_name=tuned_seed43 output.dir=outputs/tuned_seed43
```

При загрузке полностью разрешённого `best.yaml` меняйте обычные поля, например `runtime.device=cuda`, а не группы defaults. Полноценный тюнинг и финальные seed-запуски выполняются отдельно пользователем.

## Импорты и расширение

```python
from recsys.config import compose
from recsys.data import PreparedDataset
from recsys.evaluation import evaluate_model
from recsys.models import build_model
from recsys.training import run_training

config = compose(["runtime=cpu_smoke"], config_name="ml1m_sasrec")
summary = run_training(config)
```

Ядро принимает обычные словари, а каждый CLI-скрипт явно вызывает свою операцию. Hydra разрешает YAML и CLI-переопределения. `History` объединяет события, последовательности и разреженную матрицу. Инструкции по добавлению модели, лосса, метрики и источника данных описаны в руководстве. [EXTENDING.md](docs/EXTENDING.md)

## Проверки

```bash
pytest -q
ruff check src tests
ruff format --check src tests
```

Тесты используют синтетические источники, не скачивают данные и не требуют GPU. Проверяются ручные значения метрик, повторные цели, время/mapping, CE/SCE, sparse constraints, все модели, checkpoint/resume, finetune, CLI, Optuna и отчёты.

- [Техническое задание](docs/TECHNICAL_SPEC.md)
- [План и текущий статус](docs/IMPLEMENTATION_PLAN.md)
- [Выполненные проверки и ограничения](docs/VALIDATION.md)
