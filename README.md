# Universal Agentic Data Pipeline

## Обзор проекта

Этот репозиторий содержит **offline-first agentic data pipeline** для работы с текстовыми отзывами.

Проект сознательно позиционируется не как production-ready система, а как **воспроизводимый research/demo pipeline**. Он предназначен для стабильного локального запуска, демонстрации на защите, пошагового анализа качества данных и контролируемого расширения без разрушения базового сценария.

---

## Архитектурная позиция проекта

- Проект использует **offline-first стратегию**: offline-режим является основным стабильным baseline.
- **Offline-режим** — это базовый сценарий для локального воспроизводимого запуска, демо и курсовой защиты.
- **Online-режим** — это расширение baseline, а не его замена.
- Offline и online — это **две осознанные стратегии использования одного и того же pipeline**, а не конфликтующие ветки проекта.
- **Human-in-the-loop (HITL)** встроен в архитектуру как обязательный этап review после annotation.
- В pipeline есть **training stage и ML-компонент под капотом**; ML-контур является частью архитектуры, а не внешней надстройкой.
- Проект является **agentic pipeline**, потому что состоит из последовательных этапов, каждый из которых решает свою подзадачу и передаёт результат следующему шагу.

---

## Этапы pipeline

- Source discovery
- Data collection
- Data quality checks
- Annotation
- Human review queue export and merge
- Active learning
- Training
- Reporting

Агентная цепочка в проекте:

`discovery -> collection -> quality -> annotation -> review/HITL -> active learning/training -> reporting`

Каждый этап отвечает за свой участок обработки и передаёт результат дальше по pipeline.

Основная обучающая постановка в текущем baseline: `text -> effect_label`.

---

## Режимы discovery и запуска

Сейчас в проекте поддерживаются два режима работы:

- **Offline demo mode** для стабильных demo-конфигов fitness и minecraft. Это детерминированный и воспроизводимый baseline.
- **Online mode** для non-demo конфигов. Этот режим расширяет baseline и включает узкий discovery-path через Hugging Face datasets и GitHub repository search для поиска источников по теме.

Важно: online-path **не заменяет offline baseline**. Offline-режим остаётся основным сценарием проекта, а online-режим добавляет возможность работы с внешними источниками в рамках той же архитектуры.

---

## Human-in-the-Loop

Human-in-the-loop review — это **обязательный архитектурный этап**, расположенный после annotation.

На этом шаге:

- строки с низкой уверенностью экспортируются в review queue;
- исправленные человеком метки возвращаются в pipeline по каноническому `id`;
- последующие этапы продолжают работу уже с reviewed data.

HITL используется для:

- ручной проверки спорных и низкоуверенных примеров;
- исправления неоднозначной авторазметки;
- повышения надёжности итогового pipeline.

В рамках этого проекта HITL — это **не косметическое улучшение**, а встроенный механизм контроля качества.

Экспорт в Label Studio поддерживается для размеченных батчей.

Отчёт по shortlist источников формируется на русском языке и предназначен для ручного просмотра перед одобрением. Дополнительно сохраняется machine-readable артефакт `data/raw/approval_candidates.json` для review/tooling-сценариев.

Файл `data/raw/approved_sources.json` остаётся отдельным human-edited входом для approval step.

После авторазметки pipeline также сохраняет:

- `data/interim/review_queue.csv` — очередь для ручной проверки;
- `reports/review_queue_report.md` — русский отчёт-инструкция для reviewer;
- `data/interim/review_queue_context.json` — machine-readable helper artifact для tooling или лёгкого UI-слоя.

Этап annotation также сохраняет компактный русский trace-pack:

- `reports/annotation_trace_report.md`
- `data/interim/annotation_trace.json`

Он фиксирует prompt contract, ожидаемые выходные поля и поведение parser/fallback. Это слой отчётности и наблюдаемости, а не полноценный API client framework.

---

## Как включить Gemini-разметку

Gemini-path является **опциональным и узким расширением** текущего annotation flow. Он включается только тогда, когда одновременно выполнены три условия:

- `annotation.use_llm: true`
- `annotation.llm_provider: gemini`
- в текущей shell-сессии задан `GEMINI_API_KEY`

Базовый конфиг `configs/demo_fitness.yaml` намеренно закреплён на `llm_provider: mock`, чтобы demo-run по умолчанию оставался offline и детерминированным.

Если ключ отсутствует, pipeline не падает: вместо Gemini-path используется существующий `MockLLM` fallback.

---

## Пример для PowerShell

Задайте ключ в текущей PowerShell-сессии:

```powershell
$env:GEMINI_API_KEY = 'your-gemini-api-key'
```

После этого запустите pipeline с конфигом, в котором явно указан annotation.llm_provider: gemini. Для этого можно либо сделать отдельный config, либо временно поменять llm_provider в своём конфиге.

```powershell
python run_pipeline.py --config configs/your_gemini_config.yaml
```

Этот способ влияет только на текущую shell-сессию. Он не добавляет .env-механику, secret loader, vault integration или UI.

---

## Как проверить, какой путь реально отработал

После запуска проверьте annotation trace:

- Gemini-path: llm_mode == generate_parse
- Mock/offline path: llm_mode == classify_effect

Trace сохраняется в:

- data/interim/annotation_trace.json
- reports/annotation_trace_report.md

Текущая интеграция — это MVP-расширение, а не production-ready secret management и не полноценный provider framework.

Исправленный review queue по-прежнему редактируется человеком отдельно в data/interim/review_queue_corrected.csv.

Если corrected queue присутствует, pipeline дополнительно сохраняет:

- reports/review_merge_report.md
- data/interim/review_merge_context.json

Это делает human merge step видимым и аудируемым. При этом текущая реализация остаётся file-based MVP, а не полноценным UI.

После quality stage pipeline также сохраняет компактный русский EDA-pack:

- reports/eda_report.md
- data/interim/eda_context.json

Это лёгкий слой отчётности для первичного анализа данных, а не BI-система и не полноценное аналитическое приложение.

---

## Структура репозитория

- src/ — реализация pipeline, агентов, сервисов, провайдеров и ML-хелперов
- tests/ — unit, integration и end-to-end тесты
- configs/ — demo-конфиги для offline-запуска
- data/ — артефакты локальных запусков

---

## Как запускать offline demo

Запускайте pipeline из корня репозитория с одним из стабильных demo-конфигов:

```bash
python run_pipeline.py --config configs/demo_fitness.yaml
python run_pipeline.py --config configs/demo_minecraft.yaml
```

Эти demo-paths рассчитаны на offline-работу и не требуют сетевого доступа.

---

## Hugging Face Discovery MVP

Для non-demo конфигов source discovery может обращаться к публичному Hugging Face datasets search API по теме из запроса.

Это узкий discovery и shortlisting step, а не полноценный online ingestion layer.

Для реальных Hugging Face candidates discovery сохраняет:

- канонический dataset id в uri;
- page URL в metadata, когда это уместно.

Если online lookup не удался, сервис безопасно делает fallback и не ломает pipeline. Offline demo path при этом не меняется и по-прежнему использует локальные детерминированные payloads.

---

## GitHub Discovery MVP

Для non-demo конфигов source discovery также может обращаться к публичному GitHub repository search API по текущей теме запроса.

Это только discovery-level MVP. Он:

- преобразует реальные результаты поиска в shortlist candidates;
- безопасно делает fallback при lookup failure;
- не добавляет полноценную GitHub collection-логику.

---

## Hugging Face Collection MVP

Online-capability также включает узкий Hugging Face collection path для shortlisted datasets.

Loader принимает либо Hugging Face dataset id, либо Hugging Face dataset URL и нормализует его перед загрузкой.

Это всё ещё не полный online pipeline, а ограниченное расширение discovery-and-collection поверх основного offline baseline.

---

## Source Approval Gate MVP

После discovery shortlist candidates могут проходить через минимальный approval gate перед следующим collection step.

MVP использует простой файл data/raw/approved_sources.json, содержащий JSON-список одобренных source_id.

Если файл отсутствует, shortlist helper возвращает исходный shortlist без изменений.

Это discovery-side approval checkpoint, а не UI и не production approval workflow.

---

## Approval-Aware Collection MVP

Оркестрационный слой применяет approval helper между discovery и collection.

На практике pipeline:

- находит источники;
- фильтрует shortlist через approved_sources.json, если файл присутствует;
- передаёт в collection только approved subset.

Approval-aware path явно отражает file-based статусы вроде:

- missing-file
- applied
- empty-subset

Это остаётся узким MVP-решением, чтобы не ломать стабильный offline baseline.

---

## Какие артефакты создаёт pipeline

Успешный demo-run создаёт, например, такие артефакты:

- final_report.md
- data/interim/model_metrics.json
- data/interim/review_queue.csv
- reports/annotation_trace_report.md
- data/interim/annotation_trace.json
- reports/eda_report.md
- data/interim/eda_context.json
- data/raw/discovered_sources.json
- data/raw/merged_raw.parquet

---

## Текущие ограничения

- Online discovery пока не является полным.
- Online-capability начинается с Hugging Face datasets discovery и не покрывает весь collection pipeline.
- Текущий online-layer включает Hugging Face discovery, GitHub repository discovery MVP и минимальный Hugging Face collection MVP, но всё ещё не является полным production-ready online pipeline.
- Approval gate реализован в file-based виде и намеренно оставлен минимальным: он фильтрует shortlist по approved source_id.
- Source shortlist report — это русский MVP для human review, а не полноценный approval UI.
- Артефакт approval_candidates.json — это helper shortlist для review/tooling-сценариев, а не approval decision file.
- review_queue.csv — это manual review queue, review_queue_report.md — инструкция для reviewer, а review_queue_context.json — helper artifact для tooling.
- review_merge_report.md и review_merge_context.json фиксируют результат ручного merge после подачи corrected labels.
- Demo datasets намеренно маленькие и synthetic/local.
- Offline demo path предназначен для воспроизводимых coursework-style запусков, а не для production use.
- Некоторые этапы намеренно сделаны детерминированными, чтобы baseline оставался стабильным при локальном выполнении.