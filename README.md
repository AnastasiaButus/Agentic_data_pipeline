# Universal Agentic Data Pipeline

## Обзор

`Universal Agentic Data Pipeline` — это **offline-first agentic pipeline** для сбора, очистки, авторазметки, human review, active learning и обучения базовой ML-модели на текстовых данных.

Проект сделан как **воспроизводимый research/demo baseline**:

- его можно запускать локально одной командой;
- offline-сценарий остаётся основным и стабильным;
- online-путь существует как расширение, а не как замена baseline;
- human-in-the-loop встроен в архитектуру, а не добавлен формально;
- ML-контур под капотом зафиксирован и объясним.

Проект сейчас ориентирован на **text-first сценарий** и на демонстрацию полного цикла работы с данными.

---

## Supported Scope

Текущий проект сознательно зафиксирован как **text-first pipeline**.

Что поддерживается сейчас:

- текстовые датасеты и текстовые записи, которые можно привести к схеме `text -> effect_label`;
- offline demo запуск и reproducible local baseline;
- online discovery extension для Hugging Face и GitHub shortlist;
- human-in-the-loop review для low-confidence строк;
- active learning track A поверх текстового baseline.

Что пока не считается официально закрытым scope:

- мультимодальность (`audio`, `image`);
- production-grade web scraping;
- hidden API discovery/extraction;
- browser automation (`Selenium`, `Playwright`);
- `requests-html`;
- Kaggle ingestion.

Лучше всего проект подходит для задач вида:

- классификация отзывов;
- классификация коротких текстов и описаний;
- тематические текстовые выборки с небольшим набором целевых классов;
- text datasets, где пользователь заранее задаёт `effect_labels`.

Иными словами, честная формулировка проекта сейчас такая:

`Universal Agentic Data Pipeline` — это **универсальный pipeline для text-first ML-задач**, а не универсальная мультимодальная ingestion-платформа для любых данных.

---

## Архитектурная позиция проекта

- **Offline-first**: основной режим проекта — детерминированный локальный baseline.
- **Dual mode**: offline и online — это две стратегии использования одного и того же пайплайна.
- **Agentic pipeline**: система состоит из последовательности агентов и сервисов, которые передают артефакты друг другу.
- **Human-in-the-loop**: после авторазметки есть реальный шаг ручной проверки и обратного merge.
- **ML is explicit**: в проекте есть явная локальная модель, а не только LLM-обвязка.
- **Русский UX**: ключевые отчёты и инструкции для защиты и ручной проверки формируются на русском языке.

---

## Официальная ML-модель

Официальный локальный baseline проекта:

`TF-IDF + Logistic Regression`

Он используется для задачи:

`text -> effect_label`

Почему именно эта модель:

- она быстро обучается локально;
- полностью воспроизводима;
- понятна на защите;
- не требует дорогой инфраструктуры;
- хорошо подходит для baseline-классификации коротких текстов и отзывов;
- удобно сочетается с active learning loop.

LLM-path в проекте **не заменяет** эту модель. LLM используется только в слое авторазметки, а training и active learning опираются на локальный ML-baseline.

---

## Архитектура пайплайна

```mermaid
flowchart LR
    A["Source Discovery"] --> B["Data Collection"]
    B --> C["Data Quality"]
    C --> D["Annotation"]
    D --> E["Human Review / HITL"]
    E --> F["Active Learning"]
    F --> G["Training"]
    G --> H["Reporting"]
    E --> I["Следующая итерация pipeline"]
    I --> D
```

Текущая агентная цепочка:

`discovery -> collection -> quality -> annotation -> review/HITL -> active learning -> training -> reporting`

Это не набор отдельных скриптов. Каждый шаг либо создаёт новый артефакт, либо подготавливает вход для следующего этапа.

---

## Что делает каждый агент

### 1. `SourceDiscoveryService`

Ищет и ранжирует кандидатов-источников.

Сейчас поддерживаются:

- offline demo candidates;
- Hugging Face datasets discovery MVP;
- GitHub repository discovery MVP;
- file-based source approval gate.

### 2. `DataCollectionAgent`

Собирает данные из выбранных источников и приводит их к канонической схеме.

Текущая логика:

- `hf_dataset` path для Hugging Face loader;
- `api` path для JSON API endpoints через `requests`;
- `scrape` path для local/demo HTML;
- soft topic-aware filtering вместо fitness-only фильтра;
- merge нескольких frame-like источников.

### 3. `DataQualityAgent`

Ищет проблемы качества и формирует cleaned dataset.

Текущие проверки включают:

- missing values;
- duplicates;
- outliers;
- class imbalance;
- compare before/after.

### 4. `AnnotationAgent`

Делает авторазметку и подготавливает human review.

Что уже есть:

- deterministic annotation contract;
- prompt/trace layer;
- confidence;
- quality summary;
- Label Studio export helper.

### 5. `ReviewQueueService`

Делает human-in-the-loop видимым:

- экспортирует low-confidence queue;
- принимает corrected queue;
- мержит ручные правки обратно в dataset;
- формирует merge-report.

### 6. `ActiveLearningAgent`

Гоняет active learning поверх локального text baseline.

Текущая база:

- entropy;
- random;
- offline simulation loop;
- learning history.

### 7. `TrainingService`

Обучает финальную baseline-модель и сохраняет артефакты обучения.

---

## Режимы работы

### Offline demo mode

Основной и рекомендуемый режим для защиты.

Он использует стабильные demo-конфиги:

- `configs/demo_fitness.yaml`
- `configs/demo_minecraft.yaml`

Преимущества:

- не требует сети;
- детерминирован;
- воспроизводим;
- подходит для демонстрации end-to-end сценария.

### Online mode

Расширенный режим для non-demo конфигов.

Сейчас это **узкий online MVP**, а не full production ingestion:

- Hugging Face discovery;
- GitHub discovery;
- Hugging Face collection path;
- JSON API collection path;
- approval-aware shortlist.

Если online lookup не удался, пайплайн должен безопасно вернуться к стабильному пути и не ломать baseline.

Operational notes для online-path:

- GitHub discovery может работать в `unauthenticated` режиме, но он заметно чувствительнее к rate limits;
- при наличии `GITHUB_TOKEN` GitHub Search path становится стабильнее;
- итог запуска теперь фиксируется в `reports/online_governance_report.md` и `data/raw/online_governance_summary.json`;
- fallback-стратегия остаётся offline-first: пустой remote shortlist не должен ломать весь run.

### Явный runtime.mode

Теперь режим можно фиксировать прямо в конфиге через секцию:

```yaml
runtime:
  mode: offline_demo
```

Поддерживаются четыре значения:

- `offline_demo` — использует только встроенные demo-источники и сохраняет стабильный offline baseline;
- `online` — включает только удалённый discovery path и не подмешивает встроенные demo-кандидаты;
- `hybrid` — разрешает и demo baseline, и online discovery в одном запуске;
- `local_only` — запрещает удалённый discovery и оставляет только локальные/demo-артефакты, если они доступны.

Важно: source-флаги в `source.*` описывают, какие внешние типы источников проект готов использовать, а `runtime.mode` определяет, какие из них реально активны в текущем запуске.

---

## Новый Text Topic

Для запуска на новой теме теперь есть безопасный шаблон:

- `configs/text_topic_template.yaml`

Этот шаблон нужен для сценария:

1. задать свою текстовую тему в `request.topic`;
2. указать небольшой целевой набор `annotation.effect_labels`;
3. выбрать runtime-режим без изменения кода;
4. использовать тот же pipeline contract, что и в demo-конфигах.

Важно:

- шаблон рассчитан на **текстовую классификацию**, а не на audio/image задачи;
- текущий training layer и active learning layer работают с полями `text` и `effect_label`;
- если вы запускаете проект на новой теме, лучше начинать с малого числа классов и понятного `effect_labels` vocabulary.

---

## Human-in-the-Loop

HITL в проекте — это обязательный этап после annotation.

Что происходит:

1. строки с низкой уверенностью попадают в `review_queue.csv`;
2. человек редактирует corrected queue;
3. corrected labels мержатся обратно по каноническому `id`;
4. downstream шаги используют уже reviewed dataset.

Это нужно для:

- проверки спорных примеров;
- исправления неоднозначной авторазметки;
- повышения качества датасета перед active learning и training.

Ключевые HITL-артефакты:

- `data/interim/review_queue.csv`
- `reports/review_workspace.html`
- `reports/review_queue_report.md`
- `data/interim/review_queue_context.json`
- `reports/review_merge_report.md`
- `data/interim/review_merge_context.json`
- `reports/review_agreement_report.md`
- `data/interim/review_agreement_context.json`

Если `review_queue.csv` не пустой, основной человеко-ориентированный вход в HITL теперь начинается с `reports/review_workspace.html`.
После повторного запуска с `review_queue_corrected.csv` pipeline также считает честную метрику `auto-vs-human agreement` и `Cohen's kappa` на reviewed subset. Это не два независимых human annotators, а quality-control метрика для HITL-правок.

---

## Token-Saving и deterministic path

Проект специально устроен так, чтобы **не уводить всё в LLM**.

Детерминированно в Python выполняются:

- discovery ranking;
- collection;
- schema normalization;
- data quality checks;
- EDA summary;
- merge/reporting helpers;
- training;
- active learning.

LLM используется только там, где он действительно нужен:

- в слое авторазметки;
- в reasoning-sensitive annotation path.

Это делает pipeline:

- дешевле;
- воспроизводимее;
- понятнее для отладки;
- безопаснее для offline demo.

---

## Артефакты, которые создаёт pipeline

Успешный demo-run создаёт, например:

- `reports/run_dashboard.html`
- `final_report.md`
- `data/raw/discovered_sources.json`
- `data/raw/approval_candidates.json`
- `data/raw/online_governance_summary.json`
- `data/raw/merged_raw.parquet`
- `reports/source_report.md`
- `reports/online_governance_report.md`
- `reports/quality_report.md`
- `reports/eda_report.md`
- `reports/eda_report.html`
- `data/interim/eda_context.json`
- `reports/annotation_report.md`
- `reports/annotation_trace_report.md`
- `data/interim/annotation_trace.json`
- `data/interim/review_queue.csv`
- `reports/review_workspace.html`
- `reports/review_queue_report.md`
- `reports/review_merge_report.md`
- `data/interim/review_agreement_context.json`
- `reports/review_agreement_report.md`
- `data/interim/model_metrics.json`
- `data/interim/model_artifact.pkl`
- `data/interim/vectorizer_artifact.pkl`

---

## Быстрый запуск

### Требования

- Python 3.12+
- локальное виртуальное окружение `.venv`

Установка зависимостей:

```bash
pip install -r requirements.txt
```

### Запуск offline demo

```bash
python run_pipeline.py --config configs/demo_fitness.yaml
python run_pipeline.py --config configs/demo_minecraft.yaml
```

После запуска удобнее всего начинать просмотр с:

- `reports/run_dashboard.html`
- `final_report.md`
- `reports/eda_report.html`
- `reports/review_workspace.html` если нужен ручной HITL-review

### Основная CLI-команда

```bash
python run_pipeline.py --config path/to/config.yaml
```

CLI в этом блоке не меняется: пайплайн по-прежнему запускается одной командой.
Выбор между offline/online/local сценариями теперь задаётся в самом YAML-конфиге через `runtime.mode`.

---

## Запуск в VS Code

В репозитории добавлены project-level настройки для локального запуска через VS Code:

- `.vscode/settings.json`
- `.vscode/tasks.json`
- `pytest.ini`

Готовые задачи:

- `unit tests`
- `integration smoke`
- `run demo_fitness`
- `run demo_minecraft`

Что делать:

1. Откройте репозиторий в VS Code.
2. Выберите интерпретатор `.\.venv\Scripts\python.exe`.
3. Откройте `Testing`, чтобы VS Code увидел `pytest`.
4. Для ручного запуска используйте `Terminal -> Run Task`.

`pytest.ini` направляет временные pytest-артефакты в локальную рабочую директорию проекта, чтобы test workflow не зависел от системного temp-каталога.

---

## Gemini-path

Gemini-интеграция в проекте — это **опциональное расширение annotation layer**.

Она включается только если одновременно выполнены условия:

- `annotation.use_llm: true`
- `annotation.llm_provider: gemini`
- в окружении задан `GEMINI_API_KEY`

Если ключ отсутствует, пайплайн не падает и делает fallback на `MockLLM`.

PowerShell-пример:

```powershell
$env:GEMINI_API_KEY = 'your-gemini-api-key'
python run_pipeline.py --config configs/your_gemini_config.yaml
```

Проверить, какой путь реально отработал, можно в:

- `data/interim/annotation_trace.json`
- `reports/annotation_trace_report.md`

Ожидаемая семантика:

- `generate_parse` — Gemini path
- `classify_effect` или fallback path — локальный/mock path

---

## Approval gate

После discovery shortlist можно вручную ограничить через:

`data/raw/approved_sources.json`

Это простой JSON-список `source_id`, который определяет, какие источники пойдут дальше в collect stage.

Таким образом в проекте уже есть видимый approval checkpoint между discovery и collection.

Для ручного approval теперь дополнительно используются:

- `data/raw/approval_candidates.json` — shortlist в JSON с `license`, `license_status`, `robots_txt_status`, `robots_txt_url`, `approval_notes`;
- `reports/source_report.md` — человекочитаемый markdown shortlist с теми же governance-полями для просмотра перед approve.

Это позволяет не смешивать discovery с автоматическим "юридическим движком", но делает license/robots проверку видимой частью HITL approval flow.

---

## Сильные стороны текущего baseline

- стабильный offline demo;
- один pipeline entrypoint;
- явная ML-модель;
- видимый HITL;
- русскоязычные отчёты;
- active learning layer;
- traceable annotation contract;
- dual-mode архитектура без разрушения baseline.

---

## Текущие ограничения

Проект пока не претендует на full production-ready систему.

Текущие осознанные ограничения:

- text-first фокус;
- online ingestion слой пока MVP;
- governance/compliance уже покрывает базовые `license` и `robots.txt` сигналы в approval artifacts, но ещё не является полным policy engine;
- rate-limit awareness и fallback reporting уже вынесены в отдельный online governance layer, но это ещё не полный observability/monitoring stack;
- scraping не является production-ready браузерным пайплайном;
- мультимодальность пока не главный трек.

Именно поэтому архитектурная ставка в проекте сейчас такая:

**надёжный offline-first baseline + понятный agentic pipeline + видимый HITL + явная ML-модель + аккуратный online extension.**

