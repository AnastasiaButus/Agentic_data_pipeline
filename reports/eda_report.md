# EDA-пакет по данным после quality

Это расширенный честный EDA-отчет по данным, которые реально идут дальше в pipeline.
Он показывает структуру, сравнение raw/cleaned, пропуски и распределения, не подменяя пустые поля выдуманной статистикой.

- n_rows: 3
- column_count: 8
- columns: id, source, text, label, rating, created_at, split, meta_json

## Raw vs cleaned
- raw_rows: 3
- cleaned_rows: 3
- dropped_rows: 0
- kept_fraction: 1

## Дубликаты
- duplicate_rows: 0

## Распределение source
- Minecraft Instructions Offline Demo: 3

## Распределение effect_label
- колонка отсутствует

## Сводка rating
- valid_count: 3
- missing_or_invalid_count: 0
- min: 2
- max: 5
- mean: 3.667

## Распределение rating
- 5: 1, 2: 1, 4: 1

## Длина text
- valid_count: 3
- missing_or_invalid_count: 0
- min_chars: 64
- max_chars: 76
- mean_chars: 69

## Бакеты длины text
- 0-49: 0, 50-99: 3, 100-199: 0, 200+: 0

## Пропуски по ключевым колонкам
- source: missing_count=0 / 3
- effect_label: колонка отсутствует
- rating: missing_count=0 / 3
- text: missing_count=0 / 3

## Quality notes
- missing values detected
