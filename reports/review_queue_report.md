# Очередь ручной проверки

Это очередь примеров для ручной проверки после авторазметки.

## Текущий этап

- Этап pipeline: human review / HITL
- Цель: проверить low-confidence примеры до retrain и финального обучения

## Reviewer guide

- Входной файл очереди: data/interim/review_queue.csv
- Порог confidence: 0.6
- Строк в очереди: 0
- Исправленный файл положите сюда: data/interim/review_queue_corrected.csv
- Проверьте поля: id, source, text, label, effect_label, confidence, reviewed_effect_label, review_comment, human_verified
- Допустимые effect labels: crafting, combat, enchantments

Очередь пуста, ручная проверка не требуется.

## Next step

- Следующий шаг: active learning / training могут использовать текущий reviewed dataset без ручных правок.
