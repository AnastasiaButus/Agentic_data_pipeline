# Трассировка annotation contract

Это компактный отчет о prompt contract, ожидаемом формате ответа и fallback-парсинге.
Он нужен, чтобы будущий real LLM можно было подключить без угадывания контракта.

- llm_mode: classify_effect
- n_rows: 3
- n_fallback_rows: 0

## Prompt contract
- language: ru
- input_fields: text, rating
- output_fields: effect_label, sentiment_label, confidence
- sentiment_labels: negative, neutral, positive
- effect_labels: crafting, combat, enchantments

## Prompt preview
Ты разметчик текстовых данных по теме: minecraft instructions.
Верни только JSON без пояснений, markdown и лишнего текста.

Заполни ровно эти поля:
- effect_label: одно из значений списка ниже
- sentiment_label: negative, neutral или positive
- confidence: число от 0 до 1

Допустимые effect_label: crafting, combat, enchantments

Правила:
- Не добавляй новые поля.
- Если текст неполный или неоднозначный, выбирай самый безопасный вариант.
- confidence должен отражать уверенность модели, но оставаться в диапазоне 0..1.

Ожидаемый ответ:
{"effect_label": "...", "sentiment_label": "...", "confidence": 0.0}

Текст для разметки:
Minecraft guide: crafting instructions for redstone tools and starter builds

## Expected output
- example: {'effect_label': 'crafting', 'sentiment_label': 'positive', 'confidence': 0.5}

## Parser contract
- preferred_format: json
- accepted_fallbacks: key_value, partial_json, deterministic_fallback
- parse_status_counts: {'direct': 3}
- fallback_reason_counts: {}
