# JSONL Analyzer

Простой и надежный анализатор JSONL файлов с помощью OpenRouter API.

## Установка

```bash
pip install -r requirements.txt
```

## Настройка

1. Получите API ключ на [OpenRouter](https://openrouter.ai/)
2. Добавьте ключ в файл `.env`:
```
OPENROUTER_API_KEY=your_api_key_here
```

## Использование

```bash
python jsonl_analyzer.py
```

## Функции

- Автоматическое переключение между моделями при сбоях
- Повторная обработка неудачных элементов
- Настройка лимитов API для комфортной работы
- Подробное логирование процесса
- Обработка различных форматов JSONL

## Структура файлов

- `jsonl_analyzer.py` - основной скрипт
- `prompt.py` - промпт для AI
- `.env` - API ключ
- `requirements.txt` - зависимости
- `output/` - обработанные файлы