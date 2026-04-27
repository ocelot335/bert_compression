# BERT Compression for NER

Исследовательский проект по сжатию BERT-base-cased для задачи Named Entity Recognition на датасете CoNLL-2003.

**Цель**: уменьшить размер модели с 108M до ~20M параметров с минимальной потерей качества (F1).

## Методы сжатия

- Факторизация эмбеддингов (SVD, CP-разложение, Tucker)
- Knowledge Distillation (логиты, скрытые состояния, карты внимания, value-relation в стиле MiniLM)
- Шеринг весов (в стиле ALBERT)
- Прунинг (magnitude, SparseGPT, structured)
- Прунинг голов внимания с обучаемыми гейтами
- Квантизация (dynamic PTQ, static PTQ, QAT, SmoothQuant)
- LoRA восстановление после прунинга
- Гибридный пайплайн сжатия

## Структура проекта

- **configs/** — конфиги Hydra для экспериментов
- **src/** — исходный код (модульный)
- **notebooks/** — анализ и визуализация (не обучение)
- **results/** — таблицы, графики, ablation studies
- **scripts/** — вспомогательные скрипты
- **tests/** — юнит-тесты

## Установка

```bash
git clone https://github.com/ocelot335/bert_compression.git
cd bert_compression
python -m venv .venv
source .venv/Scripts/activate  
pip install -e ".[dev]"
cp .env.example .env 
```           

## Запуск обучения

```bash
# Обучение учителя
python train.py --config-name=baseline
```

## Результаты

Будет дополнено по мере проведения экспериментов.
Чекпоинты будут опубликованы на HuggingFace Hub.