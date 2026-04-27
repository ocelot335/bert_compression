from datasets import DatasetDict
from loguru import logger
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase


def tokenize_and_align(
    examples: dict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
) -> dict:
    """
    Токенизирует батч примеров и выравнивает метки.

    Args:
        examples: батч из датасета (tokens, ner_tags).
        tokenizer: токенизатор.
        max_length: максимальная длина последовательности.

    Returns:
        Токенизированный батч с выровненными метками.
    """
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        all_labels.append(_align_single(labels, word_ids))

    tokenized["labels"] = all_labels
    return tokenized


def _align_single(
    labels: list[int],
    word_ids: list[int | None],
) -> list[int]:
    """
    Выравнивает метки одного примера по word_ids.

    Args:
        labels: метки исходных слов.
        word_ids: маппинг токен -> слово из токенизатора.

    Returns:
        Выровненные метки.
    """
    aligned = []
    prev_word_id = None

    for word_id in word_ids:
        if word_id is None:
            aligned.append(-100)
        elif word_id != prev_word_id:
            aligned.append(labels[word_id])
        else:
            label = labels[word_id]
            if label % 2 == 1:
                label += 1
            aligned.append(label)
        prev_word_id = word_id

    return aligned


def preprocess_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    cfg: DictConfig,
) -> DatasetDict:
    """
    Применяет токенизацию и выравнивание меток ко всему датасету.

    Args:
        dataset: исходный датасет.
        tokenizer: токенизатор BERT.
        cfg: секция data из Hydra конфига (cfg.max_length).

    Returns:
        Токенизированный датасет.
    """
    logger.info("Tokenizing dataset...")

    tokenized = dataset.map(
        tokenize_and_align,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": cfg.max_length,
        },
        batched=True,
        remove_columns=["tokens", "ner_tags"],
        desc="Tokenizing",
    )

    logger.info("Tokenization complete")
    return tokenized
