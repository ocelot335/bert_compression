from datasets import DatasetDict, load_dataset
from loguru import logger
from omegaconf import DictConfig


LABEL_NAMES = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
]

LABEL2ID = {label: idx for idx, label in enumerate(LABEL_NAMES)}
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_NAMES)}


def load_conll2003(cfg: DictConfig) -> DatasetDict:
    """
    Загружает CoNLL-2003 и удаляет ненужные колонки.

    Args:
        cfg: секция data из Hydra конфига (cfg.dataset_name, cfg.max_length).

    Returns:
        DatasetDict с колонками tokens и ner_tags.
    """
    logger.info(f"Loading dataset: {cfg.dataset_name}")

    dataset = load_dataset(cfg.dataset_name)
    dataset = dataset.remove_columns(["id", "pos_tags", "chunk_tags"])

    logger.info(
        f"Dataset loaded: "
        f"train={len(dataset['train'])}, "
        f"val={len(dataset['validation'])}, "
        f"test={len(dataset['test'])}"
    )

    return dataset
