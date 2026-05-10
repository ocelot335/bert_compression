import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig
from transformers import AutoModelForTokenClassification

from src.data.dataset import ID2LABEL, LABEL2ID, LABEL_NAMES
from src.utils.model_utils import print_model_summary


def build_teacher(cfg: DictConfig) -> nn.Module:
    """
    Загружает BERT-base-cased с classification head для NER.

    Args:
        cfg: секция model из Hydra конфига.
            cfg.name: имя модели (bert-base-cased).

    Returns:
        BertForTokenClassification.
    """
    logger.info(f"Building teacher: {cfg.name}")

    model = AutoModelForTokenClassification.from_pretrained(
        cfg.name,
        num_labels=len(LABEL_NAMES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    print_model_summary(model, model_name="Teacher")
    return model
