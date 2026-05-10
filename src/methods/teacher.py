from loguru import logger
from omegaconf import DictConfig

from src.methods.registry import register_method
from src.models.teacher import build_teacher
from src.training.standard import train_standard
from src.utils.checkpoint import CheckpointManager
from src.utils.model_utils import get_device


@register_method("teacher")
def run(
    cfg: DictConfig,
    model=None,
    data=None,
    tokenizer=None,
    comet_experiment=None,
) -> tuple:
    """
    Fine-tuning BERT-base-cased на CoNLL-2003.

    Args:
        cfg: Hydra конфиг.
        model: игнорируется, учитель строится с нуля.
        data: токенизированный DatasetDict.
        tokenizer: токенизатор BERT.
        comet_experiment: CometML Experiment.

    Returns:
        (model, metrics).
    """
    logger.info("=== Метод: файн-тюн учителя(базовой модели) ===")

    model = build_teacher(cfg.model)
    model = model.to(get_device())

    metrics = train_standard(
        model=model,
        data=data,
        tokenizer=tokenizer,
        cfg=cfg,
        comet_experiment=comet_experiment,
    )

    ckpt = CheckpointManager(
        hf_repo_id=(cfg.hub.repo_id if cfg.hub.push_to_hub else None),
    )
    ckpt.save(
        model=model,
        name=cfg.experiment.name,
        tokenizer=tokenizer,
        metadata=metrics,
        push_to_hub=cfg.hub.push_to_hub,
    )

    return model, metrics
