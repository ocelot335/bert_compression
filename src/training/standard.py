import torch
from loguru import logger
from omegaconf import DictConfig
from transformers import (
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.evaluation.metrics import make_compute_metrics


def _compute_warmup_steps(cfg: DictConfig, data) -> int:
    total_steps = (
        len(data["train"]) // cfg.training.batch_size * cfg.training.num_epochs
    )
    return int(total_steps * cfg.training.warmup_ratio)


def train_standard(
    model,
    data,
    tokenizer,
    cfg: DictConfig,
    comet_experiment=None,
    extra_callbacks: list | None = None,
) -> dict:
    """
    Стандартное обучение через HuggingFace Trainer.

    Args:
        model: модель для обучения.
        data: токенизированный DatasetDict.
        tokenizer: токенизатор.
        cfg: полный Hydra конфиг.
        comet_experiment: CometML Experiment.
        extra_callbacks: дополнительные HF callbacks.

    Returns:
        Словарь с метриками на тестовой выборке.
    """
    logger.info("Starting standard training...")

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    report_to = ["comet_ml"] if comet_experiment is not None else []

    warmup_steps = _compute_warmup_steps(cfg, data)

    training_args = TrainingArguments(
        output_dir=cfg.paths.output_dir,
        num_train_epochs=cfg.training.num_epochs,
        max_steps=cfg.training.get("max_steps", -1),
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=warmup_steps,
        fp16=cfg.training.fp16 and torch.cuda.is_available(),
        max_grad_norm=cfg.training.gradient_clip,
        eval_strategy="steps",
        eval_steps=cfg.training.eval_steps,
        save_strategy="steps",
        save_steps=cfg.training.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=report_to,
        seed=cfg.experiment.seed,
        logging_steps=50,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(),
        callbacks=callbacks,
    )

    trainer.train()

    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(data["test"])
    logger.info(f"Test F1: {test_metrics.get('eval_f1', 0):.4f}")

    if comet_experiment is not None:
        comet_experiment.log_metrics(test_metrics)

    return test_metrics
