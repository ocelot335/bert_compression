import comet_ml  # noqa: F401

import os
import hydra
from loguru import logger
from omegaconf import DictConfig
from transformers import AutoTokenizer
from huggingface_hub import login

import src.methods
from src.data.dataset import load_conll2003
from src.data.preprocessing import preprocess_dataset
from src.methods.registry import get_method
from src.utils.logging import load_env, setup_comet, setup_logging
from src.utils.seed import seed_everything


@hydra.main(
    config_path="configs",
    config_name="baseline",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    load_env()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        login(token=hf_token)
        logger.info("HuggingFace: authenticated")
    else:
        logger.warning("HuggingFace: no token found")
    setup_logging()
    seed_everything(cfg.experiment.seed)

    experiment = setup_comet(cfg, experiment_name=cfg.experiment.name)

    logger.info("Loading data...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    raw = load_conll2003(cfg.data)
    data = preprocess_dataset(raw, tokenizer, cfg.data)

    # pipeline пока не реализован - заглушка
    if hasattr(cfg, "pipeline"):
        logger.warning("Pipeline not implemented yet")
        return

    method_fn = get_method(cfg.method)
    model, metrics = method_fn(
        cfg,
        data=data,
        tokenizer=tokenizer,
        comet_experiment=experiment,
    )

    logger.info(f"Done. Metrics: {metrics}")

    if experiment is not None:
        experiment.end()


if __name__ == "__main__":
    main()
