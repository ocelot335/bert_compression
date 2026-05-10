import comet_ml  # noqa: F401

import os
from typing import Optional

from loguru import logger
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        RichHandler(
            console=console,
            markup=True,
            rich_tracebacks=True,
        ),
        format="{message}",
        level=level,
    )
    logger.info("Logging initialized")


def load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
        logger.debug(".env loaded")
    except ImportError:
        logger.debug("python-dotenv not installed, skipping .env")


def setup_comet(
    cfg: DictConfig,
    experiment_name: str,
) -> Optional[object]:
    if not cfg.get("comet", {}).get("enabled", True):
        logger.info("CometML disabled in config")
        return None
    api_key = os.getenv("COMET_API_KEY")
    if not api_key:
        logger.warning("COMET_API_KEY not found, skipping CometML")
        return None

    try:
        experiment = comet_ml.Experiment(
            api_key=api_key,
            project_name="bert-compression-ner",
            experiment_name=experiment_name,
        )
        experiment.set_name(experiment_name)
        params = _flatten_dict(OmegaConf.to_container(cfg, resolve=True))
        experiment.log_parameters(params)
        logger.info(f"CometML experiment: {experiment_name}")
        return experiment

    except ImportError:
        logger.warning("comet-ml not installed, skipping")
        return None


def _flatten_dict(
    d: dict,
    parent_key: str = "",
    sep: str = ".",
) -> dict:
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep))
        else:
            items[new_key] = v
    return items
