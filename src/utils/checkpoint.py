import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class CheckpointManager:
    """
    Сохраняет и загружает чекпоинты локально и на HF Hub.
    """

    def __init__(
        self,
        base_dir: str = "checkpoints",
        hf_repo_id: str | None = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.hf_repo_id = hf_repo_id
        self._hf_token = os.getenv("HUGGINGFACE_TOKEN")

    def save(
        self,
        model: nn.Module,
        name: str,
        tokenizer: PreTrainedTokenizerBase | None = None,
        metadata: dict | None = None,
        push_to_hub: bool = False,
    ) -> Path:
        """
        Сохраняет чекпоинт локально и опционально на HF Hub.

        Args:
            model: модель для сохранения.
            name: имя чекпоинта (папка внутри base_dir).
            tokenizer: токенизатор (опционально).
            metadata: метрики и мета-информация.
            push_to_hub: пушить ли на HF Hub.

        Returns:
            Путь к локальному чекпоинту.
        """
        save_path = self.base_dir / name
        save_path.mkdir(parents=True, exist_ok=True)

        if isinstance(model, PreTrainedModel):
            model.save_pretrained(save_path)
        else:
            torch.save(
                model.state_dict(),
                save_path / "model.pt",
            )

        if tokenizer is not None:
            tokenizer.save_pretrained(save_path)

        if metadata is not None:
            with open(save_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

        logger.info(f"Checkpoint saved: {save_path}")

        if push_to_hub:
            self._push_to_hub(save_path, name)

        return save_path

    def load(
        self,
        name: str,
        model_class=None,
        **kwargs,
    ) -> nn.Module:
        """
        Загружает модель из локального чекпоинта или HF Hub.

        Сначала ищет локально, потом на HF Hub.

        Args:
            name: имя чекпоинта.
            model_class: класс модели для from_pretrained.
                По умолчанию AutoModelForTokenClassification.
            **kwargs: аргументы для from_pretrained.

        Returns:
            Загруженная модель.
        """
        from transformers import AutoModelForTokenClassification

        if model_class is None:
            model_class = AutoModelForTokenClassification

        local_path = self.base_dir / name

        if local_path.exists():
            logger.info(f"Loading from local: {local_path}")
            return model_class.from_pretrained(local_path, **kwargs)

        if self.hf_repo_id is not None:
            logger.info(
                f"Loading from HF Hub: " f"{self.hf_repo_id} subfolder={name}"
            )
            return model_class.from_pretrained(
                self.hf_repo_id,
                subfolder=name,
                token=self._hf_token,
                **kwargs,
            )

        raise FileNotFoundError(
            f"Checkpoint '{name}' not found locally "
            f"and no HF repo configured."
        )

    def load_metadata(self, name: str) -> dict:
        """
        Загружает метаданные чекпоинта.

        Args:
            name: имя чекпоинта.

        Returns:
            Словарь с метаданными или пустой dict.
        """
        meta_path = self.base_dir / name / "metadata.json"
        if not meta_path.exists():
            logger.warning(f"No metadata found for '{name}'")
            return {}
        with open(meta_path) as f:
            return json.load(f)

    def exists(self, name: str) -> bool:
        return (self.base_dir / name).exists()

    def _push_to_hub(self, local_path: Path, name: str) -> None:
        if self.hf_repo_id is None:
            logger.warning("hf_repo_id not set, skipping push")
            return

        if self._hf_token is None:
            logger.warning("HUGGINGFACE_TOKEN not set, skipping push")
            return

        try:
            from huggingface_hub import HfApi

            api = HfApi(token=self._hf_token)
            api.upload_folder(
                folder_path=str(local_path),
                repo_id=self.hf_repo_id,
                path_in_repo=name,
                repo_type="model",
            )
            logger.info(f"Pushed to HF Hub: {self.hf_repo_id}/{name}")
        except Exception as e:
            logger.error(f"HF Hub push failed: {e}")
