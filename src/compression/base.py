from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch.nn as nn
from omegaconf import DictConfig


@dataclass
class CompressionStats:
    method_name: str
    params_before: int
    params_after: int
    size_mb_before: float
    size_mb_after: float
    compression_ratio: float = field(init=False)
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.compression_ratio = self.params_before / max(self.params_after, 1)

    def __str__(self) -> str:
        return (
            f"{self.method_name}: "
            f"{self.params_before:,} → {self.params_after:,} params "
            f"({self.size_mb_before:.1f} → {self.size_mb_after:.1f} MB), "
            f"ratio: {self.compression_ratio:.2f}x"
        )


class CompressionMethod(ABC):
    """
    Абстрактный базовый класс для методов сжатия.
    """

    def __init__(self, cfg: DictConfig | None = None) -> None:
        self.cfg = cfg
        self._stats: CompressionStats | None = None

    @abstractmethod
    def apply(self, model: nn.Module, **kwargs) -> nn.Module:
        """
        Применить метод сжатия к модели.

        Args:
            model: исходная модель.
            **kwargs: дополнительные аргументы (dataloader для калибровки и т.д.)

        Returns:
            Сжатая модель.
        """
        ...

    @abstractmethod
    def get_stats(self) -> CompressionStats:
        """
        Вернуть статистику сжатия.

        Returns:
            CompressionStats с информацией о сжатии.
        """
        ...

    def _compute_stats(
        self,
        method_name: str,
        model_before: nn.Module,
        model_after: nn.Module,
        extra: dict | None = None,
    ) -> CompressionStats:
        from src.utils.model_utils import count_parameters, get_model_size_mb

        return CompressionStats(
            method_name=method_name,
            params_before=count_parameters(model_before),
            params_after=count_parameters(model_after),
            size_mb_before=get_model_size_mb(model_before),
            size_mb_after=get_model_size_mb(model_after),
            extra=extra or {},
        )
