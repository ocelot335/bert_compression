import torch
import torch.nn as nn
from loguru import logger
from rich.table import Table
from rich.console import Console

console = Console()


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: nn.Module) -> float:
    """
    Возвращает размер модели в мегабайтах (по параметрам в float32).

    Args:
        model: модель PyTorch.

    Returns:
        Размер в MB.
    """
    total_params = count_parameters(model)
    size_mb = total_params * 4 / (1024**2)
    return size_mb


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Возвращает доступное устройство.

    Args:
        prefer_gpu: предпочитать GPU если доступен.

    Returns:
        torch.device.
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: [bold]{torch.cuda.get_device_name(0)}[/bold]")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def print_model_summary(model: nn.Module, model_name: str = "Model") -> None:
    """
    Печатает таблицу с информацией о модели.

    Args:
        model: модель PyTorch.
        model_name: название модели для отображения.
    """
    table = Table(title=f"{model_name} Summary", show_header=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    total = count_parameters(model)
    trainable = count_parameters(model, trainable_only=True)
    frozen = total - trainable
    size_mb = get_model_size_mb(model)

    table.add_row("Total parameters", f"{total:,}")
    table.add_row("Trainable parameters", f"{trainable:,}")
    table.add_row("Frozen parameters", f"{frozen:,}")
    table.add_row("Model size (fp32)", f"{size_mb:.1f} MB")

    console.print(table)
    logger.info(
        f"{model_name}: {total:,} params ({size_mb:.1f} MB), "
        f"trainable: {trainable:,}"
    )
