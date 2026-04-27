import os
import random

import numpy as np
import torch
from loguru import logger


def seed_everything(seed: int = 42) -> None:
    """
    Фиксирует все источники случайности для воспроизводимости.

    Args:
        seed: значение seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Global seed set to {seed}")
