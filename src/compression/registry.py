from loguru import logger

from src.compression.base import CompressionMethod

_REGISTRY: dict[str, type[CompressionMethod]] = {}


def register_compression(name: str):
    """
    Декоратор для регистрации метода сжатия.
    """

    def decorator(cls: type[CompressionMethod]) -> type[CompressionMethod]:
        if name in _REGISTRY:
            logger.warning(
                f"Compression method '{name}' already registered, overwriting."
            )
        _REGISTRY[name] = cls
        logger.debug(f"Registered compression method: '{name}'")
        return cls

    return decorator


def get_compression_method(name: str) -> type[CompressionMethod]:
    """
    Получить класс метода сжатия по имени.

    Args:
        name: имя метода (должен быть зарегистрирован через @register_compression).

    Returns:
        Класс метода сжатия.

    Raises:
        ValueError: если метод не найден.
    """
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(
            f"Unknown compression method: '{name}'. "
            f"Available methods: {available}"
        )
    return _REGISTRY[name]


def list_compression_methods() -> list[str]:
    return list(_REGISTRY.keys())
