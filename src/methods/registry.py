from loguru import logger

_REGISTRY: dict[str, callable] = {}


def register_method(name: str):
    """
    Декоратор для регистрации метода.

    Использование:
        @register_method("teacher")
        def run(cfg, model=None, data=None, tokenizer=None):
            ...
    """

    def decorator(fn):
        if name in _REGISTRY:
            logger.warning(f"Method '{name}' already registered, overwriting")
        _REGISTRY[name] = fn
        logger.debug(f"Registered method: '{name}'")
        return fn

    return decorator


def get_method(name: str):
    """
    Возвращает функцию метода по имени.

    Args:
        name: имя зарегистрированного метода.

    Returns:
        Функция run().

    Raises:
        ValueError: если метод не найден.
    """
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(
            f"Unknown method: '{name}'. " f"Available: {available}"
        )
    return _REGISTRY[name]


def list_methods() -> list[str]:
    return list(_REGISTRY.keys())
