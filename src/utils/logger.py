import logging
from logging import Logger

from .config import get_settings


def get_logger(name: str) -> Logger:
    """Configure and return a logger with project defaults."""

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


