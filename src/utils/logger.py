import logging


def get_logger(name: str = "urban_eye") -> logging.Logger:
    """Return a simple reusable project logger."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
