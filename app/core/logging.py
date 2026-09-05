import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # unstructured + httpx are noisy at INFO — quiet them down
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("unstructured").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)