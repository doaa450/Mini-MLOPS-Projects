import logging

from pythonjsonlogger.json import JsonFormatter

from prodml.config import settings


def setup_logging(level: str | None = None) -> None:
    """Configure the root logger to emit one JSON object per log line."""
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            rename_fields={"asctime": "time", "levelname": "level", "name": "logger"},
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level or settings.log_level)
