"""Lightweight structured logging utilities for GMUL."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def setup_logger(
    name: str = "granunlearn",
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure and return a logger with console (and optional file) handlers.

    Parameters
    ----------
    name : str
        Logger name.
    level : int
        Logging level.
    log_file : str | Path | None
        If given, also write logs to this file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        logger.addHandler(console)

        if log_file is not None:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(log_file))
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger


def log_json(record: dict[str, Any], path: str | Path) -> None:
    """Append a JSON line to *path*, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def save_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    """Write *data* as a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)
    return path
