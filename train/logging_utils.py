from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm.auto import tqdm


def setup_logging(log_dir: Path) -> tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{timestamp}.log"
    logger = logging.getLogger("octo_train")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    return logger, log_path


def log_event(message: str, logger: Optional[logging.Logger]) -> None:
    if logger is not None:
        logger.info(message)
    tqdm.write(message)
