import logging
import os
from pathlib import Path


def setup_logger(logs_dir: str, run_id: str):
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(logs_dir) / f"run_{run_id}.log"

    log_symlink_path = log_file.parent / "latest_run.log"
    if log_symlink_path.exists():
        os.unlink(log_symlink_path)

    log_symlink_path.symlink_to(log_file.absolute())

    logger = logging.getLogger("dp")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger, str(log_file)
