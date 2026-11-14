"""One logger setup used everywhere: console + reports/logs/pipeline.log."""
import logging

from src.common.config import ROOT

_LOG_FILE = ROOT / "reports" / "logs" / "pipeline.log"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(_LOG_FILE)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger
