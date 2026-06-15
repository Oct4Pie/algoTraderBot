#!/usr/bin/env python3
"""logsetup.py — shared logger that writes to log/ and the console.

    from logsetup import get_logger
    log = get_logger()
    log.info("…")
"""
import logging
import os

from config import HERE

LOG_DIR = os.path.join(HERE, "log")
LOG_FILE = os.path.join(LOG_DIR, "bot.log")


def get_logger(name: str = "bot") -> logging.Logger:
    """A logger that appends to log/bot.log and echoes to stdout (configured
    once per name)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    os.makedirs(LOG_DIR, exist_ok=True)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger
