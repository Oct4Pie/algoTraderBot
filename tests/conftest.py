"""Shared test setup: put the repo root on sys.path and restore mutable config
globals around each test (the bot keeps settings in module-level config vars)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

_RESTORE = ["SIZE", "RISK_PER_TRADE", "MAX_CONTRACTS", "PROBA_FLOOR",
            "ACTIVATE_R", "GIVEBACK_R", "ACTIVE_STRATEGIES", "SYMBOL"]


@pytest.fixture(autouse=True)
def restore_config():
    saved = {k: getattr(config, k) for k in _RESTORE}
    yield
    for k, v in saved.items():
        setattr(config, k, v)
