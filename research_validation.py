"""Leakage guards shared by training, tuning, certification, and backtests.

These helpers deliberately fail closed.  A chronological split is not enough
when a sample's label/episode consumes future bars: the tail of the earlier
partition must be purged by the full outcome horizon.
"""
from __future__ import annotations

import json
import hashlib
import os
from typing import Iterable

import numpy as np
import pandas as pd

import config


PROTOCOL_PATH = os.path.join(config.HERE, "ppo_exit", "validation_protocol.json")
CERTIFICATIONS_DIR = os.path.join(config.HERE, "ppo_exit", "certifications")


def validation_protocol() -> dict:
    with open(PROTOCOL_PATH) as f:
        return json.load(f)


def research_contaminated_through() -> pd.Timestamp:
    """Latest bar exposed by development or a consumed certification window."""
    boundary = pd.Timestamp(validation_protocol()["contaminated_through"])
    try:
        names = os.listdir(CERTIFICATIONS_DIR)
    except FileNotFoundError:
        return boundary
    for name in names:
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(CERTIFICATIONS_DIR, name)) as f:
                report = json.load(f)
            end = report.get("test_data_through")
            if end:
                boundary = max(boundary, pd.Timestamp(end))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            # A corrupt/incomplete report must not erase the known base boundary.
            continue
    return boundary


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def policy_metadata_path(policy_path: str) -> str:
    return f"{policy_path}.validation.json"


def write_policy_metadata(policy_path: str, metadata: dict) -> str:
    """Write a hash-bound provenance sidecar for a newly trained policy."""
    path = policy_metadata_path(policy_path)
    out = dict(metadata)
    out["policy_sha256"] = file_sha256(policy_path)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, path)
    return path


def policy_is_leak_safe(policy_path: str, expected_config: dict | None = None) -> bool:
    """Only policies produced by the purged, unfiltered protocol are accepted."""
    try:
        with open(policy_metadata_path(policy_path)) as f:
            meta = json.load(f)
        protocol = validation_protocol()
        dataset_hash = meta.get("dataset_sha256")
        config_ok = isinstance(meta.get("config"), dict)
        if config_ok and expected_config is not None:
            config_ok = all(
                key in meta["config"]
                and float(meta["config"][key]) == float(value)
                for key, value in expected_config.items()
            )
        return (
            int(meta.get("protocol_version", 0)) >= int(protocol["protocol_version"])
            and int(meta.get("embargo_bars", -1))
            >= int(protocol["episode_embargo_bars"])
            and meta.get("historical_probability_filter") == "none"
            and not bool(meta.get("quick"))
            and isinstance(dataset_hash, str) and len(dataset_hash) == 64
            and int(meta.get("train_cut_bar", 0)) > 0
            and int(meta.get("train_entries", 0)) > 0
            and int(meta.get("validation_entries", 0)) > 0
            and config_ok
            and meta.get("policy_sha256") == file_sha256(policy_path)
        )
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def purged_train_eval_split(catalog: np.ndarray, cut: int, horizon: int):
    """Split entry samples at ``cut`` and purge earlier outcomes crossing it.

    An episode entered at ``i`` may read bars through ``i + horizon``.  It is
    eligible for the earlier partition only when that final bar is still
    strictly before the later partition.
    """
    if horizon < 0:
        raise ValueError("horizon must be >= 0")
    idx = np.asarray(catalog)[:, 0]
    earlier = catalog[idx + horizon < cut]
    later = catalog[idx >= cut]
    return earlier, later


def purged_window(catalog: np.ndarray, lo: int, hi: int, horizon: int):
    """Samples entering in ``[lo, hi)`` whose outcomes finish before ``hi``."""
    if not 0 <= lo < hi:
        raise ValueError("expected 0 <= lo < hi")
    idx = np.asarray(catalog)[:, 0]
    return catalog[(idx >= lo) & (idx + horizon < hi)]


def require_leak_safe_probability_filter(proba_floor: float, unsafe: bool) -> None:
    """Reject final-model historical filtering unless explicitly research-only.

    The shipped entry head was fit on much of the same history being replayed.
    Its historical probabilities are not out-of-fold and therefore cannot be
    used in a clean exit-training or certification population.
    """
    if proba_floor > 0 and not unsafe:
        raise SystemExit(
            "historical probabilities from the shipped entry model are not "
            "out-of-fold and would leak into exit training/evaluation; use "
            "--proba-floor 0 (the leak-safe default). "
            "--unsafe-final-model-filter is research-only and cannot certify results"
        )


def _span_end(span):
    if not span:
        return None
    return pd.Timestamp(span[1])


def artifact_contaminated_through(metadata: dict) -> pd.Timestamp:
    """Last timestamp exposed during fit or artifact holdout inspection."""
    if not metadata:
        raise ValueError("model has no training metadata")
    ends = [
        _span_end(metadata.get("train_span")),
        _span_end(metadata.get("holdout_span")),
    ]
    ends = [x for x in ends if x is not None]
    if not ends:
        raise ValueError("model metadata has no train/holdout span")
    return max(ends)


def backtest_contaminated_through(strategies: Iterable) -> pd.Timestamp:
    """Latest exposure boundary across every entry model in a backtest."""
    boundaries = []
    for strategy in strategies:
        bundle = strategy._load_bundle()
        try:
            boundaries.append(
                artifact_contaminated_through(bundle.get("training_metadata") or {})
            )
        except ValueError as e:
            raise SystemExit(f"cannot validate {strategy.name!r} model: {e}") from e
    if not boundaries:
        raise SystemExit("cannot validate a backtest with no strategies")
    return max(max(boundaries), research_contaminated_through())


def enforce_unseen_backtest(strategies: Iterable, first_test_bar,
                            allow_in_sample: bool = False) -> pd.Timestamp:
    """Fail unless every tested bar is later than all model-development data."""
    boundary = backtest_contaminated_through(strategies)
    first = pd.Timestamp(first_test_bar)
    if first.tzinfo is None and boundary.tzinfo is not None:
        first = first.tz_localize(boundary.tzinfo)
    if first <= boundary and not allow_in_sample:
        raise SystemExit(
            "backtest overlaps model training/inspected holdout data "
            f"(first test bar {first}, contaminated through {boundary}). "
            "Clean OOS data must start after that boundary. For strategy "
            "development only, rerun with --allow-in-sample; those results must "
            "not be reported as validation"
        )
    return boundary


def certification_test_start(value) -> pd.Timestamp:
    """Validate that a certification window begins after all exposed repo data."""
    start = pd.Timestamp(value)
    boundary = research_contaminated_through()
    if start.tzinfo is None and boundary.tzinfo is not None:
        start = start.tz_localize(boundary.tzinfo)
    if start <= boundary:
        raise SystemExit(
            f"certification start {start} is not fresh; repository development "
            f"already exposed data through {boundary}"
        )
    return start
