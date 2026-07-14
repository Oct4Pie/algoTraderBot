"""Regression tests for the research/OOS leakage firewall."""
import numpy as np
import pandas as pd
import pytest

from research_validation import (
    artifact_contaminated_through,
    certification_test_start,
    enforce_unseen_backtest,
    policy_is_leak_safe,
    purged_train_eval_split,
    purged_window,
    require_leak_safe_probability_filter,
    write_policy_metadata,
)


@pytest.fixture(autouse=True)
def _isolate_consumed_certifications(tmp_path, monkeypatch):
    import research_validation as rv
    empty = tmp_path / "certifications"
    empty.mkdir()
    monkeypatch.setattr(rv, "CERTIFICATIONS_DIR", str(empty))


def test_train_eval_split_purges_full_outcome_horizon():
    cat = np.array([[10, 1], [18, -1], [19, 1], [20, -1], [25, 1]])
    train, val = purged_train_eval_split(cat, cut=20, horizon=2)
    assert [int(r[0]) for r in train] == [10]
    assert [int(r[0]) for r in val] == [20, 25]
    assert all(int(i) + 2 < 20 for i, _ in train)


def test_purged_window_excludes_tail_that_can_cross_boundary():
    cat = np.array([[60, 1], [68, 1], [69, 1], [70, 1], [80, 1]])
    got = purged_window(cat, 60, 80, horizon=10)
    assert [int(r[0]) for r in got] == [60, 68, 69]


def test_final_model_probability_filter_fails_closed():
    with pytest.raises(SystemExit, match="not out-of-fold"):
        require_leak_safe_probability_filter(0.35, unsafe=False)
    require_leak_safe_probability_filter(0.0, unsafe=False)
    require_leak_safe_probability_filter(0.35, unsafe=True)


def _metadata():
    return {
        "train_span": ["2021-01-01T00:00:00Z", "2026-05-01T00:00:00Z"],
        "holdout_span": ["2026-05-01T00:00:00Z", "2026-06-04T23:57:00Z"],
    }


def test_artifact_boundary_includes_inspected_holdout():
    assert artifact_contaminated_through(_metadata()) == pd.Timestamp(
        "2026-06-04T23:57:00Z")


class _Strategy:
    name = "fake"

    def _load_bundle(self):
        return {"training_metadata": _metadata()}


def test_backtest_rejects_overlap_unless_explicitly_research_only():
    with pytest.raises(SystemExit, match="overlaps model training"):
        enforce_unseen_backtest([_Strategy()], "2026-06-01", False)
    boundary = enforce_unseen_backtest([_Strategy()], "2026-06-01", True)
    assert boundary == pd.Timestamp("2026-06-04T23:57:00Z")
    enforce_unseen_backtest([_Strategy()], "2026-06-05", False)


def test_certification_requires_fresh_post_protocol_data():
    with pytest.raises(SystemExit, match="not fresh"):
        certification_test_start("2026-06-04")
    assert certification_test_start("2026-06-05") > pd.Timestamp(
        "2026-06-04T23:57:00Z")


def test_consumed_certification_advances_repository_exposure(tmp_path, monkeypatch):
    import json
    import research_validation as rv

    certs = tmp_path / "certifications"
    certs.mkdir(exist_ok=True)
    (certs / "exit.json").write_text(json.dumps({
        "test_data_through": "2026-07-10T20:57:00Z",
    }))
    monkeypatch.setattr(rv, "CERTIFICATIONS_DIR", str(certs))

    assert rv.research_contaminated_through() == pd.Timestamp(
        "2026-07-10T20:57:00Z")
    with pytest.raises(SystemExit, match="not fresh"):
        rv.certification_test_start("2026-07-01")
    assert rv.certification_test_start("2026-07-11") > pd.Timestamp(
        "2026-07-10T20:57:00Z")


def test_policy_provenance_is_hash_bound_and_rejects_quick_or_tampered(tmp_path):
    policy = tmp_path / "policy.npz"
    policy.write_bytes(b"policy-v1")
    base = {"protocol_version": 1, "embargo_bars": 80,
            "historical_probability_filter": "none", "quick": False,
            "dataset_sha256": "0" * 64, "train_cut_bar": 100,
            "train_entries": 10, "validation_entries": 2, "config": {}}
    write_policy_metadata(str(policy), base)
    assert policy_is_leak_safe(str(policy))
    assert not policy_is_leak_safe(str(policy), {"STOP_ATR": 999})

    policy.write_bytes(b"tampered")
    assert not policy_is_leak_safe(str(policy))

    write_policy_metadata(str(policy), {**base, "quick": True})
    assert not policy_is_leak_safe(str(policy))
