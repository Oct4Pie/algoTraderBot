"""Optuna exit-config scanner helpers. The objective replays the give-back sim
(TrailExitSim) per config and scores expectancy on a validation slice; these check
the metric/split math and that the replay is actually sensitive to the config knobs
(otherwise the scan is meaningless)."""
import numpy as np
import pytest

import config
from ppo_exit import optimize_exit as oe


@pytest.fixture(autouse=True)
def _isolate_consumed_certifications(tmp_path, monkeypatch):
    import research_validation as rv
    empty = tmp_path / "existing_certifications"
    empty.mkdir()
    monkeypatch.setattr(rv, "CERTIFICATIONS_DIR", str(empty))


def test_metrics_basic():
    R = np.array([1.0, 1.0, -1.0, 2.0, -1.0])
    m = oe._metrics(R)
    assert m["n"] == 5
    assert abs(m["sumR"] - 2.0) < 1e-9
    assert abs(m["wr"] - 0.6) < 1e-9
    assert abs(m["pf"] - (4.0 / 2.0)) < 1e-9       # wins 4R / losses 2R


def test_metrics_empty():
    m = oe._metrics(np.empty(0))
    assert m["n"] == 0 and m["pf"] == 0.0


def test_split_is_time_ordered():
    # n_bars=100 → validation = [60,80), later historical = [80,100]
    cat = np.array([[10, 1], [65, 1], [75, -1], [85, 1], [95, -1]])
    val, test = oe._split(cat, 100, horizon=0)
    assert [int(r[0]) for r in val] == [65, 75]
    assert [int(r[0]) for r in test] == [85, 95]


def test_split_purges_validation_outcomes_that_reach_later_history():
    cat = np.array([[60, 1], [69, 1], [70, -1], [80, 1], [90, -1]])
    val, test = oe._split(cat, 100, horizon=10)
    assert [int(r[0]) for r in val] == [60, 69]
    assert [int(r[0]) for r in test] == [80, 90]
    assert all(int(i) + 10 < 80 for i, _ in val)


def test_certification_catalog_requires_complete_outcomes():
    import pandas as pd
    n = oe.MAX_HOLD + 20
    df = pd.DataFrame({
        "datetime": pd.date_range("2026-07-01", periods=n, freq="3min", tz="UTC")
    })
    cat = np.array([[1, 1], [10, -1], [20, 1]])
    got = oe._certification_catalog(df, cat, pd.Timestamp("2026-07-01", tz="UTC"))
    assert [int(r[0]) for r in got] == [1, 10]


def test_certification_is_one_shot_and_hashes_exact_policy(tmp_path, monkeypatch):
    import pandas as pd

    n = 120
    df = pd.DataFrame({
        "datetime": pd.date_range("2026-06-05", periods=n, freq="3min", tz="UTC")
    })
    csv = tmp_path / "NQ_3min.csv"
    df.to_csv(csv, index=False)
    arr = _arr(np.full(n, 100.0), np.full(n, 100.5), np.full(n, 99.5))
    cat = np.array([[1, 1], [10, -1]])

    policy = tmp_path / "policy.npz"
    actions = len(oe.TRAIL_MULTS)
    np.savez(policy, n_layers=np.int64(1),
             w0=np.zeros((actions, 6), np.float32),
             b0=np.zeros(actions, np.float32))
    from research_validation import write_policy_metadata
    write_policy_metadata(str(policy), {
        "protocol_version": 1, "embargo_bars": oe.MAX_HOLD,
        "historical_probability_filter": "none", "quick": False,
        "dataset_sha256": "0" * 64, "train_cut_bar": 100,
        "train_entries": 10, "validation_entries": 2,
        "config": {"ACTIVATE_R": oe.config.ACTIVATE_R,
                   "GIVEBACK_R": oe.config.GIVEBACK_R,
                   "STOP_ATR": oe.config.STOP_ATR},
    })
    monkeypatch.setattr(oe.config, "policy_path", lambda: str(policy))

    raw = [("NQ", arr, cat, df, {"training_context": str(csv)})]
    out = tmp_path / "certs"
    path, metrics = oe._certify(
        raw, timeframe=3, tickers=["NQ"], test_start="2026-06-05",
        min_trades=2, output_dir=str(out))
    assert metrics["ppo_policy"]["n"] == 2
    assert metrics["constant_1atr_reference"]["n"] == 2
    assert path.endswith("exit_policy_protocol1_3min.json")
    import json
    with open(path) as f:
        report = json.load(f)
    assert report["test_data_through"] == df["datetime"].iloc[-1].isoformat()
    assert report["coverage"]["NQ"]["test_entries"] == 2
    with pytest.raises(SystemExit, match="already consumed"):
        oe._certify(raw, timeframe=3, tickers=["NQ"],
                    test_start="2026-06-05", min_trades=2,
                    output_dir=str(out))


def test_load_ticker_appends_separate_fresh_data_without_mutating_base(
        tmp_path, monkeypatch):
    import pandas as pd

    base_dir = tmp_path / "repo"
    data_dir = base_dir / "data"
    fresh_dir = tmp_path / "fresh"
    data_dir.mkdir(parents=True)
    fresh_dir.mkdir()
    columns = {
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 1.0,
    }
    base = pd.DataFrame({
        "datetime": pd.date_range("2026-06-04 23:51", periods=3,
                                  freq="3min", tz="UTC"),
        **columns,
    })
    fresh = pd.DataFrame({
        "datetime": pd.date_range("2026-06-05 00:00", periods=3,
                                  freq="3min", tz="UTC"),
        **columns,
    })
    base_path = data_dir / "NQ_3min.csv"
    fresh_path = fresh_dir / "NQ_3min.csv"
    base.to_csv(base_path, index=False)
    fresh.to_csv(fresh_path, index=False)
    before = base_path.read_bytes()

    monkeypatch.setattr(oe, "REPO", str(base_dir))
    monkeypatch.setattr(oe, "build_arrays", lambda frame: {"n": len(frame)})
    monkeypatch.setattr(
        oe, "build_catalog", lambda arr: np.array([[0, 1]], dtype=int))
    arr, _cat, combined, sources = oe._load_ticker(
        "NQ", 3, 0.0, str(fresh_dir))

    assert arr["n"] == 6
    assert len(combined) == 6
    assert sources == {"training_context": str(base_path),
                       "fresh_holdout": str(fresh_path)}
    assert base_path.read_bytes() == before


def test_load_ticker_rejects_fresh_overlap(tmp_path, monkeypatch):
    import pandas as pd

    base_dir = tmp_path / "repo"
    data_dir = base_dir / "data"
    fresh_dir = tmp_path / "fresh"
    data_dir.mkdir(parents=True)
    fresh_dir.mkdir()
    values = {"open": [100.0], "high": [101.0], "low": [99.0],
              "close": [100.0], "volume": [1.0]}
    pd.DataFrame({"datetime": ["2026-06-05T00:00:00Z"], **values}).to_csv(
        data_dir / "NQ_3min.csv", index=False)
    pd.DataFrame({"datetime": ["2026-06-05T00:00:00Z"], **values}).to_csv(
        fresh_dir / "NQ_3min.csv", index=False)
    monkeypatch.setattr(oe, "REPO", str(base_dir))

    with pytest.raises(SystemExit, match="overlaps"):
        oe._load_ticker("NQ", 3, 0.0, str(fresh_dir))


def _arr(close, high, low):
    n = len(close)
    return {"close": np.array(close, float), "high": np.array(high, float),
            "low": np.array(low, float), "atr": np.full(n, 1.0),
            "atr_stop": np.full(n, 4.0), "line": np.zeros(n), "direction": np.zeros(n)}


def test_realized_r_responds_to_giveback(monkeypatch):
    # a long that rallies to a new peak then reverses — the give-back width changes
    # where it exits, so the scanner's objective must move with GIVEBACK_R. The sim
    # reads config.* at runtime (training=live), so we patch config, not the module.
    monkeypatch.setattr(config, "STOP_ATR", 0.5)   # risk = 0.5 × atr_stop(4) = 2.0
    monkeypatch.setattr(config, "ACTIVATE_R", 2.0)
    close = [100, 104, 106, 100]
    high = [100.5, 104.5, 106.5, 100.5]
    low = [99.5, 103.5, 105.5, 95.0]
    cat = np.array([[0, 1]])
    monkeypatch.setattr(config, "GIVEBACK_R", 0.5)
    tight = oe._realized_R(_arr(close, high, low), cat, action=1)[0]
    monkeypatch.setattr(config, "GIVEBACK_R", 1.5)
    loose = oe._realized_R(_arr(close, high, low), cat, action=1)[0]
    assert tight != loose                          # config genuinely drives the exit


def test_realized_r_responds_to_activate(monkeypatch):
    monkeypatch.setattr(config, "STOP_ATR", 0.5)
    monkeypatch.setattr(config, "GIVEBACK_R", 0.75)
    close = [100, 104, 100, 98]
    high = [100.5, 104.5, 100.5, 98.5]
    low = [99.5, 103.5, 99.5, 97.5]
    cat = np.array([[0, 1]])
    monkeypatch.setattr(config, "ACTIVATE_R", 1.0)   # activates (peak +2.25R ≥ 1)
    on = oe._realized_R(_arr(close, high, low), cat, action=1)[0]
    monkeypatch.setattr(config, "ACTIVATE_R", 5.0)   # never activates → rides to stop
    off = oe._realized_R(_arr(close, high, low), cat, action=1)[0]
    assert on != off
