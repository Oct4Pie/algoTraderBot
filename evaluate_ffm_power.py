#!/usr/bin/env python3
"""Exploratory predictive-power audit for the shipped FFM booster heads.

This is deliberately not a certification command.  The local June/July 2026 NQ
window was already consumed by exit-policy certification, and the shipped model
bundles omitted their labeler horizon/barrier configuration.  We therefore freeze
three plausible horizons and report robustness without tuning any threshold.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import average_precision_score, roc_auc_score

import config
import embedder
import strategies
from research_validation import file_sha256
from strategies.base import _install_pickle_compat, require_xgboost_compat


HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW = 500
HORIZONS = (20, 40, 80)
TARGET_R = 3.0
STOP_R = 1.0
FROZEN_THRESHOLD = 0.35
STRATEGIES = ("supertrend", "ema", "keltner", "bos", "orb", "cisd_ote")


def _load_bars(fresh_csv: str) -> tuple[pd.DataFrame, dict]:
    base_csv = os.path.join(HERE, "data", "NQ_3min.csv")
    base = pd.read_csv(base_csv).rename(columns={"datetime": "time"})
    fresh = pd.read_csv(fresh_csv).rename(columns={"datetime": "time"})
    required = {"time", "open", "high", "low", "close", "volume"}
    for name, frame in (("training context", base), ("fresh", fresh)):
        missing = required.difference(frame.columns)
        if missing:
            raise SystemExit(f"{name} data is missing columns: {sorted(missing)}")
        frame["time"] = pd.to_datetime(frame["time"], utc=True)
        if frame.empty or frame["time"].isna().any():
            raise SystemExit(f"{name} data is empty or has invalid timestamps")
        if not frame["time"].is_monotonic_increasing \
                or frame["time"].duplicated().any():
            raise SystemExit(f"{name} timestamps must be sorted and unique")
    if base["time"].max() >= fresh["time"].min():
        raise SystemExit("fresh data overlaps the training context")
    return pd.concat([base, fresh], ignore_index=True), {
        "training_context": {"path": os.path.relpath(base_csv, HERE),
                             "sha256": file_sha256(base_csv)},
        "exploratory_window": {"path": os.path.relpath(fresh_csv, HERE),
                               "sha256": file_sha256(fresh_csv)},
    }


def barrier_outcome(high, low, close, *, entry_idx: int, entry: float,
                    risk: float, direction: int, horizon: int,
                    include_entry_bar: bool = False) -> tuple[float, bool]:
    """Gross R and whether +TARGET_R was hit; stop wins same-bar ties."""
    if not np.isfinite(risk) or risk <= 0:
        raise ValueError("risk must be finite and positive")
    stop = entry - direction * STOP_R * risk
    target = entry + direction * TARGET_R * risk
    first = entry_idx if include_entry_bar else entry_idx + 1
    last = min(first + horizon, len(close))
    for j in range(first, last):
        hit_stop = low[j] <= stop if direction > 0 else high[j] >= stop
        hit_target = high[j] >= target if direction > 0 else low[j] <= target
        if hit_stop:
            return -STOP_R, False
        if hit_target:
            return TARGET_R, True
    terminal = max(first, last - 1)
    return direction * (close[terminal] - entry) / risk, False


def _collect_events(df: pd.DataFrame, start: pd.Timestamp,
                    strategy_name: str) -> list[dict]:
    strategy = strategies.make_strategies([strategy_name])[0]
    first = int(df["time"].searchsorted(start, side="left"))
    events = []
    # Run the deployed 500-bar inference window exactly. This is slower than a
    # vectorized detector but avoids changing session/indicator warmup semantics.
    for i in range(max(WINDOW - 1, first), len(df)):
        lo = i - WINDOW + 1
        bars = df.iloc[lo:i + 1]
        sig = strategy.detect(bars)
        if sig is None:
            continue
        embed_idx = lo + int(sig.bar_index)
        exec_idx = i if strategy_name == "cisd_ote" else embed_idx
        include_entry_bar = strategy_name == "cisd_ote"
        if exec_idx < first or exec_idx + max(HORIZONS) >= len(df):
            continue
        hand = strategy._hand_features(
            bars, int(sig.bar_index), int(sig.direction)).astype(np.float32)
        events.append({
            "strategy": strategy_name,
            "time": df["time"].iloc[exec_idx],
            "embed_idx": embed_idx,
            "entry_idx": exec_idx,
            "entry": float(sig.entry),
            "risk": float(sig.risk),
            "direction": int(sig.direction),
            "include_entry_bar": include_entry_bar,
            "hand": hand,
        })
    return events


def _score_events(df: pd.DataFrame, events: list[dict], strategy_name: str,
                  bundle: dict) -> pd.DataFrame:
    if not events:
        return pd.DataFrame()
    indices = np.asarray([e["embed_idx"] for e in events], dtype=np.int64)
    embeddings = embedder.embed_bars(
        df["close"].to_numpy(float), indices, ctx=int(bundle["ctx_window"]))
    hands = np.stack([e["hand"] for e in events]).astype(np.float32)
    X = np.hstack([embeddings, hands]).astype(np.float32)
    expected = int(bundle["feat_dim"])
    if X.shape[1] != expected:
        raise SystemExit(
            f"{strategy_name}: feature dimension {X.shape[1]} != bundle {expected}")
    proba = bundle["signal_head"].predict_proba(X)[:, 1].astype(float)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    rows = []
    for event, score in zip(events, proba):
        row = {k: event[k] for k in (
            "strategy", "time", "entry_idx", "entry", "risk", "direction")}
        row["proba"] = float(score)
        row["accepted_035"] = bool(score >= FROZEN_THRESHOLD)
        for horizon in HORIZONS:
            realized, target_hit = barrier_outcome(
                high, low, close, entry_idx=event["entry_idx"],
                entry=event["entry"], risk=event["risk"],
                direction=event["direction"], horizon=horizon,
                include_entry_bar=event["include_entry_bar"])
            row[f"r_{horizon}"] = float(realized)
            row[f"positive_{horizon}"] = bool(realized > 0)
            row[f"target_hit_{horizon}"] = bool(target_hit)
        rows.append(row)
    return pd.DataFrame(rows)


def _safe_auc(y, p):
    y = np.asarray(y, dtype=int)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None


def _bootstrap_days(frame: pd.DataFrame, horizon: int, iterations: int,
                    seed: int) -> dict:
    rng = np.random.default_rng(seed)
    day = pd.to_datetime(frame["time"], utc=True).dt.date
    days = np.asarray(sorted(day.unique()), dtype=object)
    aucs, deltas = [], []
    for _ in range(iterations):
        sampled = rng.choice(days, size=len(days), replace=True)
        parts = [frame.loc[day == value] for value in sampled]
        boot = pd.concat(parts, ignore_index=True)
        auc = _safe_auc(boot[f"positive_{horizon}"], boot["proba"])
        if auc is not None:
            aucs.append(auc)
        accepted = boot[boot["accepted_035"]][f"r_{horizon}"]
        rejected = boot[~boot["accepted_035"]][f"r_{horizon}"]
        if len(accepted) and len(rejected):
            deltas.append(float(accepted.mean() - rejected.mean()))

    def interval(values):
        if not values:
            return None
        return [float(x) for x in np.quantile(values, [0.025, 0.975])]

    return {"auc_95ci_day_bootstrap": interval(aucs),
            "accepted_minus_rejected_meanR_95ci_day_bootstrap": interval(deltas),
            "iterations": int(iterations)}


def summarize(frame: pd.DataFrame, *, bootstrap: int, seed: int) -> dict:
    out = {
        "signals": int(len(frame)),
        "days": int(pd.to_datetime(frame["time"], utc=True).dt.date.nunique()),
        "proba": {
            "mean": float(frame["proba"].mean()),
            "median": float(frame["proba"].median()),
            "min": float(frame["proba"].min()),
            "max": float(frame["proba"].max()),
        },
        "frozen_threshold": FROZEN_THRESHOLD,
        "accepted": int(frame["accepted_035"].sum()),
        "horizons": {},
    }
    for horizon in HORIZONS:
        y = frame[f"positive_{horizon}"].astype(int)
        target_y = frame[f"target_hit_{horizon}"].astype(int)
        accepted = frame[frame["accepted_035"]][f"r_{horizon}"]
        rejected = frame[~frame["accepted_035"]][f"r_{horizon}"]
        p = frame["proba"].to_numpy(float)
        out["horizons"][str(horizon)] = {
            "positive_rate": float(y.mean()),
            "target_hit_rate": float(target_y.mean()),
            "roc_auc_profitable": _safe_auc(y, p),
            "average_precision_profitable": (
                float(average_precision_score(y, p)) if y.nunique() == 2 else None),
            "brier_profitable": float(np.mean((p - y.to_numpy()) ** 2)),
            "roc_auc_target_hit": _safe_auc(target_y, p),
            "rank_corr_proba_realizedR": float(
                frame["proba"].rank().corr(frame[f"r_{horizon}"].rank())),
            "all_meanR": float(frame[f"r_{horizon}"].mean()),
            "accepted_meanR": float(accepted.mean()) if len(accepted) else None,
            "rejected_meanR": float(rejected.mean()) if len(rejected) else None,
            "accepted_minus_rejected_meanR": (
                float(accepted.mean() - rejected.mean())
                if len(accepted) and len(rejected) else None),
            "bootstrap": _bootstrap_days(frame, horizon, bootstrap,
                                         seed + horizon),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh-csv", default="data/certification/NQ_3min.csv")
    ap.add_argument("--start", default="2026-06-05")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="research_reports")
    args = ap.parse_args()
    if args.bootstrap < 100:
        raise SystemExit("--bootstrap must be >= 100")

    config.TIMEFRAME_MIN = 3
    config.SYMBOL = "NQ"
    start = pd.Timestamp(args.start, tz="UTC")
    fresh_csv = os.path.abspath(args.fresh_csv)
    df, datasets = _load_bars(fresh_csv)
    _install_pickle_compat()
    require_xgboost_compat()

    all_frames = []
    models = {}
    summaries = {}
    for name in STRATEGIES:
        strategy = strategies.make_strategies([name])[0]
        model_path = strategy.model_path()
        bundle = joblib.load(model_path)
        if bundle.get("chronos_ckpt") != "amazon/chronos-bolt-tiny":
            raise SystemExit(f"{name}: unexpected Chronos source")
        print(f"▶ {name}: detecting deployed signals", flush=True)
        events = _collect_events(df, start, name)
        print(f"  {len(events)} complete events; embedding/scoring", flush=True)
        frame = _score_events(df, events, name, bundle)
        if frame.empty:
            summaries[name] = {"signals": 0}
        else:
            all_frames.append(frame)
            summaries[name] = summarize(
                frame, bootstrap=args.bootstrap, seed=args.seed)
        models[name] = {
            "path": os.path.relpath(model_path, HERE),
            "sha256": file_sha256(model_path),
            "labeler_name": bundle.get("labeler_name"),
            "labeler_config": bundle.get("labeler_config"),
            "feat_dim": int(bundle.get("feat_dim")),
            "ctx_window": int(bundle.get("ctx_window")),
            "training_metadata": bundle.get("training_metadata"),
        }

    os.makedirs(args.output_dir, exist_ok=True)
    stem = "ffm_power_exploratory_2026-06-05_2026-07-10"
    event_path = os.path.join(args.output_dir, f"{stem}_events.csv")
    report_path = os.path.join(args.output_dir, f"{stem}.json")
    events = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    events.to_csv(event_path, index=False)
    report = {
        "kind": "EXPLORATORY / CONSUMED DATA — not OOS certification",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "symbol": "NQ",
        "timeframe_min": 3,
        "window": {"start": start.isoformat(),
                   "data_through": df["time"].iloc[-1].isoformat()},
        "frozen_design": {
            "target_R": TARGET_R,
            "stop_R": STOP_R,
            "horizons_bars": list(HORIZONS),
            "live_probability_threshold": FROZEN_THRESHOLD,
            "same_bar_tie": "stop first",
            "costs": "not included",
            "threshold_sweep": False,
        },
        "limitations": [
            "June/July outcomes were previously consumed by exit-policy certification.",
            "Bundles omitted exact labeler horizon/barrier configuration; 3R/1R "
            "and 20/40/80 bars are a predeclared robustness grid, not recovered truth.",
            "Only NQ has matching post-cutoff data.",
            "Live reconstruction emits 68/76 FFM features; missing fields remain NaN.",
            "Twenty-six resampled bars contain two source minutes and no forward fill.",
            "Booster score parity is verified for XGBoost 3.1.2–3.3.0; this run "
            "pins 3.3.0. Versions <=3.0 materially change probabilities.",
        ],
        "datasets": datasets,
        "models": models,
        "summaries": summaries,
        "events": {"path": os.path.relpath(event_path, HERE),
                   "sha256": file_sha256(event_path),
                   "rows": int(len(events))},
        "software": {"python": platform.python_version(),
                     "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit_learn": sklearn.__version__,
                     "xgboost": xgboost.__version__},
    }
    tmp = f"{report_path}.tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    os.replace(tmp, report_path)
    print(f"\nreport → {report_path}")
    print(f"events → {event_path}")


if __name__ == "__main__":
    main()
