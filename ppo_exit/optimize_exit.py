#!/usr/bin/env python3
"""optimize_exit.py — Optuna search for the best PPO trailing-exit config.

The exit shape is governed by three knobs (config.py):
    ACTIVATE_R  hold the initial stop until the peak reaches this many R
    GIVEBACK_R  once trailing, never sit more than this many R below the peak
    STOP_ATR    initial stop = STOP_ATR × ATR(ATR_P)

The PPO trail collapses to this give-back cap (the policy ≈ the best constant
trail), so we can score a config WITHOUT retraining: replay the give-back sim
(TrailExitSim, the exact training=live exit) over purged VALIDATION data. Tuning
never reads test outcomes. A separate one-shot certification command evaluates a
frozen config only after fresh post-protocol data exists.

    python optimize_exit.py --timeframe 1 --tickers NQ ES RTY YM GC --trials 200

Plug the printed config into config.py, then retrain: train_ppo_exit --timeframe N.
"""
import argparse
import hashlib
import os

import numpy as np
import pandas as pd

import json

import config
from ppo_exit.trail_exit_env import (
    MAX_HOLD, TRAIL_MULTS, NumpyMlpPolicy, TrailExitSim, build_arrays,
    build_catalog,
)
from research_validation import (
    certification_test_start, policy_is_leak_safe, purged_window,
    require_leak_safe_probability_filter, validation_protocol,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)                                # repo root (data lives here)


# ── data prep (per ticker) ─────────────────────────────────────────────────

def _load_ticker(symbol, tf, proba_floor, fresh_data_dir=None):
    csv = os.path.join(REPO, "data", f"{symbol}_{tf}min.csv")
    if not os.path.exists(csv):
        raise SystemExit(f"no data file: {csv} (copy it from the FFM data dir)")
    df = pd.read_csv(csv)
    sources = {"training_context": csv}
    if fresh_data_dir:
        fresh_csv = os.path.join(fresh_data_dir, f"{symbol}_{tf}min.csv")
        if not os.path.exists(fresh_csv):
            raise SystemExit(f"no fresh certification data file: {fresh_csv}")
        fresh = pd.read_csv(fresh_csv)
        required = {"datetime", "open", "high", "low", "close", "volume"}
        missing = required.difference(fresh.columns)
        if missing:
            raise SystemExit(
                f"fresh certification data {fresh_csv} is missing columns: "
                f"{sorted(missing)}")
        base_ts = pd.to_datetime(df["datetime"], utc=True)
        fresh_ts = pd.to_datetime(fresh["datetime"], utc=True)
        if fresh.empty or fresh_ts.isna().any():
            raise SystemExit(f"fresh certification data is empty/invalid: {fresh_csv}")
        if not fresh_ts.is_monotonic_increasing or fresh_ts.duplicated().any():
            raise SystemExit(
                f"fresh certification timestamps must be sorted and unique: {fresh_csv}")
        if base_ts.max() >= fresh_ts.min():
            raise SystemExit(
                "fresh certification data overlaps the frozen training context "
                f"({fresh_ts.min()} <= {base_ts.max()})")
        df = pd.concat([df, fresh], ignore_index=True)
        sources["fresh_holdout"] = fresh_csv
    arr = build_arrays(df)
    catalog = build_catalog(arr)
    if proba_floor > 0:                       # only the flips the bot would enter
        from ppo_exit import precompute_proba as pp
        config.TIMEFRAME_MIN, config.SYMBOL = tf, symbol   # pick model + cache
        proba = pp.read_cache(df, catalog, csv)
        if proba is None:
            pp.grade_in_subprocess(csv)
            proba = pp.read_cache(df, catalog, csv)
        if proba is None:
            raise SystemExit(f"proba grading failed for {symbol}")
        catalog = catalog[proba >= proba_floor]
    return arr, catalog, df, sources


def _split(catalog, n_bars, val_lo=0.60, val_hi=0.80, horizon=MAX_HOLD):
    """Time-ordered tuning window plus its unused later historical population.

    The early 60% is unused here (no model is trained — the config is the only
    'parameter'). Validation is purged by the full outcome horizon so none of
    its trades consume later prices. The returned later slice is not called a
    test: repository development already exposed all pre-protocol history."""
    idx = catalog[:, 0]
    lo, hi = int(val_lo * n_bars), int(val_hi * n_bars)
    val = purged_window(catalog, lo, hi, horizon)
    later = catalog[idx >= hi]
    return val, later


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _certification_catalog(df, catalog, start):
    times = pd.to_datetime(df["datetime"], utc=True)
    first = int(times.searchsorted(start, side="left"))
    idx = catalog[:, 0]
    # Do not certify entries whose full MAX_HOLD outcome is unavailable yet.
    return catalog[(idx >= first) & (idx + MAX_HOLD < len(df))]


def _certify(raw_datasets, *, timeframe, tickers, test_start,
             min_trades, output_dir):
    """One-shot, read-only-on-parameters certification on fresh data."""
    start = certification_test_start(test_start)
    cert_sets = []
    hashes = {}
    coverage = {}
    total = 0
    for symbol, arr, catalog, df, sources in raw_datasets:
        cat = _certification_catalog(df, catalog, start)
        cert_sets.append((arr, {"test": cat}))
        hashes[symbol] = {
            role: _sha256(path) for role, path in sources.items()
        }
        times = pd.to_datetime(df["datetime"], utc=True)
        coverage[symbol] = {
            "data_first": times.iloc[0].isoformat(),
            "data_last": times.iloc[-1].isoformat(),
            "test_entries": int(len(cat)),
            "first_test_entry": (
                times.iloc[int(cat[0, 0])].isoformat() if len(cat) else None),
            "last_test_entry": (
                times.iloc[int(cat[-1, 0])].isoformat() if len(cat) else None),
        }
        total += len(cat)
    if total < min_trades:
        raise SystemExit(
            f"certification population has {total} entries; need at least "
            f"{min_trades}. No outcome metrics were evaluated"
        )

    policy_path = config.policy_path()
    if not os.path.exists(policy_path):
        raise SystemExit(f"cannot certify missing policy: {policy_path}")
    expected_config = {"ACTIVATE_R": config.ACTIVATE_R,
                       "GIVEBACK_R": config.GIVEBACK_R,
                       "STOP_ATR": config.STOP_ATR}
    if not policy_is_leak_safe(policy_path, expected_config):
        raise SystemExit(
            f"cannot certify legacy/unsafe policy without hash-bound purged-training "
            f"provenance: {policy_path}. Retrain it first"
        )
    policy = NumpyMlpPolicy.load(policy_path)

    os.makedirs(output_dir, exist_ok=True)
    # Exactly one final look per protocol/timeframe, regardless of test-start,
    # universe, or config. A new attempt requires a deliberately versioned
    # protocol and a later untouched dataset—not a shifted date over old outcomes.
    protocol_version = int(validation_protocol()["protocol_version"])
    path = os.path.join(
        output_dir, f"exit_policy_protocol{protocol_version}_{timeframe}min.json")
    if os.path.exists(path):
        raise SystemExit(
            f"certification window already consumed: {path}. Do not reuse a test "
            "after seeing its outcomes; collect a new later window"
        )

    # Reserve the window BEFORE reading any outcomes. A crash leaves a consumed
    # reservation instead of silently allowing a second look at the same test.
    test_data_through = max(
        pd.Timestamp(item["data_last"]) for item in coverage.values()
    ).isoformat()
    with open(path, "x") as f:
        json.dump({"protocol": protocol_version, "status": "reserved/consumed",
                   "timeframe_min": int(timeframe),
                   "test_start": start.isoformat(),
                   "test_data_through": test_data_through}, f, indent=2)

    metrics = {
        "ppo_policy": _metrics(_pooled_policy_R(cert_sets, "test", policy)),
        "constant_1atr_reference": _metrics(_pooled_R(cert_sets, "test", 1)),
    }
    report = {
        "protocol": protocol_version,
        "kind": "one-shot fresh-data exit-policy certification",
        "timeframe_min": int(timeframe),
        "tickers": list(tickers),
        "test_start": start.isoformat(),
        "test_data_through": test_data_through,
        "coverage": coverage,
        "embargo_bars": MAX_HOLD,
        "config": {
            "ACTIVATE_R": config.ACTIVATE_R,
            "GIVEBACK_R": config.GIVEBACK_R,
            "STOP_ATR": config.STOP_ATR,
        },
        "data_sha256": hashes,
        "policy": {
            "path": os.path.relpath(policy_path, REPO),
            "sha256": _sha256(policy_path),
        },
        "metrics": metrics,
    }
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, path)
    return path, metrics


# ── give-back replay (no PPO training needed) ──────────────────────────────

def _realized_R(arr, catalog, action):
    sim = TrailExitSim(arr)
    out = np.empty(len(catalog), dtype=np.float64)
    for r, (entry_idx, sign) in enumerate(catalog):
        sim.reset(int(entry_idx), int(sign))
        done = False
        while not done:
            _o, _r, done, info = sim.step(action)
        out[r] = info["realized_R"]
    return out


def _pooled_R(datasets, which, action):
    parts = [_realized_R(arr, cats[which], action) for arr, cats in datasets]
    return np.concatenate(parts) if parts else np.empty(0)


def _policy_R(arr, catalog, policy):
    sim = TrailExitSim(arr)
    out = np.empty(len(catalog), dtype=np.float64)
    for r, (entry_idx, sign) in enumerate(catalog):
        obs = sim.reset(int(entry_idx), int(sign))
        done = False
        while not done:
            obs, _reward, done, info = sim.step(policy.action(obs))
        out[r] = info["realized_R"]
    return out


def _pooled_policy_R(datasets, which, policy):
    parts = [_policy_R(arr, cats[which], policy) for arr, cats in datasets]
    return np.concatenate(parts) if parts else np.empty(0)


def _metrics(R):
    if len(R) == 0:
        return dict(meanR=0.0, wr=0.0, pf=0.0, sumR=0.0, n=0)
    wins, losses = R[R > 0].sum(), -R[R < 0].sum()
    return dict(meanR=float(R.mean()), wr=float((R > 0).mean()),
                pf=(float(wins / losses) if losses > 0 else float("inf")),
                sumR=float(R.sum()), n=int(len(R)))


def _save_config(tf, params):
    """Write the winning config to exit_configs.json[<tf>], preserving other keys."""
    path = config.EXIT_CONFIGS_PATH
    try:
        with open(path) as f:
            cfgs = json.load(f)
    except (FileNotFoundError, ValueError):
        cfgs = {}
    cfgs[str(tf)] = {k: round(float(params[k]), 3)
                     for k in ("ACTIVATE_R", "GIVEBACK_R", "STOP_ATR")}
    with open(path, "w") as f:
        json.dump(cfgs, f, indent=2)


def _score(datasets, which, scan_mults, objective):
    """Best-achievable metric for the current tee.* config over `which` slice.
    scan_mults: if True, take the best over all trail mults (when the cap doesn't
    bind); else use a single representative mult (the cap usually dominates)."""
    actions = range(len(TRAIL_MULTS)) if scan_mults else [1]   # [1] = 1.0×ATR
    best = None
    for a in actions:
        m = _metrics(_pooled_R(datasets, which, a))
        v = m[objective]
        if best is None or v > best:
            best = v
    return best


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframe", type=int, default=config.TRAINED_TIMEFRAME_MIN)
    ap.add_argument("--tickers", nargs="+", default=["NQ"],
                    help="symbols to pool for more data (default NQ)")
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--proba-floor", type=float, default=0.0,
                    help="historical entry-probability filter. Leak-safe default "
                         "is 0 because shipped-model scores are not out-of-fold")
    ap.add_argument("--unsafe-final-model-filter", action="store_true",
                    help="RESEARCH ONLY: permit final-model historical filtering; "
                         "results cannot be certified")
    ap.add_argument("--objective", default="meanR",
                    choices=["meanR", "pf", "sumR"])
    # search bounds (widen STOP_ATR for noisier timeframes like 1-min)
    ap.add_argument("--stop-min", type=float, default=0.4)
    ap.add_argument("--stop-max", type=float, default=1.0)
    ap.add_argument("--activate-min", type=float, default=1.0)
    ap.add_argument("--activate-max", type=float, default=3.0)
    ap.add_argument("--giveback-min", type=float, default=0.25)
    ap.add_argument("--giveback-max", type=float, default=1.5)
    ap.add_argument("--scan-mults", action="store_true",
                    help="evaluate every trail mult (slower; only matters when the "
                         "give-back cap doesn't bind)")
    ap.add_argument("--progress", action="store_true",
                    help="show Optuna's live progress bar (off by default — noisy in logs)")
    ap.add_argument("--save", action="store_true",
                    help="write the winner only if it beats the current config on "
                         "validation; fresh final-test data is never read")
    ap.add_argument("--certify", action="store_true",
                    help="skip tuning and evaluate the current saved config once on "
                         "fresh post-protocol data")
    ap.add_argument("--test-start",
                    help="with --certify: first untouched test timestamp/date")
    ap.add_argument("--fresh-data-dir",
                    help="with --certify: directory containing separate fresh "
                         "<symbol>_<timeframe>min.csv files. They are appended "
                         "in memory after the frozen training CSV for indicator "
                         "warmup; the training CSV is never modified")
    ap.add_argument("--min-cert-trades", type=int, default=100,
                    help="minimum fresh entries required before outcomes are evaluated")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.certify and args.save:
        raise SystemExit("--certify is read-only on parameters; do not combine it with --save")
    if args.certify and not args.test_start:
        raise SystemExit("--certify requires --test-start on fresh, untouched data")
    if args.fresh_data_dir and not args.certify:
        raise SystemExit("--fresh-data-dir is only valid with --certify")
    require_leak_safe_probability_filter(
        args.proba_floor, args.unsafe_final_model_filter)
    if args.certify and args.unsafe_final_model_filter:
        raise SystemExit("certification forbids final-model historical probability filtering")
    if args.certify:
        # Fail before loading/replaying any data when the requested window is not fresh.
        certification_test_start(args.test_start)

    config.TIMEFRAME_MIN = args.timeframe

    print(f"▶ loading {args.tickers} @ {args.timeframe}-min "
          f"(proba≥{args.proba_floor})…")
    datasets, raw_datasets = [], []
    for sym in args.tickers:
        arr, catalog, df, sources = _load_ticker(
            sym, args.timeframe, args.proba_floor,
            fresh_data_dir=args.fresh_data_dir)
        val, later = _split(catalog, len(df))
        datasets.append((arr, {"val": val}))
        raw_datasets.append((sym, arr, catalog, df, sources))
        print(f"   {sym}: {len(catalog)} flips | {len(val)} validation | "
              f"{len(later)} later historical entries not read by tuner | "
              f"embargo={MAX_HOLD}")

    config.apply_exit_config(args.timeframe)        # baseline = this tf's saved config
    base_cfg = (config.ACTIVATE_R, config.GIVEBACK_R, config.STOP_ATR)

    if args.certify:
        path, metrics = _certify(
            raw_datasets, timeframe=args.timeframe, tickers=args.tickers,
            test_start=args.test_start, min_trades=args.min_cert_trades,
            output_dir=os.path.join(HERE, "certifications"))
        m = metrics["ppo_policy"]
        print(f"\nONE-SHOT CERTIFICATION → {path}")
        print(f"   PPO meanR={m['meanR']:+.3f} WR={m['wr']:.1%} "
              f"PF={m['pf']:.2f} sumR={m['sumR']:+.1f} n={m['n']}")
        return

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)   # one line per improvement

    def objective(trial):
        config.ACTIVATE_R = trial.suggest_float("ACTIVATE_R", args.activate_min,
                                                args.activate_max, step=0.25)
        config.GIVEBACK_R = trial.suggest_float("GIVEBACK_R", args.giveback_min,
                                                args.giveback_max, step=0.25)
        config.STOP_ATR = trial.suggest_float("STOP_ATR", args.stop_min,
                                              args.stop_max, step=0.1)
        return _score(datasets, "val", args.scan_mults, args.objective)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=args.progress)

    best = study.best_params
    best_cfg = (best["ACTIVATE_R"], best["GIVEBACK_R"], best["STOP_ATR"])

    def validation_metrics(cfg):
        config.ACTIVATE_R, config.GIVEBACK_R, config.STOP_ATR = cfg
        return _metrics(_pooled_R(datasets, "val", 1))

    base, won = validation_metrics(base_cfg), validation_metrics(best_cfg)

    def _row(tag, cfg, m):
        print(f"   {tag:<10} ACTIVATE_R={cfg[0]:.2f} GIVEBACK_R={cfg[1]:.2f} "
              f"STOP_ATR={cfg[2]:.2f} | meanR={m['meanR']:+.3f} WR={m['wr']:5.1%} "
              f"PF={m['pf']:.2f} sumR={m['sumR']:+.1f} n={m['n']}")

    print(f"\n── VALIDATION selection (no final-test outcomes were evaluated) ──")
    _row("baseline", base_cfg, base)
    _row("optuna", best_cfg, won)

    improved = won[args.objective] > base[args.objective]
    if args.save and improved:
        _save_config(args.timeframe, best)
        print(f"\nsaved → exit_configs.json[\"{args.timeframe}\"] (beats current on "
              f"validation {args.objective}; still UNCERTIFIED). Retrain: "
              f"python -m ppo_exit.train_ppo_exit "
              f"--timeframe {args.timeframe}")
    elif args.save:
        print(f"\nNOT saved — best doesn't beat the current config on validation "
              f"{args.objective}.")
    else:
        verb = "improves" if improved else "does not improve"
        print(f"\nCandidate {verb} validation {args.objective}. "
              f"Rerun with --save to write "
              f"exit_configs.json[\"{args.timeframe}\"]:")
        print(f"    ACTIVATE_R={best_cfg[0]:.2f} GIVEBACK_R={best_cfg[1]:.2f} "
              f"STOP_ATR={best_cfg[2]:.2f}")
    print("   Final test requires fresh post-protocol data. After appending it: "
          "--certify --test-start <date-after-protocol-cutoff>")


if __name__ == "__main__":
    main()
