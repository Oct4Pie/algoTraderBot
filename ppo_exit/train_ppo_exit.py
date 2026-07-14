#!/usr/bin/env python3
"""train_ppo_exit.py — train the strategy-agnostic PPO trailing-exit policy.

    pip install -r requirements.txt
    python train_ppo_exit.py                      # full train (~600k steps)
    python train_ppo_exit.py --quick              # fast smoke test
    python train_ppo_exit.py --timesteps 800000   # train harder

The policy is standard across every strategy: it manages the same 0.5×ATR(20)
stop from the trade's R-state alone, so one policy fits supertrend / ema /
keltner / bos. SuperTrend flips are used only as a representative catalog of NQ
entry points to train and benchmark on.

Pipeline:
    1. load NQ 3-min bars, precompute the trail/stop ATR
    2. catalog SuperTrend flips and make an 80-bar-purged chronological split
    3. PPO learns a trailing-stop tightness policy on the train trades
    4. benchmark on reusable validation (never presented as final OOS)
    5. export the policy to ppo_trail_exit.npz (torch-free, for the live bot)

The exported .npz (models/rl_trail_exit/) is what bot.py loads at runtime.
"""
import argparse
import datetime as dt
import os
import platform

import numpy as np
import pandas as pd

from ppo_exit.trail_exit_env import (
    build_arrays, build_catalog, make_env, TrailExitSim,
    NumpyMlpPolicy, TRAIL_MULTS, N_ACTIONS, MAX_HOLD,
)
from research_validation import (
    file_sha256, purged_train_eval_split, require_leak_safe_probability_filter,
    validation_protocol, write_policy_metadata,
)

HERE = os.path.dirname(os.path.abspath(__file__))          # the ppo_exit package
REPO = os.path.dirname(HERE)                                # repo root (data lives here)
DATA_CSV = os.path.join(REPO, "data", "NQ_3min.csv")
RL_DIR = os.path.join(HERE, "policies")                    # trained policies live in-package
OUT_NPZ = os.path.join(RL_DIR, "ppo_trail_exit.npz")
SB3_ZIP = os.path.join(RL_DIR, "ppo_trail_exit_sb3.zip")
VALIDATION_FRAC = 0.10         # reusable model-development slice; NOT final OOS


# ── evaluation helpers (pure numpy) ────────────────────────────────────

def eval_policy(arr, catalog, choose):
    """Run every trade in `catalog` through the sim, picking actions with
    `choose(obs) -> action`. Returns (mean_R, win_rate, profit_factor, n)."""
    sim = TrailExitSim(arr)
    rs = []
    for entry_idx, sign in catalog:
        obs = sim.reset(int(entry_idx), int(sign))
        done = False
        while not done:
            obs, _, done, info = sim.step(choose(obs))
        rs.append(info["realized_R"])
    rs = np.asarray(rs)
    wins, losses = rs[rs > 0].sum(), -rs[rs < 0].sum()
    pf = wins / losses if losses > 0 else float("inf")
    return rs.mean(), (rs > 0).mean(), pf, len(rs)


def eval_fixed_2r(arr, catalog, rr=2.0):
    """Baseline = a fixed exit on the same 0.5×ATR risk: stop at STOP_ATR×ATR(ATR_P),
    take-profit at `rr` R, whichever the bar hits first."""
    from config import STOP_ATR
    close, high, low = arr["close"], arr["high"], arr["low"]
    atr_stop, n = arr["atr_stop"], len(arr["close"])
    rs = []
    for entry_idx, sign in catalog:
        entry = close[entry_idx]
        risk = STOP_ATR * atr_stop[entry_idx]
        stop = entry - sign * risk
        target = entry + sign * rr * risk
        realized = None
        for k in range(1, MAX_HOLD + 1):
            j = entry_idx + k
            if j >= n:
                realized = sign * (close[j - 1] - entry) / risk
                break
            hit_stop = (low[j] <= stop) if sign > 0 else (high[j] >= stop)
            hit_tp = (high[j] >= target) if sign > 0 else (low[j] <= target)
            if hit_stop:                       # assume stop first if both touch
                realized = -1.0
                break
            if hit_tp:
                realized = rr
                break
        if realized is None:
            realized = sign * (close[entry_idx + MAX_HOLD] - entry) / risk \
                if entry_idx + MAX_HOLD < n else 0.0
        rs.append(realized)
    rs = np.asarray(rs)
    wins, losses = rs[rs > 0].sum(), -rs[rs < 0].sum()
    pf = wins / losses if losses > 0 else float("inf")
    return rs.mean(), (rs > 0).mean(), pf, len(rs)


def _report(name, stats):
    mean_R, wr, pf, n = stats
    print(f"  {name:<22} meanR={mean_R:+.3f}  WR={wr:5.1%}  "
          f"PF={pf:4.2f}  n={n}")


# ── policy export (torch-free) ─────────────────────────────────────────

def export_policy(model, path):
    """Pull the MLP weights out of the SB3 policy into a plain .npz the live
    bot can run with numpy only (no torch/SB3 next to xgboost)."""
    import torch.nn as nn

    linears = []
    for m in model.policy.mlp_extractor.policy_net:
        if isinstance(m, nn.Linear):
            linears.append(m)
    linears.append(model.policy.action_net)        # logits head

    out = {"n_layers": np.int64(len(linears))}
    for k, lin in enumerate(linears):
        out[f"w{k}"] = lin.weight.detach().cpu().numpy().astype(np.float32)
        out[f"b{k}"] = lin.bias.detach().cpu().numpy().astype(np.float32)
    np.savez(path, **out)
    print(f"✅ exported torch-free policy → {os.path.relpath(path, HERE)}")


# ── main ───────────────────────────────────────────────────────────────

def main():
    import config
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframe", type=int, default=config.TRAINED_TIMEFRAME_MIN,
                    help="bar interval to train for (default %(default)s). Reads "
                         "data/NQ_<tf>min.csv, grades with the <tf>-min SuperTrend "
                         "model, and writes ppo_trail_exit[_<tf>min].npz")
    ap.add_argument("--csv", default=None,
                    help="override the data file (default data/NQ_<timeframe>min.csv)")
    ap.add_argument("--timesteps", type=int, default=400_000)
    ap.add_argument("--quick", action="store_true",
                    help="20k steps on a slice — just proves the pipeline")
    ap.add_argument("--proba-floor", type=float, default=0.0,
                    help="historical entry-probability filter. Leak-safe default "
                         "is 0 because shipped-model scores are not out-of-fold")
    ap.add_argument("--unsafe-final-model-filter", action="store_true",
                    help="RESEARCH ONLY: allow historical filtering by the final "
                         "entry model; resulting metrics cannot be certified")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    require_leak_safe_probability_filter(
        args.proba_floor, args.unsafe_final_model_filter)

    config.TIMEFRAME_MIN = args.timeframe          # pick model/data/policy for this tf
    cfg = config.apply_exit_config()               # train on THIS timeframe's exit shaping
    print(f"  exit config ({args.timeframe}-min): "
          f"ACTIVATE_R={config.ACTIVATE_R} GIVEBACK_R={config.GIVEBACK_R} "
          f"STOP_ATR={config.STOP_ATR}" + ("" if cfg else "  (defaults — no JSON entry)"))
    csv = args.csv or os.path.join(REPO, "data", f"NQ_{args.timeframe}min.csv")
    suffix = "" if args.timeframe == config.TRAINED_TIMEFRAME_MIN else f"_{args.timeframe}min"
    artifact_suffix = suffix
    if args.quick:
        artifact_suffix += "_quick"
    if args.proba_floor > 0:
        artifact_suffix += "_unsafe"
    out_npz = os.path.join(RL_DIR, f"ppo_trail_exit{artifact_suffix}.npz")
    sb3_zip = os.path.join(RL_DIR, f"ppo_trail_exit{artifact_suffix}_sb3.zip")

    import stable_baselines3
    from stable_baselines3 import PPO

    print(f"▶ loading {csv}  (timeframe {args.timeframe}-min)")
    df = pd.read_csv(csv)
    if args.quick:
        df = df.iloc[:60_000].reset_index(drop=True)
    arr = build_arrays(df)
    catalog = build_catalog(arr)

    # Match live entries: only train the exit on flips the bot would take.
    # Grade proba with xgboost in a SUBPROCESS so this (torch/SB3) process never
    # loads xgboost — they segfault together on macOS.
    if args.proba_floor > 0:
        from ppo_exit import precompute_proba as pp
        proba = pp.read_cache(df, catalog, csv)
        if proba is None:
            pp.grade_in_subprocess(csv, rows=(60_000 if args.quick else None))
            proba = pp.read_cache(df, catalog, csv)
        if proba is None:
            raise SystemExit("proba grading failed (see subprocess output)")
        kept = proba >= args.proba_floor
        print(f"  proba floor {args.proba_floor}: kept {kept.sum()} / "
              f"{len(catalog)} flips (the bot's real entries)")
        catalog = catalog[kept]

    # Chronology alone is insufficient: an entry can consume MAX_HOLD future
    # bars. Purge the train tail so no episode reaches the validation partition.
    cut = int(len(df) * (1 - VALIDATION_FRAC))
    train_cat, val_cat = purged_train_eval_split(catalog, cut, MAX_HOLD)
    print(f"  flips: {len(catalog)} total | {len(train_cat)} train | "
          f"{len(val_cat)} validation | embargo={MAX_HOLD} bars")
    if not len(train_cat) or not len(val_cat):
        raise SystemExit("purged train/validation split produced an empty partition")

    env = make_env(arr, train_cat, seed=args.seed)
    timesteps = 20_000 if args.quick else args.timesteps
    print(f"▶ training PPO for {timesteps:,} steps "
          f"(actions = ATR mults {list(TRAIL_MULTS)})")
    model = PPO("MlpPolicy", env, seed=args.seed, verbose=1,
                n_steps=2048, batch_size=256, gamma=0.999, ent_coef=0.01)
    model.learn(total_timesteps=timesteps)
    model.save(sb3_zip)

    export_policy(model, out_npz)

    # ── benchmark on reusable validation data ──────────────────────────
    # This is intentionally NOT called OOS/test: developers may rerun it. Final
    # certification belongs on fresh post-protocol data via optimize_exit --certify.
    policy = NumpyMlpPolicy.load(out_npz)
    print("\n── VALIDATION exit comparison (reusable; not final OOS) ──")
    fixed_stats = eval_fixed_2r(arr, val_cat)
    _report("fixed 2R (current)", fixed_stats)
    constant_stats = {}
    for a in range(N_ACTIONS):
        stats = eval_policy(arr, val_cat, lambda o, a=a: a)
        constant_stats[str(float(TRAIL_MULTS[a]))] = stats
        _report(f"const trail {TRAIL_MULTS[a]:.2f}x ATR", stats)
    ppo_stats = eval_policy(arr, val_cat, policy.action)
    _report("PPO trailing exit", ppo_stats)

    def _stats_dict(stats):
        mean_r, wr, pf, n = stats
        return {"meanR": float(mean_r), "wr": float(wr),
                "pf": float(pf), "n": int(n)}

    write_policy_metadata(out_npz, {
        "protocol_version": int(validation_protocol()["protocol_version"]),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "timeframe_min": int(args.timeframe),
        "dataset": os.path.relpath(csv, REPO),
        "dataset_sha256": file_sha256(csv),
        "train_cut_bar": int(cut),
        "train_entries": int(len(train_cat)),
        "validation_entries": int(len(val_cat)),
        "embargo_bars": int(MAX_HOLD),
        "historical_probability_filter": (
            "none" if args.proba_floor == 0 else "unsafe_final_model"),
        "quick": bool(args.quick),
        "seed": int(args.seed),
        "requested_timesteps": int(timesteps),
        "actual_timesteps": int(model.num_timesteps),
        "training_device": str(model.device),
        "software": {
            "python": platform.python_version(),
            "stable_baselines3": stable_baselines3.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "config": {"ACTIVATE_R": config.ACTIVATE_R,
                   "GIVEBACK_R": config.GIVEBACK_R,
                   "STOP_ATR": config.STOP_ATR},
        "validation": {
            "fixed_2r": _stats_dict(fixed_stats),
            "constant_trails": {k: _stats_dict(v)
                                for k, v in constant_stats.items()},
            "ppo_policy": _stats_dict(ppo_stats),
        },
        "certified": False,
    })
    if args.quick or args.proba_floor > 0:
        reason = "smoke" if args.quick else "unsafe-filter research"
        print(f"\n{reason.capitalize()} artifact only → {os.path.basename(out_npz)} "
              "(live policy unchanged).")
    else:
        print(f"\nDone. The live bot loads {os.path.basename(out_npz)} "
              f"on --timeframe {args.timeframe}.")


if __name__ == "__main__":
    main()
