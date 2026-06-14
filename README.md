# SuperTrend AI Bot + PPO trailing exit (NQ futures, 3-min)

A single-file live **TopstepX bot** that trades SuperTrend flips on NQ 3-min
bars, graded by a Chronos+XGBoost AI head and exited by a **PPO-learned
trailing stop**:

```
SuperTrend flip  →  AI grades it (proba)  →  enter if proba ≥ 0.35
                 →  PPO trails the stop bar-by-bar until exit
```

- **Entry** — a SuperTrend flip is the candidate; the Chronos+XGBoost head
  scores it (`proba` = P(win)) and the bot enters only when `proba ≥ 0.35`.
- **Exit** — instead of a fixed 2R take-profit, a small PPO policy decides each
  bar *how tightly to trail the stop* (it only ever ratchets in your favor).

> ⚠️ **Educational — places LIVE orders.** Run it on a practice/evaluation
> account first. NQ 3-min only (the model's training scope).

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # first run downloads a ~45MB Chronos checkpoint
```

Add your TopstepX credentials (the API **key**, not your password) — either edit
the top of `supertrend_ai_bot.py`, or use env vars:

```bash
export TOPSTEPX_USERNAME="your_login"
export TOPSTEPX_API_KEY="your_api_key"
export TOPSTEPX_ACCOUNT=""                # blank = first tradable account
python supertrend_ai_bot.py
```

The bot prints your tradable accounts on startup — make sure it picks your
**practice** account (set `TOPSTEPX_ACCOUNT` to its id/name to pin it).

## How the PPO exit works

Once in a trade, each bar the policy picks a trailing-stop distance from a set
of ATR multiples (`[0.75, 1.0, 1.5, 2.0, 2.5, 3.5]`) and the bot updates the
live stop via `/Order/modify`. The stop never loosens. A trade ends on a stop
hit, a 4-hour max-hold, or end of data. The policy observes 7 inputs (all in
R-multiples so longs/shorts look identical): unrealized R, max favorable
excursion, stop distance, ATR/risk, time in trade, momentum, and distance from
the SuperTrend line.

Two config flags in `supertrend_ai_bot.py` control the exit:

- `USE_PPO_EXIT` (default `True`) — use the PPO trail; set `False` to fall back
  to the original fixed-2R bracket.
- `USE_TRAILING_STOP` (default `True`) — `True` enters with a broker-native
  trailing stop (follows price tick-by-tick; PPO tightens its distance each
  bar); `False` uses a plain stop the PPO reprices each bar. **For a first
  practice run, `False` is the safest — it relies only on the verified
  `stopPrice` modify.**

The live loop reconstructs its state from the broker if restarted mid-trade, so
a restart won't strand an open position.

## Retrain the exit (optional)

```bash
python train_ppo_exit.py                  # full train (~600k steps)
python train_ppo_exit.py --quick          # 20k-step smoke test
```

Trains PPO on every SuperTrend flip in `data/NQ_3min.csv`, **filtered to the
flips the bot would actually enter** (`proba ≥ 0.35`, cached in
`proba_cache.npz`), holds out the last 10% of bars, benchmarks against the
fixed-2R / constant-trail baselines, and writes `ppo_trail_exit.npz` — the
torch-free policy the live bot loads automatically.

## The AI entry head (`predict.py`)

The grading model is a two-head XGBoost on top of a Chronos embedding. Verify
the install on synthetic data:

```bash
python predict.py --demo                  # prints proba = … / r_hat = …
```

Call it directly:

```python
from predict import chronos_embedding, predict
emb = chronos_embedding(closes)           # last 128 closes -> (256,)
proba, r_hat = predict(emb, features)     # features: (78,) -> P(win), peak R
```

Inputs are `concat([embedding_256, hand_crafted_78])` (shape 334). The bot fills
only the two public hand-crafted slots (`adx`, `adx_slope`) and leaves the 76
proprietary ones as `nan` (XGBoost handles missing natively — reduced but real
accuracy). The model was trained on 95,421 flips, 2021-04-25 → 2026-05-04.

## What's in this package

| file | purpose |
|---|---|
| `supertrend_ai_bot.py` | the live bot — entry grading + PPO trailing exit |
| `trail_exit_env.py` | PPO exit env, trade simulator + torch-free policy |
| `train_ppo_exit.py` | trains the trailing-exit policy |
| `precompute_proba.py` | batch-grades every flip with the entry model (cached) |
| `predict.py` | standalone entry-head inference |
| `ppo_trail_exit.npz` | trained PPO policy (loaded live) |
| `signal_head.json` / `risk_head.json` | XGBoost entry heads (`proba` / `r_hat`) |
| `metadata.json` | training spans, dims, holdout stats |
| `data/NQ_3min.csv` | NQ 3-min OHLCV history (for retraining) |
| `requirements.txt` | python dependencies |

## Caveats

- **Scope**: NQ 3-min UTC bars only — other tickers/timeframes are out of
  distribution.
- **Native trailing stop**: the `USE_TRAILING_STOP = True` path uses the
  ProjectX trailing bracket (`type 5`) and `/Order/modify` `trailPrice` — both
  in the API docs, but the docs don't pin down `trailPrice` units (we send
  `ticks × tickSize`) or re-anchor behavior. Verify on practice, or use
  `USE_TRAILING_STOP = False`.
- **torch + xgboost OpenMP conflict (macOS)**: loading both in one process can
  segfault. The bot avoids it — Chronos runs in a subprocess and the PPO policy
  runs in pure numpy. (`KMP_DUPLICATE_LIB_OK=TRUE` is a fallback.)
- Internet needed once for the HuggingFace checkpoint; offline after.
