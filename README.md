# algoTraderBot — multi-strategy AI futures bot (3-min, TopstepX)

A live TopstepX bot that trades **mechanical entries graded by AI**, with a
**reinforcement-learned trailing exit** — for 3-min futures (`NQ`, `ES`, `RTY`,
`YM`, `GC`).

> ⚠️ **Educational — live mode places LIVE orders.** Run it on a
> practice/evaluation account first. (Backtests place **no** orders, but still
> need credentials — contract specs are fetched from the broker API.)

---

## Getting started

### 1. Requirements

- **Python 3.10+** and **git** (dependencies install the public
  [`futures_foundation`](https://github.com/johnamcruz/Futures-Foundation-Model)
  library from GitHub).
- **Internet** — downloads the ~45 MB `amazon/chronos-bolt-tiny` checkpoint on
  first run, and the bot reads contract specs from the broker API at startup.
- A **TopstepX account + API key** — required for **both live and backtest**.
  Tick size / tick value come from the broker API (`/Contract/search`); there is
  no offline mode. Backtests still use **local CSV bars** — only the contract
  specs come from the API.

### 2. Install

```bash
git clone https://github.com/johnamcruz/algoTraderBot.git
cd algoTraderBot
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Do not loosen the `xgboost==3.3.0` pin while the shipped FFM heads remain joblib
pickles. Cross-version verification found prediction parity across 3.1.2–3.3.0,
matching the artifacts' stored holdout probability scale; versions through 3.0
materially change scores. Runtime inference fails closed outside the compatible
3.1+ line instead of silently trading altered models.

> **macOS note:** if you hit a segfault (the torch/xgboost OpenMP clash), prefix
> commands with `KMP_DUPLICATE_LIB_OK=TRUE`, e.g.
> `KMP_DUPLICATE_LIB_OK=TRUE python bot.py …`.

### 3. Add your credentials (`.env`)

Credentials live in a **gitignored `.env`** file. Copy the template:

```bash
cp .env.example .env
```

Open `.env` and fill in your TopstepX details:

```ini
TOPSTEPX_USERNAME=your_login      # your TopstepX login
TOPSTEPX_API_KEY=your_api_key     # API KEY from the dashboard — NOT your password
TOPSTEPX_ACCOUNT=                 # blank = first tradable account, or an id/name
```

- Get the **API key** from your TopstepX dashboard (Settings → API). It is a
  key, not your account password.
- Leave `TOPSTEPX_ACCOUNT` blank to use your first tradable account — the bot
  prints every account on startup, so you can copy an **id or name** here to pin
  a specific one (do this to be sure it's your practice account).
- Real environment variables (`export TOPSTEPX_API_KEY=…`) override `.env` if
  set — handy for CI or secrets managers.

**Verify the install** with a short research backtest (uses your creds for the
contract spec; the first run also downloads the Chronos checkpoint, so give it a
minute). `--allow-in-sample` is required because every shipped bar has already
been exposed during model development; this proves wiring, **not performance**:

```bash
python bot.py --backtest --allow-in-sample --symbol NQ --start 2026-05-23 --end 2026-05-25
```

You should see candles/signals scroll past and a `BACKTEST NQ | trades=… win=…`
summary at the end. If you get that, the wiring is ready; it says nothing about edge.

### 4. Run (live)

```bash
python bot.py                          # config defaults
python bot.py --strategy ema           # run one strategy
python bot.py --strategy ema keltner bos   # run several (highest proba wins)
python bot.py --size 3                  # fixed: 3 contracts per trade
python bot.py --risk 500                # risk-based: ~$500 risked per trade
python bot.py --risk 500 --max-contracts 5
python bot.py --proba-floor 0.45        # only take entries graded ≥ 0.45 confidence
```

**Strategies** (`--strategy` overrides `config.ACTIVE_STRATEGIES`; pick one or
several — when more than one fires on a bar, the highest-proba signal wins):

| name | entry |
|---|---|
| `supertrend` | SuperTrend flip (period 10, mult 3.0) |
| `ema` | 9/20 EMA crossover, gated to ADX ≥ 18 |
| `keltner` | Keltner-channel breakout, gated to ADX ≥ 20 |
| `bos` | break of the last confirmed swing (break of structure) |
| `orb` | 15-min opening-range breakout (09:30 ET), gated to ADX ≥ 18 |
| `cisd_ote` | CISD displacement → OTE fib-zone pullback (mean-reversion, ICT/SMC) |

**Timeframe** (`--timeframe MIN`, default 3): the bar interval. Models are
per-timeframe and there is **no cross-timeframe fallback** — only strategies with a
model for the chosen interval are allowed, so a 3-min model is never run on 1-min
bars. Today: **3-min** has all five strategies; **1-min** has `supertrend` only.

```bash
python bot.py --strategy supertrend --timeframe 1   # 1-min SuperTrend
python bot.py --strategy ema --timeframe 1           # error: no 1-min ema model
```

A non-3-min run loads `data/<symbol>_<tf>min.csv` for backtests and the
`<model>_<tf>min.joblib` entry bundle (e.g. `supertrend_chronos_1min.joblib`). The
PPO exit is per-timeframe too — if no policy exists for the chosen timeframe yet,
the bot **flags it and trains one automatically** (`python -m ppo_exit.train_ppo_exit --timeframe <tf>`,
run once in a subprocess) before trading; it falls back to the fixed-RR exit only if
that training can't produce a policy.

On startup the bot prints your tradable accounts and a banner —
`✅ <account> | <contract> | 3-min | [ema] | conf≥0.35 | exit: PPO stop-reprice |
size: fixed 1`. **Confirm it picked your practice/eval account.** From there,
every candle, every graded signal (with a `TAKE`/`skip` flag), each entry, and
every stop move are logged to the console **and** `log/bot.log`. Stop with
`Ctrl-C`.

Pick strategies and exit shaping in `config.py`:

```python
ACTIVE_STRATEGIES = ["ema"]      # any of: supertrend, ema, keltner, bos, orb (or --strategy)
PROBA_FLOOR       = 0.35         # min entry confidence (or pass --proba-floor)
USE_PPO_EXIT      = True         # False → simple fixed-RR bracket instead
ACTIVATE_R        = 2.0          # default exit shaping; the ACTUAL per-timeframe
GIVEBACK_R        = 0.75         # ACTIVATE_R/GIVEBACK_R/STOP_ATR live in
                                 # ppo_exit/exit_configs.json (see Tune the exit config)
```

**Position sizing** — use a **fixed size** *or* **risk-based sizing** (not both).
With `--risk` (or `config.RISK_PER_TRADE`), contracts are sized from the stop so
each trade risks roughly the same dollars:

```
contracts = min(MAX_CONTRACTS, max(1, floor(risk_$ / (stop_ticks × tick_value))))
```

`tick_value` is read from the broker contract (`/Contract/search` → `tickValue`,
e.g. NQ ≈ $5/tick, MNQ ≈ $0.50/tick). A tighter stop ⇒ more contracts, a wider
stop ⇒ fewer — so dollar risk stays roughly constant, which pairs naturally with
the 0.5×ATR stop. **Micros** (MNQ, MES, MGC, M2K, MYM) just work — same models
and bars as their parent, sized at the micro's smaller `tickValue`.

### 5. Backtest (local bars, live specs)

Backtesting runs the **exact live logic** — same strategies, grading, and PPO
trailing exit — over a local CSV, with a simulated broker filling
entries/stops/trailing against history. Contract specs (tick size / value) are
still looked up from the broker API, so credentials are required.

```bash
# one month of NQ
python bot.py --backtest --allow-in-sample --symbol NQ --start 2026-05-01 --end 2026-06-01

# a micro and a different ticker
python bot.py --backtest --allow-in-sample --symbol MNQ --start 2026-05-01 --end 2026-06-01
python bot.py --backtest --allow-in-sample --symbol ES --risk 500
```

**Leakage firewall.** A backtest fails by default when its first tested bar
overlaps any active model's training span or an already-inspected artifact
holdout or a consumed certification window. Use `--allow-in-sample` only for
debugging/strategy development; the banner labels the result
`RESEARCH/IN-SAMPLE`, and it must not be reported as validation. The canonical
training CSVs end at the development cutoff (2026-06-04); the local June–July NQ
certification snapshot has already been consumed. Append later, genuinely
untouched data before running a backtest without that flag. Such a run is labeled
`UNSEEN/NOT CERTIFIED`; repeated inspection turns it into validation. Final
exit-policy certification is the locked one-shot workflow described below; a
full entry-plus-exit certification still requires a separately frozen forward run.

- `--symbol` reads `data/<symbol>_<timeframe>min.csv` (ships with NQ, ES, RTY, YM, GC at 3-min);
  **micros use their parent's bars** (MNQ → NQ) at the micro's tick value.
- `--start` / `--end` are `YYYY-MM-DD` (**start inclusive, end exclusive**); omit
  either to run from the warmup point / to the end of the file.
- `--size`, `--risk`, `--proba-floor` and the `config.py` knobs all apply, so you
  can A/B a setting by re-running.

It prints a summary — trades, win rate, mean/sum R, profit factor, plus MFE and
per-strategy / per-exit breakdowns — and writes every trade to
`log/backtest_research_<symbol>.csv` or
`log/backtest_unseen_uncertified_<symbol>.csv`. Entries fill at the signal bar's close; when a bar
straddles both stop and target the stop is assumed first. Grading embeds each
signal through Chronos, so longer ranges take a few minutes.

---

## How it works

```
each bar ─► every active strategy detects its entry
        ─► its model grades the signal  →  proba = P(win)
        ─► best signal with proba ≥ floor is taken (highest proba wins)
        ─► one shared Chronos embedding per bar feeds every strategy's grade
        ─► PPO policy trails the stop bar-by-bar until exit
```

Each strategy is a thin signal generator paired with its own Chronos+XGBoost
model; the model decides *which* signals to take, and a PPO policy decides *when
to get out*. The entry models are trained on multiple 3-min futures (NQ, ES, RTY,
YM, GC) and generalize across them; the framework is ticker- and broker-agnostic.

- **Entry** — when flat, every active strategy gets a chance to `detect()` +
  `grade()`. Signals with `proba ≥ PROBA_FLOOR` are candidates; the **highest
  proba wins** (one position per contract). The trade enters at market with a
  protective stop at `0.5×ATR(20)` — exactly how the models scored the trade.
- **Exit** — each bar the PPO policy (`ppo_exit/policies/`) reads the open
  trade's R-state (unrealized R, MFE, stop distance, ATR/risk, time, momentum) —
  **strategy-agnostic**, it never sees how the trade was entered, so one policy
  fits every strategy on the standard 0.5×ATR(20) stop. It computes a
  trailing-stop level and **reprices the live stop to it via `/Order/modify`**
  (ratcheting only in your favor). Two knobs shape it:
  **`ACTIVATE_R`** (hold the initial 1R stop until the peak reaches +2R, so
  winners survive early pullbacks) and **`GIVEBACK_R`** (once trailing, the stop
  never sits more than 0.75R below the running peak). So a trade risks 1R, and
  once it's up +2R it locks in ≥ +1.25R and rides, giving back ≤ 0.75R from the
  best point. The policy forward-pass is pure numpy, so the bot never loads
  torch/SB3 next to xgboost.
- **Trailed-stop enforcement (intra-bar).** The peak is tracked from each bar's
  favorable extreme (not just the close), and if a bar's *unfavorable* wick
  crosses the trailed stop the bot **closes at market** (`/Position/closeContract`)
  rather than rely on a resting broker stop — which the broker rejects ("Invalid
  stop price") when a fast reversal puts the lock level on the wrong side of the
  market. This fixes a bug where a big winner could ride all the way back to the
  initial −1R stop because every stop-modify was rejected. Stops are also
  direction-aware tick-snapped (floor longs / ceil shorts) so rounding never
  lands them on the wrong side.
- **Training = live parity.** The policy is trained inside `TrailExitSim`
  (`ppo_exit/trail_exit_env.py`) and run live by `ppo_exit/exit_manager.py`; the two
  implement the *same* give-back logic — activation gate, peak from the bar's
  favorable extreme, give-back cap, and the `MAX_HOLD` force-exit (the policy
  observes `bars_held/MAX_HOLD`, so live force-exits at the same horizon it was
  trained on). Crucially they share the same **two-tier fill model** (the design
  algoTraderAI uses): the protective stop is a **resting broker stop order kept at
  the give-back floor**, repriced each bar. A give-back exit therefore fills one
  of two ways, and the sim models both so the policy trains on realistic prices:
    - **resting-stop fill at the floor** — when price crosses the floor that was
      already resting from a prior bar (the common, slow give-back); accurate.
    - **market-close at the bar close** — only when a fast spike-and-reverse
      crosses a floor that was *tightened this bar* and isn't a resting order yet,
      so live closes at market (worse than the floor). The old sim recorded this
      optimistically at the floor; it now fills at the bar close, matching live.
  Entry and exit are never the same candle (both engines start evaluating exits on
  the bar *after* entry). All of this is locked by
  `tests/test_train_live_parity.py`, which drives the same trade through both
  engines and asserts an identical exit bar **and price**, including the
  spike-and-reverse market-close case. The PPO was retrained on this corrected sim.

### Order safety (live)

Trading real orders has sharp edges; these are handled so a desync can't leave an
unmanaged position. Each is covered by a test (see [Tests](#tests)):

- **Signed bracket ticks.** SL/TP distances are signed relative to the fill — a
  long's stop is *negative* ticks (below), a short's *positive* (above), TP the
  mirror — clamped to the 4-tick broker minimum. (Fixes *"Invalid stop loss ticks
  (57). Ticks should be less than zero when longing."*)
- **No orphaned brackets.** A market close (`/Position/closeContract`) does **not**
  fire the OCO, so the broker leaves the protective stop working. `close_position`
  therefore sweeps and cancels every resting order for the contract — otherwise a
  stray stop could later fill and open a brand-new **naked** position the bot never
  opened (seen live: a +0.63R short close left its buy-stop, which filled into a
  naked long at the exact stop price).
- **Mid-session reconcile.** A flat account should have no resting orders, so on
  every flat bar the bot cancels any strays (an orphaned bracket, a missed exit, a
  manual order). This is the general safety net: it keeps the account in sync even
  if a close-time cancel was missed, and is what stops the naked-position scenario
  above from persisting.
- **No silent exits.** The common exit is the **resting broker stop** filling
  intra-bar — which happens at the broker, so `manage_trail` never runs and the
  close was previously unlogged (a −1R stop-out just vanished from the logs). The
  bot now detects the in-position→flat transition and logs the exit with its
  realized R, inferred from the level the stop rested at, so every close is on the
  record.
- **ORB session gate.** The 09:30-ET opening range stays mathematically active
  until midnight ET; ORB entries are gated to the RTH window `[~09:45, 16:00)` ET
  (`ORB_CLOSE_MIN`) so the bot doesn't take stale overnight breakouts of the
  morning range.

## Architecture

Small, single-responsibility modules:

| file | responsibility |
|---|---|
| `bot.py` | entry point — `handle_bar` (detect → grade → enter → trail) + live loop + CLI |
| `config.py` | **all settings**: strategies, sizing, exit shaping, strategy params |
| `broker_base.py` | `BrokerClient` / `OrderRouter` — the broker **interface** |
| `broker.py` | `TopstepXClient` (a `BrokerClient`) over the ProjectX Gateway API + `make_broker()` |
| `sim_broker.py` | `SimBroker` (an `OrderRouter`) — fills/stops/trailing against a CSV for backtests |
| `backtest.py` | drives `handle_bar` over history with date-range selection |
| `research_validation.py` | purged splits, unsafe-filter rejection, artifact exposure/OOS firewall |
| `indicators.py` | SuperTrend / ATR / ADX / EMA / Keltner / swings / opening range |
| `embedder.py` / `embed_worker.py` | warm Chronos embedding worker — model loaded once per session |
| `strategies/` | the pluggable strategies (one file each) + shared base |
| `ppo_exit/` | the whole PPO trailing-exit subsystem (see below) |
| `logsetup.py` | logging to `log/bot.log` |
| `models/` | the entry models (joblib) + FFM feature order |

```
strategies/                 ppo_exit/   (the PPO trailing-exit subsystem)
  base.py   Strategy ABC       exit_manager.py    live exit management
  supertrend.py → supertrend   trail_exit_env.py  training env/sim + numpy policy loader
  ema_cross.py  → ema          train_ppo_exit.py  trainer (python -m ppo_exit.train_ppo_exit)
  keltner.py    → keltner      optimize_exit.py   Optuna config search
  bos.py        → bos          precompute_proba.py entry-grading for training
  orb.py        → orb          exit_configs.json  per-timeframe ACTIVATE_R/GIVEBACK_R/STOP_ATR
                               policies/          the trained .npz policies (per timeframe)

models/   supertrend_chronos.joblib + _1min  ema_cross / keltner_adx / bos / orb _chronos.joblib
          ffm_feature_columns.json (FFM feature order)
```

The bot depends only on the public **`futures_foundation`** library (Chronos
embedding + the model head classes + indicator primitives) — no proprietary
code. The joblib bundles run **inference directly**; the FFM feature block is
computed live via `futures_foundation.features.derive_features`.

**Embeddings stay warm.** Chronos runs in a persistent subprocess
(`embed_worker.py`) that loads the model **once per session** — torch isolated
from xgboost, model never reloaded. A grade drops from ~3–4 s (cold reload each
call) to ~0.03 s, so backtests/retrains run in minutes and live signal bars are
near-instant. Falls back to the one-shot library path if the worker can't start.

**Adding a strategy** = one new file in `strategies/` implementing `_fired()` /
`_hand_features()`, plus its joblib model in `models/`, then register it in
`strategies/__init__.py`. The strategy-agnostic exit applies automatically — no
exit work per strategy. Five ship today (`supertrend`, `ema`, `keltner`, `bos`, `orb`).

**Adding a broker** = implement `BrokerClient` (`broker_base.py`) in a new module
and add a case to `broker.make_broker()` + `config.BROKER`. The bar loop, sizing,
and exit all go through that interface, so nothing else changes — e.g. a Rithmic
client would just provide the same account / market-data / order methods.

## Retrain the trailing exit (optional)

```bash
python bot.py --retrain-exit          # full retrain, then exit
python bot.py --retrain-exit --quick  # fast smoke; writes _quick artifacts, never live policy
python -m ppo_exit.train_ppo_exit     # same thing, standalone
```

Catalogs a representative set of entry points in `data/NQ_3min.csv` and simulates
each trade from the live **0.5×ATR(20) stop** with the `ACTIVATE_R`/`GIVEBACK_R`
shaping while the agent learns the trail, then benchmarks vs fixed-RR /
constant-trail baselines and writes the policy into `ppo_exit/policies/`. The
exit is strategy-agnostic (it only sees the trade's R-state), so the same policy
serves every strategy. Training and reusable validation are chronological and
separated by an 80-bar purge, so no training episode can consume validation
prices. The printed table is explicitly **validation**, not final OOS.

The leak-safe default uses every representative flip (`--proba-floor 0`). Do not
filter historical samples with the shipped entry model: that model was fit on the
same history, so its scores are not out-of-fold. The old behavior is available
only behind `--unsafe-final-model-filter`, writes a separate `_unsafe` artifact,
and cannot be certified.

Each new policy gets a hash-bound `.validation.json` sidecar recording its data
hash, split, embargo, filter mode, config, seed, and validation metrics. The bot
refuses a policy without valid purged-training provenance and falls back to fixed
RR. Consequently, the legacy policies shipped before this firewall are disabled
until explicitly retrained with `python bot.py --retrain-exit`.

### Tune the exit config (Optuna)

The exit's behaviour is set by `ACTIVATE_R` / `GIVEBACK_R` / `STOP_ATR` — and the
PPO trail collapses to that give-back cap, so the config *is* the lever. Search it:

```bash
python -m ppo_exit.optimize_exit --timeframe 3 --tickers NQ --trials 200 --save
python -m ppo_exit.optimize_exit --timeframe 1 --tickers NQ ES RTY YM GC --trials 300 --save
```

It replays the exact give-back sim (`TrailExitSim`) per config — no PPO retrain per
trial — scoring expectancy on a purged **validation** slice. Test outcomes are
never read during tuning, and `--save` is controlled only by validation. All
existing pre-cutoff history is treated as development data, not relabeled as a
fresh test. Pool
multiple tickers with `--tickers` for more data. It prints the best
`ACTIVATE_R`/`GIVEBACK_R`/`STOP_ATR`; with `--save` it writes the validation
winner to that timeframe's entry in
`ppo_exit/exit_configs.json`. Then retrain so the policy matches the new shaping:
`python -m ppo_exit.train_ppo_exit --timeframe <tf>`. (1-min CSVs are local-only —
see Backtest.)

Final **exit-policy** certification is a separate, parameter-read-only operation. It requires
fresh data after the cutoff recorded in `ppo_exit/validation_protocol.json`, a
minimum population before outcomes are evaluated, and writes an immutable report
keyed by protocol and timeframe. Keep fresh bars separate from the frozen training
CSV under a directory containing `<symbol>_<timeframe>min.csv`; the certification
loader appends them only in memory for indicator warmup and hashes both inputs.
Only one final look is allowed per protocol; retrying with a shifted date, ticker
subset, or configuration is refused:

```bash
python -m ppo_exit.optimize_exit --timeframe 3 --tickers NQ \
  --certify --test-start 2026-06-05 --fresh-data-dir data/certification
```

Do not run that command until the configuration is frozen and enough untouched
post-cutoff data has accumulated. This certifies the exit policy on representative
mechanical entries; it does not certify the booster-filtered full trading stack.
Protocol 1 has now consumed NQ data through 2026-07-10 and will refuse another
look. Its frozen report is
`ppo_exit/certifications/exit_policy_protocol1_3min.json`: PPO meanR −0.00117,
PF 0.998, sumR −0.38 over 325 entries, exactly matching the deterministic 1×ATR
reference. That is a flat result, not evidence of an exit edge.

**Per-timeframe configs.** `ppo_exit/exit_configs.json` holds the exit shaping
(`ACTIVATE_R`/`GIVEBACK_R`/`STOP_ATR`) **keyed by timeframe** — 1-min and 3-min
bars want different settings (1-min is noisier, so it whipsaws into the stop far
more; a wider `STOP_ATR` helps). The active timeframe's config is applied at
runtime to **both** the live exit and the training sim, so training stays equal to
live. Defaults (no JSON entry) fall back to `config.py`'s `ACTIVATE_R`/`GIVEBACK_R`.

## Tests

```bash
pytest tests/                         # unit + end-to-end, ~1s
git config core.hooksPath .githooks   # once per clone: run tests on every commit
```

The versioned `.githooks/pre-commit` runs the suite before each commit and aborts
if anything fails (`git commit --no-verify` skips it). Enable it once per clone
with the command above.

No network, broker, or Chronos needed — everything runs against the `SimBroker`
and lightweight fakes. Coverage focuses on the order/exit money paths:

- `test_bracket_ticks` — SL/TP ticks signed by direction (the order-rejection fix)
- `test_close_cancels_brackets` / `test_no_orphan_orders` — no orphaned SL **or**
  TP: against a stateful gateway, after a market close, a give-back exit
  (`manage_trail`), or a flat reconcile, zero protective orders are left working
  for the contract (and other contracts are untouched)
- `test_reconcile` — a flat account is swept of stray orders every bar; an
  in-position bar never cancels its live protective stop (mid-session reconcile)
- `test_stop_fill_exit` — a broker-stop fill (in-position→flat with no bot close)
  is logged with its realized R, not silently dropped
- `test_exit_manager` — PPO give-back: activation gate, give-back cap, and the
  intra-bar wick-cross that closes a winner at market instead of riding it back
- `test_train_live_parity` — the trained sim (`TrailExitSim`) and the live exit
  (`manage_trail`) exit on the same bar at the same price (give-back long/short,
  MAX_HOLD timeout), and never on the entry candle
- `test_orb_gate` — ORB only fires during the RTH window (no overnight breakouts)
- `test_indicators` — indicator correctness + strict causality (no look-ahead in
  EMA / Keltner / opening-range / confirmed swings — they feed every feature)
- `test_strategy_triggers` — each strategy's `_fired` fires long/short on the
  right pattern and respects its ADX/trend gate
- `test_detect_signal` — entry/stop/risk math (`risk = STOP_ATR×ATR`, stop on the
  correct side) that drives sizing and the exit
- `test_position_size` — fixed vs risk-based sizing, cap and 1-lot floor
- `test_config_micros` — micro→parent symbol mapping (MNQ→NQ, …)
- `test_broker_contract` — active front-month selection (rollover) and
  working-stop filtering, against a stubbed gateway
- `test_sim_broker` — stop/target/trailing fills, stop-before-target tie, close
- `test_e2e_trade` — full lifecycles through `backtest.drive` → `bot.handle_bar`
  (the real driver): long/short give-back winners, stop-out loser, fixed-RR
  target + stop, the highest-proba resolver, the proba floor, and
  reconstruct-on-restart
- `test_timeframe` — per-timeframe model resolution + gating (only strategies with
  a model for the timeframe are allowed; no cross-timeframe fallback)
- `test_ensure_policy` — auto-train the PPO exit when a timeframe has no policy
  (and fall back to fixed-RR if training can't produce one)
- `test_exit_configs` — per-timeframe exit config load/save (`exit_configs.json`)
- `test_optimize_exit` — the Optuna scanner's metric/split math and that the
  give-back replay is genuinely sensitive to each config knob; validation is
  outcome-horizon purged and certification is one-shot
- `test_research_validation` — full-horizon embargoes, final-model probability
  filter rejection, inspected-holdout boundaries, strict OOS backtest blocking,
  and the fresh-data certification cutoff

## Caveats

- **Scope**: entry models are trained on NQ/ES/RTY/YM/GC 3-min UTC bars and
  generalize across them; other tickers/timeframes are out of distribution until
  retrained.
- **Certification status**: every canonical training bar through 2026-06-04 was
  used or inspected during development. Protocol 1 then consumed NQ through
  2026-07-10 for the exit-policy-only result above. The booster-filtered full
  trading stack remains **uncertified**, and neither the validation nor the flat
  exit certificate demonstrates trading edge. Passing unit tests proves
  implementation invariants, not profitability. A new honest evaluation now
  requires data strictly after 2026-07-10 and a deliberately versioned protocol.
- **Feature fidelity**: 68 of 76 FFM features are computed live; the 8
  session/time columns the current library doesn't emit are left NaN (XGBoost
  handles missing natively) — faithful but not bit-identical to training.
- **PPO exit**: one **strategy-agnostic** policy on the standard 0.5×ATR(20)
  stop — it sees only the trade's R-state, so it applies identically to every
  strategy (it trains on a representative catalog of NQ entry points).
  With `GIVEBACK_R = 0.75` the give-back cap dominates, so the exit is
  effectively a deterministic "trail 0.75R from peak" — loosen it to let
  trend-riding matter more, tighten it for consistency.
- **Exit mode**: default `USE_TRAILING_STOP = False` is the PPO-driven reprice
  (what it's trained for). `True` uses the ProjectX native trailing bracket
  (`type 5`); the PPO can only *tighten* that, so it mostly sits idle.
- **Contract rollover**: live trading always uses the broker's **active front
  month** (`/Contract/search`), and the bot re-resolves it once a day while flat,
  so a long-running session follows the quarterly roll to the new contract — and
  its clean warmup history — without a restart. The API is the source of truth;
  there is no roll calendar to drift. (Backtests use continuous CSV history, so
  they're unaffected.)
- **Online only**: the broker API is the single source of truth for contract
  specs (tick size / value) — there is no hard-coded fallback, so the bot needs
  credentials + connectivity at startup for both live and backtest. (The Chronos
  checkpoint itself is cached after the first download.)
