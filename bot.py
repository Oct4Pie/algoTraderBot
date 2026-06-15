#!/usr/bin/env python3
"""
bot.py — multi-strategy TopstepX AI bot with a PPO trailing exit.

Each bar:  every active strategy detects its mechanical entry  →  grades it with
its own Chronos+XGBoost model  →  the best graded signal (proba ≥ floor) is
taken  →  the PPO policy trails the stop until exit.

    detect (SuperTrend flip / EMA cross)  →  model grades  →  enter  →  PPO trail

Strategies and exit behaviour are configured in config.py. Run:

    pip install -r requirements.txt
    python bot.py

⚠️  EDUCATIONAL — places LIVE orders. Run it on a practice/evaluation account
    first. NQ 3-min only (the models' training scope).
"""
import os
import time

import config
import exit_manager as ex
import strategies as strat
from broker import SIDE, TopstepXClient
from logsetup import get_logger

log = get_logger()


def _creds(name, default):
    return os.environ.get(name, default)


def run():
    import trail_exit_env as tee     # numpy-only PPO policy loader

    strategies = strat.make_strategies()
    policy = None
    if config.USE_PPO_EXIT and os.path.exists(config.POLICY_PATH):
        policy = tee.NumpyMlpPolicy.load(config.POLICY_PATH)
    trailing = bool(policy) and config.USE_TRAILING_STOP
    exit_mode = ("PPO native-trail" if trailing else
                 "PPO stop-reprice" if policy else f"fixed {config.RR}R")

    client = TopstepXClient(
        _creds("TOPSTEPX_USERNAME", config.TOPSTEPX_USERNAME),
        _creds("TOPSTEPX_API_KEY", config.TOPSTEPX_API_KEY),
    )
    client.authenticate()
    acct = client.pick_account(_creds("TOPSTEPX_ACCOUNT", config.ACCOUNT))
    contract = client.get_active_contract(config.SYMBOL)
    account_id, contract_id = acct["id"], contract["id"]
    tick_size = float(contract["tickSize"])
    names = "+".join(s.name for s in strategies)
    log.info("✅ %s | %s | %d-min | [%s] | exit: %s", acct["name"], contract_id,
             config.TIMEFRAME_MIN, names, exit_mode)
    log.info("▶ running — Ctrl-C to stop")

    trade_state = None                # tracks the live trade for the trail

    while True:
        # wait for the next bar close (+2s so the API has published it)
        period = config.TIMEFRAME_MIN * 60
        time.sleep(period - (time.time() % period) + 2)

        try:
            bars = client.get_bars(contract_id, config.TIMEFRAME_MIN)
            if len(bars) < config.CTX + 30:    # need >=128 closes + warmup
                continue
            stamp = bars["time"].iloc[-1].strftime("%H:%M")
            last = bars.iloc[-1]
            log.info("candle %s  O=%.2f H=%.2f L=%.2f C=%.2f V=%s", stamp,
                     last["open"], last["high"], last["low"], last["close"],
                     last.get("volume", "?"))
            pos = client.open_position(account_id, contract_id)

            if pos:
                # In a trade — let the PPO policy trail the stop. Without a
                # policy the attached fixed bracket manages the exit itself.
                if policy is None:
                    continue
                if trade_state is None:
                    trade_state = ex.reconstruct_state(
                        client, account_id, contract_id, pos, strategies[0])
                if trade_state:
                    line = trade_state["strategy"].reference_line(bars)
                    ex.manage_trail(tee, policy, client, account_id, contract_id,
                                    tick_size, bars, line, trade_state, trailing)
                continue

            trade_state = None         # flat — clear any stale trail state

            # Detect + grade across every active strategy; keep entries the
            # bot would take, then pick the highest-proba one.
            candidates = []
            for s in strategies:
                sig = s.detect(bars)
                if sig is None:
                    continue
                sig.proba, sig.r_hat = s.grade(bars, sig)
                side_txt = "LONG" if sig.direction > 0 else "SHORT"
                take = sig.proba >= config.PROBA_FLOOR
                log.info("signal [%s] %s | proba=%.3f r_hat=%.2f | %s",
                         s.name, side_txt, sig.proba, sig.r_hat,
                         "TAKE" if take else f"skip (<{config.PROBA_FLOOR})")
                if take:
                    candidates.append((s, sig))

            if not candidates:
                continue
            s, sig = max(candidates, key=lambda c: c[1].proba)   # highest proba wins

            stop_ticks = max(1, round(sig.risk / tick_size))
            side = SIDE["BUY"] if sig.direction > 0 else SIDE["SELL"]
            side_txt = "LONG" if sig.direction > 0 else "SHORT"

            if policy is not None:
                trade_state = {"sign": sig.direction, "entry": sig.entry,
                               "risk": sig.risk, "stop": sig.stop, "bars_held": 0,
                               "mfe": 0.0, "trail_ticks": stop_ticks, "strategy": s}
                if trailing:
                    client.place_market_with_trail(
                        account_id, contract_id, side=side, size=config.SIZE,
                        trail_ticks=stop_ticks)
                    log.info("🎯 ENTER %s [%s] %d | native trail %dt | PPO (proba %.3f)",
                             side_txt, s.name, config.SIZE, stop_ticks, sig.proba)
                else:
                    client.place_market_with_stop(
                        account_id, contract_id, side=side, size=config.SIZE,
                        stop_ticks=stop_ticks)
                    log.info("🎯 ENTER %s [%s] %d | stop %dt | PPO reprice (proba %.3f)",
                             side_txt, s.name, config.SIZE, stop_ticks, sig.proba)
            else:
                target_ticks = max(1, round(config.RR * sig.risk / tick_size))
                client.place_market_with_brackets(
                    account_id, contract_id, side=side, size=config.SIZE,
                    stop_ticks=stop_ticks, target_ticks=target_ticks)
                log.info("🎯 ENTER %s [%s] %d | stop %dt | target %dt (%sR)",
                         side_txt, s.name, config.SIZE, stop_ticks, target_ticks, config.RR)

        except Exception as e:        # keep the loop alive on transient errors
            log.warning("⚠️  %s", e)


def _retrain_exit(quick: bool, timesteps: int):
    """Retrain the PPO trailing-exit policy (delegates to train_ppo_exit)."""
    import sys
    import train_ppo_exit
    sys.argv = ["train_ppo_exit.py"] + (
        ["--quick"] if quick else ["--timesteps", str(timesteps)])
    log.info("retraining PPO exit (%s)…", "quick" if quick else f"{timesteps} steps")
    train_ppo_exit.main()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="multi-strategy AI futures bot")
    ap.add_argument("--retrain-exit", action="store_true",
                    help="retrain the PPO trailing-exit policy, then exit "
                         "(writes models/rl_trail_exit/)")
    ap.add_argument("--quick", action="store_true",
                    help="with --retrain-exit: fast smoke train")
    ap.add_argument("--timesteps", type=int, default=600_000,
                    help="with --retrain-exit: PPO training steps")
    args = ap.parse_args()

    if args.retrain_exit:
        _retrain_exit(args.quick, args.timesteps)
        raise SystemExit(0)

    if config.TOPSTEPX_USERNAME == "your_login_here" \
            and not os.environ.get("TOPSTEPX_USERNAME"):
        raise SystemExit("set TOPSTEPX_USERNAME / TOPSTEPX_API_KEY in config.py "
                         "or as environment variables first")
    run()
