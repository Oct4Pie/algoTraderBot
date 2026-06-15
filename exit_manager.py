#!/usr/bin/env python3
"""exit_manager.py — PPO trailing-exit management for an open position.

Strategy-agnostic: the per-bar reference line comes from the strategy that
opened the trade (stored in trade_state). The policy picks a trail tightness;
we push it to the broker as a stop reprice or a native-trail tighten.
"""
import numpy as np
import pandas as pd

import config
import indicators as ind
from broker import POSITION_LONG
from logsetup import get_logger

log = get_logger()


def reconstruct_state(client, account_id, contract_id, pos, strategy) -> dict | None:
    """Rebuild trail state if the bot starts up already in a position (e.g.
    after a restart): infer side/entry from the position and risk from the
    distance to the working stop. bars_held resets to 0."""
    entry = pos.get("averagePrice")
    sign = 1 if pos.get("type") == POSITION_LONG else -1
    order = client.working_stop_order(account_id, contract_id)
    if entry is None or order is None or order.get("stopPrice") is None:
        return None
    stop = float(order["stopPrice"])
    risk = (float(entry) - stop) * sign
    if risk <= 0:
        return None
    return {"sign": sign, "entry": float(entry), "risk": risk, "stop": stop,
            "bars_held": 0, "mfe": 0.0, "trail_ticks": None, "strategy": strategy}


def exit_obs(tee, st: dict, bars: pd.DataFrame) -> np.ndarray:
    """Policy observation for the live open position — identical layout to
    TrailExitSim._obs in trail_exit_env.py. Strategy-agnostic: it's purely the
    trade's R-state on the standard 0.5×ATR stop, so one policy fits every
    strategy."""
    sign, entry, risk = st["sign"], st["entry"], st["risk"]
    a = float(ind.atr(bars, tee.ATR_PERIOD)[-1])
    cur = float(bars["close"].iloc[-1])
    prev = float(bars["close"].iloc[-1 - tee.MOM_LOOKBACK])
    unreal = sign * (cur - entry) / risk
    st["mfe"] = max(st["mfe"], unreal)
    obs = np.array([
        unreal,                                   # unrealized R
        st["mfe"],                                # max favorable excursion
        sign * (cur - st["stop"]) / risk,         # stop distance (R)
        a / risk,                                 # volatility vs initial risk
        st["bars_held"] / tee.MAX_HOLD,           # time in trade
        sign * (cur - prev) / risk,               # recent momentum (R)
    ], dtype=np.float32)
    return np.clip(obs, -tee.OBS_CLIP, tee.OBS_CLIP)


def manage_trail(tee, policy, client, account_id, contract_id, tick_size,
                 bars, st: dict, trailing: bool):
    """One bar of exit management: ask the policy how tight to trail, then push
    that to the broker. In `trailing` mode the order natively follows price and
    we only tighten its follow DISTANCE; otherwise we reprice a plain stop. Both
    only ever ratchet in our favor."""
    st["bars_held"] += 1
    sign = st["sign"]
    a = float(ind.atr(bars, tee.ATR_PERIOD)[-1])
    cur = float(bars["close"].iloc[-1])
    stamp = bars["time"].iloc[-1].strftime("%H:%M")

    obs = exit_obs(tee, st, bars)                             # updates st["mfe"]

    # Hold the initial stop until the trade's peak reaches ACTIVATE_R.
    if st["mfe"] < config.ACTIVATE_R:
        log.info("   %s  armed — peak %.2fR < %.1fR, holding initial stop",
                 stamp, st["mfe"], config.ACTIVATE_R)
        return

    mult = policy.trail_mult(obs)
    trail_dist = mult * a                                     # price distance
    # give-back cap: never let the stop sit > GIVEBACK_R below the running peak
    giveback = config.GIVEBACK_R * st["risk"]
    order = client.working_stop_order(account_id, contract_id)
    if order is None:
        log.warning("   %s  no working stop found — skip trail", stamp)
        return

    if trailing:
        # Native trailing stop: only ever tighten the follow distance, and cap
        # the follow distance at the give-back limit.
        new_ticks = max(1, round(min(trail_dist, giveback) / tick_size))
        cur_ticks = st.get("trail_ticks") or new_ticks
        if new_ticks < cur_ticks:
            # trailPrice is a decimal price distance, not a tick count
            client.modify_trail_price(account_id, order["id"],
                                      new_ticks * tick_size)
            st["trail_ticks"] = new_ticks
            log.info("   %s  trail tightened → %dt (%.2fx ATR, %d bars in)",
                     stamp, new_ticks, mult, st["bars_held"])
        else:
            log.info("   %s  hold trail %dt (%.2fx ATR)", stamp, cur_ticks, mult)
        # keep a stop estimate for the observation (broker trails from best price)
        best = st["entry"] + sign * st["mfe"] * st["risk"]
        est = best - sign * (st.get("trail_ticks") or new_ticks) * tick_size
        st["stop"] = max(st["stop"], est) if sign > 0 else min(st["stop"], est)
    else:
        # Plain stop: PPO trail level, but never looser than the give-back cap
        # (peak − GIVEBACK_R), then favorable ratchet only.
        peak = st["entry"] + sign * st["mfe"] * st["risk"]
        cap = peak - sign * giveback
        cand = cur - sign * trail_dist
        tightest = max(cand, cap) if sign > 0 else min(cand, cap)
        new_stop = max(st["stop"], tightest) if sign > 0 else min(st["stop"], tightest)
        if abs(new_stop - st["stop"]) >= tick_size:
            new_stop = round(new_stop / tick_size) * tick_size
            client.modify_stop_price(account_id, order["id"], new_stop)
            st["stop"] = new_stop
            log.info("   %s  trail → stop %.2f (%.2fx ATR, %d bars in)",
                     stamp, new_stop, mult, st["bars_held"])
        else:
            log.info("   %s  hold (stop %.2f, %.2fx ATR)", stamp, st["stop"], mult)
