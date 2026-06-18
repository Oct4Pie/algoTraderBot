"""End-to-end trade lifecycle through the REAL bot.handle_bar + PPO exit, driven
by the SimBroker — no network, no broker, no Chronos. A fake strategy supplies one
long signal and a fake policy supplies the trail tightness, so this exercises the
full flow: detect → grade → size → enter → manage_trail → give-back exit.

It proves the fix end-to-end: a +2R winner is protected to a profitable exit
instead of riding back to −1R."""
import types

import pandas as pd

import bot
import config
import trail_exit_env as tee
from broker_base import SIDE
from sim_broker import SimBroker

TICK = 0.25


class _FakePolicy:
    def trail_mult(self, obs):
        return 1.0


class _OneShotLong:
    """Fires a single long signal the first time it's asked while flat."""
    name = "fake"

    def __init__(self):
        self.fired = False

    def detect(self, bars):
        if self.fired or len(bars) < 16:
            return None
        self.fired = True
        entry = float(bars["close"].iloc[-1])
        risk = 10.0                                  # 40 ticks @ 0.25
        return types.SimpleNamespace(direction=1, entry=entry, stop=entry - risk,
                                     risk=risk, bar_index=len(bars) - 1,
                                     bar_time=bars["time"].iloc[-1])

    def grade(self, bars, sig, emb=None):
        return 0.90, 5.0                             # proba well above the floor


def _price_path():
    """15 flat warmup bars, entry ~bar 15 @ 100, rally to +2R (120), then reverse
    down so the give-back trail closes the winner."""
    closes = ([100.0] * 16
              + [104.0, 108.0, 112.0, 116.0, 120.0]      # rally to +2R
              + [116.0, 110.0, 104.0, 100.0, 98.0])      # reversal
    rows = []
    for i, c in enumerate(closes):
        hi = c + (2.0 if 16 <= i <= 20 else 1.0)         # highs lead on the way up
        lo = c - 2.0
        rows.append((hi, lo, c))
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01 09:30", periods=len(rows), freq="3min", tz="UTC"),
        "open": [c for _, _, c in rows],
        "high": [h for h, _, _ in rows],
        "low": [l for _, l, _ in rows],
        "close": [c for _, _, c in rows],
        "volume": [1] * len(rows),
    })


def test_full_lifecycle_protects_a_winner(monkeypatch):
    config.RISK_PER_TRADE = 0.0          # fixed size
    config.SIZE = 1
    config.MAX_CONTRACTS = 5
    config.PROBA_FLOOR = 0.35
    config.ACTIVATE_R = 2.0
    config.GIVEBACK_R = 0.75
    config.USE_PPO_EXIT = True
    monkeypatch.setattr(bot.strat, "embed_context", lambda bars, i: None)

    df = _price_path()
    sim = SimBroker(df, TICK)
    ctx = types.SimpleNamespace(
        client=sim, account_id=1, contract_id="NQ", tick_size=TICK,
        tick_value=5.0, log_candles=False, policy=_FakePolicy(), tee=tee,
        strategies=[_OneShotLong()], trailing=False,
    )

    trade_state = None
    for i in range(len(df)):
        sim.set_bar(i)
        sim.process_exits()                          # broker exits first (like --backtest)
        sim.tag_strategy("fake")
        trade_state = bot.handle_bar(ctx, df.iloc[: i + 1], trade_state)
    sim.close_open()                                 # flatten anything still open

    assert len(sim.trades) == 1, "exactly one trade over the path"
    t = sim.trades[0]
    assert t.direction == 1
    assert t.r >= 1.0, f"give-back should protect the winner, got {t.r:.2f}R"
    assert sim.pos is None
