import numpy as np

from evaluate_ffm_power import barrier_outcome


def test_barrier_outcome_stop_wins_same_bar_tie():
    high = np.array([100.0, 104.0])
    low = np.array([100.0, 98.0])
    close = np.array([100.0, 102.0])
    realized, target = barrier_outcome(
        high, low, close, entry_idx=0, entry=100.0, risk=1.0,
        direction=1, horizon=1)
    assert realized == -1.0
    assert not target


def test_barrier_outcome_target_and_vertical_mark():
    high = np.array([100.0, 101.0, 103.0])
    low = np.array([100.0, 99.5, 100.0])
    close = np.array([100.0, 100.5, 102.0])
    realized, target = barrier_outcome(
        high, low, close, entry_idx=0, entry=100.0, risk=1.0,
        direction=1, horizon=2)
    assert realized == 3.0 and target

    realized, target = barrier_outcome(
        high, low, close, entry_idx=0, entry=100.0, risk=2.0,
        direction=1, horizon=1)
    assert realized == 0.25 and not target


def test_barrier_outcome_can_include_execution_bar_for_cisd():
    high = np.array([103.0, 100.0])
    low = np.array([99.5, 100.0])
    close = np.array([102.0, 100.0])
    realized, target = barrier_outcome(
        high, low, close, entry_idx=0, entry=100.0, risk=1.0,
        direction=1, horizon=1, include_entry_bar=True)
    assert realized == 3.0 and target
