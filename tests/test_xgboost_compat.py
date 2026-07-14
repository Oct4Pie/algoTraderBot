import sys
import types

import pytest

from strategies import base


def test_boosters_reject_xgboost_3_0(monkeypatch):
    monkeypatch.setitem(sys.modules, "xgboost",
                        types.SimpleNamespace(__version__="3.0.5"))
    with pytest.raises(RuntimeError, match="require xgboost==3.3.0"):
        base.require_xgboost_compat()


def test_boosters_reject_xgboost_2(monkeypatch):
    monkeypatch.setitem(sys.modules, "xgboost",
                        types.SimpleNamespace(__version__="2.1.4"))
    with pytest.raises(RuntimeError, match="require xgboost==3.3.0"):
        base.require_xgboost_compat()


@pytest.mark.parametrize("version", ["3.1.2", "3.2.0", "3.3.0"])
def test_boosters_accept_verified_xgboost_line(monkeypatch, version):
    monkeypatch.setitem(sys.modules, "xgboost",
                        types.SimpleNamespace(__version__=version))
    base.require_xgboost_compat()
