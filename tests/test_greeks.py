"""Tests for analytical and numerical Greeks."""

import pytest

from greeks.analytical import BSMGreeks
from greeks.greek_engine import GreekEngine
from greeks.numerical import NumericalGreeks
from pricing.base import OptionContract
from pricing.black_scholes import BlackScholesMerton


@pytest.fixture
def greeks():
    return BSMGreeks()


def test_call_delta_in_unit_interval(greeks):
    g = greeks.compute(100, 100, 1.0, 0.05, 0.2, "call")
    assert 0.0 < g.delta < 1.0


def test_put_delta_in_negative_unit_interval(greeks):
    g = greeks.compute(100, 100, 1.0, 0.05, 0.2, "put")
    assert -1.0 < g.delta < 0.0


def test_gamma_positive_and_equal_for_call_put(greeks):
    call = greeks.compute(100, 100, 1.0, 0.05, 0.2, "call")
    put = greeks.compute(100, 100, 1.0, 0.05, 0.2, "put")
    assert call.gamma > 0
    assert call.gamma == pytest.approx(put.gamma, abs=1e-12)


def test_vega_equal_for_call_and_put(greeks):
    call = greeks.compute(100, 100, 1.0, 0.05, 0.2, "call")
    put = greeks.compute(100, 100, 1.0, 0.05, 0.2, "put")
    assert call.vega == pytest.approx(put.vega, abs=1e-12)
    assert call.vega > 0


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_numerical_matches_analytical(option_type):
    """Finite-difference Greeks agree with closed-form BSM Greeks."""
    args = (100, 100, 1.0, 0.05, 0.2, option_type)
    analytical = BSMGreeks().compute(*args)
    numerical = NumericalGreeks(BlackScholesMerton()).compute(*args)
    assert numerical.delta == pytest.approx(analytical.delta, abs=1e-4)
    assert numerical.gamma == pytest.approx(analytical.gamma, abs=1e-4)
    assert numerical.vega == pytest.approx(analytical.vega, abs=1e-4)
    assert numerical.rho == pytest.approx(analytical.rho, abs=1e-4)
    assert numerical.theta == pytest.approx(analytical.theta, abs=1e-4)


def test_engine_uses_analytical_for_bsm():
    engine = GreekEngine(BlackScholesMerton())
    assert engine.is_analytical is True
    g = engine.compute_all(OptionContract(S=100, K=100, T=1.0, r=0.05, sigma=0.2))
    assert 0 < g.delta < 1


def test_pnl_attribution_total_equals_actual():
    contract = OptionContract(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
    res = GreekEngine().pnl_explain(contract, dS=1.0, dsigma=0.0, dT=0.0, dr=0.0)
    assert res["total_pnl"] == pytest.approx(res["actual_pnl"])
    assert res["explained_pnl"] + res["residual_pnl"] == pytest.approx(
        res["actual_pnl"]
    )
