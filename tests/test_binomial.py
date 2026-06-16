"""Tests for the binomial and trinomial tree pricers."""

import pytest

from pricing.binomial import CRRBinomial
from pricing.black_scholes import BlackScholesMerton
from pricing.trinomial import BoyleTrinomial


@pytest.fixture
def bsm():
    return BlackScholesMerton()


def test_binomial_converges_to_bsm(bsm):
    """Binomial error decreases as the number of steps grows."""
    target = bsm.price(100, 100, 1.0, 0.05, 0.2, "call")
    err_coarse = abs(CRRBinomial(25).price(100, 100, 1.0, 0.05, 0.2, "call") - target)
    err_fine = abs(CRRBinomial(500).price(100, 100, 1.0, 0.05, 0.2, "call") - target)
    assert err_fine < err_coarse
    assert err_fine < 1e-2


def test_trinomial_converges_to_bsm(bsm):
    target = bsm.price(100, 100, 1.0, 0.05, 0.2, "call")
    price = BoyleTrinomial(300).price(100, 100, 1.0, 0.05, 0.2, "call")
    assert price == pytest.approx(target, abs=1e-2)


def test_american_put_exceeds_european_put():
    """Early-exercise premium on an American put is strictly positive."""
    crr = CRRBinomial(300)
    american = crr.price(100, 100, 1.0, 0.05, 0.2, "put", exercise_style="american")
    european = crr.price(100, 100, 1.0, 0.05, 0.2, "put", exercise_style="european")
    assert american > european


def test_american_call_no_dividend_equals_european():
    """Without dividends an American call should not be exercised early."""
    crr = CRRBinomial(300)
    american = crr.price(
        100, 100, 1.0, 0.05, 0.2, "call", q=0.0, exercise_style="american"
    )
    european = crr.price(
        100, 100, 1.0, 0.05, 0.2, "call", q=0.0, exercise_style="european"
    )
    assert american == pytest.approx(european, abs=1e-6)


def test_price_with_tree_shapes():
    price, stock, opt = CRRBinomial(50).price_with_tree(100, 100, 1.0, 0.05, 0.2)
    assert stock.shape == (51, 51)
    assert opt.shape == (51, 51)
    assert price > 0


def test_convergence_helper_lengths():
    steps, prices = CRRBinomial().convergence(100, 100, 1.0, 0.05, 0.2)
    assert len(steps) == len(prices)
    assert all(p > 0 for p in prices)
