"""Tests for the Black-Scholes-Merton pricer and implied vol."""

import math

import numpy as np
import pytest

from pricing.black76 import Black76
from pricing.black_scholes import BlackScholesMerton


@pytest.fixture
def model():
    return BlackScholesMerton()


def test_atm_one_year_call_known_value(model):
    """ATM 1y call with S=K=100, r=5%, sigma=20% has a known value."""
    price = model.price(100, 100, 1.0, 0.05, 0.2, "call")
    assert price == pytest.approx(10.4506, abs=1e-3)


def test_atm_one_year_put_known_value(model):
    price = model.price(100, 100, 1.0, 0.05, 0.2, "put")
    assert price == pytest.approx(5.5735, abs=1e-3)


def test_zero_time_returns_intrinsic(model):
    assert model.price(120, 100, 0.0, 0.05, 0.2, "call") == pytest.approx(20.0)
    assert model.price(80, 100, 0.0, 0.05, 0.2, "put") == pytest.approx(20.0)


def test_implied_vol_roundtrip(model):
    """Pricing then inverting recovers the input vol to high precision."""
    true_sigma = 0.27
    price = model.price(100, 110, 0.5, 0.03, true_sigma, "call", q=0.01)
    recovered = model.implied_vol(
        price, 100, 110, 0.5, 0.03, q=0.01, option_type="call"
    )
    assert recovered == pytest.approx(true_sigma, abs=1e-4)


def test_price_surface_shape(model):
    surf = model.price_surface(np.array([90, 100, 110]), np.array([95, 105]), 1.0, 0.05)
    assert surf.shape == (3, 2)
    assert np.all(surf >= 0)


def test_black76_equivalent_to_bsm():
    """Black-76 reproduces BSM when F = S*exp((r-q)T)."""
    res = Black76().compare_to_bsm(100, 100, 1.0, 0.05, 0.2, q=0.02)
    assert res["abs_diff"] < 1e-8


def test_dividend_reduces_call_value(model):
    no_div = model.price(100, 100, 1.0, 0.05, 0.2, "call", q=0.0)
    with_div = model.price(100, 100, 1.0, 0.05, 0.2, "call", q=0.05)
    assert with_div < no_div


# ---------------------------------------------------------------------------
# Edge cases and monotonicity (audit additions)
# ---------------------------------------------------------------------------
def test_sigma_zero_returns_discounted_intrinsic(model):
    """As sigma->0 the call equals max(S*e^{-qT} - K*e^{-rT}, 0)."""
    val = model.price(100, 90, 1.0, 0.05, 0.0, "call", q=0.02)
    import numpy as np

    expected = max(100 * np.exp(-0.02) - 90 * np.exp(-0.05), 0.0)
    assert val == pytest.approx(expected, abs=1e-9)


def test_sigma_zero_atm_zero_rate_is_finite(model):
    """The degenerate 0/0 case (sigma=0, r=q=0, ATM) must not be NaN."""
    val = model.price(100, 100, 1.0, 0.0, 0.0, "call")
    assert math.isfinite(val)
    assert val == pytest.approx(0.0, abs=1e-12)


def test_time_zero_returns_immediate_intrinsic(model):
    assert model.price(115, 100, 0.0, 0.05, 0.2, "call") == pytest.approx(15.0)


def test_rate_monotonicity(model):
    assert model.price(100, 100, 1.0, 0.07, 0.2, "call") > model.price(
        100, 100, 1.0, 0.03, 0.2, "call"
    )
    assert model.price(100, 100, 1.0, 0.07, 0.2, "put") < model.price(
        100, 100, 1.0, 0.03, 0.2, "put"
    )


def test_dividend_monotonicity(model):
    assert model.price(100, 100, 1.0, 0.05, 0.2, "call", q=0.05) < model.price(
        100, 100, 1.0, 0.05, 0.2, "call", q=0.0
    )
    assert model.price(100, 100, 1.0, 0.05, 0.2, "put", q=0.05) > model.price(
        100, 100, 1.0, 0.05, 0.2, "put", q=0.0
    )


def test_implied_vol_below_intrinsic_raises(model):
    """A price below the no-arbitrage bound cannot be inverted."""
    with pytest.raises(ValueError):
        model.implied_vol(0.5, 100, 80, 1.0, 0.05, 0.0, "call")


def test_implied_vol_above_spot_raises(model):
    with pytest.raises(ValueError):
        model.implied_vol(101.0, 100, 80, 1.0, 0.05, 0.0, "call")
