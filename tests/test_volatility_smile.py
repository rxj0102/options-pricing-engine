"""Tests for volatility smile construction (SVI) and SABR calibration."""

import numpy as np
import pytest

from pricing.black_scholes import BlackScholesMerton
from volatility.sabr import SABRModel, SABRParams
from volatility.smile import VolatilitySmile

S, T, r = 100.0, 1.0, 0.05
FORWARD = S * np.exp(r * T)


def _synthetic_quotes(alpha=0.2, beta=1.0, rho=-0.3, nu=0.5):
    """Build arbitrage-free call quotes from a SABR smile."""
    sabr = SABRModel()
    bsm = BlackScholesMerton()
    quotes = []
    for k in np.linspace(70, 130, 13):
        iv = sabr.implied_vol(FORWARD, k, T, alpha, beta, rho, nu)
        quotes.append((float(k), float(bsm.price(S, k, T, r, iv, "call")), "call"))
    return quotes


@pytest.fixture
def smile():
    return VolatilitySmile(_synthetic_quotes(), S=S, T=T, r=r)


def test_svi_total_variance_nonnegative(smile):
    """Fitted SVI total variance w(k) must be >= 0 across the grid."""
    params = smile.svi_fit()
    k = np.linspace(-0.6, 0.6, 121)
    w = params.a + params.b * (
        params.rho * (k - params.m) + np.sqrt((k - params.m) ** 2 + params.sigma**2)
    )
    assert np.all(w >= -1e-10)


def test_svi_butterfly_convexity(smile):
    """w(k) is convex (d^2w/dk^2 >= 0) for the raw SVI form -> no butterfly."""
    params = smile.svi_fit()
    k = np.linspace(-0.5, 0.5, 201)
    w = params.a + params.b * (
        params.rho * (k - params.m) + np.sqrt((k - params.m) ** 2 + params.sigma**2)
    )
    second_diff = np.diff(w, 2)
    assert np.all(second_diff >= -1e-8)


def test_sabr_atm_recovery():
    """Calibrated SABR reproduces the ATM market vol within 0.1 vol point."""
    quotes = _synthetic_quotes()
    smile = VolatilitySmile(quotes, S=S, T=T, r=r)
    df = smile.extract_ivols()
    atm_market = float(np.interp(FORWARD, df["strike"], df["ivol"]))
    sabr = SABRModel()
    params = sabr.calibrate(
        df["strike"].to_numpy(), df["ivol"].to_numpy(), FORWARD, T, beta=1.0
    )
    atm_fit = sabr.implied_vol(
        FORWARD, FORWARD, T, params.alpha, params.beta, params.rho, params.nu
    )
    assert abs(atm_fit - atm_market) < 0.001


def test_extracted_ivols_match_flat_input():
    """A flat-vol set of quotes inverts back to the same constant vol."""
    bsm = BlackScholesMerton()
    quotes = [
        (k, bsm.price(S, k, T, r, 0.22, "call"), "call")
        for k in (80, 90, 100, 110, 120)
    ]
    df = VolatilitySmile(quotes, S=S, T=T, r=r).extract_ivols()
    assert np.allclose(df["ivol"], 0.22, atol=1e-4)
