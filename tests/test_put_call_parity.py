"""Put-call parity tests across all four pricing models."""

import numpy as np
import pytest

from pricing.binomial import CRRBinomial
from pricing.black76 import Black76
from pricing.black_scholes import BlackScholesMerton
from pricing.trinomial import BoyleTrinomial
from validation.put_call_parity import PutCallParityCheck

S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.2


def _parity_holds(call, put):
    res = PutCallParityCheck().check(call, put, S, K, T, r, q)
    return res["passes"]


def test_parity_black_scholes():
    m = BlackScholesMerton()
    call = m.price(S, K, T, r, sigma, "call", q)
    put = m.price(S, K, T, r, sigma, "put", q)
    assert _parity_holds(call, put)


def test_parity_black76():
    forward = S * np.exp((r - q) * T)
    m = Black76()
    call = m.price(forward, K, T, r, sigma, "call")
    put = m.price(forward, K, T, r, sigma, "put")
    assert _parity_holds(call, put)


def test_parity_binomial_european():
    m = CRRBinomial(500)
    call = m.price(S, K, T, r, sigma, "call", q, "european")
    put = m.price(S, K, T, r, sigma, "put", q, "european")
    # Trees carry small discretisation error; allow a looser bps tolerance.
    res = PutCallParityCheck().check(call, put, S, K, T, r, q, tolerance_bps=5.0)
    assert res["passes"]


def test_parity_trinomial_european():
    m = BoyleTrinomial(300)
    call = m.price(S, K, T, r, sigma, "call", q, "european")
    put = m.price(S, K, T, r, sigma, "put", q, "european")
    res = PutCallParityCheck().check(call, put, S, K, T, r, q, tolerance_bps=5.0)
    assert res["passes"]


def test_implied_forward_recovery():
    m = BlackScholesMerton()
    call = m.price(S, K, T, r, sigma, "call", q)
    put = m.price(S, K, T, r, sigma, "put", q)
    fwd = PutCallParityCheck().implied_forward(call, put, K, T, r)
    assert fwd == pytest.approx(S * np.exp((r - q) * T), abs=1e-6)


# ---------------------------------------------------------------------------
# Tight-tolerance and American parity-break checks (audit additions)
# ---------------------------------------------------------------------------
def test_parity_bsm_within_1e8():
    m = BlackScholesMerton()
    c = m.price(S, K, T, r, sigma, "call", q)
    p = m.price(S, K, T, r, sigma, "put", q)
    lhs = c - p
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert abs(lhs - rhs) < 1e-8


def test_parity_black76_within_1e8():
    forward = S * np.exp((r - q) * T)
    m = Black76()
    c = m.price(forward, K, T, r, sigma, "call")
    p = m.price(forward, K, T, r, sigma, "put")
    assert abs((c - p) - np.exp(-r * T) * (forward - K)) < 1e-8


def test_american_options_break_european_parity():
    """American calls/puts need NOT satisfy European put-call parity."""
    crr = CRRBinomial(400)
    # Use dividends so the American call also carries early-exercise value.
    c = crr.price(S, K, T, r, sigma, "call", 0.06, "american")
    p = crr.price(S, K, T, r, sigma, "put", 0.06, "american")
    european_diff = S * np.exp(-0.06 * T) - K * np.exp(-r * T)
    assert abs((c - p) - european_diff) > 1e-3
