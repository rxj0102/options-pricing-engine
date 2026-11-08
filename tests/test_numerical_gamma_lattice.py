import math

import pytest
from scipy.stats import norm

from greeks import NumericalGreeks
from pricing.binomial import CRRBinomial


@pytest.mark.parametrize("K", [90, 100, 110])
def test_lattice_gamma_matches_analytic(K):
    S, T, r, s = 100.0, 1.0, 0.05, 0.2
    d1 = (math.log(S / K) + (r + s * s / 2) * T) / (s * math.sqrt(T))
    exact = norm.pdf(d1) / (S * s * math.sqrt(T))
    est = NumericalGreeks(CRRBinomial(800)).gamma(S, K, T, r, s)
    assert est == pytest.approx(exact, rel=0.05)
