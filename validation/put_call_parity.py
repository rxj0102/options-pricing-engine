"""Put-call parity verification and implied-forward extraction."""

from __future__ import annotations

import numpy as np

#: A parity break beyond this many basis points is flagged as a failure.
DEFAULT_TOLERANCE_BPS: float = 1.0
BPS_PER_UNIT: float = 1e4


class PutCallParityCheck:
    r"""Verify European put-call parity for a quoted call/put pair.

    Parity states

    .. math::

        C - P = S e^{-qT} - K e^{-rT}.

    Examples
    --------
    >>> from pricing.black_scholes import BlackScholesMerton
    >>> m = BlackScholesMerton()
    >>> c = m.price(100, 100, 1.0, 0.05, 0.2, "call")
    >>> p = m.price(100, 100, 1.0, 0.05, 0.2, "put")
    >>> PutCallParityCheck().check(c, p, 100, 100, 1.0, 0.05)["passes"]
    True
    """

    def check(
        self,
        call_price: float,
        put_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float = 0.0,
        tolerance_bps: float = DEFAULT_TOLERANCE_BPS,
    ) -> dict:
        """Check parity and report the violation in basis points.

        Parameters
        ----------
        call_price, put_price : float
            Quoted call and put prices.
        S, K, T, r, q : float
            Spot, strike, time to expiry, rate, dividend yield.
        tolerance_bps : float, optional
            Pass/fail threshold in basis points (relative to spot).
            Default ``1.0``.

        Returns
        -------
        dict
            ``theoretical_diff``, ``actual_diff``, ``violation``,
            ``violation_bps``, ``passes``.

        Examples
        --------
        >>> res = PutCallParityCheck().check(10.0, 10.0, 100, 100, 1.0, 0.0)
        >>> res["theoretical_diff"]
        0.0
        """
        theoretical = S * np.exp(-q * T) - K * np.exp(-r * T)
        actual = call_price - put_price
        violation = actual - theoretical
        violation_bps = abs(violation) / S * BPS_PER_UNIT
        return {
            "theoretical_diff": float(theoretical),
            "actual_diff": float(actual),
            "violation": float(violation),
            "violation_bps": float(violation_bps),
            "passes": bool(violation_bps <= tolerance_bps),
        }

    def implied_forward(
        self,
        call_price: float,
        put_price: float,
        K: float,
        T: float,
        r: float,
    ) -> float:
        r"""Recover the implied forward from a call/put pair.

        From parity, :math:`F = (C - P) e^{rT} + K`.

        Parameters
        ----------
        call_price, put_price : float
            Quoted call and put prices.
        K, T, r : float
            Strike, time to expiry, rate.

        Returns
        -------
        float
            Implied forward price of the underlying.

        Examples
        --------
        >>> from pricing.black_scholes import BlackScholesMerton
        >>> m = BlackScholesMerton()
        >>> c = m.price(100, 100, 1.0, 0.05, 0.2, "call")
        >>> p = m.price(100, 100, 1.0, 0.05, 0.2, "put")
        >>> round(PutCallParityCheck().implied_forward(c, p, 100, 1.0, 0.05), 2)
        105.13
        """
        return float((call_price - put_price) * np.exp(r * T) + K)
