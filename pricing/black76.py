"""Black (1976) model for options on futures and forwards."""

from __future__ import annotations

from typing import Union

import numpy as np

from pricing.base import OptionPricer, OptionType
from pricing.black_scholes import BlackScholesMerton
from utils.distributions import norm_cdf


class Black76(OptionPricer):
    r"""Black-76 model for European options on a forward/futures price.

    The underlying is the forward price :math:`F`, so no dividend yield is
    required. Prices are

    .. math::

        d_1 &= \frac{\ln(F/K) + \tfrac12\sigma^2 T}{\sigma\sqrt{T}}, \quad
        d_2 = d_1 - \sigma\sqrt{T}, \\
        C &= e^{-rT}\bigl(F N(d_1) - K N(d_2)\bigr).

    Typical applications include interest-rate caps/floors, commodity
    options, and swaptions.

    Examples
    --------
    >>> model = Black76()
    >>> round(model.price(100, 100, 1.0, 0.05, 0.2), 4)
    7.5771
    """

    name = "Black-76"

    @staticmethod
    def d1_d2(F: float, K: float, T: float, sigma: float) -> tuple:
        r"""Compute :math:`d_1` and :math:`d_2` for the forward measure.

        Parameters
        ----------
        F, K, T, sigma : float
            Forward price, strike, time to expiry, volatility.

        Returns
        -------
        tuple of (float or numpy.ndarray)
            ``(d1, d2)``.

        Examples
        --------
        >>> d1, d2 = Black76.d1_d2(100, 100, 1.0, 0.2)
        >>> round(float(d1), 4), round(float(d2), 4)
        (0.1, -0.1)
        """
        F = np.asarray(F, dtype=float)
        K = np.asarray(K, dtype=float)
        vol_sqrt_t = sigma * np.sqrt(T)
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
        return d1, d2

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Union[OptionType, str] = OptionType.CALL,
        q: float = 0.0,
    ) -> float:
        """Price a European option on a forward price.

        Parameters
        ----------
        S : float
            The **forward price** ``F`` of the underlying (named ``S`` to
            satisfy the common :class:`~pricing.base.OptionPricer` interface).
        K, T, r, sigma : float
            Strike, time to expiry, discount rate, volatility.
        option_type : OptionType or str, optional
            Call or put. Default call.
        q : float, optional
            Unused for Black-76 (kept for interface compatibility).

        Returns
        -------
        float
            Present value of the option.

        Examples
        --------
        >>> round(Black76().price(100, 95, 1.0, 0.05, 0.2, "put"), 4)
        5.2503
        """
        option_type = OptionType.coerce(option_type)
        F = S
        if T <= 0:
            if option_type is OptionType.CALL:
                return max(float(F) - float(K), 0.0)
            return max(float(K) - float(F), 0.0)

        d1, d2 = self.d1_d2(F, K, T, sigma)
        disc = np.exp(-r * T)
        if option_type is OptionType.CALL:
            value = disc * (F * norm_cdf(d1) - K * norm_cdf(d2))
        else:
            value = disc * (K * norm_cdf(-d2) - F * norm_cdf(-d1))
        return float(value)

    def price_from_spot(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Union[OptionType, str] = OptionType.CALL,
        q: float = 0.0,
    ) -> float:
        """Price using a spot price by first computing the forward.

        The forward is :math:`F = S e^{(r - q)T}`.

        Parameters
        ----------
        S, K, T, r, sigma, q : float
            Spot, strike, time to expiry, rate, volatility, dividend yield.
        option_type : OptionType or str, optional
            Call or put. Default call.

        Returns
        -------
        float
            Present value of the option.

        Examples
        --------
        >>> b76 = Black76()
        >>> round(b76.price_from_spot(100, 100, 1.0, 0.05, 0.2), 4)
        10.4506
        """
        forward = S * np.exp((r - q) * T)
        return self.price(forward, K, T, r, sigma, option_type)

    def compare_to_bsm(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: Union[OptionType, str] = OptionType.CALL,
    ) -> dict:
        r"""Demonstrate Black-76 / BSM equivalence when :math:`F = S e^{(r-q)T}`.

        Parameters
        ----------
        S, K, T, r, sigma, q : float
            Spot, strike, time to expiry, rate, volatility, dividend yield.
        option_type : OptionType or str, optional
            Call or put. Default call.

        Returns
        -------
        dict
            Keys ``forward``, ``black76_price``, ``bsm_price``, and
            ``abs_diff``. The two prices coincide to numerical precision.

        Examples
        --------
        >>> res = Black76().compare_to_bsm(100, 100, 1.0, 0.05, 0.2)
        >>> res["abs_diff"] < 1e-8
        True
        """
        forward = S * np.exp((r - q) * T)
        b76_price = self.price(forward, K, T, r, sigma, option_type)
        bsm_price = BlackScholesMerton().price(S, K, T, r, sigma, option_type, q)
        return {
            "forward": float(forward),
            "black76_price": b76_price,
            "bsm_price": bsm_price,
            "abs_diff": abs(b76_price - bsm_price),
        }
