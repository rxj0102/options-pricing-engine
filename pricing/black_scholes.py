"""Black-Scholes-Merton (1973) closed-form European option pricer."""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.optimize import brentq

from pricing.base import OptionPricer, OptionType
from utils.distributions import norm_cdf

# Bounds and tolerances for the implied-volatility root solve.
_IV_LOWER_BOUND: float = 1e-6
_IV_UPPER_BOUND: float = 5.0
_IV_TOLERANCE: float = 1e-8

# Degenerate-input thresholds used to short-circuit d1/d2 evaluation.
_MIN_TIME: float = 1e-10
_MIN_VOL: float = 1e-10
# Tolerance (in price units) for the no-arbitrage bounds in implied_vol.
_ARB_TOLERANCE: float = 1e-8


class BlackScholesMerton(OptionPricer):
    r"""Black-Scholes-Merton model for European options with continuous yield.

    The model assumes the underlying follows geometric Brownian motion with
    constant volatility and a continuous dividend yield ``q``. Prices are

    .. math::

        d_1 &= \frac{\ln(S/K) + (r - q + \tfrac12\sigma^2)T}{\sigma\sqrt{T}}, \\
        d_2 &= d_1 - \sigma\sqrt{T}, \\
        C &= S e^{-qT} N(d_1) - K e^{-rT} N(d_2).

    Examples
    --------
    >>> model = BlackScholesMerton()
    >>> round(model.price(100, 100, 1.0, 0.05, 0.2), 4)
    10.4506
    """

    name = "Black-Scholes-Merton"

    @staticmethod
    def d1_d2(
        S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
    ) -> tuple:
        r"""Compute the auxiliary terms :math:`d_1` and :math:`d_2`.

        Parameters
        ----------
        S, K, T, r, sigma, q : float
            Spot, strike, time to expiry, rate, volatility, dividend yield.

        Returns
        -------
        tuple of (float or numpy.ndarray)
            ``(d1, d2)``.

        Examples
        --------
        >>> d1, d2 = BlackScholesMerton.d1_d2(100, 100, 1.0, 0.05, 0.2)
        >>> round(float(d1), 4), round(float(d2), 4)
        (0.35, 0.15)
        """
        S = np.asarray(S, dtype=float)
        K = np.asarray(K, dtype=float)
        vol_sqrt_t = sigma * np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
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
        """Price a European call or put.

        Parameters
        ----------
        S, K, T, r, sigma : float
            Spot, strike, time to expiry (years), rate, volatility.
        option_type : OptionType or str, optional
            Call or put. Default call.
        q : float, optional
            Continuous dividend yield. Default ``0.0``.

        Returns
        -------
        float
            Present value of the option.

        Examples
        --------
        >>> round(BlackScholesMerton().price(100, 100, 1, 0.05, 0.2, "put"), 4)
        5.5735
        """
        option_type = OptionType.coerce(option_type)
        if T <= _MIN_TIME:
            # At expiry the option is worth its (undiscounted) intrinsic value.
            S_arr, K_arr = float(S), float(K)
            if option_type is OptionType.CALL:
                return max(S_arr - K_arr, 0.0)
            return max(K_arr - S_arr, 0.0)
        if sigma <= _MIN_VOL:
            # With zero volatility the payoff is deterministic: discount the
            # forward intrinsic value. Avoids the 0/0 in d1 when r=q=0 ATM.
            fwd = float(S) * np.exp(-q * T)
            strike_pv = float(K) * np.exp(-r * T)
            if option_type is OptionType.CALL:
                return max(fwd - strike_pv, 0.0)
            return max(strike_pv - fwd, 0.0)

        d1, d2 = self.d1_d2(S, K, T, r, sigma, q)
        disc_q = np.exp(-q * T)
        disc_r = np.exp(-r * T)
        if option_type is OptionType.CALL:
            value = S * disc_q * norm_cdf(d1) - K * disc_r * norm_cdf(d2)
        else:
            value = K * disc_r * norm_cdf(-d2) - S * disc_q * norm_cdf(-d1)
        return float(value)

    def implied_vol(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float = 0.0,
        option_type: Union[OptionType, str] = OptionType.CALL,
    ) -> float:
        """Recover Black-Scholes implied volatility via Brent's method.

        Parameters
        ----------
        market_price : float
            Observed option price to invert.
        S, K, T, r, q : float
            Spot, strike, time to expiry, rate, dividend yield.
        option_type : OptionType or str, optional
            Call or put. Default call.

        Returns
        -------
        float
            Implied volatility, or ``nan`` if the solver fails to bracket a
            root within ``[1e-6, 5.0]`` for an otherwise arbitrage-free price.

        Raises
        ------
        ValueError
            If ``market_price`` violates the no-arbitrage bounds — i.e. it is
            below the discounted intrinsic value or above the maximum
            attainable option value (``S·e^{-qT}`` for a call, ``K·e^{-rT}``
            for a put). Inverting such a price is impossible.

        Examples
        --------
        >>> m = BlackScholesMerton()
        >>> p = m.price(100, 100, 1.0, 0.05, 0.2)
        >>> round(m.implied_vol(p, 100, 100, 1.0, 0.05), 4)
        0.2
        """
        option_type = OptionType.coerce(option_type)
        disc_q = np.exp(-q * T)
        disc_r = np.exp(-r * T)
        if option_type is OptionType.CALL:
            lower_bound = max(S * disc_q - K * disc_r, 0.0)
            upper_bound = S * disc_q
        else:
            lower_bound = max(K * disc_r - S * disc_q, 0.0)
            upper_bound = K * disc_r
        if market_price < lower_bound - _ARB_TOLERANCE:
            raise ValueError(
                f"market_price {market_price:.6g} is below the no-arbitrage "
                f"lower bound {lower_bound:.6g}; no implied vol exists."
            )
        if market_price > upper_bound + _ARB_TOLERANCE:
            raise ValueError(
                f"market_price {market_price:.6g} exceeds the no-arbitrage "
                f"upper bound {upper_bound:.6g}; no implied vol exists."
            )

        def objective(sigma: float) -> float:
            return self.price(S, K, T, r, sigma, option_type, q) - market_price

        try:
            return float(
                brentq(
                    objective,
                    _IV_LOWER_BOUND,
                    _IV_UPPER_BOUND,
                    xtol=_IV_TOLERANCE,
                    maxiter=200,
                )
            )
        except (ValueError, RuntimeError):
            return float("nan")

    def price_surface(
        self,
        S_range: np.ndarray,
        K_range: np.ndarray,
        T: float,
        r: float,
        q: float = 0.0,
        sigma: float = 0.2,
        option_type: Union[OptionType, str] = OptionType.CALL,
    ) -> np.ndarray:
        """Vectorised price surface over a spot/strike grid.

        Parameters
        ----------
        S_range : numpy.ndarray
            1-D array of spot prices (rows of the output).
        K_range : numpy.ndarray
            1-D array of strikes (columns of the output).
        T, r, q, sigma : float
            Time to expiry, rate, dividend yield, volatility.
        option_type : OptionType or str, optional
            Call or put. Default call.

        Returns
        -------
        numpy.ndarray
            2-D array of shape ``(len(S_range), len(K_range))``.

        Examples
        --------
        >>> import numpy as np
        >>> surf = BlackScholesMerton().price_surface(
        ...     np.array([90, 100]), np.array([100]), 1.0, 0.05)
        >>> surf.shape
        (2, 1)
        """
        option_type = OptionType.coerce(option_type)
        S_grid, K_grid = np.meshgrid(
            np.asarray(S_range, dtype=float),
            np.asarray(K_range, dtype=float),
            indexing="ij",
        )
        d1, d2 = self.d1_d2(S_grid, K_grid, T, r, sigma, q)
        disc_q = np.exp(-q * T)
        disc_r = np.exp(-r * T)
        if option_type is OptionType.CALL:
            return S_grid * disc_q * norm_cdf(d1) - K_grid * disc_r * norm_cdf(d2)
        return K_grid * disc_r * norm_cdf(-d2) - S_grid * disc_q * norm_cdf(-d1)
