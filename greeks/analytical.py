"""Closed-form Black-Scholes-Merton Greeks, including higher-order sensitivities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Union

import numpy as np

from pricing.base import DAYS_PER_YEAR, OptionType
from pricing.black_scholes import BlackScholesMerton
from utils.distributions import norm_cdf, norm_pdf

#: Greeks are quoted per 1% move in vol/rate where noted.
PCT_MOVE: float = 100.0


@dataclass
class GreekSet:
    """Full set of analytical Greeks for a single contract.

    Attributes
    ----------
    delta : float
        Sensitivity to spot, :math:`\\partial V/\\partial S`.
    gamma : float
        Curvature in spot, :math:`\\partial^2 V/\\partial S^2`.
    vega : float
        Sensitivity to a 1% move in volatility.
    theta : float
        Daily time decay (per calendar day).
    rho : float
        Sensitivity to a 1% move in the rate.
    vanna : float
        :math:`\\partial\\Delta/\\partial\\sigma`.
    volga : float
        :math:`\\partial\\nu/\\partial\\sigma` (a.k.a. vomma).
    charm : float
        :math:`\\partial\\Delta/\\partial T` (delta bleed, per calendar day).
    speed : float
        :math:`\\partial\\Gamma/\\partial S` (third-order spot sensitivity).
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    vanna: float
    volga: float
    charm: float
    speed: float

    def as_dict(self) -> dict:
        """Return the Greeks as a plain dictionary.

        Returns
        -------
        dict
            Mapping of Greek name to value.
        """
        return asdict(self)


class BSMGreeks:
    """Analytical Greek calculator for the Black-Scholes-Merton model.

    Examples
    --------
    >>> g = BSMGreeks().compute(100, 100, 1.0, 0.05, 0.2, "call")
    >>> 0 < g.delta < 1 and g.gamma > 0
    True
    """

    def __init__(self) -> None:
        self._model = BlackScholesMerton()

    def compute(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Union[OptionType, str] = OptionType.CALL,
        q: float = 0.0,
    ) -> GreekSet:
        """Compute the full :class:`GreekSet` analytically.

        Parameters
        ----------
        S, K, T, r, sigma, q : float
            Spot, strike, time to expiry, rate, volatility, dividend yield.
        option_type : OptionType or str, optional
            Call or put. Default call.

        Returns
        -------
        GreekSet
            All first-, second- and selected third-order Greeks.

        Examples
        --------
        >>> g = BSMGreeks().compute(100, 100, 1.0, 0.05, 0.2, "put")
        >>> -1 < g.delta < 0
        True
        """
        option_type = OptionType.coerce(option_type)
        d1, d2 = self._model.d1_d2(S, K, T, r, sigma, q)
        d1 = float(d1)
        d2 = float(d2)
        sqrt_t = np.sqrt(T)
        disc_q = np.exp(-q * T)
        disc_r = np.exp(-r * T)
        pdf_d1 = float(norm_pdf(d1))
        is_call = option_type is OptionType.CALL

        # First-order
        if is_call:
            delta = disc_q * norm_cdf(d1)
        else:
            delta = -disc_q * norm_cdf(-d1)

        gamma = disc_q * pdf_d1 / (S * sigma * sqrt_t)
        vega = S * disc_q * pdf_d1 * sqrt_t / PCT_MOVE

        if is_call:
            theta_annual = (
                -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrt_t)
                - r * K * disc_r * norm_cdf(d2)
                + q * S * disc_q * norm_cdf(d1)
            )
            rho = K * T * disc_r * norm_cdf(d2) / PCT_MOVE
        else:
            theta_annual = (
                -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrt_t)
                + r * K * disc_r * norm_cdf(-d2)
                - q * S * disc_q * norm_cdf(-d1)
            )
            rho = -K * T * disc_r * norm_cdf(-d2) / PCT_MOVE

        theta = theta_annual / DAYS_PER_YEAR

        # Higher-order
        vanna = -disc_q * pdf_d1 * d2 / sigma
        volga = vega * d1 * d2 / sigma
        if is_call:
            charm_annual = -disc_q * (
                pdf_d1
                * (2.0 * (r - q) * T - d2 * sigma * sqrt_t)
                / (2.0 * T * sigma * sqrt_t)
                - q * norm_cdf(d1)
            )
        else:
            charm_annual = -disc_q * (
                pdf_d1
                * (2.0 * (r - q) * T - d2 * sigma * sqrt_t)
                / (2.0 * T * sigma * sqrt_t)
                + q * norm_cdf(-d1)
            )
        charm = charm_annual / DAYS_PER_YEAR
        speed = -gamma / S * (d1 / (sigma * sqrt_t) + 1.0)

        return GreekSet(
            delta=float(delta),
            gamma=float(gamma),
            vega=float(vega),
            theta=float(theta),
            rho=float(rho),
            vanna=float(vanna),
            volga=float(volga),
            charm=float(charm),
            speed=float(speed),
        )
