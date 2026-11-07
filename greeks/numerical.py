"""Finite-difference Greeks for any pricer implementing the base interface."""

from __future__ import annotations

from typing import Callable, Union

from greeks.analytical import GreekSet
from pricing.base import DAYS_PER_YEAR, OptionPricer, OptionType

# Bump sizes for finite differences.
REL_SPOT_BUMP: float = 0.01
VOL_BUMP: float = 0.001
TIME_BUMP: float = 1.0 / DAYS_PER_YEAR
RATE_BUMP: float = 0.0001


class NumericalGreeks:
    """Finite-difference Greeks usable with tree models or any pricer.

    Parameters
    ----------
    pricer : OptionPricer
        Any model exposing ``price(S, K, T, r, sigma, option_type, q)``.

    Examples
    --------
    >>> from pricing.binomial import CRRBinomial
    >>> ng = NumericalGreeks(CRRBinomial(n_steps=300))
    >>> g = ng.compute(100, 100, 1.0, 0.05, 0.2, "call")
    >>> 0 < g.delta < 1
    True
    """

    def __init__(self, pricer: OptionPricer) -> None:
        self.pricer = pricer

    def _price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        q: float,
    ) -> float:
        return self.pricer.price(S, K, T, r, sigma, option_type, q)

    def delta(self, S, K, T, r, sigma, option_type=OptionType.CALL, q=0.0) -> float:
        """Central-difference delta, ``(V(S+dS) - V(S-dS)) / (2 dS)``.

        Parameters
        ----------
        S, K, T, r, sigma, q : float
            Standard pricing inputs.
        option_type : OptionType or str, optional
            Call or put. Default call.

        Returns
        -------
        float
            Estimated delta.
        """
        option_type = OptionType.coerce(option_type)
        dS = REL_SPOT_BUMP * S
        up = self._price(S + dS, K, T, r, sigma, option_type, q)
        dn = self._price(S - dS, K, T, r, sigma, option_type, q)
        return (up - dn) / (2.0 * dS)

    def gamma(self, S, K, T, r, sigma, option_type=OptionType.CALL, q=0.0) -> float:
        """Central-difference gamma, ``(V+ - 2V + V-) / dS^2``.

        Parameters
        ----------
        S, K, T, r, sigma, q : float
            Standard pricing inputs.
        option_type : OptionType or str, optional
            Call or put. Default call.

        Returns
        -------
        float
            Estimated gamma.
        """
        option_type = OptionType.coerce(option_type)
        dS = REL_SPOT_BUMP * S
        up = self._price(S + dS, K, T, r, sigma, option_type, q)
        mid = self._price(S, K, T, r, sigma, option_type, q)
        dn = self._price(S - dS, K, T, r, sigma, option_type, q)
        return (up - 2.0 * mid + dn) / (dS**2)

    def vega(self, S, K, T, r, sigma, option_type=OptionType.CALL, q=0.0) -> float:
        """Central-difference vega per 1% vol move.

        Parameters
        ----------
        S, K, T, r, sigma, q : float
            Standard pricing inputs.
        option_type : OptionType or str, optional
            Call or put. Default call.

        Returns
        -------
        float
            Estimated vega (per 1% vol move, matching :class:`BSMGreeks`).
        """
        option_type = OptionType.coerce(option_type)
        up = self._price(S, K, T, r, sigma + VOL_BUMP, option_type, q)
        dn = self._price(S, K, T, r, sigma - VOL_BUMP, option_type, q)
        return (up - dn) / (2.0 * VOL_BUMP) / 100.0

    def theta(self, S, K, T, r, sigma, option_type=OptionType.CALL, q=0.0) -> float:
        """Forward-difference daily theta, ``(V(T-dt) - V(T)) / dt`` per day.

        Parameters
        ----------
        S, K, T, r, sigma, q : float
            Standard pricing inputs.
        option_type : OptionType or str, optional
            Call or put. Default call.

        Returns
        -------
        float
            Estimated daily theta.
        """
        option_type = OptionType.coerce(option_type)
        base = self._price(S, K, T, r, sigma, option_type, q)
        bumped = self._price(S, K, T - TIME_BUMP, r, sigma, option_type, q)
        return bumped - base

    def rho(self, S, K, T, r, sigma, option_type=OptionType.CALL, q=0.0) -> float:
        """Central-difference rho per 1% rate move.

        Parameters
        ----------
        S, K, T, r, sigma, q : float
            Standard pricing inputs.
        option_type : OptionType or str, optional
            Call or put. Default call.

        Returns
        -------
        float
            Estimated rho (per 1% rate move).
        """
        option_type = OptionType.coerce(option_type)
        up = self._price(S, K, T, r + RATE_BUMP, sigma, option_type, q)
        dn = self._price(S, K, T, r - RATE_BUMP, sigma, option_type, q)
        return (up - dn) / (2.0 * RATE_BUMP) / 100.0

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
        """Compute the standard Greeks numerically as a :class:`GreekSet`.

        Higher-order Greeks (vanna, volga, charm, speed) are returned as
        ``nan`` since they are not produced by the finite-difference scheme.

        Parameters
        ----------
        S, K, T, r, sigma, q : float
            Standard pricing inputs.
        option_type : OptionType or str, optional
            Call or put. Default call.

        Returns
        -------
        GreekSet
            Delta, gamma, vega, theta and rho populated numerically.

        Examples
        --------
        >>> from pricing.black_scholes import BlackScholesMerton
        >>> g = NumericalGreeks(BlackScholesMerton()).compute(
        ...     100, 100, 1.0, 0.05, 0.2)
        >>> round(g.delta, 2)
        0.64
        """
        option_type = OptionType.coerce(option_type)
        nan = float("nan")
        return GreekSet(
            delta=self.delta(S, K, T, r, sigma, option_type, q),
            gamma=self.gamma(S, K, T, r, sigma, option_type, q),
            vega=self.vega(S, K, T, r, sigma, option_type, q),
            theta=self.theta(S, K, T, r, sigma, option_type, q),
            rho=self.rho(S, K, T, r, sigma, option_type, q),
            vanna=nan,
            volga=nan,
            charm=nan,
            speed=nan,
        )
