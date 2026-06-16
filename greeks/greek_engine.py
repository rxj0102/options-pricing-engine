"""Unified Greek computation interface across analytical and numerical methods."""

from __future__ import annotations

from typing import Union

import numpy as np

from greeks.analytical import BSMGreeks, GreekSet
from greeks.numerical import NumericalGreeks
from pricing.base import OptionContract, OptionPricer, OptionType
from pricing.black_scholes import BlackScholesMerton


class GreekEngine:
    """Compute Greeks, surfaces and P&L attribution for any pricer.

    When the wrapped pricer is :class:`~pricing.black_scholes.BlackScholesMerton`
    the engine uses closed-form Greeks; otherwise it falls back to
    finite differences.

    Parameters
    ----------
    pricer : OptionPricer, optional
        Pricing model to differentiate. Default
        :class:`~pricing.black_scholes.BlackScholesMerton`.

    Examples
    --------
    >>> from pricing.base import OptionContract
    >>> eng = GreekEngine()
    >>> g = eng.compute_all(OptionContract(S=100, K=100, T=1.0, r=0.05, sigma=0.2))
    >>> 0 < g.delta < 1
    True
    """

    def __init__(self, pricer: OptionPricer | None = None) -> None:
        self.pricer = pricer or BlackScholesMerton()
        self._analytical = BSMGreeks()
        self._numerical = NumericalGreeks(self.pricer)

    @property
    def is_analytical(self) -> bool:
        """Whether closed-form Greeks are used for the wrapped pricer."""
        return isinstance(self.pricer, BlackScholesMerton)

    def compute_all(self, contract: OptionContract) -> GreekSet:
        """Compute the full Greek set for a contract.

        Parameters
        ----------
        contract : OptionContract
            Contract to differentiate.

        Returns
        -------
        GreekSet
            Closed-form Greeks for BSM, otherwise finite-difference Greeks.
        """
        if self.is_analytical:
            return self._analytical.compute(
                contract.S,
                contract.K,
                contract.T,
                contract.r,
                contract.sigma,
                contract.option_type,
                contract.q,
            )
        return self._numerical.compute(
            contract.S,
            contract.K,
            contract.T,
            contract.r,
            contract.sigma,
            contract.option_type,
            contract.q,
        )

    def greek_surface(
        self,
        greek_name: str,
        S_range: np.ndarray,
        sigma_range: np.ndarray,
        K: float,
        T: float,
        r: float,
        q: float = 0.0,
        option_type: Union[OptionType, str] = OptionType.CALL,
    ) -> np.ndarray:
        """Evaluate a single Greek over a spot/vol grid.

        Parameters
        ----------
        greek_name : str
            One of ``delta, gamma, vega, theta, rho, vanna, volga, charm, speed``.
        S_range : numpy.ndarray
            1-D array of spots (rows).
        sigma_range : numpy.ndarray
            1-D array of vols (columns).
        K, T, r, q : float
            Strike, time to expiry, rate, dividend yield.
        option_type : OptionType or str, optional
            Call or put. Default call.

        Returns
        -------
        numpy.ndarray
            2-D array of shape ``(len(S_range), len(sigma_range))``.

        Examples
        --------
        >>> import numpy as np
        >>> eng = GreekEngine()
        >>> surf = eng.greek_surface("delta", np.array([90, 100, 110]),
        ...                          np.array([0.1, 0.2]), 100, 1.0, 0.05)
        >>> surf.shape
        (3, 2)
        """
        option_type = OptionType.coerce(option_type)
        S_range = np.asarray(S_range, dtype=float)
        sigma_range = np.asarray(sigma_range, dtype=float)
        surface = np.empty((S_range.size, sigma_range.size))
        for i, s in enumerate(S_range):
            for jj, vol in enumerate(sigma_range):
                if self.is_analytical:
                    g = self._analytical.compute(s, K, T, r, vol, option_type, q)
                else:
                    g = self._numerical.compute(s, K, T, r, vol, option_type, q)
                surface[i, jj] = getattr(g, greek_name)
        return surface

    def greek_heatmap_data(
        self,
        greek_name: str,
        S_range: np.ndarray,
        sigma_range: np.ndarray,
        K: float,
        T: float,
        r: float,
        q: float = 0.0,
        option_type: Union[OptionType, str] = OptionType.CALL,
    ) -> dict:
        """Package a Greek surface for Streamlit/plotly heatmaps.

        Parameters
        ----------
        greek_name : str
            Greek to evaluate (see :meth:`greek_surface`).
        S_range, sigma_range : numpy.ndarray
            Grid axes.
        K, T, r, q : float
            Strike, time to expiry, rate, dividend yield.
        option_type : OptionType or str, optional
            Call or put. Default call.

        Returns
        -------
        dict
            Keys ``x`` (vols), ``y`` (spots), ``z`` (2-D values), ``greek``.
        """
        z = self.greek_surface(
            greek_name, S_range, sigma_range, K, T, r, q, option_type
        )
        return {
            "x": np.asarray(sigma_range, dtype=float),
            "y": np.asarray(S_range, dtype=float),
            "z": z,
            "greek": greek_name,
        }

    def pnl_explain(
        self,
        contract: OptionContract,
        dS: float,
        dsigma: float,
        dT: float,
        dr: float,
    ) -> dict:
        """First/second-order Taylor P&L attribution plus a residual term.

        Parameters
        ----------
        contract : OptionContract
            Base contract.
        dS : float
            Absolute spot shock.
        dsigma : float
            Absolute volatility shock (e.g. ``0.05`` for +5 vol points).
        dT : float
            Time shock in years (positive means time passing).
        dr : float
            Absolute rate shock.

        Returns
        -------
        dict
            ``delta_pnl, gamma_pnl, vega_pnl, theta_pnl, rho_pnl,
            explained_pnl, actual_pnl, residual_pnl, total_pnl``.

        Examples
        --------
        >>> from pricing.base import OptionContract
        >>> c = OptionContract(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        >>> res = GreekEngine().pnl_explain(c, dS=1.0, dsigma=0.0, dT=0.0, dr=0.0)
        >>> res["total_pnl"] == res["actual_pnl"]
        True
        """
        g = self.compute_all(contract)
        # Vega/rho stored per 1% move; theta is per calendar day.
        delta_pnl = g.delta * dS
        gamma_pnl = 0.5 * g.gamma * dS**2
        vega_pnl = g.vega * (dsigma * 100.0)
        theta_pnl = g.theta * (dT * 365.0)
        rho_pnl = g.rho * (dr * 100.0)
        explained = delta_pnl + gamma_pnl + vega_pnl + theta_pnl + rho_pnl

        base_price = self.pricer.price_contract(contract)
        shocked = contract.with_(
            S=contract.S + dS,
            sigma=max(contract.sigma + dsigma, 1e-6),
            T=max(contract.T - dT, 1e-6),
            r=contract.r + dr,
        )
        actual = self.pricer.price_contract(shocked) - base_price

        return {
            "delta_pnl": float(delta_pnl),
            "gamma_pnl": float(gamma_pnl),
            "vega_pnl": float(vega_pnl),
            "theta_pnl": float(theta_pnl),
            "rho_pnl": float(rho_pnl),
            "explained_pnl": float(explained),
            "actual_pnl": float(actual),
            "residual_pnl": float(actual - explained),
            "total_pnl": float(actual),
        }
