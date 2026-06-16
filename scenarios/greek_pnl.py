"""Greek-based P&L attribution with waterfall-ready output."""

from __future__ import annotations

from typing import List

import pandas as pd

from greeks.greek_engine import GreekEngine
from pricing.base import OptionContract, OptionPricer

#: Conversions: vega/rho are stored per 1% move; theta per calendar day.
PCT_MOVE: float = 100.0
DAYS_PER_YEAR: int = 365


class GreekPnLAttribution:
    """Decompose option P&L into Greek contributions plus a residual.

    Parameters
    ----------
    pricer : OptionPricer, optional
        Pricing model used for both Greeks and the realised reprice.
        Default Black-Scholes-Merton via :class:`GreekEngine`.

    Examples
    --------
    >>> from pricing.base import OptionContract
    >>> c = OptionContract(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
    >>> att = GreekPnLAttribution().attribute(c, dS=2.0, dsigma=0.01,
    ...                                       dT=1/365, dr=0.0)
    >>> set(["delta_pnl", "gamma_pnl", "residual_pnl", "total_pnl"]) <= set(att)
    True
    """

    def __init__(self, pricer: OptionPricer | None = None) -> None:
        self.engine = GreekEngine(pricer)

    def attribute(
        self,
        contract: OptionContract,
        dS: float,
        dsigma: float,
        dT: float,
        dr: float,
    ) -> dict:
        """Attribute realised P&L to first/second-order Greeks.

        Parameters
        ----------
        contract : OptionContract
            Base contract.
        dS : float
            Absolute spot shock.
        dsigma : float
            Absolute volatility shock.
        dT : float
            Time shock in years (positive = time elapsed).
        dr : float
            Absolute rate shock.

        Returns
        -------
        dict
            ``delta_pnl, gamma_pnl, vega_pnl, theta_pnl, rho_pnl,
            residual_pnl, total_pnl``.
        """
        return self.engine.pnl_explain(contract, dS, dsigma, dT, dr)

    def waterfall_data(
        self,
        contract: OptionContract,
        dS: float,
        dsigma: float,
        dT: float,
        dr: float,
    ) -> pd.DataFrame:
        """Build a waterfall table summing components to the total P&L.

        Parameters
        ----------
        contract : OptionContract
            Base contract.
        dS, dsigma, dT, dr : float
            Shocks as in :meth:`attribute`.

        Returns
        -------
        pandas.DataFrame
            Columns ``component``, ``pnl``, ``measure`` (``relative`` for
            building blocks, ``total`` for the final bar) — ready for a
            plotly waterfall.

        Examples
        --------
        >>> from pricing.base import OptionContract
        >>> c = OptionContract(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        >>> df = GreekPnLAttribution().waterfall_data(c, 2.0, 0.01, 1/365, 0.0)
        >>> df["component"].tolist()[-1]
        'Total'
        """
        att = self.attribute(contract, dS, dsigma, dT, dr)
        components: List[dict] = [
            {"component": "Delta", "pnl": att["delta_pnl"], "measure": "relative"},
            {"component": "Gamma", "pnl": att["gamma_pnl"], "measure": "relative"},
            {"component": "Vega", "pnl": att["vega_pnl"], "measure": "relative"},
            {"component": "Theta", "pnl": att["theta_pnl"], "measure": "relative"},
            {"component": "Rho", "pnl": att["rho_pnl"], "measure": "relative"},
            {
                "component": "Residual",
                "pnl": att["residual_pnl"],
                "measure": "relative",
            },
            {"component": "Total", "pnl": att["total_pnl"], "measure": "total"},
        ]
        return pd.DataFrame(components)
