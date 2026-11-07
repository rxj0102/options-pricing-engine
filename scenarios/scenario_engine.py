"""Unified scenario runner combining stress tests and Greek attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pricing.base import OptionContract, OptionPricer
from scenarios.greek_pnl import GreekPnLAttribution
from scenarios.stress_test import StressTester


class ScenarioEngine:
    """One entry point for stress grids, named scenarios and attribution.

    Parameters
    ----------
    pricer : OptionPricer, optional
        Model used throughout. Default Black-Scholes-Merton.

    Examples
    --------
    >>> from pricing.base import OptionContract
    >>> c = OptionContract(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
    >>> eng = ScenarioEngine()
    >>> report = eng.full_report(c)
    >>> set(["scenarios", "spot_vol_grid", "attribution"]) <= set(report)
    True
    """

    def __init__(self, pricer: OptionPricer | None = None) -> None:
        self.stress = StressTester(pricer)
        self.attribution = GreekPnLAttribution(pricer)

    def heatmap_data(
        self,
        contract: OptionContract,
        spot_pct_range=np.linspace(-0.3, 0.3, 13),
        vol_pct_range=np.linspace(-0.5, 0.5, 11),
    ) -> dict:
        """Build a spot/vol P&L heatmap payload for plotly.

        Parameters
        ----------
        contract : OptionContract
            Base contract.
        spot_pct_range : array-like, optional
            Fractional spot shocks (y-axis).
        vol_pct_range : array-like, optional
            Fractional vol shocks (x-axis).

        Returns
        -------
        dict
            Keys ``x`` (vol shocks), ``y`` (spot shocks), ``z`` (2-D P&L).
        """
        grid = self.stress.spot_vol_grid(
            contract, list(spot_pct_range), list(vol_pct_range)
        )
        return {
            "x": np.asarray(vol_pct_range, dtype=float),
            "y": np.asarray(spot_pct_range, dtype=float),
            "z": grid.to_numpy(),
        }

    def full_report(
        self,
        contract: OptionContract,
        dS: float = 0.05,
        dsigma: float = 0.05,
        dT: float = 1.0 / 365.0,
        dr: float = 0.0,
    ) -> dict:
        """Run the standard scenario suite for a contract.

        Parameters
        ----------
        contract : OptionContract
            Base contract.
        dS : float, optional
            Fractional spot shock for the attribution leg. Default ``0.05``.
        dsigma : float, optional
            Absolute vol shock for attribution. Default ``0.05``.
        dT : float, optional
            Time shock in years for attribution. Default one day.
        dr : float, optional
            Absolute rate shock for attribution. Default ``0.0``.

        Returns
        -------
        dict
            ``scenarios`` (DataFrame), ``spot_vol_grid`` (DataFrame),
            ``attribution`` (DataFrame waterfall).
        """
        return {
            "scenarios": self.stress.scenario_pnl(contract),
            "spot_vol_grid": self.stress.spot_vol_grid(
                contract,
                list(np.linspace(-0.2, 0.2, 9)),
                list(np.linspace(-0.3, 0.3, 7)),
            ),
            "attribution": self.attribution.waterfall_data(
                contract, dS=contract.S * dS, dsigma=dsigma, dT=dT, dr=dr
            ),
        }
